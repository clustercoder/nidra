"""The primary benchmark: natural-prevalence, leakage-resistant evaluation of
NIDRA against its ablations and the classifier baselines, on one split.

    # select the operating point on validation (writes operating_point.json)
    python -m nidra.eval.benchmark --config config/default.yaml --split val --select-operating-point
    # then freeze it and evaluate
    python -m nidra.eval.benchmark --config config/default.yaml --split test
    python -m nidra.eval.benchmark --config config/default.yaml --split holdout

Everything decision-like (pooling statistic, calibration, threshold) comes
from the operating-point file written on validation; a test/holdout run
without one uses the config pooling and the mandated threshold and says so.
The historical sampled-prevalence numbers are produced by run_eval.py and
are reported separately; the two must never be conflated.

Outputs <metrics_dir>/<split>/benchmark.json with: provenance record, the
evaluation-set composition, every system's metrics on the three task labels
(published risk label, detection, onset-within-H forecasting), per-horizon
progression metrics, state-forecast skill, the world-model attribution
(paired bootstrap differences), per-episode warning/latency, calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nidra.data.dataset import build_windowed_arrays
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import STAGE_LABELS
from nidra.eval.episode_metrics import per_episode_report
from nidra.eval.eval_set import EvalSet, build_eval_set
from nidra.eval.metrics import brier_score, reliability_diagram
from nidra.eval.metrics_natural import bootstrap_ap, bootstrap_ap_difference, summarize_scores, weighted_ap
from nidra.eval.operating_point import (
    OPERATING_POINT_FILENAME,
    apply_operating_point,
    load_operating_point,
    pooling_key,
    save_operating_point,
    select_operating_point,
)
from nidra.eval.run_eval import _build_model, resolve_pooling
from nidra.eval.state_metrics import state_forecast_report
from nidra.eval.systems import (
    ScoreBundle,
    fit_classifier_baselines,
    fit_ridge_two_lag,
    score_classifier_baselines,
    score_ridge,
    score_world_model,
    system_scores,
)
from nidra.models.world_model import WorldModel
from nidra.train.pipeline import build_all_splits, geometry_from_config, scale_arrays
from nidra.utils.config import load_config, resolve_path
from nidra.utils.provenance import experiment_record, write_json

logger = logging.getLogger(__name__)

ATTRIBUTION_PAIRS = [
    ("world_model", "persistence"),
    ("world_model", "persistence_rollout"),
    ("world_model", "noised_persistence"),
    ("world_model", "isotropic_noise_persistence"),
    ("world_model", "world_model_deterministic"),
    ("world_model", "ridge_two_lag"),
    ("world_model_deterministic", "persistence"),
    ("world_model_deterministic", "ridge_two_lag"),
    ("noised_persistence", "persistence"),
    ("world_model", "lr_current_state"),
    ("world_model", "lr_flattened_history"),
    ("world_model", "gbdt_current_state"),
    ("world_model", "gru_classifier"),
]
BOOTSTRAP_SYSTEMS = ("world_model", "world_model_deterministic", "persistence", "persistence_rollout", "noised_persistence",
                     "ridge_two_lag", "lr_current_state", "lr_flattened_history", "gbdt_current_state", "gru_classifier")


PER_GROUP_SYSTEMS = ("world_model_calibrated", "world_model", "persistence", "persistence_rollout", "oracle_true_future",
                    "gru_classifier", "lr_flattened_history")


def _per_attack_group(ev: EvalSet, scores: dict[str, np.ndarray], y: np.ndarray, threshold: float,
                      bundle, X_scaled: np.ndarray, Y_scaled: np.ndarray, model_mask: np.ndarray,
                      day_meta: dict | None = None) -> dict[str, Any]:
    """One-vs-rest breakdown by the kind of attack a positive row forecasts.

    An aggregate AP over a split that concatenates several captures hides
    exactly what Run 8 ran into: 833 of Friday's 946 positives were Bot-C2 on
    five workstations, and the aggregate number was theirs. Each group here
    keeps every negative in the split and only that group's positives, so the
    numbers are comparable to each other and to the aggregate, and it is
    visible which attack the system can and cannot see.

    Reported per group, because the four answers need different fixes:
    the oracle AP (head cannot recognise this attack even on the true future),
    the world-model AP (forecast quality on top of that), the state forecast
    error (the transition model on these rows), and the mean predicted
    log-variance (what the model thinks it knows).
    """
    groups = ev.attack_group
    if groups is None:
        return {}
    negatives = y == 0
    out: dict[str, Any] = {}
    for name in sorted({g for g in groups if g}):
        in_group = (groups == name) & (y == 1)
        rows = np.where(in_group | negatives)[0]
        yy = y[rows]
        if yy.sum() == 0:
            continue
        w = ev.weight[rows]
        entry: dict[str, Any] = {
            "n_positive_rows": int(in_group.sum()),
            "n_rows": int(len(rows)),
            "prevalence_natural": float(np.average(yy, weights=w)),
            "n_episodes": int(len({k for k in ev.episode_key[in_group] if k})),
            "n_hosts": int(len(set(ev.arrays.host_id[in_group]))),
            "systems": {},
        }
        if day_meta:
            capture = name.split(":")[0]
            meta = day_meta.get(capture, {})
            entry["capture"] = capture
            entry["family"] = meta.get("family")
            entry["dataset"] = meta.get("dataset")
        for system in PER_GROUP_SYSTEMS:
            if system in scores:
                entry["systems"][system] = summarize_scores(yy, scores[system][rows], w, threshold)
        # Where the failure is: the transition model on this group's rows.
        pos_rows = np.where(in_group)[0]
        if len(pos_rows) and bundle.states_det is not None:
            err = ((bundle.states_det[pos_rows] - Y_scaled[pos_rows]) ** 2)[..., model_mask].mean()
            persist = ((X_scaled[pos_rows, -1:, :] - Y_scaled[pos_rows]) ** 2)[..., model_mask].mean()
            entry["state_mse_predicted"] = float(err)
            entry["state_mse_persistence"] = float(persist)
            entry["state_skill_vs_persistence"] = float(1.0 - err / persist) if persist > 0 else float("nan")
        if len(pos_rows) and bundle.logvar_mean is not None:
            entry["mean_predicted_logvar"] = float(bundle.logvar_mean[pos_rows].mean())
        out[name] = entry
    return out


def _host_identity(ev: EvalSet, scores: dict[str, np.ndarray], y: np.ndarray) -> dict[str, Any]:
    """How much of each system's ranking is "which host" rather than "which
    window"?

    Both corpora put every attack group's positives on a single host, so an
    aggregate AP cannot separate recognising a family's behaviour from
    recognising the machine it ran on. Collapsing every score to its host's
    mean keeps only identity; restricting to the hosts that carry positives
    keeps only timing. A system whose host-mean ROC is ~1.0 and whose
    within-host ROC is ~0.5 has learned the host.

    Reported per system rather than computed afterwards because the scores,
    the labels and the host ids are all in hand here and nowhere else.
    Within-host ROC is prevalence-independent and so comparable across splits
    and datasets; the lift is not, and the base rate is reported beside it.
    """
    from nidra.scripts.silent_positive_audit import floor_stratum_probe

    out: dict[str, Any] = {}
    every_row = np.ones(len(y), dtype=bool)
    for name, s in scores.items():
        try:
            p = floor_stratum_probe(s, y, ev.arrays.host_id, every_row)
        except Exception as exc:
            # A reporting block, not a result. It is about to run unattended
            # over the whole cross-dataset matrix, and a diagnostic that can
            # fail an evaluation run is worse than a missing column.
            logger.warning("host/timing decomposition skipped for %s: %s", name, exc)
            continue
        out[name] = {"ap": p.ap, "roc": p.roc, "host_mean_ap": p.host_mean_ap, "host_mean_roc": p.host_mean_roc,
                     "n_positive_hosts": p.n_positive_hosts, "within_host_rows": p.within_host_rows,
                     "within_host_prevalence": p.within_host_prevalence, "within_host_ap": p.within_host_ap,
                     "within_host_lift": p.within_host_lift, "within_host_roc": p.within_host_roc,
                     "is_host_identity": p.is_host_identity}
    return out


def _with_forced_pooling(op: dict, key: str) -> dict:
    """A copy of the operating point reading a different pooling rule.

    Each run selects its own pooling on its own validation, which is the
    protocol and the right unit for "what would you deploy". It is the wrong
    unit for "does adding CTU help the dynamics", because a row-to-row
    comparison then differs in the readout as well as the training set. This
    forces a common readout so that question can be asked; the threshold and
    calibration still come from the source run's selection, which is why the
    result is a controlled comparison and NOT a deployable configuration.

    `key` is a `pooling_key` string: "<method>|q=<quantile>|<horizon_reduction>".
    """
    parts = key.split("|")
    if len(parts) != 3:
        raise ValueError(f"force_pooling {key!r} is not '<method>|q=<quantile>|<horizon_reduction>'")
    method, q, horizon = parts
    quantile = None if q in ("q=-", "-") else float(q.removeprefix("q="))
    pooling = {"method": method, "quantile": quantile, "horizon_reduction": horizon}
    if pooling_key(pooling) != key:
        raise ValueError(f"force_pooling {key!r} did not round-trip to {pooling_key(pooling)!r}")
    return {**op, "pooling": pooling, "pooling_key": key,
            "pooling_key_selected": op.get("pooling_key"), "pooling_forced": True}


def assert_scaler_matches_checkpoint(scaler_dropped: list[str], meta: dict | None, checkpoint: str) -> None:
    """Refuse to evaluate a model with a scaler fitted under a different feature
    regime.

    A dropped feature is ZEROED, not removed, so the shapes always match and
    nothing raises — the model simply receives a different input space than it
    was trained on and produces a plausible wrong number. This was a near-miss:
    a probe resolved `config/combined_eval_ctu.yaml` without the queue's
    `--set artifacts.scaler_dir=...` and got the shared `artifacts/scaler`,
    which a later run had rewritten at the full-45 regime against a model
    trained at cross_core's 32. The cell's AP came out 2.9x off.

    Checkpoints written before `dropped_features` existed are passed through:
    refusing them would break replaying Run 8's artifacts, which §29 forbids.
    """
    if not meta or "dropped_features" not in meta:
        return
    trained, loaded = set(meta["dropped_features"]), set(scaler_dropped)
    if trained == loaded:
        return
    only_trained, only_loaded = sorted(trained - loaded), sorted(loaded - trained)
    raise ValueError(
        f"scaler/checkpoint feature-regime mismatch for {checkpoint}: the model was trained with "
        f"{len(trained)} dropped feature(s) and this scaler drops {len(loaded)}. "
        f"Trained-but-not-dropped-now: {only_trained or 'none'}. "
        f"Dropped-now-but-not-in-training: {only_loaded or 'none'}. "
        "A dropped feature is zeroed rather than removed, so this would not have raised on its own — "
        "point artifacts.scaler_dir at the scaler this checkpoint was trained with.")


def load_models(cfg: dict, seeds: list[int]) -> list[WorldModel]:
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    dropped = FeatureScaler.load(*FeatureScaler.default_paths(scaler_dir)).dropped_features
    models = []
    for s in seeds:
        meta_path = weights_dir / f"model_seed_{s}_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        assert_scaler_matches_checkpoint(dropped, meta, f"model_seed_{s}.pt in {weights_dir}")
        m = _build_model(cfg)
        m.load_state_dict(torch.load(weights_dir / f"model_seed_{s}.pt", map_location="cpu"))
        m.eval()
        models.append(m)
    return models


def _task_table(ev: EvalSet, scores: dict[str, np.ndarray], y: np.ndarray, threshold: float, mask: np.ndarray | None,
                clusters_bootstrap: bool, n_resamples: int, span_hours: float | None) -> dict[str, Any]:
    rows = np.arange(len(ev)) if mask is None else np.where(mask)[0]
    w = ev.weight[rows]
    yy = y[rows]
    active_benign = (ev.stratum[rows] == "active_negative") | (ev.stratum[rows] == "pre_onset")
    active_benign &= (yy == 0)
    out: dict[str, Any] = {"n_rows": int(len(rows)), "n_pos": int(yy.sum()),
                           "prevalence_natural": float(np.average(yy, weights=w)) if len(rows) else float("nan"),
                           "systems": {}}
    for name, s in scores.items():
        ss = s[rows]
        entry = summarize_scores(yy, ss, w, threshold, active_benign_mask=active_benign, span_hours=span_hours)
        if clusters_bootstrap and name in BOOTSTRAP_SYSTEMS and yy.sum() > 0:
            entry["auc_pr_bootstrap"] = bootstrap_ap(yy, ss, w, ev.cluster[rows], n_resamples=n_resamples)
        out["systems"][name] = entry
    return out


def _score_onset_heads(cfg: dict, seeds: list[int], s_t_scaled: np.ndarray,
                       models: list[WorldModel] | None = None,
                       X_scaled: np.ndarray | None = None) -> dict[int, dict[str, np.ndarray]]:
    """Explicit onset supervision (train/train_onset.py) scored on the
    observed state: {horizon_min: {"onset_head_direct": [N]}} averaged over
    the seeds whose head exists. Empty when none was trained."""
    from nidra.train.train_onset import onset_head_path, score_onset_head
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    probs, horizons = [], None
    for i, seed in enumerate(seeds):
        path = onset_head_path(weights_dir, seed)
        if path.exists():
            p, horizons = score_onset_head(path, s_t_scaled,
                                           _onset_context(path, models, i, X_scaled))
            probs.append(p)
    if not probs:
        return {}
    mean = np.mean(np.stack(probs, 0), axis=0)
    return {int(h): {"onset_head_direct": mean[:, j]} for j, h in enumerate(horizons)}


@torch.no_grad()
def _onset_context(path, models, i: int, X_scaled: np.ndarray | None) -> dict | None:
    """The observed context for a history-aware onset head, from the SAME
    ensemble member whose head this is — member j's encoder state is not an
    input member i's head was ever trained against."""
    from nidra.models.heads import OnsetHead
    head = OnsetHead.load(path)
    if head.components == ("state",):
        return None
    if not models or X_scaled is None:
        raise ValueError(f"{path.name} declares components {head.components} but no model was available "
                         "to rebuild the encoder context")
    m = models[min(i, len(models) - 1)]
    return m.observed_context(torch.from_numpy(X_scaled).float())


def _attribution(ev: EvalSet, scores: dict[str, np.ndarray], y: np.ndarray, n_resamples: int) -> dict[str, Any]:
    out = {}
    for a, b in ATTRIBUTION_PAIRS:
        if a in scores and b in scores:
            out[f"{a} - {b}"] = bootstrap_ap_difference(y, scores[a], scores[b], ev.weight, ev.cluster, n_resamples=n_resamples)
    return out


def _progression(ev: EvalSet, bundle: ScoreBundle, risk_k: np.ndarray, threshold: float) -> dict[str, Any]:
    """Task C: per-horizon risk vs future_is_attack[k]; predicted stage at
    t+k vs the true stage where the future is an attack."""
    K = risk_k.shape[1]
    fut = ev.arrays.future_is_attack
    per_k = []
    for k in range(K):
        entry = {"k": k + 1, "minutes_ahead": (k + 1) * ev.window_seconds / 60.0,
                 "auc_pr_natural": weighted_ap(fut[:, k], risk_k[:, k], ev.weight),
                 "base_rate_natural": float(np.average(fut[:, k], weights=ev.weight)),
                 "brier": brier_score(fut[:, k], np.clip(risk_k[:, k], 0, 1)),
                 "reliability": reliability_diagram(fut[:, k], np.clip(risk_k[:, k], 0, 1), n_bins=10)}
        if bundle.risk_det_k is not None:
            entry["auc_pr_natural_deterministic"] = weighted_ap(fut[:, k], bundle.risk_det_k[:, k], ev.weight)
        if bundle.risk_true_k is not None:
            entry["auc_pr_natural_oracle"] = weighted_ap(fut[:, k], bundle.risk_true_k[:, k], ev.weight)
        if bundle.stage_traj_mean_k is not None:
            attack_rows = fut[:, k] == 1
            if attack_rows.any():
                pred_stage = bundle.stage_traj_mean_k[attack_rows, k, :].argmax(axis=1)
                true_stage = ev.arrays.future_stage_idx[attack_rows, k]
                entry["stage_top1_accuracy_on_attack_futures"] = float((pred_stage == true_stage).mean())
                non_benign_pred = bundle.stage_traj_mean_k[attack_rows, k, 1:].argmax(axis=1) + 1
                entry["stage_top1_accuracy_excluding_benign"] = float((non_benign_pred == true_stage).mean())
                entry["n_attack_futures"] = int(attack_rows.sum())
                entry["predicted_stage_histogram"] = {STAGE_LABELS[i]: int((pred_stage == i).sum()) for i in range(len(STAGE_LABELS))}
        per_k.append(entry)
    return {"per_horizon": per_k}


def _calibration(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    from nidra.eval.metrics_natural import brier
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
    rel = []
    for b in range(10):
        m = bin_idx == b
        rel.append({"bin_center": float((bins[b] + bins[b + 1]) / 2), "n": int(m.sum()),
                    "weighted_n": float(w[m].sum()),
                    "observed_frequency_natural": float(np.average(y[m], weights=w[m])) if m.any() else None})
    return {"brier_natural": brier(y, s, w), "reliability": rel}


def run(cfg: dict, split: str, seeds: list[int], n_samples: int, out_dir: Path, operating_point_path: Path | None,
        select_op: bool, caps: dict | None, n_resamples: int, chunk: int, gru_classifier: Path | None,
        with_ablations: bool = True, eval_seed: int = 0, force_pooling: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    window_seconds, L, K = geometry_from_config(cfg)
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(scaler_dir))
    models = load_models(cfg, seeds)
    eval_cfg = cfg.get("eval", {})
    threshold_mandated = float(eval_cfg.get("risk_threshold", 0.75))
    merge_gap = int(eval_cfg.get("episode_merge_gap_windows", 5))
    set_cfg = eval_cfg.get("benchmark_set", {})
    caps = {**{"active_negative": set_cfg.get("active_negative_cap", 6000),
               "silent_negative": set_cfg.get("silent_negative_cap", 2000)}, **(caps or {})}

    splits = build_all_splits(cfg)
    table = getattr(splits, split)
    ev = build_eval_set(table, L, K, window_seconds, split, caps=caps, seed=eval_seed, merge_gap_windows=merge_gap,
                        pre_onset_minutes=int(set_cfg.get("pre_onset_minutes", 30)))
    logger.info("eval set %s: %s (%.0fs)", split, json.dumps(ev.summary()["strata"]), time.time() - t0)
    X_scaled, Y_scaled = scale_arrays(ev.arrays, scaler)

    # baselines fit on the training split's stratified sample (the head-training sample at seed 0)
    cap = cfg.get("training_data", {}).get("max_train_samples")
    train = build_windowed_arrays(splits.train, L=L, K=K, max_samples=cap, seed=0)
    X_train, Y_train = scale_arrays(train, scaler)
    logger.info("train sample for baselines: %d rows, %d positives (%.0fs)", len(train.X), int(train.risk_label.sum()), time.time() - t0)
    ridges = fit_ridge_two_lag(X_train, Y_train, K)
    fitted = fit_classifier_baselines(train, X_train, seed=0)
    del X_train, Y_train, train

    bundle = ScoreBundle(K=K)
    bundle = score_world_model(bundle, models, X_scaled, Y_scaled, n_samples, chunk=chunk, with_ablations=with_ablations, seed=eval_seed)
    bundle = score_ridge(bundle, models, ridges, X_scaled, clip=float(cfg["model"]["transition"]["state_clamp"]))
    bundle = score_classifier_baselines(bundle, fitted, X_scaled)
    if gru_classifier is not None and Path(gru_classifier).exists():
        from nidra.eval.gru_classifier import load_and_score
        bundle.baselines["gru_classifier"] = load_and_score(gru_classifier, X_scaled)
    onset_direct = _score_onset_heads(cfg, seeds, X_scaled[:, -1, :], models, X_scaled)
    logger.info("scoring done (%.0fs)", time.time() - t0)

    # operating point
    if select_op:
        if split != "val":
            raise ValueError("the operating point is selected on the validation split only")
        op = select_operating_point(bundle, ev, mandated_threshold=threshold_mandated)
        op_path = operating_point_path or (resolve_path(cfg, cfg["artifacts"]["weights_dir"]) / OPERATING_POINT_FILENAME)
        save_operating_point(op, op_path)
        logger.info("selected operating point %s, threshold %.3f (val F1 %.3f) -> %s", op["pooling_key"],
                    op["threshold"]["f1_optimal_calibrated"], op["threshold"]["val_f1_calibrated"], op_path)
        op_source = f"selected on val, written to {op_path}"
    elif operating_point_path is not None and Path(operating_point_path).exists():
        op = load_operating_point(operating_point_path)
        op_source = f"loaded from {operating_point_path} (selected on {op.get('selected_on')})"
    else:
        default_pool = resolve_pooling(cfg)
        op = {"pooling": {"method": default_pool["method"], "quantile": default_pool["quantile"], "horizon_reduction": "max"},
              "pooling_key": pooling_key({"method": default_pool["method"], "quantile": default_pool["quantile"], "horizon_reduction": "max"}),
              "calibration": {"params_by_k": [{"a": 1.0, "b": 0.0, "n": 0, "degenerate": True}] * K},
              "threshold": {"f1_optimal_calibrated": threshold_mandated, "mandated": threshold_mandated}}
        op_source = "NO operating point file: config pooling, identity calibration, mandated threshold"
        logger.warning(op_source)
    if force_pooling is not None:
        op = _with_forced_pooling(op, force_pooling)
        op_source += f"; pooling OVERRIDDEN to {force_pooling}"
        logger.warning("pooling forced to %s — the threshold and calibration were selected under "
                       "%s, so this run answers 'does the training set help at a FIXED readout', "
                       "not 'what would you deploy'", force_pooling, op.get("pooling_key_selected"))

    applied = apply_operating_point(bundle, op)
    threshold = float(op["threshold"]["f1_optimal_calibrated"])

    scores = system_scores(bundle, op["pooling"])
    scores["world_model_calibrated"] = applied["calibrated"]
    y_pub, y_det = ev.y_published, ev.y_detect

    results: dict[str, Any] = {
        "eval_set": ev.summary(),
        "operating_point": {"source": op_source, "pooling": op["pooling"], "pooling_key": op.get("pooling_key"),
                            "pooling_forced": bool(op.get("pooling_forced")),
                            "pooling_key_selected": op.get("pooling_key_selected"),
                            "threshold_used": threshold, "threshold_mandated": threshold_mandated,
                            "calibration_degenerate": [p.get("degenerate", False) for p in op["calibration"]["params_by_k"]],
                            "calibration_fitted_horizons": len(op["calibration"]["params_by_k"]),
                            "calibration_horizons_extended": bool(applied["calibration_horizons_extended"])},
        "systems_scored": sorted(scores),
        "iso_sigma": bundle.iso_sigma,
        "task_published_label": _task_table(ev, scores, y_pub, threshold, None, True, n_resamples, ev.span_hours),
        "task_published_label_at_mandated_threshold": _task_table(ev, {"world_model_calibrated": scores["world_model_calibrated"],
                                                                        "world_model": scores["world_model"]},
                                                                  y_pub, threshold_mandated, None, False, 0, ev.span_hours),
        "task_A_detection": _task_table(ev, scores, y_det, threshold, None, False, 0, ev.span_hours),
        "task_B_onset_forecast": {
            str(h): _task_table(ev, {**scores, **onset_direct.get(h, {})}, ev.onset_labels[h], threshold, ev.mask_forecast,
                                h in (3, 5, 10), n_resamples, ev.span_hours)
            for h in ev.onset_labels
        },
        "task_C_progression": _progression(ev, bundle, applied["risk_k_calibrated"], threshold),
        "per_attack_group": _per_attack_group(ev, scores, y_pub, threshold, bundle, X_scaled, Y_scaled,
                                              scaler.model_mask, cfg.get("dataset", {}).get("days")),
        "host_identity_published_label": _host_identity(ev, scores, y_pub),
        "attribution_published_label": _attribution(ev, scores, y_pub, n_resamples),
        "attribution_detection": _attribution(ev, scores, y_det, n_resamples),
        "state_forecast": state_forecast_report(
            bundle.states_det, bundle.states_ridge, X_scaled, Y_scaled, scaler.model_mask,
            active_origin=(ev.stratum != "silent_negative") & (np.abs(X_scaled[:, -1, :]).sum(1) > 0),
            states_std=bundle.states_std,
        ),
        "per_episode": {
            "world_model_calibrated": per_episode_report(ev, scores["world_model_calibrated"], threshold),
            "world_model_calibrated_at_mandated": per_episode_report(ev, scores["world_model_calibrated"], threshold_mandated),
            "persistence": per_episode_report(ev, scores["persistence"], threshold),
        },
        "calibration_published_label": {
            "raw": _calibration(y_pub, np.clip(scores["world_model"], 0, 1), ev.weight),
            "calibrated": _calibration(y_pub, scores["world_model_calibrated"], ev.weight),
        },
        "pooling_sweep_val_only": op.get("candidates") if split == "val" else None,
        "wall_seconds": time.time() - t0,
    }
    if split == "val" and select_op:
        results["operating_point_selection"] = {k: v for k, v in op.items() if k != "candidates"}

    record = experiment_record(cfg, stage="benchmark", split=split, seeds=seeds, threshold=threshold,
                               pooling=op["pooling"], calibration={"source": op_source},
                               metrics=results, extra={"n_samples_per_member": n_samples, "n_trajectories": n_samples * len(seeds),
                                                       "eval_seed": eval_seed, "caps": caps,
                                                       "config_overrides": cfg.get("_config_overrides", [])})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "benchmark.json", record)
    # the per-row scores, for plots and for re-analysis without re-rolling
    np.savez_compressed(out_dir / "benchmark_scores.npz", **{k: v.astype("float32") for k, v in scores.items()},
                        y_published=y_pub, y_detect=y_det, weight=ev.weight, stratum=ev.stratum, cluster=ev.cluster,
                        host_id=ev.arrays.host_id, origin_ts=ev.arrays.origin_ts, inside_episode=ev.inside_episode,
                        minutes_to_onset=ev.minutes_to_onset, risk_k_calibrated=applied["risk_k_calibrated"],
                        risk_k_raw=applied["risk_k_raw"], future_is_attack=ev.arrays.future_is_attack,
                        **{f"onset_{h}": v for h, v in ev.onset_labels.items()})
    logger.info("wrote %s (%.0fs)", out_dir / "benchmark.json", time.time() - t0)
    print_summary(results, split)
    return record


def print_summary(results: dict[str, Any], split: str) -> None:
    pub = results["task_published_label"]["systems"]
    det = results["task_A_detection"]["systems"]
    print(f"\n== benchmark {split}: {results['eval_set']['n_rows']} rows, prevalence {results['eval_set']['natural_prevalence_published']:.5f}, "
          f"operating point {results['operating_point']['pooling_key']} thr={results['operating_point']['threshold_used']:.3f}")
    print(f"{'system':32s} {'AP-nat':>8s} {'CI':>17s} {'ROC':>6s} {'P@thr':>6s} {'R@thr':>6s} {'F1':>6s} {'FA-act':>7s} | det AP  | onset AP 3/5/10/15")
    for name in sorted(pub, key=lambda n: -np.nan_to_num(pub[n]["auc_pr"], nan=-1)):
        r = pub[name]
        ci = r.get("auc_pr_bootstrap")
        ci_s = f"[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]" if ci else ""
        if ci and ci.get("n_positive_clusters", 99) < 5:
            ci_s = f"{ci_s}*"  # * = fewer than 5 positive clusters: cluster interval is not informative
        at = r["at_threshold"]
        onset = " ".join(f"{results['task_B_onset_forecast'][h]['systems'][name]['auc_pr']:.3f}" for h in ("3", "5", "10", "15")
                         if name in results["task_B_onset_forecast"][h]["systems"])
        print(f"{name:32s} {r['auc_pr']:8.3f} {ci_s:>17s} {r['roc_auc']:6.3f} {at['precision']:6.3f} {at['recall']:6.3f} {at['f1']:6.3f} "
              f"{r.get('active_benign_false_alarm_rate', float('nan')):7.3f} | {det[name]['auc_pr']:6.3f}  | {onset}")
    sf = results["state_forecast"]
    print("state skill vs persistence:", {k: round(v, 3) for k, v in sf["skill_vs_persistence"].items()},
          "| vs ridge:", round(sf.get("skill_vs_ridge", float("nan")), 3))
    for k, v in results["attribution_published_label"].items():
        print(f"  attribution {k:58s} {v['point']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    groups = results.get("per_attack_group") or {}
    if groups:
        print(f"{'attack group':38s} {'pos':>5s} {'eps':>4s} {'AP':>6s} {'ROC':>6s} {'R@thr':>6s} {'oracle':>6s} {'persist':>7s} {'stateskill':>10s}")
        for name in sorted(groups, key=lambda n: -groups[n]["n_positive_rows"]):
            g = groups[name]
            sysm = g["systems"]
            def _ap(k):
                return sysm.get(k, {}).get("auc_pr", float("nan"))
            print(f"{name:38s} {g['n_positive_rows']:5d} {g['n_episodes']:4d} {_ap('world_model_calibrated'):6.3f} "
                  f"{sysm.get('world_model_calibrated', {}).get('roc_auc', float('nan')):6.3f} "
                  f"{sysm.get('world_model_calibrated', {}).get('at_threshold', {}).get('recall', float('nan')):6.3f} "
                  f"{_ap('oracle_true_future'):6.3f} {_ap('persistence'):7.3f} "
                  f"{g.get('state_skill_vs_persistence', float('nan')):10.3f}")
    pe = results["per_episode"]["world_model_calibrated"]
    print(f"episodes: {pe['n_episodes']}, warned pre-onset {pe['n_warned_pre_onset']}, detected within {pe['n_detected_within_episode']}, "
          f"median latency {pe['median_latency_min']} min, median lead {pe['median_lead_time_s']} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test", "holdout"])
    parser.add_argument("--seeds", default=None, help="comma-separated; default cfg ensemble seeds")
    parser.add_argument("--n-samples", type=int, default=None, help="trajectories per member; default rollout.n_samples_per_member")
    parser.add_argument("--out-dir", default=None, help="default <metrics_dir>/<split>")
    parser.add_argument("--operating-point", default=None, help="default <weights_dir>/operating_point.json")
    parser.add_argument("--select-operating-point", action="store_true")
    parser.add_argument("--active-negative-cap", type=int, default=None)
    parser.add_argument("--silent-negative-cap", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=300)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--gru-classifier", default=None)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="config override key.path=value (e.g. windowing.horizon_length=6 for a K-extension check; "
                             "labels.risk_threshold_windows must be set to match)")
    parser.add_argument("--force-pooling", default=None,
                        help="score with this pooling_key instead of the selected one, e.g. "
                             "'mean|q=-|max'. For holding the readout fixed while comparing "
                             "training sets; NOT a deployable configuration, since the threshold "
                             "and calibration were selected under a different rule.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    if args.overrides:
        from nidra.scripts.run_experiment import apply_overrides
        cfg = apply_overrides(cfg, args.overrides)
        cfg["_config_overrides"] = list(args.overrides)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["ensemble"]["seeds"])
    n_samples = args.n_samples or int(cfg["rollout"].get("n_samples_per_member", 100))
    out_dir = Path(args.out_dir) if args.out_dir else resolve_path(cfg, cfg["artifacts"]["metrics_dir"]) / args.split
    op_path = Path(args.operating_point) if args.operating_point else resolve_path(cfg, cfg["artifacts"]["weights_dir"]) / OPERATING_POINT_FILENAME
    caps = {}
    if args.active_negative_cap is not None:
        caps["active_negative"] = args.active_negative_cap
    if args.silent_negative_cap is not None:
        caps["silent_negative"] = args.silent_negative_cap
    gru = Path(args.gru_classifier) if args.gru_classifier else resolve_path(cfg, cfg["artifacts"]["weights_dir"]) / "gru_classifier.pt"
    run(cfg, args.split, seeds, n_samples, out_dir, op_path, args.select_operating_point, caps or None,
        args.n_resamples, args.chunk, gru, with_ablations=not args.no_ablations, force_pooling=args.force_pooling)


if __name__ == "__main__":
    main()
