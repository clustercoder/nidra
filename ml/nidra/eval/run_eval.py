"""Top-level evaluation CLI: runs baselines, ablations, calibration, and the
lead-time report against a trained model, and writes JSON artifacts to
artifacts/metrics/. This is the "python -m nidra.eval.*" entry point
referenced by the build order (IMPLEMENTATION-ML.md §8, day 4).

Usage:
    python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split test
    python -m nidra.eval.run_eval --seed 0 --split holdout   # Infiltration / generalization claim
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from nidra.data.dataset import build_windowed_arrays, subsample_stratified_by_risk
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH
from nidra.eval.ablations import horizon_curve, persistence_ablation, surprise_signal, time_shuffle_ablation
from nidra.eval.baselines import (
    baseline_lr_current_state,
    baseline_lr_flattened_history,
    baseline_oracle,
    baseline_persistence,
    ensemble_baseline_oracle,
    ensemble_baseline_persistence,
    ensemble_world_model_forecast,
    world_model_forecast,
)
from nidra.eval.calibrate import apply_platt_by_horizon, calibration_pooling_mismatch, load_calibration
from nidra.eval.calibration import calibration_by_horizon
from nidra.eval.lead_time_runner import compute_lead_time_report
from nidra.eval.metrics import standard_metrics
from nidra.models.world_model import WorldModel
from nidra.train.pipeline import build_all_splits, scale_arrays
from nidra.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _build_model(cfg: dict) -> WorldModel:
    mcfg = cfg["model"]
    return WorldModel(
        n_features=mcfg["n_features"],
        hidden_size=mcfg["encoder"]["hidden_size"],
        encoder_layers=mcfg["encoder"]["num_layers"],
        encoder_dropout=mcfg["encoder"]["dropout"],
        transition_mlp_hidden=mcfg["transition"]["mlp_hidden"],
        logvar_min=mcfg["transition"]["logvar_min"],
        logvar_max=mcfg["transition"]["logvar_max"],
        risk_hidden=mcfg["risk_head"]["hidden"],
        stage_hidden=mcfg["stage_head"]["hidden"],
        n_stages=mcfg["stage_head"]["n_stages"],
        state_clamp=mcfg["transition"]["state_clamp"],
    )


def _warn_if_single_class(y_true: np.ndarray, split_name: str, max_eval_samples: int | None) -> None:
    """subsample_stratified_by_risk keeps ALL positive-risk samples, only
    filling the remainder with negatives — if max_eval_samples is smaller
    than the split's actual positive-candidate pool, the eval set silently
    becomes 100% positive with zero negatives. standard_metrics() correctly
    reports auc_pr as NaN in that case (undefined, not fabricated) rather
    than a real bug, but a NaN AUC-PR across every baseline is easy to
    mistake for something broken — this warns loudly at the actual root
    cause instead."""
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "run_eval: eval set for split=%s has only one risk_label class present (n=%d) — "
            "max_eval_samples=%s is smaller than this split's positive-candidate pool, so every "
            "auc_pr below will be NaN (undefined for a single-class set, not a bug). Pass a larger "
            "--max-eval-samples to get a real negative/positive mix.",
            split_name, len(y_true), max_eval_samples,
        )


def resolve_pooling(cfg: dict) -> dict:
    """The trajectory-pooling settings, read from the one place they live.

    These keys sit under `rollout:` in the config. This helper exists because
    they were being read in two places — once to drive the forecast calls and
    once to stamp the metrics files — and the second read looked at the top
    level, found nothing, and silently fell back to the `"mean"` default. The
    numbers were quantile-pooled and the provenance block said "mean", which
    is worse than recording nothing at all. One reader, used by both.
    """
    rollout = cfg.get("rollout", {})
    return {
        "method": rollout.get("risk_pooling_method", "mean"),
        "quantile": rollout.get("risk_pooling_quantile", 0.9),
        "head_reduction": rollout.get("risk_pooling_head_reduction", "before_pooling"),
    }


def build_run_params(cfg: dict, seed: int, split_name: str, n_samples: int,
                      max_eval_samples: int | None, ensemble_seeds: list[int] | None,
                      n_eval_rows: int) -> dict:
    """The provenance block stamped into every metrics file this run writes.

    Bare numbers are not reproducible: a committed F1 of 0.835 from a cheap
    20-rollout smoke run and 0.837 from the published 200-per-member run look
    identical to two decimal places but are different experiments, and the
    published headline once disagreed with the committed artifacts for
    exactly that reason. `n_trajectories` is the count the served statistic
    actually pools — n_samples per ensemble member, times the members.
    """
    pooling = resolve_pooling(cfg)
    eval_cfg = cfg.get("eval", {})
    seeds = [int(s) for s in ensemble_seeds] if ensemble_seeds is not None else None
    n_members = len(seeds) if seeds else 1
    return {
        "split": split_name,
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_trajectories": int(n_samples) * n_members,
        "max_eval_samples": None if max_eval_samples is None else int(max_eval_samples),
        "n_eval_rows": int(n_eval_rows),
        "ensemble_seeds": seeds,
        "risk_pooling": pooling,
        "risk_threshold": eval_cfg.get("risk_threshold"),
        "forecast_chunk_size": eval_cfg.get("forecast_chunk_size"),
        "config_hash": cfg.get("_config_hash"),
        "git_commit": cfg.get("_git_commit"),
    }


def run(cfg: dict, seed: int, split_name: str, n_samples: int, max_eval_samples: int | None = 5000,
        ensemble_seeds: list[int] | None = None) -> dict:
    """`ensemble_seeds`, if given, additionally loads every listed seed's
    checkpoint and computes an `ensemble_*`-prefixed set of baselines and a
    `"ensemble"`/`"ensemble_calibrated"` lead-time section using the real
    pooled-ensemble statistic (`ensemble_world_model_forecast`) — the exact
    statistic `NidraPredictor` serves, not the single-seed approximation
    the rest of this function uses. This is opt-in (default None = skip)
    both because it multiplies rollout cost by len(ensemble_seeds) and
    because the single-seed baselines above remain independently useful
    (e.g. for a quick per-seed sanity check). See REAL_DATA_RESULTS.md's
    Run 3 calibration section for why this was added: the single-seed
    approximation can overstate a calibrated-recall/lead-time improvement
    at production scale, not just differ from it in magnitude."""
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    # Split-specific subdirectory: the documented reproduction commands run
    # this script once per split (test, then holdout) against the SAME
    # metrics_dir — flat filenames would let the holdout run silently
    # overwrite the test run's baselines.json/ablations.json/etc, which is
    # exactly what happened running this by hand before this fix.
    metrics_dir = resolve_path(cfg, cfg["artifacts"]["metrics_dir"]) / split_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _pooling = resolve_pooling(cfg)
    risk_pooling_method = _pooling["method"]
    risk_pooling_quantile = _pooling["quantile"]
    risk_pooling_head_reduction = _pooling["head_reduction"]
    # Bounds peak memory in the forecast functions, which tile the whole batch
    # by n_samples in one allocation — an unchunked large split at a high
    # sample count gets OOM-killed by the OS with no Python traceback.
    forecast_chunk_size = cfg["eval"].get("forecast_chunk_size")
    logger.info(
        "run_eval: risk_pooling_method=%s risk_pooling_quantile=%.2f "
        "risk_pooling_head_reduction=%s forecast_chunk_size=%s",
        risk_pooling_method, risk_pooling_quantile, risk_pooling_head_reduction, forecast_chunk_size,
    )

    scaler = FeatureScaler.load(scaler_dir / "robust_scaler.joblib", scaler_dir / "scaler_metadata.json")
    model = _build_model(cfg)
    model.load_state_dict(torch.load(weights_dir / f"model_seed_{seed}.pt", map_location="cpu"))
    model.eval()

    ensemble_models = None
    if ensemble_seeds:
        ensemble_models = []
        for eseed in ensemble_seeds:
            m = _build_model(cfg)
            m.load_state_dict(torch.load(weights_dir / f"model_seed_{eseed}.pt", map_location="cpu"))
            m.eval()
            ensemble_models.append(m)
        logger.info("run_eval: loaded %d ensemble member(s) for pooled-ensemble baselines: seeds=%s",
                    len(ensemble_models), ensemble_seeds)

    # Optional post-hoc calibration (nidra.scripts.fit_calibration), fit on
    # the validation split, applied here to a "world_model_calibrated"
    # variant alongside the raw/uncalibrated numbers — never silently
    # replacing them, since it's an honest before/after comparison, not a
    # correction. Absent file => every "_calibrated" section below is
    # simply omitted, not an error (a model can be fully evaluated without
    # ever running fit_calibration).
    calibration_loaded = load_calibration(weights_dir / "risk_calibration.json")
    calibration_params = calibration_loaded[0] if calibration_loaded else None
    calibration_meta = calibration_loaded[1] if calibration_loaded else None
    # A Platt fit only describes the pooled statistic it was fit against. An
    # artifact fit under different pooling is not a weaker refinement, it is
    # the wrong function applied to the wrong numbers — measured to make
    # quantile-pooled scores slightly WORSE than raw (see REAL_DATA_RESULTS.md).
    # Dropping it is the honest fallback; the run then reports raw scores and
    # says why, rather than publishing a silently mismatched "_calibrated" row.
    if calibration_params is not None:
        mismatch = calibration_pooling_mismatch(calibration_meta or {}, cfg)
        if mismatch is not None:
            logger.warning(
                "run_eval: IGNORING stale calibration at %s — %s. Every '_calibrated' metric is omitted "
                "from this run. Re-run nidra.scripts.fit_calibration against this config to restore it.",
                weights_dir / "risk_calibration.json", mismatch,
            )
            calibration_params = None
            calibration_meta = {**(calibration_meta or {}), "ignored_reason": mismatch}
        else:
            logger.info("run_eval: applying post-hoc calibration from %s", weights_dir / "risk_calibration.json")

    splits = build_all_splits(cfg)
    eval_split_df = getattr(splits, split_name)

    # Only two splits are needed here: train (to fit the LR baselines) and the
    # split under evaluation. This used to call build_windowed_splits, which
    # windowizes all four — including val, which is never used, and the
    # UNCAPPED train split, ~6.9M candidate origins at full production scale.
    # Materializing those as float32 [L=30,F=45] slices needs ~35GB and is
    # OOM-killed with no traceback before anything is evaluated, which made
    # full-scale eval unrunnable on an ordinary machine. build_windowed_arrays
    # applies the identical stratified-by-risk cap BEFORE building any slice
    # (see its docstring), so this is the same selection, just affordable.
    #
    # The flattened-history LR baseline fits on [n_train, L*F] — 1350 cols
    # for L=30 — and a real day's train split can carry hundreds of
    # thousands of rows (confirmed against real CIC-IDS2017 data: ~1M rows
    # for a single day slice), which is both slow and memory-heavy to fit
    # on directly. Cap it the same stratified way eval samples are capped.
    train_arrays = build_windowed_arrays(
        splits.train, L=CONTEXT_LENGTH, K=HORIZON_LENGTH, max_samples=max_eval_samples, seed=seed
    )
    logger.info("run_eval: train split windowed to %d samples for baseline fitting (cap=%s)",
                len(train_arrays.X), max_eval_samples)

    # The split under evaluation stays UNCAPPED: lead time needs each attacked
    # host's complete chronological sequence, not a random subset of windows.
    # The capped copy used for baselines/ablations is derived below.
    full_eval_arrays = build_windowed_arrays(eval_split_df, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)

    if len(full_eval_arrays.X) == 0:
        logger.warning("run_eval: split %s is empty, nothing to evaluate", split_name)
        return {}

    # Rollout sampling is the expensive step and it runs once per
    # baseline/ablation/calibration sample, so cap the evaluation set size
    # the same stratified way training samples are capped — full-scale real
    # splits can carry hundreds of thousands of samples per day. Lead time
    # is computed separately below against the FULL (unsampled) split,
    # since it needs each attacked host's complete chronological sequence,
    # not a random subset of windows — but that cost is naturally bounded
    # by the (small) number of hosts that are ever attacked.
    eval_arrays = subsample_stratified_by_risk(full_eval_arrays, max_eval_samples, seed=seed)
    if len(eval_arrays.X) < len(full_eval_arrays.X):
        logger.info(
            "run_eval: capped split=%s from %d to %d samples for baselines/ablations/calibration "
            "(stratified: all positive risk_label samples kept)",
            split_name, len(full_eval_arrays.X), len(eval_arrays.X),
        )

    X_eval, Y_eval = scale_arrays(eval_arrays, scaler)
    X_train, _ = scale_arrays(train_arrays, scaler)
    X_eval_last, X_train_last = X_eval[:, -1, :], X_train[:, -1, :]

    # State nRMSE must be normalized by a stable, population-level per-feature
    # scale (see eval/metrics.state_nrmse) rather than the std of whatever
    # small eval batch happens to be sampled — many of the 45 features are
    # structurally near-constant on large slices of this dataset, and a
    # batch-recomputed std collapses toward zero for them, inflating nRMSE by
    # orders of magnitude for reasons unrelated to forecast quality.
    feature_scale = scaler.reference_std_

    run_params = build_run_params(cfg, seed=seed, split_name=split_name, n_samples=n_samples,
                                   max_eval_samples=max_eval_samples, ensemble_seeds=ensemble_seeds,
                                   n_eval_rows=len(eval_arrays.X))
    results: dict = {**run_params}

    # --- Baselines ---
    logger.info("running baselines on split=%s (n=%d)", split_name, len(eval_arrays.X))
    _, probs_lr1 = baseline_lr_current_state(X_train_last, train_arrays.risk_label, X_eval_last)
    _, probs_lr2 = baseline_lr_flattened_history(X_train, train_arrays.risk_label, X_eval)
    probs_persistence = baseline_persistence(X_eval_last, model)
    probs_oracle, _, _ = baseline_oracle(Y_eval, model)
    world = world_model_forecast(X_eval, model, K=Y_eval.shape[1], n_samples=n_samples,
                                  risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
                                  chunk_size=forecast_chunk_size)
    probs_world_model = world["risk_over_horizon"]

    y_true = eval_arrays.risk_label
    _warn_if_single_class(y_true, split_name, max_eval_samples)
    baselines = {
        "lr_current_state": standard_metrics(y_true, probs_lr1, threshold=cfg["eval"]["risk_threshold"]),
        "lr_flattened_history": standard_metrics(y_true, probs_lr2, threshold=cfg["eval"]["risk_threshold"]),
        "persistence": standard_metrics(y_true, probs_persistence, threshold=cfg["eval"]["risk_threshold"]),
        "world_model": standard_metrics(y_true, probs_world_model, threshold=cfg["eval"]["risk_threshold"]),
        "oracle": standard_metrics(y_true, probs_oracle, threshold=cfg["eval"]["risk_threshold"]),
    }
    if calibration_params is not None:
        # NOTE: this seed's OWN world_model_forecast is a single-model
        # statistic, while calibration was fit against the pooled-ensemble
        # statistic (ensemble_world_model_forecast) — applying it here to a
        # single seed is an approximation (documented in
        # calibration_meta/"applied_to_single_seed_approximation"), useful
        # for a quick per-seed sanity check; the number that matters for
        # serving is what NidraPredictor actually reports, which pools the
        # full ensemble before calibrating, exactly matching the fit.
        calibrated_risk_mean_k = apply_platt_by_horizon(world["risk_mean_k"], calibration_params)
        probs_world_model_calibrated = calibrated_risk_mean_k.max(axis=1)
        baselines["world_model_calibrated"] = standard_metrics(
            y_true, probs_world_model_calibrated, threshold=cfg["eval"]["risk_threshold"]
        )
        baselines["calibration_fit_metadata"] = {
            **calibration_meta,
            "applied_to_single_seed_approximation": True,
            "applied_to_seed": seed,
        }

    ensemble_world = None
    if ensemble_models is not None:
        logger.info("running pooled-ensemble baselines on split=%s (n=%d, %d members)",
                    split_name, len(eval_arrays.X), len(ensemble_models))
        probs_ensemble_persistence = ensemble_baseline_persistence(X_eval_last, ensemble_models)
        probs_ensemble_oracle, _, _ = ensemble_baseline_oracle(Y_eval, ensemble_models)
        ensemble_world = ensemble_world_model_forecast(X_eval, ensemble_models, K=Y_eval.shape[1],
                                                        n_samples_per_member=n_samples,
                                                        risk_pooling_method=risk_pooling_method,
                                                        risk_pooling_quantile=risk_pooling_quantile,
                                                        head_reduction=risk_pooling_head_reduction,
                                                        chunk_size=forecast_chunk_size)
        probs_ensemble_world_model = ensemble_world["risk_mean_k"].max(axis=1)
        baselines["ensemble_persistence"] = standard_metrics(
            y_true, probs_ensemble_persistence, threshold=cfg["eval"]["risk_threshold"]
        )
        baselines["ensemble_oracle"] = standard_metrics(
            y_true, probs_ensemble_oracle, threshold=cfg["eval"]["risk_threshold"]
        )
        baselines["ensemble_world_model"] = standard_metrics(
            y_true, probs_ensemble_world_model, threshold=cfg["eval"]["risk_threshold"]
        )
        if calibration_params is not None:
            # This is the REAL calibrated statistic — pooled across the
            # ensemble exactly as fit, not the single-seed approximation
            # above. See the run() docstring for why both are kept.
            ensemble_calibrated_risk_mean_k = apply_platt_by_horizon(ensemble_world["risk_mean_k"], calibration_params)
            probs_ensemble_world_model_calibrated = ensemble_calibrated_risk_mean_k.max(axis=1)
            baselines["ensemble_world_model_calibrated"] = standard_metrics(
                y_true, probs_ensemble_world_model_calibrated, threshold=cfg["eval"]["risk_threshold"]
            )
    with open(metrics_dir / "baselines.json", "w") as f:
        json.dump({**run_params, "baselines": baselines}, f, indent=2)
    results["baselines"] = baselines

    # --- Ablations ---
    logger.info("running ablations")
    ablations = {
        "persistence": persistence_ablation(
            X_eval, Y_eval, y_true, model, feature_scale=feature_scale,
            risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
            chunk_size=forecast_chunk_size),
        "time_shuffle": time_shuffle_ablation(
            X_eval, y_true, model, K=Y_eval.shape[1],
            risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
            chunk_size=forecast_chunk_size),
        "horizon_curve": horizon_curve(
            eval_arrays.future_is_attack, Y_eval, model, X_eval, n_samples=n_samples,
            feature_scale=feature_scale, risk_pooling_method=risk_pooling_method,
            risk_pooling_quantile=risk_pooling_quantile, chunk_size=forecast_chunk_size
        ),
        "surprise_signal": surprise_signal(X_eval, Y_eval, eval_arrays.future_is_attack, model),
    }
    if feature_scale is not None:
        from nidra.data.schema import FEATURE_ORDER
        low_variance = [
            FEATURE_ORDER[i] for i in range(len(FEATURE_ORDER)) if feature_scale[i] <= 0.05
        ]
        ablations["nrmse_normalization"] = {
            "method": "train_population_reference_std",
            "scale_floor": 0.05,
            "low_variance_features_floored": low_variance,
        }
    with open(metrics_dir / "ablations.json", "w") as f:
        json.dump({**run_params, "ablations": ablations}, f, indent=2)
    results["ablations"] = ablations

    # --- Calibration (Brier/reliability) ---
    logger.info("running calibration")
    calibration = calibration_by_horizon(X_eval, eval_arrays.future_is_attack, model, n_samples=n_samples)
    calibration_out = {**run_params, "calibration": calibration}
    if calibration_params is not None:
        from nidra.eval.metrics import brier_score, reliability_diagram
        calibrated_risk_mean_k = apply_platt_by_horizon(world["risk_mean_k"], calibration_params)
        per_k_calibrated = []
        for k in range(Y_eval.shape[1]):
            y_true_k = eval_arrays.future_is_attack[:, k]
            y_prob_k = calibrated_risk_mean_k[:, k]
            per_k_calibrated.append({
                "k": k,
                "brier": brier_score(y_true_k, y_prob_k),
                "reliability": reliability_diagram(y_true_k, y_prob_k, n_bins=10),
            })
        calibration_out["calibration_recalibrated"] = {
            "per_horizon": per_k_calibrated,
            "mean_brier": float(np.mean([p["brier"] for p in per_k_calibrated])),
        }
    with open(metrics_dir / "calibration.json", "w") as f:
        json.dump(calibration_out, f, indent=2)
    results["calibration"] = calibration_out["calibration"]

    # --- Lead time --- (uses the FULL split, not the capped eval_arrays —
    # see the comment above on why lead time needs complete per-host history)
    logger.info("computing lead-time report (this scans every pre-onset window per attacked host)")
    lead_time_report = compute_lead_time_report(
        full_eval_arrays, eval_split_df, model, scaler,
        threshold=cfg["eval"]["risk_threshold"], m=cfg["eval"]["lead_time_persistence_windows"], n_samples=n_samples,
        risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
        risk_pooling_head_reduction=risk_pooling_head_reduction,
    )
    lead_time_out = {**run_params, "raw": lead_time_report.to_dict()}
    if calibration_params is not None:
        lead_time_report_calibrated = compute_lead_time_report(
            full_eval_arrays, eval_split_df, model, scaler,
            threshold=cfg["eval"]["risk_threshold"], m=cfg["eval"]["lead_time_persistence_windows"],
            n_samples=n_samples, calibration=calibration_params,
            risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
            risk_pooling_head_reduction=risk_pooling_head_reduction,
        )
        lead_time_out["calibrated"] = lead_time_report_calibrated.to_dict()
    if ensemble_models is not None:
        logger.info("computing pooled-ensemble lead-time report (%d members)", len(ensemble_models))
        ensemble_lead_time_report = compute_lead_time_report(
            full_eval_arrays, eval_split_df, ensemble_models, scaler,
            threshold=cfg["eval"]["risk_threshold"], m=cfg["eval"]["lead_time_persistence_windows"],
            n_samples=n_samples,
            risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
            risk_pooling_head_reduction=risk_pooling_head_reduction,
        )
        lead_time_out["ensemble"] = ensemble_lead_time_report.to_dict()
        if calibration_params is not None:
            ensemble_lead_time_report_calibrated = compute_lead_time_report(
                full_eval_arrays, eval_split_df, ensemble_models, scaler,
                threshold=cfg["eval"]["risk_threshold"], m=cfg["eval"]["lead_time_persistence_windows"],
                n_samples=n_samples, calibration=calibration_params,
                risk_pooling_method=risk_pooling_method, risk_pooling_quantile=risk_pooling_quantile,
                risk_pooling_head_reduction=risk_pooling_head_reduction,
            )
            lead_time_out["ensemble_calibrated"] = ensemble_lead_time_report_calibrated.to_dict()
    with open(metrics_dir / "lead_time.json", "w") as f:
        json.dump(lead_time_out, f, indent=2)
    results["lead_time"] = lead_time_out

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the full NIDRA evaluation harness.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["test", "holdout", "val"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-eval-samples", type=int, default=5000,
                         help="cap on baseline/ablation/calibration sample count (rollout cost scales with this); "
                              "lead time always uses the full split. Pass 0 to disable capping.")
    parser.add_argument("--use-ensemble", action="store_true",
                         help="also compute ensemble_*-prefixed baselines and an 'ensemble'/'ensemble_calibrated' "
                              "lead-time section using the real pooled-ensemble statistic (loads every seed in "
                              "cfg['ensemble']['seeds'], multiplying rollout cost by the ensemble size) — this is "
                              "the statistic NidraPredictor actually serves, not the single-seed approximation "
                              "the rest of this script's baselines use.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(2)

    cfg = load_config(args.config)
    max_eval_samples = None if args.max_eval_samples == 0 else args.max_eval_samples
    ensemble_seeds = cfg["ensemble"]["seeds"] if args.use_ensemble else None
    results = run(cfg, args.seed, args.split, args.n_samples, max_eval_samples=max_eval_samples,
                  ensemble_seeds=ensemble_seeds)
    print(json.dumps({k: v for k, v in results.items() if k not in ("baselines", "ablations", "calibration", "lead_time")}, indent=2))
    if "baselines" in results:
        # Not every entry is a per-model metrics dict — "calibration_fit_metadata"
        # (present only when a risk_calibration.json was applied) is
        # provenance info, not a {f1, auc_pr, ...} row, so it's skipped here.
        print("baselines:", json.dumps(
            {k: {"f1": v["f1"], "auc_pr": v["auc_pr"]} for k, v in results["baselines"].items() if isinstance(v, dict) and "f1" in v},
            indent=2,
        ))
    if "lead_time" in results:
        print("lead_time:", json.dumps(results["lead_time"], indent=2))


if __name__ == "__main__":
    main()
