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

from nidra.data.dataset import subsample_stratified_by_risk
from nidra.data.normalize import FeatureScaler
from nidra.eval.ablations import horizon_curve, persistence_ablation, surprise_signal, time_shuffle_ablation
from nidra.eval.baselines import (
    baseline_lr_current_state,
    baseline_lr_flattened_history,
    baseline_oracle,
    baseline_persistence,
    world_model_forecast,
)
from nidra.eval.calibration import calibration_by_horizon
from nidra.eval.lead_time_runner import compute_lead_time_report
from nidra.eval.metrics import standard_metrics
from nidra.models.world_model import WorldModel
from nidra.train.pipeline import build_all_splits, build_windowed_splits, scale_arrays
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


def run(cfg: dict, seed: int, split_name: str, n_samples: int, max_eval_samples: int | None = 5000) -> dict:
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    metrics_dir = resolve_path(cfg, cfg["artifacts"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scaler = FeatureScaler.load(scaler_dir / "robust_scaler.joblib", scaler_dir / "scaler_metadata.json")
    model = _build_model(cfg)
    model.load_state_dict(torch.load(weights_dir / f"model_seed_{seed}.pt", map_location="cpu"))
    model.eval()

    splits = build_all_splits(cfg)
    windowed_all = build_windowed_splits(splits)
    eval_split_df = getattr(splits, split_name)
    full_eval_arrays = windowed_all[split_name]

    if len(full_eval_arrays.X) == 0:
        logger.warning("run_eval: split %s is empty, nothing to evaluate", split_name)
        return {}

    # The flattened-history LR baseline fits on [n_train, L*F] — 1350 cols
    # for L=30 — and a real day's train split can carry hundreds of
    # thousands of rows (confirmed against real CIC-IDS2017 data: ~1M rows
    # for a single day slice), which is both slow and memory-heavy to fit
    # on directly. Cap it the same stratified way eval samples are capped.
    train_arrays = subsample_stratified_by_risk(windowed_all["train"], max_eval_samples, seed=seed)
    if len(train_arrays.X) < len(windowed_all["train"].X):
        logger.info(
            "run_eval: capped train split from %d to %d samples for baseline fitting",
            len(windowed_all["train"].X), len(train_arrays.X),
        )

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

    results: dict = {"split": split_name, "seed": seed, "n_samples": int(len(eval_arrays.X))}

    # --- Baselines ---
    logger.info("running baselines on split=%s (n=%d)", split_name, len(eval_arrays.X))
    _, probs_lr1 = baseline_lr_current_state(X_train_last, train_arrays.risk_label, X_eval_last)
    _, probs_lr2 = baseline_lr_flattened_history(X_train, train_arrays.risk_label, X_eval)
    probs_persistence = baseline_persistence(X_eval_last, model)
    probs_oracle, _, _ = baseline_oracle(Y_eval, model)
    world = world_model_forecast(X_eval, model, K=Y_eval.shape[1], n_samples=n_samples)
    probs_world_model = world["risk_over_horizon"]

    y_true = eval_arrays.risk_label
    baselines = {
        "lr_current_state": standard_metrics(y_true, probs_lr1, threshold=cfg["eval"]["risk_threshold"]),
        "lr_flattened_history": standard_metrics(y_true, probs_lr2, threshold=cfg["eval"]["risk_threshold"]),
        "persistence": standard_metrics(y_true, probs_persistence, threshold=cfg["eval"]["risk_threshold"]),
        "world_model": standard_metrics(y_true, probs_world_model, threshold=cfg["eval"]["risk_threshold"]),
        "oracle": standard_metrics(y_true, probs_oracle, threshold=cfg["eval"]["risk_threshold"]),
    }
    with open(metrics_dir / "baselines.json", "w") as f:
        json.dump({"split": split_name, "seed": seed, "baselines": baselines}, f, indent=2)
    results["baselines"] = baselines

    # --- Ablations ---
    logger.info("running ablations")
    ablations = {
        "persistence": persistence_ablation(X_eval, Y_eval, y_true, model),
        "time_shuffle": time_shuffle_ablation(X_eval, y_true, model, K=Y_eval.shape[1]),
        "horizon_curve": horizon_curve(eval_arrays.future_is_attack, Y_eval, model, X_eval, n_samples=n_samples),
        "surprise_signal": surprise_signal(X_eval, Y_eval, eval_arrays.future_is_attack, model),
    }
    with open(metrics_dir / "ablations.json", "w") as f:
        json.dump({"split": split_name, "seed": seed, "ablations": ablations}, f, indent=2)
    results["ablations"] = ablations

    # --- Calibration ---
    logger.info("running calibration")
    calibration = calibration_by_horizon(X_eval, eval_arrays.future_is_attack, model, n_samples=n_samples)
    with open(metrics_dir / "calibration.json", "w") as f:
        json.dump({"split": split_name, "seed": seed, "calibration": calibration}, f, indent=2)
    results["calibration"] = calibration

    # --- Lead time --- (uses the FULL split, not the capped eval_arrays —
    # see the comment above on why lead time needs complete per-host history)
    logger.info("computing lead-time report (this scans every pre-onset window per attacked host)")
    lead_time_report = compute_lead_time_report(
        full_eval_arrays, eval_split_df, model, scaler,
        threshold=cfg["eval"]["risk_threshold"], m=cfg["eval"]["lead_time_persistence_windows"], n_samples=n_samples,
    )
    with open(metrics_dir / "lead_time.json", "w") as f:
        json.dump({"split": split_name, "seed": seed, **lead_time_report.to_dict()}, f, indent=2)
    results["lead_time"] = lead_time_report.to_dict()

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(2)

    cfg = load_config(args.config)
    max_eval_samples = None if args.max_eval_samples == 0 else args.max_eval_samples
    results = run(cfg, args.seed, args.split, args.n_samples, max_eval_samples=max_eval_samples)
    print(json.dumps({k: v for k, v in results.items() if k not in ("baselines", "ablations", "calibration", "lead_time")}, indent=2))
    if "baselines" in results:
        print("baselines:", json.dumps({k: {"f1": v["f1"], "auc_pr": v["auc_pr"]} for k, v in results["baselines"].items()}, indent=2))
    if "lead_time" in results:
        print("lead_time:", json.dumps(results["lead_time"], indent=2))


if __name__ == "__main__":
    main()
