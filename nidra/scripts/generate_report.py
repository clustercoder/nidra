"""Generates the plots (reports/*.png) and metadata JSONs
(artifacts/metadata/*.json) described in the project's artifact spec, from
metrics/weights that a real train_dynamics -> train_heads -> run_eval run
already produced.

This script never computes a metric itself — it only reads the JSON/weight
metadata those stages already wrote and renders/re-packages them. If a given
input file is missing (e.g. a holdout run was never executed), the
corresponding plot/field is skipped and logged, never fabricated.

Usage:
    python -m nidra.scripts.generate_report --config config/mvp_2017.yaml \
        --holdout-metrics-dir artifacts_mvp_2017/metrics_holdout \
        --reports-dir reports --metadata-dir artifacts_mvp_2017/metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nidra.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("generate_report: %s not found, skipping anything that depends on it", path)
        return None
    with open(path) as f:
        return json.load(f)


def _savefig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)


def plot_training_curves(weights_dir: Path, seed: int, reports_dir: Path) -> None:
    meta = _load_json(weights_dir / f"model_seed_{seed}_metadata.json")
    if meta is None or "history" not in meta:
        return
    hist = meta["history"]
    epochs = [h["epoch"] for h in hist]
    train_nll = [h["train_nll"] for h in hist]
    val_nll = [h["val_nll"] for h in hist]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_nll, label="train NLL", color="tab:blue")
    ax.set_xlabel("epoch"); ax.set_ylabel("multi-step Gaussian NLL")
    ax.set_title(f"Stage-1 dynamics training loss (seed {seed})")
    ax.legend()
    _savefig(fig, reports_dir, "training_loss.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, val_nll, label="val NLL", color="tab:orange")
    ax.set_xlabel("epoch"); ax.set_ylabel("multi-step Gaussian NLL")
    ax.set_title(f"Stage-1 dynamics validation loss (seed {seed})")
    ax.legend()
    _savefig(fig, reports_dir, "validation_loss.png")

    if "heads_history" in meta:
        hh = meta["heads_history"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h["epoch"] for h in hh], [h["train_loss"] for h in hh], label="heads train loss")
        ax.plot([h["epoch"] for h in hh], [h["val_loss"] for h in hh], label="heads val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("risk BCE + stage CE")
        ax.set_title(f"Stage-2 head training (seed {seed})")
        ax.legend()
        _savefig(fig, reports_dir, "heads_training_loss.png")


def plot_state_nrmse_horizon(ablations: dict, reports_dir: Path, split_label: str) -> None:
    hc = ablations.get("horizon_curve")
    if hc is None or "nrmse_by_k" not in hc:
        return
    k = list(range(1, len(hc["nrmse_by_k"]) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, hc["nrmse_by_k"], marker="o", color="tab:blue")
    ax.set_xlabel("forecast horizon k (30s windows)"); ax.set_ylabel("mean state nRMSE")
    ax.set_title(f"State forecast nRMSE by horizon ({split_label})")
    _savefig(fig, reports_dir, "state_nrmse_horizon.png")


def plot_auc_pr_horizon(ablations: dict, reports_dir: Path, split_label: str) -> None:
    hc = ablations.get("horizon_curve")
    if hc is None or "auc_pr_by_k" not in hc:
        return
    k = list(range(1, len(hc["auc_pr_by_k"]) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, hc["auc_pr_by_k"], marker="o", color="tab:green")
    ax.set_xlabel("forecast horizon k (30s windows)"); ax.set_ylabel("AUC-PR")
    ax.set_title(f"Risk AUC-PR by horizon ({split_label})" +
                 (" — FLAT-CURVE LEAKAGE WARNING" if hc.get("flat_curve_leakage_warning") else ""))
    _savefig(fig, reports_dir, "auc_pr_horizon.png")


def plot_baseline_comparison(baselines: dict, reports_dir: Path, out_name: str, split_label: str) -> None:
    # baselines.json can also carry non-metrics provenance entries (e.g.
    # "calibration_fit_metadata", added once run_eval.py started attaching
    # calibration info to this dict) — only plot entries that are actually
    # {f1, auc_pr} metrics rows.
    names = [n for n, v in baselines.items() if isinstance(v, dict) and "f1" in v and "auc_pr" in v]
    f1 = [baselines[n]["f1"] for n in names]
    auc = [baselines[n]["auc_pr"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, f1, width=0.36, label="F1")
    ax.bar(x + 0.18, auc, width=0.36, label="AUC-PR")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Baseline comparison — {split_label}")
    ax.legend()
    _savefig(fig, reports_dir, out_name)


def plot_persistence_ablation(ablations: dict, reports_dir: Path, split_label: str) -> None:
    p = ablations.get("persistence")
    if p is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["persistence", "world model"], [p["auc_pr_persistence"], p["auc_pr_world_model"]],
           color=["tab:gray", "tab:blue"])
    ax.set_ylabel("AUC-PR"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Persistence ablation ({split_label})\n{p['interpretation']}", fontsize=8, wrap=True)
    _savefig(fig, reports_dir, "persistence_ablation.png")


def plot_time_shuffle_ablation(ablations: dict, reports_dir: Path, split_label: str) -> None:
    t = ablations.get("time_shuffle")
    if t is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["normal order", "shuffled order"], [t["auc_pr_normal_order"], t["auc_pr_shuffled_order"]],
           color=["tab:blue", "tab:red"])
    ax.set_ylabel("AUC-PR"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Time-shuffle ablation ({split_label})\n{t['interpretation']}", fontsize=8, wrap=True)
    _savefig(fig, reports_dir, "time_shuffle_ablation.png")


def plot_lead_time_distribution(lead_time: dict, reports_dir: Path, split_label: str) -> None:
    dist = lead_time.get("lead_time_distribution_s")
    if not dist:
        logger.warning("generate_report: no lead-time distribution to plot for %s", split_label)
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dist, bins=min(10, len(dist)), color="tab:purple", alpha=0.8)
    ax.axvline(lead_time["median_lead_time_s"], color="black", linestyle="--", label="median")
    ax.set_xlabel("lead time (s)"); ax.set_ylabel("episodes")
    ax.set_title(
        f"Lead-time distribution ({split_label})\n"
        f"n_episodes={lead_time['n_episodes']}, no_warning_fraction={lead_time['fraction_no_warning']:.2f}",
        fontsize=9,
    )
    ax.legend()
    _savefig(fig, reports_dir, "lead_time_distribution.png")


def plot_calibration_reliability(calibration: dict, reports_dir: Path, split_label: str) -> None:
    per_k = calibration.get("per_horizon")
    if not per_k:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, entry in zip(axes.flat, per_k):
        rel = entry["reliability"]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.plot(rel["bin_centers"], rel["observed_frequency"], marker="o", markersize=3)
        ax.set_title(f"k={entry['k']+1}, brier={entry['brier']:.3f}", fontsize=8)
    fig.suptitle(f"Reliability diagrams by horizon ({split_label}) — mean_brier={calibration.get('mean_brier'):.3f}")
    fig.text(0.5, 0.02, "predicted p_compromise", ha="center")
    fig.text(0.02, 0.5, "observed frequency", va="center", rotation="vertical")
    _savefig(fig, reports_dir, "calibration_reliability.png")


def build_dataset_metadata(cfg: dict) -> dict:
    dataset_cfg = cfg["dataset"]
    return {
        "dataset": "CIC-IDS2017",
        "flow_dir": dataset_cfg.get("cic2017_flow_dir"),
        "packets_dir": dataset_cfg.get("packets_dir"),
        "days": dataset_cfg["days"],
        "train_days": cfg["splits"]["train_days"],
        "test_days": cfg["splits"]["test_days"],
        "holdout_days": cfg["splits"]["holdout_days"],
        "window_seconds": cfg["windowing"]["window_seconds"],
        "context_length": cfg["windowing"]["context_length"],
        "horizon_length": cfg["windowing"]["horizon_length"],
        "min_windows_per_host": cfg["windowing"]["min_windows_per_host"],
    }


def build_model_metadata(cfg: dict, weights_dir: Path, seeds: list[int]) -> dict:
    trained_seeds = [s for s in seeds if (weights_dir / f"model_seed_{s}.pt").exists()]
    return {
        "model_name": "NIDRAWorldModel",
        "architecture": "2-layer GRU + Gaussian transition + frozen heads",
        "input_features": cfg["model"]["n_features"],
        "context_windows": cfg["windowing"]["context_length"],
        "window_seconds": cfg["windowing"]["window_seconds"],
        "forecast_horizon": cfg["windowing"]["horizon_length"],
        "hidden_size": cfg["model"]["encoder"]["hidden_size"],
        "latent_size": cfg["model"]["encoder"]["hidden_size"],
        "ensemble_seeds_configured": seeds,
        "ensemble_seeds_trained": trained_seeds,
        "dataset": "CIC-IDS2017",
        "train_days": cfg["splits"]["train_days"],
        "test_days": cfg["splits"]["test_days"],
        "holdout": [d for d in cfg["splits"]["holdout_days"] if "infilt" in d],
        "risk_threshold": cfg["eval"]["risk_threshold"],
        "lead_rule_windows": cfg["eval"]["lead_time_persistence_windows"],
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate report plots and metadata JSONs from existing artifacts.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-metrics-dir", type=str, default=None, help="defaults to cfg artifacts.metrics_dir/test")
    parser.add_argument("--holdout-metrics-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--metadata-dir", type=str, default=None, help="defaults to <artifacts.root>/metadata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    test_metrics_dir = Path(args.test_metrics_dir) if args.test_metrics_dir else resolve_path(cfg, cfg["artifacts"]["metrics_dir"]) / "test"
    reports_dir = Path(args.reports_dir)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else resolve_path(cfg, cfg["artifacts"]["root"]) / "metadata"

    plot_training_curves(weights_dir, args.seed, reports_dir)

    test_baselines = _load_json(test_metrics_dir / "baselines.json")
    test_ablations = _load_json(test_metrics_dir / "ablations.json")
    test_calibration = _load_json(test_metrics_dir / "calibration.json")
    test_lead_time = _load_json(test_metrics_dir / "lead_time.json")

    if test_baselines:
        plot_baseline_comparison(test_baselines["baselines"], reports_dir, "baseline_comparison.png", "test (Friday)")
    if test_ablations:
        plot_state_nrmse_horizon(test_ablations["ablations"], reports_dir, "test (Friday)")
        plot_auc_pr_horizon(test_ablations["ablations"], reports_dir, "test (Friday)")
        plot_persistence_ablation(test_ablations["ablations"], reports_dir, "test (Friday)")
        plot_time_shuffle_ablation(test_ablations["ablations"], reports_dir, "test (Friday)")
    if test_calibration:
        plot_calibration_reliability(test_calibration["calibration"], reports_dir, "test (Friday)")
    if test_lead_time:
        plot_lead_time_distribution(test_lead_time, reports_dir, "test (Friday)")

    if args.holdout_metrics_dir:
        holdout_dir = Path(args.holdout_metrics_dir)
        holdout_baselines = _load_json(holdout_dir / "baselines.json")
        if holdout_baselines:
            plot_baseline_comparison(holdout_baselines["baselines"], reports_dir, "infiltration_holdout.png",
                                      "holdout (Thursday/Infiltration)")

    metadata_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta = build_dataset_metadata(cfg)
    model_meta = build_model_metadata(cfg, weights_dir, cfg["ensemble"]["seeds"])
    experiment_config = {
        "config_path": cfg.get("_config_path"),
        "config_hash": cfg.get("_config_hash"),
        "git_commit": git_commit(),
    }
    (metadata_dir / "dataset_metadata.json").write_text(json.dumps(dataset_meta, indent=2))
    (metadata_dir / "model_metadata.json").write_text(json.dumps(model_meta, indent=2))
    (metadata_dir / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2))
    logger.info("wrote metadata JSONs to %s", metadata_dir)


if __name__ == "__main__":
    main()
