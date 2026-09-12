"""Reality-overlay demo (task spec: "Reality overlay"): generate a forecast
at time t using ONLY observations up to t, then overlay it against what was
subsequently actually observed at t+1..t+K.

This is a reproducible check on the same discipline the rollout itself
follows — NidraPredictor.forecast() never receives Y (the future), only the
[L, F] history — and it is the same forecast object shape the backend
consumes (origin_ts, horizons, p_compromise, ci_low, ci_high,
predicted_features, stage distribution, model_version).

Usage:
    python -m nidra.scripts.reality_overlay --config config/mvp_2017.yaml \
        --weights-dir artifacts_mvp_2017/weights --scaler-dir artifacts_mvp_2017/scaler \
        --split test --out-json reality_overlay.json --out-png reports/reality_overlay.png
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER
from nidra.eval.metrics import state_nrmse
from nidra.serve.predictor import NidraPredictor
from nidra.train.pipeline import build_all_splits, build_windowed_splits
from nidra.utils.config import load_config

logger = logging.getLogger(__name__)

# A few human-legible features to plot; the full 45-feature comparison is
# still written to the JSON output regardless of which ones get plotted.
PLOT_FEATURES = ["syn_ratio", "out_degree", "dst_port_entropy", "bytes_total"]


def pick_pre_attack_sample(arrays, seed: int = 0):
    """Picks a sample whose origin window is benign but whose forecast
    horizon contains an attack — the exact regime-change case the reality
    overlay is meant to illustrate. Falls back to a random sample if none
    exists in this split (e.g. an all-benign slice)."""
    rng = np.random.default_rng(seed)
    if arrays.future_is_attack is not None:
        pre_attack = np.where(arrays.future_is_attack.any(axis=1) & (arrays.risk_label == 1))[0]
        if len(pre_attack) > 0:
            return int(rng.choice(pre_attack))
    return int(rng.integers(0, len(arrays.X)))


def run(cfg: dict, weights_dir: str, scaler_dir: str, split: str, seed: int) -> dict:
    # seeds=None -> the predictor loads every ensemble member configured
    # (cfg["ensemble"]["seeds"]), pooling their trajectories — this demo
    # should show the real ensemble forecast, not a single seed. `seed` is
    # used only below, to pick which sample to overlay deterministically.
    predictor = NidraPredictor(weights_dir=weights_dir, scaler_path=Path(scaler_dir) / "robust_scaler.joblib",
                                config_path=cfg.get("_config_path"))

    splits = build_all_splits(cfg)
    windowed = build_windowed_splits(splits)[split]
    if len(windowed.X) == 0:
        raise RuntimeError(f"reality_overlay: split={split} has no samples")

    idx = pick_pre_attack_sample(windowed, seed=seed)
    X = windowed.X[idx]           # [L, F] raw, unscaled
    Y_true = windowed.Y[idx]      # [K, F] raw, unscaled — NEVER passed to forecast()
    host_id = str(windowed.host_id[idx])
    origin_ts = datetime.fromtimestamp(int(windowed.origin_ts[idx]), tz=timezone.utc)

    forecast = predictor.forecast(X, host_id=host_id, origin_ts=origin_ts)

    # Primary state-forecast quality metric: nRMSE in the model's scaled
    # operating space (see eval/metrics.state_nrmse) — a stable, unit-
    # normalized comparison across all 45 features. A single BLENDED raw-unit
    # RMSE across features is not used as the primary number here: several
    # features (flow_duration_var, iat_var, and their delta/slope
    # derivatives) are variance-of-a-duration quantities in squared
    # microsecond units, so their natural raw-unit dynamic range spans many
    # orders of magnitude independent of forecast quality — a single
    # blended raw RMSE would be dominated by whichever of those happens to
    # be large in this window, exactly the scaled-vs-raw-units pitfall
    # flagged in the project's own metric-auditing guidance.
    scaler = FeatureScaler.load(Path(scaler_dir) / "robust_scaler.joblib", Path(scaler_dir) / "scaler_metadata.json")
    Y_true_scaled = scaler.transform(Y_true)
    per_horizon_error = []
    for k, h in enumerate(forecast["horizons"]):
        pred_raw = np.array([h["predicted_features"][f] for f in FEATURE_ORDER])
        pred_scaled = scaler.transform(pred_raw[None, :])[0]
        nrmse_scaled = float(state_nrmse(Y_true_scaled[k][None, :], pred_scaled[None, :], scaler.reference_std_).mean())
        raw_abs_error = {name: float(abs(pred_raw[i] - Y_true[k, i])) for i, name in enumerate(FEATURE_ORDER)}
        per_horizon_error.append({
            "k": h["k"],
            "state_nrmse_scaled": nrmse_scaled,
            "raw_abs_error_per_feature": raw_abs_error,
        })

    return {
        "host_id": host_id,
        "origin_ts": forecast["origin_ts"],
        "split": split,
        "forecast": forecast,
        "actual_future_features": [
            {name: float(v) for name, v in zip(FEATURE_ORDER, Y_true[k])} for k in range(Y_true.shape[0])
        ],
        "actual_future_is_attack": windowed.future_is_attack[idx].tolist() if windowed.future_is_attack is not None else None,
        "per_horizon_state_error": per_horizon_error,
        "note": "Forecast generated from observations up to origin_ts only. "
                "actual_future_features/actual_future_is_attack were observed AFTER forecast() returned "
                "and are used here only to score the forecast, never as model input. "
                "state_nrmse_scaled is the primary quality metric (unit-normalized, comparable across "
                "features); raw_abs_error_per_feature is illustrative only and will look large for "
                "variance-type features (flow_duration_var, iat_var, and their derivatives) by "
                "construction, since those are squared-microsecond quantities with a naturally huge "
                "dynamic range independent of forecast quality.",
    }


def plot_overlay(result: dict, out_png: str) -> None:
    horizons = result["forecast"]["horizons"]
    k = [h["k"] for h in horizons]

    fig, axes = plt.subplots(len(PLOT_FEATURES) + 1, 1, figsize=(7, 2.2 * (len(PLOT_FEATURES) + 1)), sharex=True)

    ax = axes[0]
    p_mean = [h["p_compromise"] for h in horizons]
    ci_low = [h["ci_low"] for h in horizons]
    ci_high = [h["ci_high"] for h in horizons]
    ax.plot(k, p_mean, marker="o", color="tab:red", label="projected p_compromise")
    ax.fill_between(k, ci_low, ci_high, color="tab:red", alpha=0.2, label="confidence band")
    is_attack = result.get("actual_future_is_attack")
    if is_attack:
        for ki, a in zip(k, is_attack):
            if a:
                ax.axvspan(ki - 0.5, ki + 0.5, color="black", alpha=0.08)
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="risk threshold")
    ax.set_ylabel("p_compromise"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    ax.set_title(f"Reality overlay — host {result['host_id']}, origin {result['origin_ts']}", fontsize=9)

    for ax, feat in zip(axes[1:], PLOT_FEATURES):
        predicted = [h["predicted_features"][feat] for h in horizons]
        actual = [result["actual_future_features"][i][feat] for i in range(len(horizons))]
        ax.plot(k, predicted, marker="o", color="tab:blue", label="predicted (simulated)")
        ax.plot(k, actual, marker="x", color="tab:green", label="actual (observed after forecast)")
        ax.set_ylabel(feat, fontsize=8)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("forecast horizon k (30s windows ahead of NOW)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out_png)


def main():
    parser = argparse.ArgumentParser(description="Reality-overlay demo: forecast at t vs. subsequently observed states.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, required=True)
    parser.add_argument("--scaler-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "holdout", "val"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=str, default="reality_overlay.json")
    parser.add_argument("--out-png", type=str, default="reports/reality_overlay.png")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    result = run(cfg, args.weights_dir, args.scaler_dir, args.split, args.seed)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", args.out_json)

    plot_overlay(result, args.out_png)


if __name__ == "__main__":
    main()
