"""Fits post-hoc Platt-scaling calibration for the risk head's ensemble
forecast, on the VALIDATION split only, and saves it alongside the trained
weights for both `run_eval.py` and `NidraPredictor` to pick up.

Run this once, after training (Stage 2 heads must already be frozen), and
again any time the weights are retrained. See `nidra/eval/calibrate.py` for
why this exists and what it does/doesn't change.

Usage:
    python -m nidra.scripts.fit_calibration --config config/mvp_2017.yaml
"""

from __future__ import annotations

import argparse
import logging
import time

import torch

import numpy as np

from nidra.data.dataset import build_windowed_arrays, subsample_stratified_by_risk
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH
from nidra.eval.baselines import ensemble_world_model_forecast
from nidra.eval.calibrate import fit_platt_by_horizon, save_calibration
from nidra.eval.run_eval import _build_model
from nidra.train.pipeline import build_all_splits, scale_arrays
from nidra.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def fit_and_save(
    cfg: dict, seeds: list[int] | None = None, n_samples_per_member: int = 100,
    max_val_samples: int | None = 4000, batch_size: int = 64,
) -> dict:
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    seeds = seeds if seeds is not None else cfg["ensemble"]["seeds"]

    scaler = FeatureScaler.load(scaler_dir / "robust_scaler.joblib", scaler_dir / "scaler_metadata.json")
    models = []
    for seed in seeds:
        path = weights_dir / f"model_seed_{seed}.pt"
        if not path.exists():
            logger.warning("fit_calibration: missing weights for seed %d at %s, skipping", seed, path)
            continue
        model = _build_model(cfg)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError(f"no ensemble weights found in {weights_dir}")

    # Only the val split's windowed tensors are needed here — unlike
    # run_eval.py (which also needs train for baseline fitting and
    # test/holdout for the split under evaluation), calibration fitting
    # uses val alone. build_windowed_splits() would windowize ALL FOUR
    # splits, including the uncapped multi-million-window train split — a
    # 10+ minute cost for no benefit to this script.
    splits = build_all_splits(cfg)
    val = build_windowed_arrays(splits.val, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    if len(val.X) == 0:
        raise RuntimeError("validation split is empty — cannot fit calibration without held-out labels")

    # Rollout cost scales with n_samples * n_ensemble_members * n_val_samples
    # — the same reason run_eval.py caps its eval set the same stratified
    # way (nidra.data.dataset.subsample_stratified_by_risk keeps every
    # positive risk_label row and fills the rest randomly, so the fit still
    # sees every positive example even on a capped slice).
    val = subsample_stratified_by_risk(val, max_val_samples, seed=0)
    X_val, _ = scale_arrays(val, scaler)
    K = val.future_is_attack.shape[1]

    logger.info(
        "fit_calibration: pooling %d ensemble member(s) x %d samples/member over %d val windows (batch_size=%d)",
        len(models), n_samples_per_member, len(X_val), batch_size,
    )
    # Batched, not one giant forward pass: the pooled rollout tensor is
    # [n_in_batch * n_ensemble_members * n_samples_per_member, L or K, F] —
    # at the FULL (uncapped) val split size this would be tens of GB in one
    # shot. batch_size=64 keeps peak memory to a few hundred MB regardless
    # of how large max_val_samples is set.
    risk_mean_k_parts = []
    for i in range(0, len(X_val), batch_size):
        batch = X_val[i : i + batch_size]
        out = ensemble_world_model_forecast(batch, models, K=K, n_samples_per_member=n_samples_per_member)
        risk_mean_k_parts.append(out["risk_mean_k"])
    risk_mean_k = np.concatenate(risk_mean_k_parts, axis=0)

    params_by_k = fit_platt_by_horizon(risk_mean_k, val.future_is_attack)

    metadata = {
        "fit_split": "val",
        "n_val_samples": int(len(X_val)),
        "n_ensemble_members": len(models),
        "seeds": seeds,
        "n_samples_per_member": n_samples_per_member,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "platt_scaling_per_horizon",
        "note": (
            "Fit on the validation split only, using the same pooled-ensemble "
            "rollout statistic (`ensemble_world_model_forecast`) that "
            "NidraPredictor serves at inference time. Does not retrain or "
            "modify the frozen risk head in any way — see eval/calibrate.py."
        ),
    }
    out_path = weights_dir / "risk_calibration.json"
    save_calibration(out_path, params_by_k, metadata)
    logger.info("fit_calibration: wrote %s", out_path)
    for k, p in enumerate(params_by_k):
        logger.info("  k=%d: a=%.4f b=%.4f n=%d degenerate=%s", k, p["a"], p["b"], p["n"], p["degenerate"])
    return {"path": str(out_path), "params_by_k": params_by_k, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description="Fit post-hoc risk-head calibration on the validation split.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-samples-per-member", type=int, default=100)
    parser.add_argument("--max-val-samples", type=int, default=4000,
                         help="cap on val samples used to fit calibration (stratified, keeps all positives); 0 disables capping")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(2)

    cfg = load_config(args.config)
    max_val_samples = None if args.max_val_samples == 0 else args.max_val_samples
    fit_and_save(cfg, n_samples_per_member=args.n_samples_per_member, max_val_samples=max_val_samples, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
