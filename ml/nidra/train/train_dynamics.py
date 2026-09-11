"""Stage-1 training: encoder + transition on multi-step unrolled Gaussian
NLL with scheduled sampling. Heads are NOT touched in this stage.

Usage:
    python -m nidra.train.train_dynamics --config config/default.yaml
    python -m nidra.train.train_dynamics --seed 0 --epochs 5 --max-train-samples 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nidra.data.dataset import WorldModelDataset, build_windowed_arrays
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH
from nidra.explain.shap_runner import build_shap_background, save_background
from nidra.models.world_model import WorldModel
from nidra.train.losses import dynamics_loss, teacher_forcing_schedule
from nidra.train.pipeline import build_all_splits, fit_scaler, scale_arrays
from nidra.utils.config import load_config, resolve_path
from nidra.utils.seed import set_seed

logger = logging.getLogger(__name__)


def prepare_training_data(cfg: dict, max_train_samples: int | None, max_val_samples: int | None, subsample_seed: int = 0):
    """Builds splits, windowed arrays, and the fitted scaler ONCE. Reused
    across every ensemble seed — the scaler in particular must be fit
    exactly once and never refit per seed."""
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    scaler_dir.mkdir(parents=True, exist_ok=True)

    logger.info("building splits from raw data")
    splits = build_all_splits(cfg)
    # Windowize train/val only (test/holdout are never used by this
    # function) and cap DURING construction, not after — building the full
    # uncapped [N,L,F] float32 tensor first (N ~ 6.9M at full production
    # scale) needs ~35GB before any cap is applied, which reliably OOM-kills
    # a machine with well under that much RAM. See build_windowed_arrays's
    # docstring for the two-pass design that avoids this.
    windowed = {
        "train": build_windowed_arrays(splits.train, L=CONTEXT_LENGTH, K=HORIZON_LENGTH,
                                        max_samples=max_train_samples, seed=subsample_seed),
        "val": build_windowed_arrays(splits.val, L=CONTEXT_LENGTH, K=HORIZON_LENGTH,
                                      max_samples=max_val_samples, seed=subsample_seed),
    }

    scaler_path = scaler_dir / "robust_scaler.joblib"
    meta_path = scaler_dir / "scaler_metadata.json"
    if scaler_path.exists() and meta_path.exists():
        logger.info("loading existing scaler (fit once across the ensemble, never refit per seed)")
        scaler = FeatureScaler.load(scaler_path, meta_path)
    else:
        logger.info("fitting scaler on TRAIN split only (%d train samples)", len(windowed["train"].X))
        scaler = fit_scaler(windowed["train"])
        scaler.save(scaler_path, meta_path, extra_metadata={"n_train_samples": int(len(windowed["train"].X))})

    background_path = scaler_dir / "shap_background.npy"
    if not background_path.exists():
        train_arrays = windowed["train"]
        benign_mask = train_arrays.stage_label == "benign"
        if benign_mask.any():
            benign_last_states = scaler.transform(train_arrays.X[benign_mask, -1, :])
            n_centroids = cfg.get("explain", {}).get("shap_background_centroids", 100)
            background = build_shap_background(benign_last_states, n_centroids=n_centroids)
            save_background(background, background_path)
            logger.info("saved SHAP background (%d centroids) to %s", background.shape[0], background_path)

    return windowed, scaler


def train_one_seed(cfg: dict, seed: int, epochs_override: int | None, windowed: dict, scaler: FeatureScaler,
                    device: str) -> dict:
    set_seed(seed)
    tcfg = cfg["train_dynamics"]
    mcfg = cfg["model"]
    epochs = epochs_override or tcfg["epochs"]

    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    X_train, Y_train = scale_arrays(windowed["train"], scaler)
    X_val, Y_val = scale_arrays(windowed["val"], scaler)

    train_ds = WorldModelDataset(windowed["train"], X_train, Y_train)
    val_ds = WorldModelDataset(windowed["val"], X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = WorldModel(
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
    ).to(device)

    params = list(model.encoder.parameters()) + list(model.transition.parameters())
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    K = cfg["windowing"]["horizon_length"]
    best_val = float("inf")
    patience_left = tcfg["patience"]
    history = []

    for epoch in range(epochs):
        model.train()
        tf_p = teacher_forcing_schedule(
            epoch, epochs, tcfg["teacher_forcing"]["start_p"], tcfg["teacher_forcing"]["end_p"],
            tcfg["teacher_forcing"]["anneal_fraction_of_epochs"],
        )
        train_losses = []
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            loss = dynamics_loss(model, x, y, K=K, horizon_discount=tcfg["horizon_discount"], teacher_forcing_p=tf_p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                loss = dynamics_loss(model, x, y, K=K, horizon_discount=tcfg["horizon_discount"], teacher_forcing_p=1.0)
                val_losses.append(loss.item())

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_nll": train_mean, "val_nll": val_mean, "teacher_forcing_p": tf_p})
        logger.info("seed=%d epoch=%d train_nll=%.4f val_nll=%.4f tf_p=%.2f", seed, epoch, train_mean, val_mean, tf_p)

        if val_mean < best_val - 1e-4:
            best_val = val_mean
            patience_left = tcfg["patience"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("seed=%d: early stopping at epoch %d (best val_nll=%.4f)", seed, epoch, best_val)
                break

    model.load_state_dict(best_state)

    weights_path = weights_dir / f"model_seed_{seed}.pt"
    torch.save(model.state_dict(), weights_path)

    metadata = {
        "seed": seed,
        "stage": "dynamics_only",
        "best_val_multistep_nll": best_val,
        "n_train_samples": int(len(windowed["train"].X)),
        "n_val_samples": int(len(windowed["val"].X)),
        "config_hash": cfg.get("_config_hash"),
        "git_commit": cfg.get("_git_commit"),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "history": history,
    }
    with open(weights_dir / f"model_seed_{seed}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Stage-1 NIDRA dynamics training.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="train a single seed; default trains the full ensemble")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(2)

    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg["ensemble"]["seeds"]

    windowed, scaler = prepare_training_data(cfg, args.max_train_samples, args.max_val_samples)
    for seed in seeds:
        train_one_seed(cfg, seed, args.epochs, windowed, scaler, args.device)


if __name__ == "__main__":
    main()
