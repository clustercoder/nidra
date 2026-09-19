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
from nidra.data.preprocessing_audit import write_audit
from nidra.data.schema import FEATURE_INDEX
from nidra.data.windowize import CIC2017_TIMEBASE_TAG
from nidra.explain.shap_runner import build_shap_background, save_background
from nidra.models.world_model import WorldModel
from nidra.train.losses import dynamics_loss, free_running_metrics, teacher_forcing_schedule
from nidra.train.pipeline import build_all_splits, fit_scaler, geometry_from_config, scale_arrays
from nidra.utils.config import load_config, resolve_path
from nidra.utils.seed import set_seed

logger = logging.getLogger(__name__)

#: The scaler artifact's `method` stamp that this pipeline produces and will
#: reuse. A scaler saved by an older normalize.py is refit, never reused.
SCALER_METHOD_TAG = "feature_transforms+zscore_on_active_rows"


def resolve_sample_caps(cfg: dict, max_train_samples: int | None,
                         max_val_samples: int | None) -> tuple[int | None, int | None]:
    """CLI caps win; otherwise fall back to the config's `training_data`
    caps. Without a config fallback, the documented full-scale command
    (`--config config/default.yaml`, no flags) windowed the whole ~6.9M-window
    train split uncapped and was OOM-killed before the first epoch, while the
    published Run 3 checkpoints record 500000/50000 — i.e. the documented
    command could not produce the documented numbers."""
    training_data = cfg.get("training_data", {})
    resolved_train = max_train_samples if max_train_samples is not None else training_data.get("max_train_samples")
    resolved_val = max_val_samples if max_val_samples is not None else training_data.get("max_val_samples")
    return resolved_train, resolved_val


def _scaler_matches_data(meta_path: Path) -> bool:
    """Is a saved scaler safe to reuse for the data now in hand?

    Reuse across ENSEMBLE SEEDS is required — five members must share one input
    space. Reuse across a change to the DATA is silently wrong, and this is
    what tells the two apart.

    The flow timebase is the stamp because it is the thing that has actually
    changed the input distribution: correcting the CSV clock (see
    windowize.parse_cic_timestamp) turned 11 of the 45 features from
    ~always-zero into populated ones. A RobustScaler fit on the zero version
    has a degenerate spread for exactly those columns, and reusing it neither
    errors nor looks wrong in a log line — it just trains the model on a
    mangled input space. An artifact with no stamp predates the correction, so
    a missing key is a mismatch rather than a pass.
    """
    try:
        meta = json.loads(Path(meta_path).read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("scaler metadata at %s is unreadable — refitting", meta_path)
        return False
    found = meta.get("flow_timebase")
    method = meta.get("method")
    if found == CIC2017_TIMEBASE_TAG and method == SCALER_METHOD_TAG:
        return True
    if method != SCALER_METHOD_TAG:
        logger.warning("saved scaler uses method %r, this run is %r — refitting", method, SCALER_METHOD_TAG)
        return False
    logger.warning(
        "saved scaler was fit on flow timebase %r, this run is %r — refitting rather "
        "than scaling the corrected features by a spread measured before the fix",
        found, CIC2017_TIMEBASE_TAG,
    )
    return False


def prepare_training_data(cfg: dict, max_train_samples: int | None, max_val_samples: int | None, subsample_seed: int = 0):
    """Builds splits, windowed arrays, and the fitted scaler ONCE. Reused
    across every ensemble seed — the scaler in particular must be fit
    exactly once and never refit per seed."""
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    scaler_dir.mkdir(parents=True, exist_ok=True)

    max_train_samples, max_val_samples = resolve_sample_caps(cfg, max_train_samples, max_val_samples)
    logger.info("building splits from raw data (max_train_samples=%s, max_val_samples=%s)",
                max_train_samples, max_val_samples)
    splits = build_all_splits(cfg)
    _, L, K = geometry_from_config(cfg)
    # Windowize train/val only (test/holdout are never used by this
    # function) and cap DURING construction, not after — building the full
    # uncapped [N,L,F] float32 tensor first (N ~ 6.9M at full production
    # scale) needs ~35GB before any cap is applied, which reliably OOM-kills
    # a machine with well under that much RAM. See build_windowed_arrays's
    # docstring for the two-pass design that avoids this.
    windowed = {
        "train": build_windowed_arrays(splits.train, L=L, K=K,
                                        max_samples=max_train_samples, seed=subsample_seed),
        "val": build_windowed_arrays(splits.val, L=L, K=K,
                                      max_samples=max_val_samples, seed=subsample_seed),
    }

    scaler_path, meta_path = FeatureScaler.default_paths(scaler_dir)
    if scaler_path.exists() and meta_path.exists() and _scaler_matches_data(meta_path):
        logger.info("loading existing scaler (fit once across the ensemble, never refit per seed)")
        scaler = FeatureScaler.load(scaler_path, meta_path)
    else:
        logger.info("fitting scaler on TRAIN split only (%d train samples)", len(windowed["train"].X))
        scaler = fit_scaler(windowed["train"])
        scaler.save(scaler_path, meta_path, extra_metadata={
            "n_train_samples": int(len(windowed["train"].X)),
            "flow_timebase": CIC2017_TIMEBASE_TAG,
            "config_hash": cfg.get("_config_hash"),
            "git_commit": cfg.get("_git_commit"),
            "window_seconds": cfg.get("windowing", {}).get("window_seconds"),
        })
        # The audit is part of the artifact: which transform each feature got,
        # what it did to skew/saturation, what was dropped and why.
        last_states = windowed["train"].X[:, -1, :]
        write_audit(last_states, scaler, scaler_dir, active_mask=last_states[:, FEATURE_INDEX["is_active"]] > 0)
        logger.info("wrote preprocessing audit to %s (dropped: %s)", scaler_dir, scaler.dropped_features)
        # A background built from the previous scaler's output is in the
        # previous input space; it has to go with the scaler that made it.
        (scaler_dir / "shap_background.npy").unlink(missing_ok=True)

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


VALID_DYNAMICS_SELECTION = ("val_free_running_nll", "val_free_running_mse", "val_multistep_nll")


def dynamics_selection_score(epoch_metrics: dict, metric: str) -> float:
    """Lower is better. `val_multistep_nll` is the original teacher-forced
    criterion (kept for reproducing the Δ=30 artifacts); the free-running
    criteria score the model under its deployment behaviour — its own
    predictions fed back for K steps — which is what the forecast is."""
    if metric == "val_free_running_nll":
        return float(epoch_metrics["val_free_nll"])
    if metric == "val_free_running_mse":
        return float(epoch_metrics["val_free_mse"])
    if metric == "val_multistep_nll":
        return float(epoch_metrics["val_nll"])
    raise ValueError(f"unknown train_dynamics.selection_metric {metric!r}, expected one of {VALID_DYNAMICS_SELECTION}")


def nonsilent_sample_weights(windowed_split, weight: float) -> np.ndarray | None:
    """Per-sample weights: `weight` for samples whose origin window or any
    future window is active (something is happening), 1.0 for silent-to-
    silent samples. None when the weight is 1 (off)."""
    if weight is None or float(weight) == 1.0:
        return None
    active_origin = windowed_split.X[:, -1, :].astype("float64").__abs__().sum(axis=1) > 0
    active_future = np.abs(windowed_split.Y.astype("float64")).sum(axis=(1, 2)) > 0
    w = np.ones(len(windowed_split.X), dtype="float32")
    w[active_origin | active_future] = float(weight)
    return w


def evaluate_dynamics(model: WorldModel, loader: DataLoader, K: int, horizon_discount: float,
                      feature_mask: torch.Tensor | None, device: str) -> dict:
    """Teacher-forced multi-step NLL (the original validation number) plus
    the free-running per-horizon metrics, averaged over the loader."""
    model.eval()
    tf_losses = []
    sums = {"nll": 0.0, "mse": 0.0, "mse_persistence": 0.0, "coverage90": 0.0}
    per_k = {"nll": None, "mse": None, "mse_persistence": None, "coverage90": None}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            tf_losses.append(dynamics_loss(model, x, y, K=K, horizon_discount=horizon_discount,
                                           teacher_forcing_p=1.0, feature_mask=feature_mask).item())
            fr = free_running_metrics(model, x, y, K=K, feature_mask=feature_mask)
            b = x.shape[0]
            for key in sums:
                vec = fr[key].cpu().numpy() * b
                per_k[key] = vec if per_k[key] is None else per_k[key] + vec
            n += b
    out = {"val_nll": float(np.mean(tf_losses)) if tf_losses else float("nan")}
    if n:
        for key in sums:
            vec = per_k[key] / n
            out[f"val_free_{key}_by_k"] = [float(v) for v in vec]
        out["val_free_nll"] = float(np.mean(out["val_free_nll_by_k"]))
        out["val_free_mse"] = float(np.mean(out["val_free_mse_by_k"]))
        mse_p = np.array(out["val_free_mse_persistence_by_k"])
        mse_m = np.array(out["val_free_mse_by_k"])
        out["val_free_skill_by_k"] = [float(1.0 - m / p) if p > 0 else float("nan") for m, p in zip(mse_m, mse_p)]
        out["val_free_skill"] = float(1.0 - mse_m.sum() / mse_p.sum()) if mse_p.sum() > 0 else float("nan")
        out["val_free_coverage90"] = float(np.mean(out["val_free_coverage90_by_k"]))
    return out


def train_one_seed(cfg: dict, seed: int, epochs_override: int | None, windowed: dict, scaler: FeatureScaler,
                    device: str) -> dict:
    set_seed(seed)
    tcfg = cfg["train_dynamics"]
    mcfg = cfg["model"]
    epochs = epochs_override or tcfg["epochs"]
    selection_metric = tcfg.get("selection_metric", "val_free_running_nll")
    if selection_metric not in VALID_DYNAMICS_SELECTION:
        raise ValueError(f"unknown train_dynamics.selection_metric {selection_metric!r}")
    beta_nll = float(tcfg.get("beta_nll", 0.0))
    mse_aux_weight = float(tcfg.get("mse_aux_weight", 0.0))
    nonsilent_weight = float(tcfg.get("nonsilent_sample_weight", 1.0))

    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    X_train, Y_train = scale_arrays(windowed["train"], scaler)
    X_val, Y_val = scale_arrays(windowed["val"], scaler)
    sample_w = nonsilent_sample_weights(windowed["train"], nonsilent_weight)

    train_ds = WorldModelDataset(windowed["train"], X_train, Y_train, sample_weight=sample_w)
    val_ds = WorldModelDataset(windowed["val"], X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)
    feature_mask = torch.from_numpy(scaler.model_mask.astype("bool")).to(device)

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
    best_score = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    patience_left = tcfg["patience"]
    history = []
    logger.info("seed=%d dynamics: selecting on %s (beta_nll=%.2f mse_aux=%.2f nonsilent_weight=%.1f, %d/%d features kept)",
                seed, selection_metric, beta_nll, mse_aux_weight, nonsilent_weight,
                int(scaler.model_mask.sum()), len(scaler.model_mask))

    for epoch in range(epochs):
        model.train()
        tf_p = teacher_forcing_schedule(
            epoch, epochs, tcfg["teacher_forcing"]["start_p"], tcfg["teacher_forcing"]["end_p"],
            tcfg["teacher_forcing"]["anneal_fraction_of_epochs"],
        )
        train_losses = []
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            sw = batch["sample_weight"].to(device) if "sample_weight" in batch else None
            optimizer.zero_grad()
            loss = dynamics_loss(model, x, y, K=K, horizon_discount=tcfg["horizon_discount"], teacher_forcing_p=tf_p,
                                 feature_mask=feature_mask, beta_nll=beta_nll, mse_aux_weight=mse_aux_weight,
                                 sample_weight=sw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_metrics = evaluate_dynamics(model, val_loader, K, tcfg["horizon_discount"], feature_mask, device)
        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_record = {"epoch": epoch, "train_nll": train_mean, "teacher_forcing_p": tf_p, **val_metrics}
        history.append(epoch_record)
        logger.info("seed=%d epoch=%d train=%.4f val_tf_nll=%.4f val_free_nll=%.4f val_free_mse=%.4f skill=%.3f cov90=%.3f tf_p=%.2f",
                    seed, epoch, train_mean, val_metrics["val_nll"], val_metrics.get("val_free_nll", float("nan")),
                    val_metrics.get("val_free_mse", float("nan")), val_metrics.get("val_free_skill", float("nan")),
                    val_metrics.get("val_free_coverage90", float("nan")), tf_p)

        score = dynamics_selection_score(epoch_record, selection_metric)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            patience_left = tcfg["patience"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("seed=%d: early stopping at epoch %d (best %s=%.4f at epoch %d)",
                            seed, epoch, selection_metric, best_score, best_epoch)
                break

    model.load_state_dict(best_state)

    weights_path = weights_dir / f"model_seed_{seed}.pt"
    torch.save(model.state_dict(), weights_path)

    metadata = {
        "seed": seed,
        "stage": "dynamics_only",
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_val_multistep_nll": float(best_metrics.get("val_nll", float("nan"))),
        "best_val_free_running": {k: v for k, v in best_metrics.items() if k.startswith("val_free")},
        "loss_options": {"beta_nll": beta_nll, "mse_aux_weight": mse_aux_weight,
                         "nonsilent_sample_weight": nonsilent_weight, "horizon_discount": tcfg["horizon_discount"]},
        "features_kept": int(scaler.model_mask.sum()),
        "dropped_features": scaler.dropped_features,
        "n_train_samples": int(len(windowed["train"].X)),
        "n_val_samples": int(len(windowed["val"].X)),
        "geometry": {"window_seconds": cfg["windowing"].get("window_seconds"),
                     "context_length": cfg["windowing"].get("context_length"), "horizon_length": K},
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
