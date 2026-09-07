"""Stage-2 training: risk head + stage head, trained ONLY on observed
states (S_t = the last window of each sample's input history), with the
encoder and transition frozen beforehand.

After this stage completes, every parameter in the model is frozen —
nothing is trained after `model.freeze_all()` runs at the end of this
script. There is no code path here that trains a head on rollout output.

Usage:
    python -m nidra.train.train_heads --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from nidra.data.dataset import WorldModelDataset
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import STAGE_INDEX, STAGE_LABELS
from nidra.models.world_model import WorldModel
from nidra.train.losses import risk_head_loss, stage_head_loss
from nidra.train.pipeline import build_all_splits, build_windowed_splits, scale_arrays
from nidra.utils.config import load_config, resolve_path
from nidra.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _stage_indices(stage_labels: np.ndarray) -> np.ndarray:
    return np.array([STAGE_INDEX[s] for s in stage_labels], dtype="int64")


def compute_pos_weight(risk_labels: np.ndarray) -> torch.Tensor:
    n_pos = max(int(risk_labels.sum()), 1)
    n_neg = max(len(risk_labels) - n_pos, 1)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def compute_class_weights(stage_idx: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(stage_idx, minlength=n_classes).astype("float64")
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_heads_for_seed(cfg: dict, seed: int, windowed: dict, scaler: FeatureScaler, device: str) -> dict:
    set_seed(seed)
    hcfg = cfg["train_heads"]
    mcfg = cfg["model"]
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    weights_path = weights_dir / f"model_seed_{seed}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"expected stage-1 weights at {weights_path} — run train_dynamics first")

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
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.freeze_dynamics()

    X_train, _ = scale_arrays(windowed["train"], scaler)
    X_val, _ = scale_arrays(windowed["val"], scaler)
    s_train = X_train[:, -1, :]  # S_t: observed state at the sample's origin window
    s_val = X_val[:, -1, :]

    risk_train = windowed["train"].risk_label
    risk_val = windowed["val"].risk_label
    stage_train = _stage_indices(windowed["train"].stage_label)
    stage_val = _stage_indices(windowed["val"].stage_label)

    pos_weight = compute_pos_weight(risk_train).to(device)
    class_weights = compute_class_weights(stage_train, len(STAGE_LABELS)).to(device)
    logger.info("seed=%d: pos_weight=%.3f class_weights=%s", seed, pos_weight.item(), class_weights.tolist())

    s_train_t = torch.from_numpy(s_train).float()
    risk_train_t = torch.from_numpy(risk_train.astype("float32"))
    stage_train_t = torch.from_numpy(stage_train)
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(s_train_t, risk_train_t, stage_train_t),
        batch_size=hcfg["batch_size"], shuffle=True,
    )
    s_val_t = torch.from_numpy(s_val).float().to(device)
    risk_val_t = torch.from_numpy(risk_val.astype("float32")).to(device)
    stage_val_t = torch.from_numpy(stage_val).to(device)

    head_params = list(model.risk_head.parameters()) + list(model.stage_head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])

    best_val = float("inf")
    patience_left = hcfg["patience"]
    best_state = None
    history = []

    for epoch in range(hcfg["epochs"]):
        model.risk_head.train()
        model.stage_head.train()
        epoch_losses = []
        for s, risk_y, stage_y in train_loader:
            s, risk_y, stage_y = s.to(device), risk_y.to(device), stage_y.to(device)
            optimizer.zero_grad()
            risk_logits = model.risk_head(s)
            stage_logits = model.stage_head(s)
            loss = risk_head_loss(risk_logits, risk_y, pos_weight) + stage_head_loss(stage_logits, stage_y, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, hcfg["grad_clip"])
            optimizer.step()
            epoch_losses.append(loss.item())

        model.risk_head.eval()
        model.stage_head.eval()
        with torch.no_grad():
            val_risk_logits = model.risk_head(s_val_t)
            val_stage_logits = model.stage_head(s_val_t)
            val_loss = (risk_head_loss(val_risk_logits, risk_val_t, pos_weight)
                        + stage_head_loss(val_stage_logits, stage_val_t, class_weights)).item()

        train_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_mean, "val_loss": val_loss})
        logger.info("seed=%d heads epoch=%d train_loss=%.4f val_loss=%.4f", seed, epoch, train_mean, val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_left = hcfg["patience"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("seed=%d heads: early stopping at epoch %d", seed, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.freeze_heads()  # nothing is trained after this point
    torch.save(model.state_dict(), weights_path)

    metadata_path = weights_dir / f"model_seed_{seed}_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({
        "stage": "dynamics_and_frozen_heads",
        "heads_best_val_loss": best_val,
        "heads_pos_weight": pos_weight.item(),
        "heads_class_weights": class_weights.tolist(),
        "heads_trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heads_history": history,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Stage-2 NIDRA head training (observed states only, then frozen).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(2)

    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg["ensemble"]["seeds"]

    from nidra.train.train_dynamics import prepare_training_data
    windowed, scaler = prepare_training_data(cfg, args.max_train_samples, args.max_val_samples)

    for seed in seeds:
        train_heads_for_seed(cfg, seed, windowed, scaler, args.device)


if __name__ == "__main__":
    main()
