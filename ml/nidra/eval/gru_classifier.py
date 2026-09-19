"""The conventional-classifier baseline NIDRA must be compared against:
a GRU that reads the same L-window history and maps it straight to the
risk label. No transition model, no future state, no rollout — the thing
the problem statement says a world model should move beyond, and the
strongest fair non-world-model baseline on the same input.

Trained on the same stratified training sample the heads see, selected on
validation AP, saved next to the ensemble as `gru_classifier.pt`. It is a
BASELINE: nothing in serving reads it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from nidra.data.schema import FEATURE_ORDER

logger = logging.getLogger(__name__)


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int = 45, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_gru_classifier(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         out_path: str | Path, epochs: int = 8, batch: int = 256, lr: float = 1e-3,
                         hidden: int = 64, pos_repeat: int = 20, neg_ratio: int = 10, input_noise: float = 0.0,
                         seed: int = 0, threads: int | None = None) -> dict:
    """Balanced epochs (every positive repeated `pos_repeat` times, negatives
    freshly sampled at `neg_ratio` : 1), selection on validation AP."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if threads:
        torch.set_num_threads(threads)
    model = GRUClassifier(X_train.shape[-1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xv = torch.from_numpy(np.ascontiguousarray(X_val)).float()
    pos = np.where(y_train == 1)[0]
    neg = np.where(y_train == 0)[0]
    best = (-1.0, None, -1)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = np.concatenate([np.repeat(pos, pos_repeat),
                              rng.choice(neg, size=min(len(neg), len(pos) * pos_repeat * neg_ratio), replace=False)])
        rng.shuffle(idx)
        losses = []
        for lo in range(0, len(idx), batch):
            b = idx[lo:lo + batch]
            xb = torch.from_numpy(np.ascontiguousarray(X_train[b])).float()
            yb = torch.from_numpy(y_train[b].astype("float32"))
            if input_noise > 0:
                xb = xb + input_noise * torch.randn_like(xb)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(torch.cat([model(Xv[lo:lo + 4096]) for lo in range(0, len(Xv), 4096)])).numpy()
        ap = float(average_precision_score(y_val, pv)) if y_val.sum() > 0 else float("nan")
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_auc_pr": ap})
        logger.info("gru_classifier epoch=%d loss=%.4f val_ap=%.4f (%.0fs)", ep, np.mean(losses), ap, time.time() - t0)
        if ap > best[0]:
            best = (ap, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    if best[1] is not None:
        model.load_state_dict(best[1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": X_train.shape[-1], "hidden": hidden,
                "L": X_train.shape[1], "best_epoch": best[2], "best_val_auc_pr": best[0], "history": history,
                "feature_order": FEATURE_ORDER}, out_path)
    meta = {"best_epoch": best[2], "best_val_auc_pr": best[0], "history": history, "epochs": epochs,
            "hidden": hidden, "pos_repeat": pos_repeat, "neg_ratio": neg_ratio, "input_noise": input_noise}
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=2))
    return meta


def load_and_score(path: str | Path, X_scaled: np.ndarray) -> np.ndarray:
    ckpt = torch.load(path, map_location="cpu")
    model = GRUClassifier(ckpt["n_features"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    X = torch.from_numpy(np.ascontiguousarray(X_scaled)).float()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([model(X[lo:lo + 4096]) for lo in range(0, len(X), 4096)])).numpy()
