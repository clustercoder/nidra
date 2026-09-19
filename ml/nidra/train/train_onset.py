"""Stage 2b: the onset head — explicit future-target supervision.

Targets are P(an attack episode begins within h minutes | S_t) for
h in `onset.horizons_min`, defined only at origins strictly before an
onset (rows inside an episode are excluded — an ongoing attack is not a
forecast, see data/onset.py). Trained on OBSERVED states, selected on
validation natural-prevalence AP at `onset.selection_horizon_min`, then
frozen and saved as `onset_head_seed_<s>.pt` next to the world model.

The number that matters is written to the metadata as
`n_train_positives_by_horizon`: on CIC-IDS2017 there are only a few dozen
pre-onset windows in the training days, and a head that cannot learn from
them is a finding about the data, not something to paper over.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from nidra.data.normalize import FeatureScaler
from nidra.data.onset import DEFAULT_ONSET_HORIZONS_MIN
from nidra.models.heads import OnsetHead
from nidra.train.pipeline import scale_arrays
from nidra.train.train_heads import balanced_epoch_indices
from nidra.utils.config import resolve_path
from nidra.utils.seed import set_seed

logger = logging.getLogger(__name__)


def onset_head_path(weights_dir: Path, seed: int) -> Path:
    return Path(weights_dir) / f"onset_head_seed_{seed}.pt"


def _eligible_states(windowed, scaler: FeatureScaler, horizons: tuple[int, ...]):
    X, _ = scale_arrays(windowed, scaler)
    states = X[:, -1, :]
    del X
    targets = windowed.onset_targets(horizons)
    eligible = ~windowed.inside_episode
    weight = windowed.sample_weight if windowed.sample_weight is not None else np.ones(len(states))
    return states[eligible], targets[eligible], np.asarray(weight)[eligible], eligible


def train_onset_head_for_seed(cfg: dict, seed: int, windowed: dict, scaler: FeatureScaler, device: str = "cpu",
                              head_data: dict | None = None) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ocfg = cfg.get("onset", {})
    horizons = tuple(int(h) for h in ocfg.get("horizons_min", DEFAULT_ONSET_HORIZONS_MIN))
    sel_h = int(ocfg.get("selection_horizon_min", 5))
    if sel_h not in horizons:
        raise ValueError(f"onset.selection_horizon_min={sel_h} not in horizons {horizons}")
    sel_j = horizons.index(sel_h)
    hidden = int(ocfg.get("hidden", 64))
    epochs = int(ocfg.get("epochs", 30))
    patience = int(ocfg.get("patience", 6))
    lr = float(ocfg.get("lr", 1e-3))
    weight_decay = float(ocfg.get("weight_decay", 1e-4))
    batch_size = int(ocfg.get("batch_size", 256))
    input_noise = float(ocfg.get("input_noise", 0.3))
    pos_repeat = int(ocfg.get("pos_repeat", 20))
    neg_ratio = int(ocfg.get("neg_ratio", 10))
    hard_frac = float(ocfg.get("hard_negative_fraction", 0.5))
    broadest = len(horizons) - 1  # sampling treats "onset within max(h)" as the positive class

    if head_data is not None:
        tr, va = head_data["train"], head_data["val"]
        el_tr, el_va = ~tr.inside_episode, ~va.inside_episode
        s_tr, y_tr = tr.states[el_tr], tr.onset_targets(horizons)[el_tr]
        s_va, y_va, w_va = va.states[el_va], va.onset_targets(horizons)[el_va], np.ones(int(el_va.sum()))
        active_tr = tr.active[el_tr]
    else:
        s_tr, y_tr, _, _ = _eligible_states(windowed["train"], scaler, horizons)
        s_va, y_va, w_va, _ = _eligible_states(windowed["val"], scaler, horizons)
        active_tr = (windowed["train"].X[:, -1, :][~windowed["train"].inside_episode][:, _active_index()] > 0)
    n_pos_h = {str(h): int(y_tr[:, j].sum()) for j, h in enumerate(horizons)}
    n_pos_val_h = {str(h): int(y_va[:, j].sum()) for j, h in enumerate(horizons)}
    logger.info("seed=%d onset head: train eligible %d (positives by horizon %s), val eligible %d (%s)",
                seed, len(s_tr), n_pos_h, len(s_va), n_pos_val_h)

    head = OnsetHead(s_tr.shape[1], hidden, horizons).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    s_tr_t = torch.from_numpy(s_tr).float()
    y_tr_t = torch.from_numpy(y_tr.astype("float32"))
    s_va_t = torch.from_numpy(s_va).float().to(device)
    best = {"score": -np.inf, "state": None, "epoch": -1, "val_ap_by_horizon": None}
    history, patience_left = [], patience
    t0 = time.time()
    trainable = y_tr[:, broadest].sum() > 0
    for epoch in range(epochs if trainable else 0):
        head.train()
        idx = balanced_epoch_indices(y_tr[:, broadest], active_tr, rng, pos_repeat, neg_ratio, hard_frac)
        losses = []
        for lo in range(0, len(idx), batch_size):
            b = torch.from_numpy(idx[lo:lo + batch_size])
            s = s_tr_t[b].to(device)
            if input_noise > 0:
                s = s + input_noise * torch.randn_like(s)
            loss = F.binary_cross_entropy_with_logits(head(s), y_tr_t[b].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(head(s_va_t)).cpu().numpy()
        ap_h = {}
        for j, h in enumerate(horizons):
            ap_h[str(h)] = float(average_precision_score(y_va[:, j], p_va[:, j], sample_weight=w_va)) if y_va[:, j].sum() > 0 else float("nan")
        score = ap_h[str(sel_h)]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ap_natural_by_horizon": ap_h})
        logger.info("seed=%d onset epoch=%d loss=%.4f val_ap_nat@%dmin=%.4f (%.0fs)", seed, epoch, np.mean(losses), sel_h,
                    score if np.isfinite(score) else float("nan"), time.time() - t0)
        if np.isfinite(score) and score > best["score"] + 1e-5:
            best.update(score=score, epoch=epoch, val_ap_by_horizon=ap_h,
                        state={k: v.clone() for k, v in head.state_dict().items()})
            patience_left = patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = onset_head_path(weights_dir, seed)
    head.save(path)
    meta = {
        "horizons_min": list(horizons), "selection_horizon_min": sel_h, "best_epoch": best["epoch"],
        "best_val_ap_natural_by_horizon": best["val_ap_by_horizon"],
        "n_train_eligible": int(len(s_tr)), "n_train_positives_by_horizon": n_pos_h,
        "n_val_eligible": int(len(s_va)), "n_val_positives_by_horizon": n_pos_val_h,
        "trainable": bool(trainable), "recipe": {"input_noise": input_noise, "pos_repeat": pos_repeat, "neg_ratio": neg_ratio,
                                                  "hard_negative_fraction": hard_frac, "hidden": hidden},
        "history": history, "path": str(path),
    }
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=2))
    return meta


def _active_index() -> int:
    from nidra.data.schema import FEATURE_INDEX
    return FEATURE_INDEX["is_active"]


def score_onset_head(path: Path, states_scaled: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """[N, H] probabilities from a saved head, plus its horizons."""
    head = OnsetHead.load(path)
    with torch.no_grad():
        p = torch.sigmoid(head(torch.from_numpy(np.ascontiguousarray(states_scaled)).float())).numpy()
    return p, head.horizons_min
