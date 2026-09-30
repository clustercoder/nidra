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
from nidra.data.onset import DEFAULT_ONSET_HORIZONS_MIN, discrete_hazard_targets, survival_to_cumulative
from nidra.models.heads import OnsetHead, ordered_components
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


def _onset_parts(components: tuple[str, ...], state: torch.Tensor, ctx, idx, device: str) -> dict:
    """One batch's inputs for a history-aware onset head. The state carries
    whatever noise the caller added; the context components are left clean,
    exactly as in train_heads._risk_logits."""
    parts = {"state": state}
    for name in components:
        if name == "state":
            continue
        value = torch.from_numpy(getattr(ctx, name))
        parts[name] = (value[idx] if idx is not None else value).float().to(device)
    return parts


def train_onset_head_for_seed(cfg: dict, seed: int, windowed: dict, scaler: FeatureScaler, device: str = "cpu",
                              head_data: dict | None = None, model=None, head_tables: dict | None = None) -> dict:
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
    components = ordered_components(tuple(ocfg.get("components", ("state",))))
    needs_context = components != ("state",)
    if needs_context and model is None:
        raise ValueError(
            f"onset.components={list(components)} needs the encoder context, so the frozen "
            f"stage-1 `model` must be passed to train_onset_head_for_seed")
    parameterisation = str(ocfg.get("parameterisation", "independent"))
    if parameterisation not in OnsetHead.PARAMETERISATIONS:
        raise ValueError(f"unknown onset.parameterisation {parameterisation!r}; "
                         f"expected one of {OnsetHead.PARAMETERISATIONS}")

    if head_data is not None:
        tr, va = head_data["train"], head_data["val"]
        el_tr, el_va = ~tr.inside_episode, ~va.inside_episode
        s_tr, y_tr = tr.states[el_tr], tr.onset_targets(horizons)[el_tr]
        s_va, y_va, w_va = va.states[el_va], va.onset_targets(horizons)[el_va], np.ones(int(el_va.sum()))
        active_tr = tr.active[el_tr]
        hazard_event, hazard_at_risk = discrete_hazard_targets(
            tr.inside_episode[el_tr], tr.minutes_to_onset[el_tr], horizons)
    else:
        s_tr, y_tr, _, el = _eligible_states(windowed["train"], scaler, horizons)
        s_va, y_va, w_va, el_va = _eligible_states(windowed["val"], scaler, horizons)
        el_tr = el
        active_tr = (windowed["train"].X[:, -1, :][~windowed["train"].inside_episode][:, _active_index()] > 0)
        hazard_event, hazard_at_risk = discrete_hazard_targets(
            windowed["train"].inside_episode[el], windowed["train"].minutes_to_onset[el], horizons)
    ctx_tr = ctx_va = None
    if needs_context:
        from nidra.train.train_heads import _trajectory_context
        from nidra.train.pipeline import geometry_from_config
        _, L, _ = geometry_from_config(cfg)
        full_tr, full_va = _trajectory_context(cfg, model, scaler, L, windowed, head_data, head_tables, device)
        # The onset head trains on ELIGIBLE rows only (origins outside every
        # episode), so the context has to be masked identically or a row's
        # history would belong to a different row's state.
        ctx_tr, ctx_va = full_tr.select(el_tr), full_va.select(el_va)
        if len(ctx_tr) != len(s_tr) or len(ctx_va) != len(s_va):
            raise ValueError(f"onset head context misaligned: train {len(ctx_tr)} vs {len(s_tr)}, "
                             f"val {len(ctx_va)} vs {len(s_va)}")

    n_pos_h = {str(h): int(y_tr[:, j].sum()) for j, h in enumerate(horizons)}
    n_pos_val_h = {str(h): int(y_va[:, j].sum()) for j, h in enumerate(horizons)}
    logger.info("seed=%d onset head: train eligible %d (positives by horizon %s), val eligible %d (%s)",
                seed, len(s_tr), n_pos_h, len(s_va), n_pos_val_h)

    hidden_size = int(getattr(getattr(model, "encoder", None), "hidden_size", 0) or
                      cfg.get("model", {}).get("encoder", {}).get("hidden_size", 128))
    head = OnsetHead(s_tr.shape[1], hidden, horizons, components, hidden_size).to(device)
    head.parameterisation = parameterisation
    event_t = torch.from_numpy(hazard_event.astype("float32"))
    at_risk_t = torch.from_numpy(hazard_at_risk.astype("float32"))
    logger.info("seed=%d onset head parameterisation=%s; rows at risk per bucket %s, events %s",
                seed, parameterisation, hazard_at_risk.sum(0).tolist(), hazard_event.sum(0).tolist())
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
            logits = head(s) if not needs_context else head(**_onset_parts(components, s, ctx_tr, b, device))
            if parameterisation == "hazard":
                # Discrete-time survival: bucket j contributes only for the
                # rows still at risk in it. Censored rows (no onset within the
                # last horizon) are at risk everywhere and carry no event,
                # which is what they actually tell us.
                mask = at_risk_t[b].to(device)
                per = F.binary_cross_entropy_with_logits(logits, event_t[b].to(device), reduction="none")
                denom = mask.sum().clamp(min=1.0)
                loss = (per * mask).sum() / denom
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y_tr_t[b].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(
                head(s_va_t) if not needs_context
                else head(**_onset_parts(components, s_va_t, ctx_va, None, device))).cpu().numpy()
        if parameterisation == "hazard":
            p_va = survival_to_cumulative(p_va)
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
    head.save(path, parameterisation=parameterisation)
    meta = {
        "horizons_min": list(horizons), "selection_horizon_min": sel_h, "best_epoch": best["epoch"],
        "components": list(components), "input_dim": int(head.input_dim),
        "parameterisation": parameterisation,
        "n_at_risk_by_bucket": hazard_at_risk.sum(0).tolist(),
        "n_events_by_bucket": hazard_event.sum(0).tolist(),
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


def score_onset_head(path: Path, states_scaled: np.ndarray,
                     context: dict | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    """[N, H] probabilities from a saved head, plus its horizons.

    `context` supplies the non-state components a history-aware head declares
    (as `WorldModel.observed_context` returns them). A head that needs one and
    is not given it is refused rather than scored on a guess."""
    head = OnsetHead.load(path)
    needed = [c for c in head.components if c != "state"]
    if needed and context is None:
        raise ValueError(f"this onset head reads {needed} as well as the state; pass `context`")
    with torch.no_grad():
        state = torch.from_numpy(np.ascontiguousarray(states_scaled)).float()
        if not needed:
            logits = head(state)
        else:
            missing = [c for c in needed if context.get(c) is None]
            if missing:
                raise ValueError(f"`context` is missing the component(s) {missing} this onset head declares")
            parts = {"state": state}
            for c in needed:
                v = context[c]
                parts[c] = (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))).float()
            logits = head(**parts)
        p = torch.sigmoid(logits).numpy()
    if getattr(head, "parameterisation", "independent") == "hazard":
        p = survival_to_cumulative(p)
    return p, head.horizons_min
