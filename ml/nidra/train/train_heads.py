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
from sklearn.metrics import average_precision_score

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_INDEX, STAGE_INDEX, STAGE_LABELS
from nidra.models.world_model import RiskHead, StageHead, WorldModel
from nidra.train.losses import risk_head_loss, stage_head_loss
from nidra.train.pipeline import scale_arrays
from nidra.utils.config import load_config, resolve_path
from nidra.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _stage_indices(stage_labels: np.ndarray) -> np.ndarray:
    return np.array([STAGE_INDEX[s] for s in stage_labels], dtype="int64")


VALID_SELECTION_METRICS = ("val_auc_pr", "val_auc_pr_natural", "weighted_val_loss")
VALID_SAMPLING = ("imbalanced", "balanced")


def head_selection_score(val_loss: float, val_auc_pr: float, metric: str) -> float:
    """Score for "is this epoch's head better?", where LOWER is better.

    `"weighted_val_loss"` is the original criterion (risk BCE at a huge
    pos_weight plus stage CE with class weights up to ~83,000), documented in
    REAL_DATA_RESULTS.md Run 6 as an accidental regularizer on a validation
    split that then held one Heartbleed episode. `"val_auc_pr"` selects on
    the risk head's validation AUC-PR; `"val_auc_pr_natural"` on the same
    statistic reweighted to the split's natural prevalence (the benchmark's
    primary metric — see eval/metrics_natural.py), which is the default from
    the Δ=60 rebuild on, now that validation holds two attack episodes of
    two families (DECISIONS.md D105). The stage head is selected separately.
    """
    if metric in ("val_auc_pr", "val_auc_pr_natural"):
        return -val_auc_pr if np.isfinite(val_auc_pr) else float("inf")
    if metric == "weighted_val_loss":
        return val_loss
    raise ValueError(f"unknown head selection metric {metric!r}, expected one of {VALID_SELECTION_METRICS}")


def compute_pos_weight(risk_labels: np.ndarray) -> torch.Tensor:
    n_pos = max(int(risk_labels.sum()), 1)
    n_neg = max(len(risk_labels) - n_pos, 1)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def compute_class_weights(stage_idx: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(stage_idx, minlength=n_classes).astype("float64")
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def balanced_epoch_indices(risk: np.ndarray, active: np.ndarray, rng: np.random.Generator,
                           pos_repeat: int, neg_ratio: int, hard_negative_fraction: float) -> np.ndarray:
    """One epoch of balanced batches: every positive `pos_repeat` times, and
    `neg_ratio` negatives per (repeated) positive, of which
    `hard_negative_fraction` are drawn from ACTIVE benign windows — the rows
    a detector actually has to reject — and the rest from silent ones."""
    pos = np.where(risk == 1)[0]
    neg = np.where(risk == 0)[0]
    n_neg = min(len(neg), len(pos) * pos_repeat * neg_ratio)
    hard_pool = neg[active[neg]]
    easy_pool = neg[~active[neg]]
    n_hard = min(len(hard_pool), int(round(n_neg * hard_negative_fraction)))
    n_easy = min(len(easy_pool), n_neg - n_hard)
    parts = [np.repeat(pos, pos_repeat)]
    if n_hard:
        parts.append(rng.choice(hard_pool, size=n_hard, replace=len(hard_pool) < n_hard))
    if n_easy:
        parts.append(rng.choice(easy_pool, size=n_easy, replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return idx


def _natural_weights(arrays) -> np.ndarray:
    w = getattr(arrays, "sample_weight", None)
    return np.ones(len(arrays.risk_label)) if w is None else np.asarray(w, dtype="float64")


def _head_inputs(windowed: dict, scaler: FeatureScaler, head_data: dict | None):
    """(s_train, risk_train, stage_train, active_train, s_val, risk_val, stage_val, w_val).
    With `head_data` (train/val HeadArrays over EVERY row of the split —
    train/head_data.py) validation needs no weights; without it the windowed
    subsample and its natural-prevalence weights are used (tests, legacy)."""
    if head_data is not None:
        tr, va = head_data["train"], head_data["val"]
        return (tr.states, tr.risk_label, tr.stage_idx, tr.active,
                va.states, va.risk_label, va.stage_idx, np.ones(len(va), dtype="float64"))
    X_train, _ = scale_arrays(windowed["train"], scaler)
    X_val, _ = scale_arrays(windowed["val"], scaler)
    s_train = X_train[:, -1, :]  # S_t: observed state at the sample's origin window
    s_val = X_val[:, -1, :]
    del X_train, X_val
    return (s_train, windowed["train"].risk_label, _stage_indices(windowed["train"].stage_label),
            windowed["train"].X[:, -1, FEATURE_INDEX["is_active"]] > 0,
            s_val, windowed["val"].risk_label, _stage_indices(windowed["val"].stage_label), _natural_weights(windowed["val"]))


def train_heads_for_seed(cfg: dict, seed: int, windowed: dict, scaler: FeatureScaler, device: str,
                         head_data: dict | None = None) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
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
        linear_skip=bool(mcfg["transition"].get("linear_skip", False)),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Stage 2 always starts from FRESHLY INITIALIZED heads. The checkpoint at
    # this path may already carry trained heads (it does whenever this script
    # is re-run against a completed pipeline), and continuing from those would
    # make the result depend on how many times the script had been run rather
    # than on (dynamics weights, data, seed) alone.
    model.risk_head = RiskHead(mcfg["n_features"], mcfg["risk_head"]["hidden"]).to(device)
    model.stage_head = StageHead(
        mcfg["n_features"], mcfg["stage_head"]["hidden"], mcfg["stage_head"]["n_stages"]
    ).to(device)
    model.freeze_dynamics()

    s_train, risk_train, stage_train, active_train, s_val, risk_val, stage_val, w_val = _head_inputs(windowed, scaler, head_data)

    sampling = hcfg.get("risk_sampling", "imbalanced")
    if sampling not in VALID_SAMPLING:
        raise ValueError(f"unknown train_heads.risk_sampling {sampling!r}, expected one of {VALID_SAMPLING}")
    input_noise = float(hcfg.get("input_noise", 0.0))
    pos_repeat = int(hcfg.get("pos_repeat", 20))
    neg_ratio = int(hcfg.get("neg_ratio", 10))
    hard_frac = float(hcfg.get("hard_negative_fraction", 0.5))
    selection_metric = hcfg.get("selection_metric", "val_auc_pr_natural")
    if selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"unknown train_heads.selection_metric {selection_metric!r}")

    pos_weight = compute_pos_weight(risk_train).to(device) if sampling == "imbalanced" else None
    class_weights = compute_class_weights(stage_train, len(STAGE_LABELS)).to(device)
    logger.info("seed=%d heads: sampling=%s input_noise=%.2f pos_repeat=%d neg_ratio=%d hard_neg=%.2f selecting on %s; "
                "train pos=%d/%d (active %.3f) val pos=%d/%d (%s)",
                seed, sampling, input_noise, pos_repeat, neg_ratio, hard_frac, selection_metric,
                int(risk_train.sum()), len(risk_train), float(active_train.mean()), int(risk_val.sum()), len(risk_val),
                "every row of the split" if head_data is not None else "windowed subsample, natural weights")

    s_train_t = torch.from_numpy(s_train).float()
    risk_train_t = torch.from_numpy(risk_train.astype("float32"))
    stage_train_t = torch.from_numpy(stage_train)
    s_val_t = torch.from_numpy(s_val).float().to(device)
    risk_val_t = torch.from_numpy(risk_val.astype("float32")).to(device)
    stage_val_t = torch.from_numpy(stage_val).to(device)
    batch_size = int(hcfg["batch_size"])

    # Independent optimizers and independent selection: the stage head's
    # class-weighted CE must not steer the risk head's checkpoint (or its
    # gradient clip), and vice versa.
    opt_risk = torch.optim.AdamW(model.risk_head.parameters(), lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])
    opt_stage = torch.optim.AdamW(model.stage_head.parameters(), lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])

    best_risk = {"score": float("inf"), "state": None, "epoch": -1, "val_auc_pr": float("nan"),
                 "val_auc_pr_natural": float("nan"), "val_loss": float("inf")}
    best_stage = {"score": float("inf"), "state": None, "epoch": -1}
    patience_left = hcfg["patience"]
    history = []

    for epoch in range(hcfg["epochs"]):
        model.risk_head.train()
        model.stage_head.train()
        if sampling == "balanced":
            idx = balanced_epoch_indices(risk_train, active_train, rng, pos_repeat, neg_ratio, hard_frac)
        else:
            idx = rng.permutation(len(risk_train))
        risk_losses, stage_losses = [], []
        for lo in range(0, len(idx), batch_size):
            b = torch.from_numpy(idx[lo:lo + batch_size])
            s = s_train_t[b].to(device)
            risk_y = risk_train_t[b].to(device)
            stage_y = stage_train_t[b].to(device)
            s_in = s + input_noise * torch.randn_like(s) if input_noise > 0 else s

            opt_risk.zero_grad()
            r_loss = risk_head_loss(model.risk_head(s_in), risk_y, pos_weight)
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), hcfg["grad_clip"])
            opt_risk.step()

            opt_stage.zero_grad()
            st_loss = stage_head_loss(model.stage_head(s_in), stage_y, class_weights)
            st_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.stage_head.parameters(), hcfg["grad_clip"])
            opt_stage.step()
            risk_losses.append(r_loss.item())
            stage_losses.append(st_loss.item())

        model.risk_head.eval()
        model.stage_head.eval()
        with torch.no_grad():
            val_risk_logits = model.risk_head(s_val_t)
            val_stage_logits = model.stage_head(s_val_t)
            val_risk_loss = risk_head_loss(val_risk_logits, risk_val_t, pos_weight).item()
            val_stage_loss = stage_head_loss(val_stage_logits, stage_val_t, class_weights).item()
            val_probs = torch.sigmoid(val_risk_logits.squeeze(-1)).cpu().numpy()
            val_stage_pred = val_stage_logits.argmax(dim=-1).cpu().numpy()

        two_class = len(np.unique(risk_val)) > 1
        val_auc_pr = float(average_precision_score(risk_val, val_probs)) if two_class else float("nan")
        val_auc_pr_nat = float(average_precision_score(risk_val, val_probs, sample_weight=w_val)) if two_class else float("nan")
        attack_val = stage_val != 0
        stage_acc_attack = float((val_stage_pred[attack_val] == stage_val[attack_val]).mean()) if attack_val.any() else float("nan")
        stage_macro_f1 = _macro_f1(stage_val, val_stage_pred, len(STAGE_LABELS))
        val_loss = val_risk_loss + val_stage_loss

        history.append({"epoch": epoch, "train_risk_loss": float(np.mean(risk_losses)), "train_stage_loss": float(np.mean(stage_losses)),
                        "val_loss": val_loss, "val_risk_loss": val_risk_loss, "val_stage_loss": val_stage_loss,
                        "val_auc_pr": val_auc_pr, "val_auc_pr_natural": val_auc_pr_nat,
                        "val_stage_acc_on_attacks": stage_acc_attack, "val_stage_macro_f1": stage_macro_f1})
        logger.info("seed=%d heads epoch=%d risk_loss=%.4f val_auc_pr=%.4f val_auc_pr_nat=%.4f | stage_loss=%.4f val_stage_acc_attack=%.3f macro_f1=%.3f",
                    seed, epoch, np.mean(risk_losses), val_auc_pr, val_auc_pr_nat, np.mean(stage_losses), stage_acc_attack, stage_macro_f1)

        metric_value = val_auc_pr_nat if selection_metric == "val_auc_pr_natural" else val_auc_pr
        score = head_selection_score(val_loss, metric_value, selection_metric)
        if not np.isfinite(score):
            # single-class validation (tiny fixtures, degenerate splits): AP is
            # undefined, so fall back to the risk BCE for this run and say so.
            if epoch == 0:
                logger.warning("seed=%d heads: validation has one risk class; selecting on val_risk_loss instead of %s",
                               seed, selection_metric)
            score = val_risk_loss
        improved = score < best_risk["score"] - 1e-5
        if improved:
            best_risk.update(score=score, epoch=epoch, val_auc_pr=val_auc_pr, val_auc_pr_natural=val_auc_pr_nat,
                             val_loss=val_loss, state={k: v.clone() for k, v in model.risk_head.state_dict().items()})
            patience_left = hcfg["patience"]
        else:
            patience_left -= 1
        stage_score = -stage_macro_f1 if np.isfinite(stage_macro_f1) else val_stage_loss
        if stage_score < best_stage["score"] - 1e-5:
            best_stage.update(score=stage_score, epoch=epoch, state={k: v.clone() for k, v in model.stage_head.state_dict().items()})
        if patience_left <= 0:
            logger.info("seed=%d heads: early stopping at epoch %d", seed, epoch)
            break

    if best_risk["state"] is not None:
        model.risk_head.load_state_dict(best_risk["state"])
    if best_stage["state"] is not None:
        model.stage_head.load_state_dict(best_stage["state"])

    model.freeze_heads()  # nothing is trained after this point
    torch.save(model.state_dict(), weights_path)

    metadata_path = weights_dir / f"model_seed_{seed}_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({
        "stage": "dynamics_and_frozen_heads",
        "heads_best_val_loss": best_risk["val_loss"],
        "heads_best_val_auc_pr": best_risk["val_auc_pr"],
        "heads_best_val_auc_pr_natural": best_risk["val_auc_pr_natural"],
        "heads_best_epoch_risk": best_risk["epoch"],
        "heads_best_epoch_stage": best_stage["epoch"],
        "heads_selection_metric": selection_metric,
        "heads_data": "all_split_rows" if head_data is not None else "windowed_subsample",
        "heads_n_train": int(len(risk_train)), "heads_n_train_pos": int(risk_train.sum()),
        "heads_n_val": int(len(risk_val)), "heads_n_val_pos": int(risk_val.sum()),
        "heads_recipe": {"risk_sampling": sampling, "input_noise": input_noise, "pos_repeat": pos_repeat,
                         "neg_ratio": neg_ratio, "hard_negative_fraction": hard_frac,
                         "separate_optimizers": True},
        "heads_pos_weight": None if pos_weight is None else pos_weight.item(),
        "heads_class_weights": class_weights.tolist(),
        "heads_trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heads_history": history,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


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
