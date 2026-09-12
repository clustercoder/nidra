"""The falsification suite (IMPLEMENTATION-ML.md §5.4 / PRD §7.4).

Each of these is a real experiment with a real possible failure. Rule 3
(CLAUDE.md): if persistence matches the model, or time-shuffle doesn't
collapse performance, that is the reported result — never hidden, never
engineered away.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from nidra.eval.baselines import baseline_persistence, world_model_forecast
from nidra.eval.metrics import state_nrmse_by_horizon
from nidra.models.world_model import WorldModel


def persistence_ablation(
    X: np.ndarray, Y: np.ndarray, risk_label: np.ndarray, model: WorldModel, feature_scale: np.ndarray | None = None
) -> dict:
    """Replace the transition model with copy-forward (mu=0): S_hat[t+k]=S_t
    for every k. Compares risk-prediction AUC-PR and state nRMSE against
    the full world model's rollout on the SAME inputs.

    Expected: substantial collapse in AUC/lead-time relative to the world
    model. If it does NOT collapse, that means no useful dynamics were
    learned — reported honestly, not hidden.
    """
    K = Y.shape[1]
    X_last = X[:, -1, :]
    persistence_risk = baseline_persistence(X_last, model)
    persisted_states = np.repeat(X_last[:, None, :], K, axis=1)  # S_hat[t+k] = S_t for all k

    world = world_model_forecast(X, model, K=K, n_samples=50)

    auc_persistence = float(average_precision_score(risk_label, persistence_risk)) if len(np.unique(risk_label)) > 1 else float("nan")
    auc_world_model = float(average_precision_score(risk_label, world["risk_over_horizon"])) if len(np.unique(risk_label)) > 1 else float("nan")

    nrmse_persistence = state_nrmse_by_horizon(Y, persisted_states, feature_scale)
    nrmse_world_model = state_nrmse_by_horizon(Y, world["predicted_states_mean"], feature_scale)

    return {
        "auc_pr_persistence": auc_persistence,
        "auc_pr_world_model": auc_world_model,
        "auc_collapse": auc_world_model - auc_persistence,
        "nrmse_persistence_mean": float(nrmse_persistence.mean()),
        "nrmse_world_model_mean": float(nrmse_world_model.mean()),
        "interpretation": (
            "world model beats persistence" if auc_world_model > auc_persistence + 0.02
            else "NO MEANINGFUL GAP over persistence — dynamics may not be adding value; "
                 "this is a reportable negative result, not a bug to hide"
        ),
    }


def time_shuffle_ablation(X: np.ndarray, risk_label: np.ndarray, model: WorldModel, K: int = 6, seed: int = 0) -> dict:
    """Shuffle window order WITHIN each input sequence (per-sample
    permutation of the L axis) and re-run the world-model forecast.

    Expected: performance collapses, because the encoder can no longer read
    a trend. If it does NOT collapse, the model is using per-window
    features only and the temporal claim is false — this must be reported,
    not suppressed.
    """
    rng = np.random.default_rng(seed)
    X_shuffled = X.copy()
    L = X.shape[1]
    for i in range(X.shape[0]):
        perm = rng.permutation(L)
        X_shuffled[i] = X_shuffled[i, perm]

    world_normal = world_model_forecast(X, model, K=K, n_samples=50)
    world_shuffled = world_model_forecast(X_shuffled, model, K=K, n_samples=50)

    has_two_classes = len(np.unique(risk_label)) > 1
    auc_normal = float(average_precision_score(risk_label, world_normal["risk_over_horizon"])) if has_two_classes else float("nan")
    auc_shuffled = float(average_precision_score(risk_label, world_shuffled["risk_over_horizon"])) if has_two_classes else float("nan")

    return {
        "auc_pr_normal_order": auc_normal,
        "auc_pr_shuffled_order": auc_shuffled,
        "collapse": auc_normal - auc_shuffled,
        "interpretation": (
            "temporal order matters (collapse observed)" if auc_normal > auc_shuffled + 0.02
            else "NO COLLAPSE under shuffling — the model may be using per-window features only, "
                 "not real temporal structure; this is a reportable negative result, not a bug to hide"
        ),
    }


def horizon_curve(
    future_is_attack: np.ndarray,
    Y: np.ndarray,
    model: WorldModel,
    X: np.ndarray,
    n_samples: int = 50,
    feature_scale: np.ndarray | None = None,
) -> dict:
    """AUC-PR and state nRMSE plotted against horizon k. Smooth degradation
    is expected; a FLAT curve across all k is a leakage red flag, not a
    good result (IMPLEMENTATION-ML.md §5.4)."""
    K = Y.shape[1]
    world = world_model_forecast(X, model, K=K, n_samples=n_samples)
    risk_mean_k = world["risk_mean_k"]          # [N, K]
    pred_states_k = world["predicted_states_mean"]  # [N, K, F]

    auc_per_k = []
    for k in range(K):
        labels_k = future_is_attack[:, k]
        if len(np.unique(labels_k)) > 1:
            auc_per_k.append(float(average_precision_score(labels_k, risk_mean_k[:, k])))
        else:
            auc_per_k.append(float("nan"))

    nrmse_per_k = state_nrmse_by_horizon(Y, pred_states_k, feature_scale).mean(axis=1)  # mean over features -> [K]

    is_flat = (np.nanmax(auc_per_k) - np.nanmin(auc_per_k)) < 0.02 if not all(np.isnan(auc_per_k)) else True

    return {
        "auc_pr_by_k": auc_per_k,
        "nrmse_by_k": nrmse_per_k.tolist(),
        "flat_curve_leakage_warning": bool(is_flat),
    }


def surprise_signal(
    X: np.ndarray,
    Y: np.ndarray,
    future_is_attack: np.ndarray,
    model: WorldModel,
) -> dict:
    """Compares one-step-ahead state-forecast error (nRMSE at k=1) between
    benign windows and PRE-ATTACK windows (samples whose horizon contains
    an attack, i.e. risk_label=1 but the origin window itself is benign).
    A rise in forecast error before onset is a regime-change signal a
    classifier cannot produce, since it never emits a state."""
    with torch.no_grad():
        out = model.rollout(torch.from_numpy(X).float(), K=1, n_samples=1, stochastic=False)
    pred_k1 = out.states[:, 0, 0, :].numpy()   # [N, F]
    true_k1 = Y[:, 0, :]

    per_sample_error = np.sqrt(np.mean((pred_k1 - true_k1) ** 2, axis=1))
    # "pre-attack": the origin window is benign but an attack occurs
    # somewhere in its forecast horizon — the regime-change window this
    # ablation is meant to detect.
    pre_attack_mask = future_is_attack.any(axis=1)
    benign_mask = ~pre_attack_mask

    benign_error = per_sample_error[benign_mask]
    pre_attack_error = per_sample_error[pre_attack_mask]

    return {
        "mean_error_benign": float(benign_error.mean()) if len(benign_error) else float("nan"),
        "mean_error_pre_attack": float(pre_attack_error.mean()) if len(pre_attack_error) else float("nan"),
        "n_benign": int(benign_mask.sum()),
        "n_pre_attack": int(pre_attack_mask.sum()),
        "error_rises_before_onset": bool(
            len(pre_attack_error) and len(benign_error) and pre_attack_error.mean() > benign_error.mean()
        ),
    }
