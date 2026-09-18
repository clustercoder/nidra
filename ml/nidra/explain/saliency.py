"""Temporal saliency: "why is THIS FUTURE predicted?"

Input-gradient saliency (via Captum) from a predicted future-state feature
back to the historical input windows. This is mechanism (b) of the three
explainability questions in IMPLEMENTATION-ML.md §6 — deliberately distinct
from (a) KernelSHAP over the current observed state and (c) SHAP on the
stage head over a predicted state. Conflating the three is flagged in the
spec as the most common weakness in submissions of this type.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import Saliency

from nidra.data.schema import FEATURE_ORDER
from nidra.models.world_model import WorldModel


def _deterministic_rollout_feature(model: WorldModel, K: int, horizon_k: int, feature_idx: int):
    """Builds a differentiable forward function x -> predicted_state_feature
    at horizon `horizon_k`, feature `feature_idx`. Deterministic (no
    sampling noise) so the gradient reflects the mean trajectory, not one
    noisy draw."""

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = model.rollout(x, K=K, n_samples=1, stochastic=False)
        return out.states[:, 0, horizon_k, feature_idx]

    return forward


def temporal_saliency(
    model: WorldModel,
    x: np.ndarray,
    target_feature: str,
    horizon_k: int = 0,
    K: int = 6,
) -> dict:
    """x: [L, F] raw single sample (already scaled — caller's responsibility,
    matching how the model was trained). Returns per-window importance
    (summed absolute gradient across features) and the driving window
    index/indices — "the forecast is driven by the change between windows
    t-3 and t-1" is exactly this signal.
    """
    feature_idx = FEATURE_ORDER.index(target_feature)
    x_t = torch.from_numpy(x).float().unsqueeze(0)
    x_t.requires_grad_(True)

    forward_fn = _deterministic_rollout_feature(model, K, horizon_k, feature_idx)
    saliency = Saliency(forward_fn)
    grads = saliency.attribute(x_t)  # [1, L, F]

    window_importance = grads.abs().sum(dim=2).squeeze(0).detach().numpy()  # [L]
    L = window_importance.shape[0]
    driving_window = int(np.argmax(window_importance))

    return {
        "target_feature": target_feature,
        "horizon_k": horizon_k,
        "window_importance": window_importance.tolist(),
        "driving_window": driving_window,
        "driving_window_offset_from_now": driving_window - (L - 1),  # e.g. -3 means "3 windows before t"
    }
