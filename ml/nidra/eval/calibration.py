"""Calibration evaluation: per-horizon Brier score and reliability diagram.

Calibration is FIT (if at all — this module only evaluates) on validation
data, never on test data, so the reported test-set calibration numbers are
never contaminated by the same labels used to tune anything.
"""

from __future__ import annotations

import numpy as np

from nidra.eval.baselines import world_model_forecast
from nidra.eval.metrics import brier_score, reliability_diagram
from nidra.models.world_model import WorldModel


def calibration_by_horizon(
    X: np.ndarray,
    future_is_attack: np.ndarray,
    model: WorldModel,
    n_samples: int = 50,
    n_bins: int = 10,
) -> dict:
    """Returns per-horizon Brier score + reliability diagram data, computed
    on whichever split `X`/`future_is_attack` come from — caller is
    responsible for passing VALIDATION data if this is meant to inform any
    downstream calibration fitting, and TEST data only for final reporting.
    """
    K = future_is_attack.shape[1]
    world = world_model_forecast(X, model, K=K, n_samples=n_samples)
    risk_mean_k = world["risk_mean_k"]

    per_k = []
    for k in range(K):
        y_true = future_is_attack[:, k]
        y_prob = risk_mean_k[:, k]
        per_k.append({
            "k": k,
            "brier": brier_score(y_true, y_prob),
            "reliability": reliability_diagram(y_true, y_prob, n_bins=n_bins),
        })
    return {"per_horizon": per_k, "mean_brier": float(np.mean([p["brier"] for p in per_k]))}
