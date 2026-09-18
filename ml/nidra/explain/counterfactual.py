"""Counterfactual rollout: "what would THIS MODEL predict if this feature
were held at this value?"

This is explicitly NOT causal inference, NOT an intervention on the real
network, and NOT a claim about attacker response. It answers a question
about the MODEL, not about the network — every result from this module
must be labelled "model-internal what-if" wherever it surfaces (API
response, UI, docs). See CLAUDE.md claims-discipline rule and
IMPLEMENTATION-ML.md §6(d).
"""

from __future__ import annotations

import numpy as np
import torch

from nidra.data.schema import FEATURE_ORDER
from nidra.eval.baselines import world_model_forecast
from nidra.models.world_model import WorldModel

COUNTERFACTUAL_LABEL = "model-internal what-if"


@torch.no_grad()
def counterfactual_rollout(
    x: np.ndarray,
    model: WorldModel,
    feature_name: str,
    clamp_value: float,
    K: int = 6,
    n_samples: int = 200,
    stochastic: bool = True,
) -> dict:
    """x: [L, F] or [B, L, F], already scaled the same way training data
    was. Clamps `feature_name` to `clamp_value` at the START of the rollout
    AND after every subsequent predicted step, then re-simulates — identical
    machinery to WorldModel.rollout, with one line different, exactly as
    specified in IMPLEMENTATION-ML.md §6(d).
    """
    if feature_name not in FEATURE_ORDER:
        raise ValueError(f"unknown feature {feature_name!r}; must be one of FEATURE_ORDER")
    feature_idx = FEATURE_ORDER.index(feature_name)

    x_t = torch.from_numpy(x).float()
    if x_t.dim() == 2:
        x_t = x_t.unsqueeze(0)
    B = x_t.shape[0]

    x_tiled = x_t.repeat_interleave(n_samples, dim=0) if n_samples > 1 else x_t
    h_t, h = model.encoder(x_tiled)
    cur = x_tiled[:, -1, :].clone()
    cur[:, feature_idx] = clamp_value

    traj = []
    for _ in range(K):
        mu, logvar = model.transition(h_t)
        nxt = cur + mu
        if stochastic:
            nxt = nxt + torch.randn_like(mu) * (0.5 * logvar).exp()
        nxt = nxt.clamp(-model.state_clamp, model.state_clamp)
        nxt[:, feature_idx] = clamp_value  # re-clamp after every step
        traj.append(nxt)
        h_t, h = model.encoder(nxt.unsqueeze(1), h)
        cur = nxt

    states = torch.stack(traj, dim=1)  # [B*S, K, F]
    BS = B * max(n_samples, 1)
    S = BS // B
    states = states.reshape(B, S, K, model.n_features)

    risk, stage = model.score_states(states.reshape(B * S, K, model.n_features))
    risk = risk.reshape(B, S, K)
    stage = stage.reshape(B, S, K, -1)

    return {
        "label": COUNTERFACTUAL_LABEL,
        "clamped_feature": feature_name,
        "clamp_value": clamp_value,
        "risk_mean_k": risk.mean(dim=1).numpy(),
        "risk_ci_low_k": risk.quantile(0.05, dim=1).numpy(),
        "risk_ci_high_k": risk.quantile(0.95, dim=1).numpy(),
        "stage_mean_k": stage.mean(dim=1).numpy(),
        "predicted_states_mean": states.mean(dim=1).numpy(),
    }


def compare_to_baseline(
    x: np.ndarray,
    model: WorldModel,
    feature_name: str,
    clamp_value: float,
    K: int = 6,
    n_samples: int = 200,
) -> dict:
    """Runs the normal rollout AND the counterfactual rollout on the same
    input and returns both curves side by side, so the caller (API/UI) can
    overlay them. Both are explicitly labelled — the baseline is not
    "ground truth", it is also a model output.
    """
    x_batched = x if x.ndim == 3 else x[None, ...]
    baseline = world_model_forecast(x_batched, model, K=K, n_samples=n_samples)
    counterfactual = counterfactual_rollout(x, model, feature_name, clamp_value, K=K, n_samples=n_samples)
    return {
        "label": COUNTERFACTUAL_LABEL,
        "baseline_risk_mean_k": baseline["risk_mean_k"],
        "counterfactual_risk_mean_k": counterfactual["risk_mean_k"],
        "clamped_feature": feature_name,
        "clamp_value": clamp_value,
    }
