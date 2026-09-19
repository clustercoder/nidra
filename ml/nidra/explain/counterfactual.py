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


#: Seed used for the sampled trajectories in a what-if comparison. Fixed so
#: that the baseline and the clamped rollout draw the SAME noise (common
#: random numbers) and so that re-running a what-if gives the same answer —
#: see `compare_to_baseline`.
COUNTERFACTUAL_SEED = 20170707


@torch.no_grad()
def counterfactual_rollout(
    x: np.ndarray,
    model: WorldModel,
    feature_name: str | None,
    clamp_value: float,
    K: int = 6,
    n_samples: int = 200,
    stochastic: bool = True,
    seed: int | None = None,
) -> dict:
    """x: [L, F] or [B, L, F], already scaled the same way training data
    was. Clamps `feature_name` to `clamp_value` at the START of the rollout
    AND after every subsequent predicted step, then re-simulates — identical
    machinery to WorldModel.rollout, with one line different, exactly as
    specified in IMPLEMENTATION-ML.md §6(d).

    `feature_name=None` clamps nothing and runs the same code on the same
    path: that is how `compare_to_baseline` gets a baseline whose only
    difference from the clamped run is the clamp itself. Producing the
    baseline from a different function would reintroduce exactly the
    discrepancy this is meant to remove.

    `seed`, when given, seeds the trajectory sampler. Two calls with the same
    seed draw the same noise, which is what makes the difference between two
    curves attributable to the clamp rather than to Monte Carlo error.
    """
    feature_idx: int | None = None
    if feature_name is not None:
        if feature_name not in FEATURE_ORDER:
            raise ValueError(f"unknown feature {feature_name!r}; must be one of FEATURE_ORDER")
        feature_idx = FEATURE_ORDER.index(feature_name)
    if seed is not None:
        torch.manual_seed(seed)

    x_t = torch.from_numpy(x).float()
    if x_t.dim() == 2:
        x_t = x_t.unsqueeze(0)
    B = x_t.shape[0]

    x_tiled = x_t.repeat_interleave(n_samples, dim=0) if n_samples > 1 else x_t
    h_t, h = model.encoder(x_tiled)
    cur = x_tiled[:, -1, :].clone()
    if feature_idx is not None:
        cur[:, feature_idx] = clamp_value

    traj = []
    for _ in range(K):
        mu, logvar = model.transition(h_t)
        nxt = cur + mu
        if stochastic:
            nxt = nxt + torch.randn_like(mu) * (0.5 * logvar).exp()
        nxt = nxt.clamp(-model.state_clamp, model.state_clamp)
        if feature_idx is not None:
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
    seed: int | None = COUNTERFACTUAL_SEED,
) -> dict:
    """Runs the normal rollout AND the counterfactual rollout on the same
    input and returns both curves side by side, so the caller (API/UI) can
    overlay them. Both are explicitly labelled — the baseline is not
    "ground truth", it is also a model output.
    """
    # Both curves come from the same function, on the same code path, seeded
    # identically — so they draw the same trajectory noise and the only thing
    # that differs between them is the clamp. Running the baseline through
    # `world_model_forecast` instead left each curve with its own Monte Carlo
    # error, which at a step where the clamp's real effect is small is enough
    # to flip the sign of the difference the caller is being shown.
    baseline = counterfactual_rollout(
        x, model, None, 0.0, K=K, n_samples=n_samples, seed=seed,
    )
    counterfactual = counterfactual_rollout(
        x, model, feature_name, clamp_value, K=K, n_samples=n_samples, seed=seed,
    )
    return {
        "label": COUNTERFACTUAL_LABEL,
        "baseline_risk_mean_k": baseline["risk_mean_k"],
        "counterfactual_risk_mean_k": counterfactual["risk_mean_k"],
        # The full rollouts too, so a caller that needs the confidence bands
        # (the serving plane draws them) does not have to re-pair the two runs
        # itself and risk getting the pairing wrong.
        "baseline": baseline,
        "counterfactual": counterfactual,
        "clamped_feature": feature_name,
        "clamp_value": clamp_value,
    }
