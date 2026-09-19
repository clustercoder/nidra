"""Signed, per-forecast attributions of the projected risk to the input
history, with a faithfulness check.

The three existing mechanisms answer other questions: KernelSHAP on the
risk head explains the CURRENT state's risk, KernelSHAP on the stage head
explains a projected stage, and input-gradient saliency ranks history
windows for one predicted feature. None of them says, with sign, which
input cells (window, feature) pushed the FORECAST — the composite
"attack within the horizon" score that the threshold is applied to — up
or down. This module does, with integrated gradients through the
deterministic rollout and the frozen heads:

    f(x) = reduce_k  mean_m risk_m( rollout_m(x)_k )        (mean trajectory)

    IG_ij(x) = (x_ij - b_ij) * ∫_0^1 ∂f(b + α(x - b))/∂x_ij dα

with the baseline b the scaled all-silent state repeated over the history
(the state a host sits in when it does nothing). Completeness holds:
sum_ij IG_ij ≈ f(x) - f(b), so the attributions are in the units of the
forecast probability and are signed.

The faithfulness check is a deletion test on the SAME forward function:
replace the top-m attributed cells by their baseline value and measure how
much the forecast falls, against the same count of random cells. An
explanation whose top cells do not move the forecast more than random
ones is decoration; the number is returned so a consumer (and a test) can
see which it is. Attributions are in the model's scaled feature space; the
feature-level summary sums over the history so it can be read next to the
feature names.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from nidra.data.schema import FEATURE_ORDER
from nidra.models.world_model import WorldModel


def forecast_score_fn(models: list[WorldModel], K: int, horizon_reduction: str = "max") -> Callable[[torch.Tensor], torch.Tensor]:
    """Differentiable x [B, L, F] -> composite forecast score [B] on the
    mean (deterministic) rollout, heads soft-voted across members."""

    def f(x: torch.Tensor) -> torch.Tensor:
        risks = []
        for m in models:
            states = m.rollout(x, K=K, n_samples=1, stochastic=False).states[:, 0]      # [B, K, F]
            r, _ = m.score_states(states)                                                # [B, K]
            risks.append(r)
        risk_k = torch.stack(risks).mean(0)
        if horizon_reduction == "integrated":
            return 1.0 - torch.prod(1.0 - risk_k.clamp(0, 1), dim=1)
        return risk_k.max(dim=1).values

    return f


def integrated_gradients(f: Callable[[torch.Tensor], torch.Tensor], x: np.ndarray, baseline: np.ndarray,
                         steps: int = 32) -> tuple[np.ndarray, float, float]:
    """IG of scalar f over one input x [L, F] from baseline [L, F].
    Returns attributions [L, F], f(x), f(baseline)."""
    x_t = torch.from_numpy(np.asarray(x, dtype="float32"))
    b_t = torch.from_numpy(np.asarray(baseline, dtype="float32"))
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps                    # midpoint rule
    path = b_t.unsqueeze(0) + alphas.view(-1, 1, 1) * (x_t - b_t).unsqueeze(0)             # [steps, L, F]
    path.requires_grad_(True)
    out = f(path)
    grads = torch.autograd.grad(out.sum(), path)[0]                                        # [steps, L, F]
    avg_grad = grads.mean(dim=0)
    attributions = ((x_t - b_t) * avg_grad).detach().numpy()
    with torch.no_grad():
        fx = float(f(x_t.unsqueeze(0))[0])
        fb = float(f(b_t.unsqueeze(0))[0])
    return attributions, fx, fb


def summarize_attributions(attr: np.ndarray, x_scaled: np.ndarray, n_features: int = 10, n_cells: int = 10) -> dict:
    """Feature-level signed totals (summed over history) and the top cells."""
    L = attr.shape[0]
    per_feature = attr.sum(axis=0)
    order = np.argsort(-np.abs(per_feature))[:n_features]
    features = [{"feature": FEATURE_ORDER[j], "contribution": float(per_feature[j]),
                 "direction": "up" if per_feature[j] > 0 else "down",
                 "value_at_t_scaled": float(x_scaled[-1, j])} for j in order]
    flat = np.argsort(-np.abs(attr).ravel())[:n_cells]
    cells = [{"window_offset": int(i // attr.shape[1]) - (L - 1), "feature": FEATURE_ORDER[int(i % attr.shape[1])],
              "contribution": float(attr.ravel()[i])} for i in flat]
    per_window = attr.sum(axis=1)
    return {"features": features, "cells": cells,
            "per_window_total": [float(v) for v in per_window],
            "driving_window_offset": int(np.argmax(np.abs(per_window))) - (L - 1)}


def faithfulness_check(f: Callable[[torch.Tensor], torch.Tensor], x: np.ndarray, baseline: np.ndarray, attr: np.ndarray,
                       m: int = 8, n_random: int = 20, seed: int = 0) -> dict:
    """Deletion test: set the top-m |attribution| cells to their baseline
    value and measure the forecast drop, against n_random draws of m random
    cells. Returns both drops and the fraction of random draws the top-m
    deletion beats."""
    x = np.asarray(x, dtype="float32")
    baseline = np.asarray(baseline, dtype="float32")
    L, F = x.shape
    m = min(m, L * F)
    with torch.no_grad():
        fx = float(f(torch.from_numpy(x).unsqueeze(0))[0])
    top = np.argsort(-np.abs(attr).ravel())[:m]

    def _deleted(cells: np.ndarray) -> float:
        x2 = x.copy().ravel()
        x2[cells] = baseline.ravel()[cells]
        with torch.no_grad():
            return float(f(torch.from_numpy(x2.reshape(L, F)).unsqueeze(0))[0])

    drop_top = fx - _deleted(top)
    rng = np.random.default_rng(seed)
    drops_random = np.array([fx - _deleted(rng.choice(L * F, size=m, replace=False)) for _ in range(n_random)])
    return {"m": int(m), "forecast": fx, "drop_top_m": float(drop_top),
            "drop_random_m_mean": float(drops_random.mean()), "drop_random_m_max": float(drops_random.max()),
            "beats_random_fraction": float((drop_top > drops_random).mean()),
            "faithful": bool(drop_top > drops_random.mean())}


def explain_forecast(models: list[WorldModel], x_scaled: np.ndarray, baseline_state: np.ndarray, K: int,
                     horizon_reduction: str = "max", steps: int = 32, check_faithfulness: bool = True,
                     m: int = 8, n_random: int = 20) -> dict:
    """The per-forecast explanation object: integrated-gradient attributions
    of the composite forecast score to every (history window, feature) cell,
    their feature-level summary, completeness, and the deletion check."""
    f = forecast_score_fn(models, K, horizon_reduction)
    baseline = np.tile(np.asarray(baseline_state, dtype="float32").reshape(1, -1), (x_scaled.shape[0], 1))
    attr, fx, fb = integrated_gradients(f, x_scaled, baseline, steps=steps)
    out = {"method": "integrated_gradients", "steps": steps, "baseline": "all-silent scaled state over the history",
           "forecast_score": fx, "baseline_score": fb, "attribution_sum": float(attr.sum()),
           "completeness_gap": float(abs(attr.sum() - (fx - fb))),
           "space": "scaled features; contributions are in forecast-probability units",
           **summarize_attributions(attr, x_scaled)}
    if check_faithfulness:
        out["faithfulness"] = faithfulness_check(f, x_scaled, baseline, attr, m=m, n_random=n_random)
    return out
