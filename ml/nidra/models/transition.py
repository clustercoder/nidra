"""Gaussian transition model: emits a diagonal-Gaussian distribution over the
NEXT state's delta, given the encoder's recurrent hidden summary.

Two design choices are load-bearing, not stylistic:

  - logvar is clamped to [logvar_min, logvar_max]. Without this the model
    discovers that predicting infinite variance on hard features minimises
    NLL, variance explodes, and rollout produces NaN by k~3.
  - The head predicts a DELTA (S_hat[t+1] = S[t] + mu), not the absolute
    state. This hands the model a strong autocorrelation prior — capacity
    goes into modelling change — and makes the persistence baseline exactly
    the mu=0 case, which is the clean ablation story in eval/ablations.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Transition(nn.Module):
    """`linear_skip=True` adds a two-lag linear term to the mean delta,
    mu = MLP(h_t) + A [s_t, s_{t-1}] + b, initialised to zero so training
    starts from the plain model. Motivation (benchmark on the Δ=60
    validation set, geomA_L15K3): a ridge regression on the same two lags
    forecast the state with skill 0.63 against persistence where the GRU
    transition reached 0.47 — the recurrent model was not reproducing a
    map a linear model finds by least squares. With the skip the MLP only
    has to model the residual over that map; the model is otherwise the
    same encoder -> transition -> rollout."""

    def __init__(self, hidden_size: int = 128, n_features: int = 45, mlp_hidden: int = 256,
                 logvar_min: float = -6.0, logvar_max: float = 3.0, linear_skip: bool = False):
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.linear_skip = bool(linear_skip)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(mlp_hidden, n_features)
        self.logvar_head = nn.Linear(mlp_hidden, n_features)
        if self.linear_skip:
            self.skip = nn.Linear(2 * n_features, n_features)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor | None = None,
                s_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """h_t: [B, H]; s_t, s_prev: [B, F] the current and previous state
        (required when linear_skip is on). Returns (mu, logvar) each [B, F].
        mu is the predicted DELTA — callers must add the current state."""
        z = self.net(h_t)
        mu = self.mu_head(z)
        if self.linear_skip:
            if s_t is None:
                raise ValueError("Transition(linear_skip=True) needs s_t (and s_prev)")
            s_prev = s_t if s_prev is None else s_prev
            mu = mu + self.skip(torch.cat([s_t, s_prev], dim=-1))
        logvar = self.logvar_head(z).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar
