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
    def __init__(self, hidden_size: int = 128, n_features: int = 45, mlp_hidden: int = 256,
                 logvar_min: float = -6.0, logvar_max: float = 3.0):
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(mlp_hidden, n_features)
        self.logvar_head = nn.Linear(mlp_hidden, n_features)

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_t: [B, H]. Returns (mu, logvar) each [B, F]. mu is the
        predicted DELTA — callers must add the current state themselves."""
        z = self.net(h_t)
        mu = self.mu_head(z)
        logvar = self.logvar_head(z).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar
