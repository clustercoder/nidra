"""Risk and stage heads.

Deliberately small, and deliberately operating on the 45-dim FEATURE VECTOR
rather than the GRU's hidden state, for two reasons: (1) they must be
directly applicable to a decoded predicted state produced by rollout, and
(2) small models on named features make KernelSHAP tractable.

Rule 1 (see CLAUDE.md / IMPLEMENTATION-ML.md §0): these heads are trained
ONLY on observed states, then frozen. Nothing in this module enforces that
by itself — the freezing discipline lives in train/train_heads.py — but the
architecture here (operating on raw feature vectors) is what makes a head
applicable to both an observed state and a predicted one without
distinguishing the two, which is the whole point.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """F -> 64 -> 1, sigmoid (via BCEWithLogitsLoss at training time)."""

    def __init__(self, n_features: int = 45, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [..., F]. Returns raw logits [..., 1] — apply sigmoid
        outside for a probability, or use BCEWithLogitsLoss directly."""
        return self.net(state)


class StageHead(nn.Module):
    """F -> 64 -> n_stages, softmax (via CrossEntropyLoss at training time)."""

    def __init__(self, n_features: int = 45, hidden: int = 64, n_stages: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_stages),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [..., F]. Returns raw logits [..., n_stages]."""
        return self.net(state)
