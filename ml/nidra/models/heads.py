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


class OnsetHead(nn.Module):
    """F -> hidden -> H logits: P(an attack episode BEGINS within h_j minutes
    | state), one output per horizon in `horizons_min`. Trained on observed
    states at origins OUTSIDE any episode (data/onset.py), then frozen —
    the same discipline as the risk head. Stored as its own artifact
    (`onset_head_seed_<s>.pt`) so the WorldModel checkpoint format is
    unchanged."""

    def __init__(self, n_features: int = 45, hidden: int = 64, horizons_min: tuple[int, ...] = (1, 3, 5, 10, 15, 30)):
        super().__init__()
        self.horizons_min = tuple(int(h) for h in horizons_min)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.horizons_min)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [..., F]. Returns raw logits [..., H]."""
        return self.net(state)

    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "n_features": self.net[0].in_features,
                    "hidden": self.net[0].out_features, "horizons_min": self.horizons_min}, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "OnsetHead":
        ckpt = torch.load(path, map_location=map_location)
        head = cls(ckpt["n_features"], ckpt["hidden"], tuple(ckpt["horizons_min"]))
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        return head
