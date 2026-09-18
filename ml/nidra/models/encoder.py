"""GRU encoder: history of observed states -> recurrent hidden summary h_t.

Raw single-window observations are not Markov — one window cannot tell you
whether SYN ratio is rising or falling. The usable formulation is
P(S_t+1 | h_t) where h_t = f(S_1..S_t); this module produces h_t and, just
as importantly, returns the recurrent hidden state itself so rollout can
re-enter the GRU one step at a time while carrying that state forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features: int = 45, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, F] (T may be 1 during rollout re-entry). Returns
        (h_t: [B, H] the last-step hidden summary, h: [num_layers, B, H] the
        full recurrent state to pass back in on the next call)."""
        out, h_next = self.gru(x, h)
        h_t = out[:, -1, :]
        return h_t, h_next
