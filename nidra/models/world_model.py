"""Assembly of encoder + transition + frozen heads, and the recursive
K-step rollout.

The rollout is the entire "world model" claim: `nxt` — the model's own
prediction — is fed back into the encoder as if it were an observation
(`self.encoder(nxt.unsqueeze(1), h)`), exactly what "forward simulation"
means and exactly what a per-window classifier structurally cannot do.
No real data after the input's last window is consumed anywhere in this
function.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from nidra.models.encoder import Encoder
from nidra.models.heads import RiskHead, StageHead
from nidra.models.transition import Transition


@dataclass
class RolloutOutput:
    states: torch.Tensor        # [B, S, K, F] predicted states (S = n_samples)
    mus: torch.Tensor           # [B, S, K, F] per-step predicted deltas
    logvars: torch.Tensor       # [B, S, K, F] per-step predicted log-variances


class WorldModel(nn.Module):
    def __init__(
        self,
        n_features: int = 45,
        hidden_size: int = 128,
        encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
        transition_mlp_hidden: int = 256,
        logvar_min: float = -6.0,
        logvar_max: float = 3.0,
        risk_hidden: int = 64,
        stage_hidden: int = 64,
        n_stages: int = 6,
        state_clamp: float = 10.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.state_clamp = state_clamp
        self.encoder = Encoder(n_features, hidden_size, encoder_layers, encoder_dropout)
        self.transition = Transition(hidden_size, n_features, transition_mlp_hidden, logvar_min, logvar_max)
        self.risk_head = RiskHead(n_features, risk_hidden)
        self.stage_head = StageHead(n_features, stage_hidden, n_stages)

    def encode(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, h)

    def freeze_dynamics(self) -> None:
        """Stage 2 entry point: freeze encoder + transition before training
        heads on observed states only."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.transition.parameters():
            p.requires_grad = False

    def freeze_heads(self) -> None:
        """Called after stage-2 head training completes. Nothing is trained
        after this point."""
        for p in self.risk_head.parameters():
            p.requires_grad = False
        for p in self.stage_head.parameters():
            p.requires_grad = False

    def freeze_all(self) -> None:
        self.freeze_dynamics()
        self.freeze_heads()

    def rollout(
        self,
        x: torch.Tensor,
        K: int = 6,
        n_samples: int = 1,
        stochastic: bool = True,
    ) -> RolloutOutput:
        """x: [B, L, F] observed history (already scaled), oldest-first.
        Consumes NO data after the last window of x. Returns predicted
        states for `n_samples` trajectories per input row.

        For n_samples > 1, the batch is tiled (repeat_interleave) rather
        than looped in Python, so sampling stays a single vectorized
        forward pass through the rollout.

        Deliberately NOT wrapped in torch.no_grad(): explain/saliency.py
        needs gradients to flow through the rollout (input-gradient temporal
        saliency). Inference/eval call sites that don't need gradients
        (serving, baselines, ablations) wrap their own calls in
        `torch.no_grad()` instead — see eval/baselines.py.
        """
        B = x.shape[0]
        if n_samples > 1:
            x_tiled = x.repeat_interleave(n_samples, dim=0)
        else:
            x_tiled = x

        h_t, h = self.encoder(x_tiled)
        cur = x_tiled[:, -1, :]

        traj, mus, logvars = [], [], []
        for _ in range(K):
            mu, logvar = self.transition(h_t)
            nxt = cur + mu
            if stochastic:
                noise = torch.randn_like(mu) * (0.5 * logvar).exp()
                nxt = nxt + noise
            nxt = nxt.clamp(-self.state_clamp, self.state_clamp)

            traj.append(nxt)
            mus.append(mu)
            logvars.append(logvar)

            h_t, h = self.encoder(nxt.unsqueeze(1), h)
            cur = nxt

        states = torch.stack(traj, dim=1)      # [B*S, K, F]
        mus_t = torch.stack(mus, dim=1)
        logvars_t = torch.stack(logvars, dim=1)

        BS = B * max(n_samples, 1)
        states = states.reshape(B, BS // B, K, self.n_features)
        mus_t = mus_t.reshape(B, BS // B, K, self.n_features)
        logvars_t = logvars_t.reshape(B, BS // B, K, self.n_features)

        return RolloutOutput(states=states, mus=mus_t, logvars=logvars_t)

    def score_states(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the (frozen, at inference time) heads to arbitrary states —
        observed or rolled-out. states: [..., F]. Returns
        (risk_prob [...], stage_probs [..., n_stages])."""
        risk_logits = self.risk_head(states).squeeze(-1)
        stage_logits = self.stage_head(states)
        risk_prob = torch.sigmoid(risk_logits)
        stage_probs = torch.softmax(stage_logits, dim=-1)
        return risk_prob, stage_probs
