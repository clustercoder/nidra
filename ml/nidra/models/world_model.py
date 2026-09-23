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
from nidra.models.heads import RiskHead, StageHead, TrajectoryRiskHead
from nidra.models.transition import Transition


@dataclass
class RolloutOutput:
    states: torch.Tensor        # [B, S, K, F] predicted states (S = n_samples)
    mus: torch.Tensor           # [B, S, K, F] per-step predicted deltas
    logvars: torch.Tensor       # [B, S, K, F] per-step predicted log-variances
    hiddens: torch.Tensor       # [B, S, K, H] encoder state AFTER ingesting step k
    anchor: torch.Tensor        # [B, S, F] the observed S_t the rollout started from

    @staticmethod
    def pool(outs: "list[RolloutOutput]") -> "RolloutOutput":
        """Concatenate several members' rollouts along the sample axis.

        The predictor pools every member's trajectories and then scores the
        pooled set with each member's head. Pooling only `states` and calling
        `score_states` was enough for the per-state head and raises a bare
        TypeError for a trajectory head, which reads three more tensors — so
        the whole output is pooled and scored through `score_trajectory`, and
        the per-state head's numbers are unchanged because it still sees only
        the states.
        """
        if not outs:
            raise ValueError("pool() needs at least one rollout")
        return RolloutOutput(
            states=torch.cat([o.states for o in outs], dim=1),
            mus=torch.cat([o.mus for o in outs], dim=1),
            logvars=torch.cat([o.logvars for o in outs], dim=1),
            hiddens=torch.cat([o.hiddens for o in outs], dim=1),
            anchor=torch.cat([o.anchor for o in outs], dim=1),
        )

    def realized_deltas(self) -> torch.Tensor:
        """[B, S, K, F] backward difference of the trajectory itself,
        states[k] - states[k-1], with states[-1] taken as the anchor S_t.

        This, rather than `mus`, is what a trajectory head reads. Under a
        stochastic rollout the realized change is mu + noise, and the head's
        training pairs are observed backward differences — so using mu would
        feed the head a quantity at inference that it never saw in training.
        """
        prev = torch.cat([self.anchor.unsqueeze(2), self.states[:, :, :-1, :]], dim=2)
        return self.states - prev


#: Where the next state of a rollout comes from. Persistence and the oracle
#: are the same forward simulation with this one thing changed, so a
#: comparison against them isolates the transition model rather than
#: comparing two differently-wired systems. See eval/baselines.py.
STATE_SOURCES = ("model", "persist", "truth")


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
        linear_skip: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.state_clamp = state_clamp
        self.encoder = Encoder(n_features, hidden_size, encoder_layers, encoder_dropout)
        self.transition = Transition(hidden_size, n_features, transition_mlp_hidden, logvar_min, logvar_max,
                                     linear_skip=linear_skip)
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
        state_source: str = "model",
        truth: torch.Tensor | None = None,
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

        `state_source` swaps out ONLY where the next state comes from, so
        the two reference systems run through this identical code path:

            "model"    S_hat[t+k] = S_hat[t+k-1] + mu  — the system under test
            "persist"  S_hat[t+k] = S_t for every k    — the transition does
                       nothing; the encoder still advances, so a
                       history-aware head keeps its history and the only
                       thing removed is the predicted change
            "truth"    S_hat[t+k] = the observed S[t+k], from `truth`
                       [B, >=K, F]  — the oracle upper bound

        "truth" is the ONLY path that reads data after time t, and it exists
        to bound head quality, never to produce a forecast. `truth` is
        refused on any other source so no call site can leak one in.
        """
        if state_source not in STATE_SOURCES:
            raise ValueError(f"unknown state_source {state_source!r}; expected one of {STATE_SOURCES}")
        if state_source == "truth":
            if truth is None:
                raise ValueError("state_source='truth' needs the observed future states in `truth`")
            if truth.shape[1] < K:
                raise ValueError(f"truth has {truth.shape[1]} steps but the rollout is {K} steps long")
        elif truth is not None:
            raise ValueError(f"state_source={state_source!r} must not be given `truth` — "
                             "only the oracle may read data after time t")
        B = x.shape[0]
        if n_samples > 1:
            x_tiled = x.repeat_interleave(n_samples, dim=0)
        else:
            x_tiled = x

        truth_tiled = None
        if truth is not None:
            truth_tiled = truth.repeat_interleave(n_samples, dim=0) if n_samples > 1 else truth

        h_t, h = self.encoder(x_tiled)
        cur = x_tiled[:, -1, :]
        prev = x_tiled[:, -2, :] if x_tiled.shape[1] > 1 else cur
        anchor = x_tiled[:, -1, :]

        traj, mus, logvars, hiddens = [], [], [], []
        for k in range(K):
            mu, logvar = self.transition(h_t, cur, prev)
            if state_source == "model":
                nxt = cur + mu
                if stochastic:
                    noise = torch.randn_like(mu) * (0.5 * logvar).exp()
                    nxt = nxt + noise
            elif state_source == "persist":
                nxt = anchor
                mu = torch.zeros_like(mu)
            else:
                nxt = truth_tiled[:, k, :]
            nxt = nxt.clamp(-self.state_clamp, self.state_clamp)

            traj.append(nxt)
            mus.append(mu)
            logvars.append(logvar)

            h_t, h = self.encoder(nxt.unsqueeze(1), h)
            hiddens.append(h_t)
            prev, cur = cur, nxt

        anchor_out = x_tiled[:, -1, :]
        states = torch.stack(traj, dim=1)      # [B*S, K, F]
        mus_t = torch.stack(mus, dim=1)
        logvars_t = torch.stack(logvars, dim=1)
        hiddens_t = torch.stack(hiddens, dim=1)

        BS = B * max(n_samples, 1)
        S = BS // B
        states = states.reshape(B, S, K, self.n_features)
        mus_t = mus_t.reshape(B, S, K, self.n_features)
        logvars_t = logvars_t.reshape(B, S, K, self.n_features)
        hiddens_t = hiddens_t.reshape(B, S, K, hiddens_t.shape[-1])
        anchor_out = anchor_out.reshape(B, S, self.n_features)

        return RolloutOutput(states=states, mus=mus_t, logvars=logvars_t, hiddens=hiddens_t,
                             anchor=anchor_out)

    def score_trajectory(self, out: RolloutOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """Heads applied to a rollout, giving a trajectory-aware risk head the
        context `score_states` cannot pass it.

        With the default per-state `RiskHead` this is exactly
        `score_states(out.states)` — every existing call site's numbers are
        unchanged — and with a `TrajectoryRiskHead` it additionally passes the
        hidden state, predicted delta and predicted log-variance of each step.
        Returns (risk_prob [B,S,K], stage_probs [B,S,K,n_stages]).
        """
        stage_probs = torch.softmax(self.stage_head(out.states), dim=-1)
        if isinstance(self.risk_head, TrajectoryRiskHead):
            logits = self.risk_head(state=out.states, hidden=out.hiddens,
                                    delta=out.realized_deltas(), logvar=out.logvars).squeeze(-1)
        else:
            logits = self.risk_head(out.states).squeeze(-1)
        return torch.sigmoid(logits), stage_probs

    def observed_context(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The trajectory-head inputs for the OBSERVED origin state of each
        history window. x: [B, L, F] scaled history, oldest-first.

        The exact quantities `train.head_context` builds for training, so a
        head scores an observed state the same way in both places:
        `hidden` is the encoder over all L windows, `delta` the observed
        backward difference, `logvar` the transition's prediction for this
        step conditioned on everything before it.
        """
        out, _ = self.encoder.gru(x)
        h_prev = out[:, -2, :] if x.shape[1] >= 2 else torch.zeros_like(out[:, -1, :])
        prev1 = x[:, -2, :] if x.shape[1] >= 2 else torch.zeros_like(x[:, -1, :])
        prev2 = x[:, -3, :] if x.shape[1] >= 3 else torch.zeros_like(x[:, -1, :])
        _, logvar = self.transition(h_prev, prev1, prev2)
        return {"state": x[:, -1, :], "hidden": out[:, -1, :], "delta": x[:, -1, :] - prev1, "logvar": logvar}

    def score_observed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Heads applied to the observed state at the end of each history
        window. x: [B, L, F]. Equal to `score_states(x[:, -1, :])` for a
        per-state head; a trajectory head additionally gets the observed
        context. Returns (risk_prob [B], stage_probs [B, n_stages])."""
        stage_probs = torch.softmax(self.stage_head(x[:, -1, :]), dim=-1)
        if isinstance(self.risk_head, TrajectoryRiskHead):
            parts = self.observed_context(x)
            logits = self.risk_head(**parts).squeeze(-1)
        else:
            logits = self.risk_head(x[:, -1, :]).squeeze(-1)
        return torch.sigmoid(logits), stage_probs

    def score_stage(self, states: torch.Tensor) -> torch.Tensor:
        """Stage probabilities for arbitrary states: [..., F] -> [..., n_stages].

        The stage head is per-state under every configuration, so this is
        always well defined. Call sites that want only the stage distribution
        use this rather than discarding the risk half of `score_states`,
        which a trajectory head cannot produce from a bare state.
        """
        return torch.softmax(self.stage_head(states), dim=-1)

    def score_states(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the (frozen, at inference time) heads to arbitrary states —
        observed or rolled-out. states: [..., F]. Returns
        (risk_prob [...], stage_probs [..., n_stages]).

        Per-state risk heads only. A trajectory head reads context a bare
        state does not carry, so this raises rather than guessing at it.
        """
        if isinstance(self.risk_head, TrajectoryRiskHead):
            raise TypeError(
                f"score_states cannot score a risk head that reads "
                f"{self.risk_head.components}: a bare state carries only 'state'. "
                "Use score_trajectory(RolloutOutput) for rolled-out states, "
                "score_observed(x) for a history window, or score_stage(states) "
                "if only the stage distribution is wanted.")
        risk_logits = self.risk_head(states).squeeze(-1)
        risk_prob = torch.sigmoid(risk_logits)
        return risk_prob, self.score_stage(states)
