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


#: Everything a rollout step can offer a risk head, and the name each goes by
#: in `TrajectoryRiskHead`'s component list.
#:
#:   state   S_hat[t+k]         the predicted state — the Run 8 head's only input
#:   hidden  h[t+k]             the encoder's recurrent summary AFTER ingesting
#:                              that predicted state; the host's whole history
#:                              plus the model's own simulation of the future
#:   delta   mu[t+k]            the transition's predicted change at that step
#:   logvar  log sigma^2[t+k]   the transition's predicted per-feature variance
#:
#: There is deliberately NO horizon component. A head trained under the
#: frozen-head discipline only ever sees OBSERVED states, where "how many of
#: the ingested windows were predictions" is identically zero — a horizon
#: input would be a constant in training and an out-of-distribution value at
#: inference. Horizon dependence enters legitimately through `logvar` (the
#: transition's uncertainty does grow with k) and through the per-horizon
#: Platt calibration already fit on validation.
TRAJECTORY_COMPONENTS: tuple[str, ...] = ("state", "hidden", "delta", "logvar")




def component_widths(components: tuple[str, ...], n_features: int, hidden_size: int) -> list[int]:
    """Per-component input width, in TRAJECTORY_COMPONENTS order. One table,
    so the risk head and the onset head cannot disagree about what `hidden`
    is worth."""
    widths = {"state": int(n_features), "hidden": int(hidden_size),
              "delta": int(n_features), "logvar": int(n_features)}
    unknown = [c for c in components if c not in widths]
    if unknown:
        raise ValueError(f"unknown component(s) {unknown}; expected from {TRAJECTORY_COMPONENTS}")
    return [widths[c] for c in ordered_components(components)]


def ordered_components(components: tuple[str, ...]) -> tuple[str, ...]:
    """Declared order, deduplicated. The concatenation layout must not depend
    on how a caller spelled the list."""
    unknown = [c for c in components if c not in TRAJECTORY_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown component(s) {unknown}; expected from {TRAJECTORY_COMPONENTS}")
    ordered = tuple(c for c in TRAJECTORY_COMPONENTS if c in set(components))
    if not ordered:
        raise ValueError("a component head needs at least one component")
    return ordered


def concat_components(components: tuple[str, ...], parts: dict) -> torch.Tensor:
    """Concatenate the declared components along the last axis. A declared
    component that is not supplied is an error, never a zero fill — a head
    silently scoring on zeros looks like a working head with a puzzling loss
    curve."""
    pieces = []
    for name in ordered_components(components):
        value = parts.get(name)
        if value is None:
            raise ValueError(f"this head requires component {name!r}, which was not supplied")
        pieces.append(value)
    return torch.cat(pieces, dim=-1)


class OnsetHead(nn.Module):
    """F -> hidden -> H logits: P(an attack episode BEGINS within h_j minutes
    | state), one output per horizon in `horizons_min`. Trained on observed
    states at origins OUTSIDE any episode (data/onset.py), then frozen —
    the same discipline as the risk head. Stored as its own artifact
    (`onset_head_seed_<s>.pt`) so the WorldModel checkpoint format is
    unchanged."""

    def __init__(self, n_features: int = 45, hidden: int = 64,
                 horizons_min: tuple[int, ...] = (1, 3, 5, 10, 15, 30),
                 components: tuple[str, ...] = ("state",), hidden_size: int = 128):
        super().__init__()
        self.horizons_min = tuple(int(h) for h in horizons_min)
        self.parameterisation = "independent"
        self.components = ordered_components(tuple(components))
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.input_dim = sum(component_widths(self.components, n_features, hidden_size))
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.horizons_min)),
        )

    def forward(self, state: torch.Tensor | None = None, **parts: torch.Tensor) -> torch.Tensor:
        """state: [..., F], positionally, for the state-only head every
        existing call site builds. A head declaring more components takes them
        all as keywords and refuses a bare tensor. Returns raw logits
        [..., H]."""
        if self.components == ("state",):
            if state is None:
                state = parts.get("state")
            return self.net(state)
        if state is not None:
            parts = {**parts, "state": state}
        return self.net(concat_components(self.components, parts))

    #: How the H outputs are read. "independent": output j is the logit of
    #: P(onset within h_j) directly, each fit with its own BCE — the Run 8
    #: parameterisation, whose probabilities need not be monotone in j and
    #: were observed not to be. "hazard": output j is the logit of
    #: P(onset in bucket j | none before it), fit only on rows still at risk
    #: in bucket j, and read out as 1 - prod(1 - p) — coherent by
    #: construction and able to use censored rows. The ARCHITECTURE is
    #: identical either way, which is what makes the comparison controlled.
    PARAMETERISATIONS = ("independent", "hazard")

    def save(self, path, parameterisation: str = "independent") -> None:
        if parameterisation not in self.PARAMETERISATIONS:
            raise ValueError(f"unknown onset parameterisation {parameterisation!r}")
        torch.save({"state_dict": self.state_dict(), "n_features": self.n_features,
                    "hidden": self.net[0].out_features, "horizons_min": self.horizons_min,
                    "components": list(self.components), "hidden_size": self.hidden_size,
                    "parameterisation": parameterisation}, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "OnsetHead":
        ckpt = torch.load(path, map_location=map_location)
        head = cls(ckpt["n_features"], ckpt["hidden"], tuple(ckpt["horizons_min"]),
                   tuple(ckpt.get("components", ("state",))), int(ckpt.get("hidden_size", 128)))
        head.parameterisation = ckpt.get("parameterisation", "independent")
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        return head


class TrajectoryRiskHead(nn.Module):
    """Risk head over a declared subset of the rollout context.

    Run 8's diagnosis (REAL_DATA_RESULTS.md §8.7) was that the per-state head
    is the ceiling: on Friday it ranked Bot-C2 windows below silence
    (ROC-AUC 0.37) while a GRU sequence classifier on the identical rows
    reached 0.976. The classifier's advantage is history. This head can read
    the encoder's hidden state, which carries that history — and, during a
    rollout, carries the model's own simulated continuation of it.

    The frozen-head discipline is unchanged and is what keeps the forecasting
    claim falsifiable. Training pairs come from OBSERVED states only: the
    hidden state is the encoder over observed history, the delta is the
    observed backward difference, the logvar is the frozen transition's
    prediction for that step. At inference the same head reads the rollout's
    own hidden state, predicted delta and predicted logvar. Nothing after
    time t reaches the input in either case.

    `components=("state",)` reproduces `RiskHead` exactly in shape, which is
    what makes the ablation in the CTU-13 phase a controlled one.
    """

    def __init__(self, components: tuple[str, ...] = ("state", "hidden"), n_features: int = 45,
                 hidden_size: int = 128, hidden: int = 64):
        super().__init__()
        self.components = ordered_components(tuple(components))
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.input_dim = sum(component_widths(self.components, n_features, hidden_size))
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, **parts: torch.Tensor) -> torch.Tensor:
        """Every declared component as a keyword, each [..., width]. Returns
        raw logits [..., 1]. A declared component that is not supplied is an
        error, never a zero fill — a head silently scoring on zeros would
        look like a working head with a puzzling loss curve."""
        return self.net(concat_components(self.components, parts))

    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "components": list(self.components),
                    "n_features": self.n_features, "hidden_size": self.hidden_size,
                    "hidden": self.net[0].out_features}, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "TrajectoryRiskHead":
        ckpt = torch.load(path, map_location=map_location)
        head = cls(tuple(ckpt["components"]), ckpt["n_features"], ckpt["hidden_size"], ckpt["hidden"])
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        return head
