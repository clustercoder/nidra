"""One place that turns a config into a WorldModel.

There were five: train_dynamics, train_heads, eval.run_eval, eval.benchmark
(via run_eval) and serve.predictor each spelled out the same constructor. A
model variant added in one of them and forgotten in another does not raise —
the checkpoint loads, the shapes match, and the evaluation scores a different
architecture than the one that was trained. The risk-head variants of the
CTU-13 phase are exactly that kind of change, so the constructor is here.
"""

from __future__ import annotations

from nidra.models.heads import ordered_components, TRAJECTORY_COMPONENTS, TrajectoryRiskHead
from nidra.models.world_model import WorldModel


def risk_head_components(cfg: dict) -> tuple[str, ...]:
    """Which rollout-context components this config's risk head reads.

    Absent or `["state"]` means the Run 8 per-state `RiskHead`, unchanged.
    Anything else builds a `TrajectoryRiskHead`; the two are checkpoint-
    compatible at `("state",)` (identical layer shapes and parameter names),
    which is what makes the head ablation a controlled comparison.
    """
    declared = cfg.get("model", {}).get("risk_head", {}).get("components")
    if not declared:
        return ("state",)
    components = tuple(str(c) for c in declared)
    unknown = [c for c in components if c not in TRAJECTORY_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown model.risk_head.components {unknown}; expected from {TRAJECTORY_COMPONENTS}")
    return tuple(c for c in TRAJECTORY_COMPONENTS if c in set(components))


def uses_trajectory_head(cfg: dict) -> bool:
    return risk_head_components(cfg) != ("state",)


def onset_head_components(cfg: dict) -> tuple[str, ...]:
    return ordered_components(tuple(cfg.get("onset", {}).get("components", ("state",))))


def uses_head_context(cfg: dict) -> bool:
    """Does ANY head in this run read more than the state?

    The observed encoder context is rebuilt from the labelled tables, and the
    tables are kept only when something needs them. Asking the risk head
    alone was wrong the moment the onset head could declare components too:
    an onset-only run would finish its data pass and then stop."""
    return uses_trajectory_head(cfg) or onset_head_components(cfg) != ("state",)


def world_model_from_config(cfg: dict) -> WorldModel:
    mcfg = cfg["model"]
    model = WorldModel(
        n_features=mcfg["n_features"],
        hidden_size=mcfg["encoder"]["hidden_size"],
        encoder_layers=mcfg["encoder"]["num_layers"],
        encoder_dropout=mcfg["encoder"]["dropout"],
        transition_mlp_hidden=mcfg["transition"]["mlp_hidden"],
        logvar_min=mcfg["transition"]["logvar_min"],
        logvar_max=mcfg["transition"]["logvar_max"],
        risk_hidden=mcfg["risk_head"]["hidden"],
        stage_hidden=mcfg["stage_head"]["hidden"],
        n_stages=mcfg["stage_head"]["n_stages"],
        state_clamp=mcfg["transition"]["state_clamp"],
        linear_skip=bool(mcfg["transition"].get("linear_skip", False)),
    )
    if uses_trajectory_head(cfg):
        model.risk_head = TrajectoryRiskHead(
            components=risk_head_components(cfg),
            n_features=mcfg["n_features"],
            hidden_size=mcfg["encoder"]["hidden_size"],
            hidden=mcfg["risk_head"]["hidden"],
        )
    return model
