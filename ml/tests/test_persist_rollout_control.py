"""The strict transition ablation, and what it must equal for each head type."""

from __future__ import annotations

import numpy as np
import torch

from nidra.eval.systems import ScoreBundle, score_world_model
from nidra.models.heads import TrajectoryRiskHead
from nidra.models.world_model import WorldModel


def _model(seed: int = 0) -> WorldModel:
    torch.manual_seed(seed)
    m = WorldModel(n_features=45, hidden_size=8, encoder_layers=1, transition_mlp_hidden=16)
    m.eval()
    return m


def _bundle(K: int = 3) -> ScoreBundle:
    return ScoreBundle(K=K)


def test_for_a_per_state_head_the_two_persistence_definitions_agree():
    """With a head that reads only the state, a rollout that repeats S_t
    scores exactly what the head scores on S_t — so the new control is a
    no-op there, which is what makes it safe to add to every run."""
    m = _model()
    X = (torch.randn(6, 5, 45) * 0.3).numpy()
    Y = (torch.randn(6, 3, 45) * 0.3).numpy()
    b = score_world_model(_bundle(), [m], X, Y, n_samples_per_member=2, chunk=4, with_ablations=False)
    assert np.allclose(b.risk_persist_k, b.risk_obs[:, None].repeat(3, axis=1), atol=1e-5)


def test_for_a_history_aware_head_they_differ_because_the_encoder_advances():
    m = _model()
    m.risk_head = TrajectoryRiskHead(components=("state", "hidden"), n_features=45, hidden_size=8)
    X = (torch.randn(6, 5, 45) * 0.3).numpy()
    Y = (torch.randn(6, 3, 45) * 0.3).numpy()
    b = score_world_model(_bundle(), [m], X, Y, n_samples_per_member=2, chunk=4, with_ablations=False)
    assert not np.allclose(b.risk_persist_k, b.risk_obs[:, None].repeat(3, axis=1), atol=1e-4)
    assert np.isfinite(b.risk_persist_k).all()


def test_the_control_appears_as_its_own_system():
    from nidra.eval.systems import system_scores

    m = _model()
    X = (torch.randn(4, 5, 45) * 0.3).numpy()
    Y = (torch.randn(4, 3, 45) * 0.3).numpy()
    b = score_world_model(_bundle(), [m], X, Y, n_samples_per_member=2, chunk=4, with_ablations=False)
    scores = system_scores(b, {"method": "mean"})
    assert "persistence_rollout" in scores and scores["persistence_rollout"].shape == (4,)
