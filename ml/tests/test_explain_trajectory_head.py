"""Explainability when the risk head reads more than the state.

The three explanation mechanisms (KernelSHAP on the observed state,
integrated gradients through the rollout, the model-internal what-if) all
assumed a head whose only input is the 45-feature vector. A history-aware
head breaks that assumption in a specific way: the score depends on the
encoder's hidden state, which SHAP is not perturbing.

The answer is not to pretend the extra inputs are zero — an explanation of a
head fed zeros is an explanation of a different head. SHAP over the state is
made a CONDITIONAL attribution, holding the observed history fixed, and a
trajectory head with no context supplied is an error rather than a plausible
wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nidra.data.schema import FEATURE_ORDER
from nidra.explain.counterfactual import counterfactual_rollout as simulate_counterfactual
from nidra.explain.shap_runner import explain_current_risk
from nidra.models.build import world_model_from_config


def _cfg(components=None) -> dict:
    cfg = {"model": {
        "n_features": 45,
        "encoder": {"hidden_size": 12, "num_layers": 1, "dropout": 0.0},
        "transition": {"mlp_hidden": 16, "logvar_min": -6.0, "logvar_max": 3.0, "state_clamp": 10.0},
        "risk_head": {"hidden": 8},
        "stage_head": {"hidden": 8, "n_stages": 6},
    }}
    if components is not None:
        cfg["model"]["risk_head"]["components"] = components
    return cfg


def _model(components=None):
    m = world_model_from_config(_cfg(components))
    m.eval()
    return m


def _x(L=8, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(L, 45)).astype("float32") * 0.3


class TestCounterfactual:
    def test_it_runs_for_a_per_state_head(self):
        out = simulate_counterfactual(_x(), _model(), "syn_ratio", 0.0, K=3, n_samples=4, stochastic=False)
        assert out["risk_mean_k"].shape == (1, 3)

    def test_it_runs_for_a_history_aware_head(self):
        out = simulate_counterfactual(_x(), _model(["state", "hidden", "delta", "logvar"]),
                                      "syn_ratio", 0.0, K=3, n_samples=4, stochastic=False)
        assert out["risk_mean_k"].shape == (1, 3)

    def test_the_clamped_feature_stays_clamped_in_the_simulated_states(self):
        j = FEATURE_ORDER.index("syn_ratio")
        out = simulate_counterfactual(_x(), _model(["state", "hidden"]), "syn_ratio", 0.0,
                                      K=3, n_samples=2, stochastic=False)
        assert np.allclose(out["predicted_states_mean"][0, :, j], 0.0)

    def test_it_still_carries_the_model_internal_label(self):
        out = simulate_counterfactual(_x(), _model(["state", "hidden"]), "syn_ratio", 0.0, K=2, n_samples=2)
        assert "model-internal" in out["label"]

    def test_a_per_state_head_is_scored_identically_to_before(self):
        """score_trajectory must reduce to score_states for a plain head, or
        this change moved a shipped number."""
        m, x = _model(), _x()
        torch.manual_seed(0)
        out = simulate_counterfactual(x, m, None, 0.0, K=4, n_samples=1, stochastic=False)
        with torch.no_grad():
            ro = m.rollout(torch.from_numpy(x).float().unsqueeze(0), K=4, n_samples=1, stochastic=False)
            direct = m.score_states(ro.states[:, 0])[0].numpy()
        assert np.allclose(out["risk_mean_k"], direct, atol=1e-6)


class TestShapOverTheState:
    def test_a_per_state_head_needs_no_context(self):
        bg = np.random.default_rng(1).normal(size=(6, 45)).astype("float32") * 0.2
        attrs = explain_current_risk(_x()[-1], bg, _model(), nsamples=32)
        assert len(attrs) == 45

    def test_a_history_aware_head_without_context_is_refused(self):
        bg = np.random.default_rng(1).normal(size=(6, 45)).astype("float32") * 0.2
        with pytest.raises(ValueError, match="context"):
            explain_current_risk(_x()[-1], bg, _model(["state", "hidden"]), nsamples=32)

    def test_a_history_aware_head_explains_conditionally_on_its_context(self):
        m = _model(["state", "hidden", "logvar"])
        x = _x()
        with torch.no_grad():
            ctx = m.observed_context(torch.from_numpy(x).float().unsqueeze(0))
        bg = np.random.default_rng(1).normal(size=(6, 45)).astype("float32") * 0.2
        attrs = explain_current_risk(x[-1], bg, m, nsamples=32, context=ctx)
        assert len(attrs) == 45
        assert all(np.isfinite(a["shap_value"]) for a in attrs)

    def test_a_context_missing_a_declared_component_is_refused(self):
        m = _model(["state", "hidden", "logvar"])
        x = _x()
        with torch.no_grad():
            ctx = m.observed_context(torch.from_numpy(x).float().unsqueeze(0))
        bg = np.random.default_rng(1).normal(size=(6, 45)).astype("float32") * 0.2
        with pytest.raises(ValueError, match="logvar"):
            explain_current_risk(x[-1], bg, m, nsamples=32, context={"hidden": ctx["hidden"]})

    def test_context_for_a_per_state_head_is_ignored_not_an_error(self):
        bg = np.random.default_rng(1).normal(size=(6, 45)).astype("float32") * 0.2
        attrs = explain_current_risk(_x()[-1], bg, _model(), nsamples=32,
                                     context={"hidden": torch.zeros(1, 12)})
        assert len(attrs) == 45
