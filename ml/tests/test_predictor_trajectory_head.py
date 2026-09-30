"""The served surface, driven with the history-aware risk head this phase
selects.

`NidraPredictor` reached the CTU-13 phase able to serve only the Run 8
per-state head: four of its call sites asked `score_states` for a risk
number, and a `TrajectoryRiskHead` cannot produce one from a bare state —
it reads the encoder hidden state as well. Every one of `forecast`,
`forecast_batch`, `counterfactual` and `explain` raised. Nothing caught it,
because every predictor test built the default head.

These tests assert only that the plumbing carries the head's context end to
end and that the contract still holds; the head here is randomly
initialised, so its NUMBERS mean nothing and none are asserted.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from nidra.data.schema import FEATURE_ORDER, HORIZON_LENGTH
from nidra.explain.counterfactual import COUNTERFACTUAL_LABEL
from nidra.models.heads import TrajectoryRiskHead


@pytest.fixture(scope="module")
def trajectory_predictor(trained_predictor):
    """The trained predictor with each member's risk head replaced by a
    `("state", "hidden")` one. Deep-copied: `trained_predictor` is
    session-scoped and shared with every other serving test."""
    predictor, windowed = trained_predictor
    predictor = copy.deepcopy(predictor)
    for model in predictor.models:
        model.risk_head = TrajectoryRiskHead(
            components=("state", "hidden"),
            n_features=predictor.cfg["model"]["n_features"],
            hidden_size=predictor.cfg["model"]["encoder"]["hidden_size"],
            hidden=predictor.cfg["model"]["risk_head"]["hidden"],
        )
        model.eval()
    return predictor, windowed


def test_forecast_completes_with_a_history_aware_head(trajectory_predictor):
    predictor, windowed = trajectory_predictor
    states = windowed["train"].X[0]
    result = predictor.forecast(
        states, host_id="10.0.0.1", origin_ts=datetime(2017, 7, 4, 9, 0, tzinfo=timezone.utc))

    assert len(result["horizons"]) == HORIZON_LENGTH
    for h in result["horizons"]:
        assert 0.0 <= h["p_compromise"] <= 1.0
        assert len(h["predicted_features"]) == len(FEATURE_ORDER)
    assert 0.0 <= result["observed_risk"] <= 1.0


def test_the_observed_risk_reads_the_history_not_the_last_state_alone(trajectory_predictor):
    """`observed_risk` is scored through `score_observed`, so it must respond
    to the history behind the origin. Two windows sharing a final state but
    differing earlier have to score differently — if they do not, the
    predictor is feeding the head a zeroed or truncated context."""
    predictor, windowed = trajectory_predictor
    a = windowed["train"].X[0].copy()
    b = windowed["train"].X[1].copy()
    b[-1] = a[-1]  # identical origin state, different history

    origin_ts = datetime(2017, 7, 4, 9, 0, tzinfo=timezone.utc)
    risk_a = predictor.forecast(a, host_id="h", origin_ts=origin_ts)["observed_risk"]
    risk_b = predictor.forecast(b, host_id="h", origin_ts=origin_ts)["observed_risk"]
    assert risk_a != pytest.approx(risk_b, abs=1e-9)


def test_forecast_batch_completes_and_matches_single_forecast_geometry(trajectory_predictor):
    predictor, windowed = trajectory_predictor
    batch = windowed["train"].X[:3]
    out = predictor.forecast_batch(batch, chunk=2)

    assert out["p_attack_at_k"].shape == (3, predictor.K)
    assert out["stage_mean_k"].shape[:2] == (3, predictor.K)
    assert np.isfinite(out["p_attack_at_k"]).all()
    assert ((0.0 <= out["p_attack_at_k"]) & (out["p_attack_at_k"] <= 1.0)).all()


def test_counterfactual_completes_with_a_history_aware_head(trajectory_predictor):
    predictor, windowed = trajectory_predictor
    out = predictor.counterfactual(windowed["train"].X[0], feature_name=FEATURE_ORDER[0], clamp_value=0.0)
    assert out["label"] == COUNTERFACTUAL_LABEL
    assert len(out["counterfactual"]) == HORIZON_LENGTH


def test_explain_completes_with_a_history_aware_head(trajectory_predictor):
    """`explain_current_risk` refuses to attribute a trajectory head without
    the context it holds fixed — the predictor has to supply it rather than
    letting the head be attributed against an implicit zero context."""
    predictor, windowed = trajectory_predictor
    out = predictor.explain(windowed["train"].X[0], horizon_k=1)
    assert {"current_risk_attributions", "predicted_stage_attributions", "temporal_saliency"} <= out.keys()
    # the two lists spell the feature key differently ("name" vs "feature");
    # that predates this fix and is not what this test is about
    for key in ("current_risk_attributions", "predicted_stage_attributions"):
        named = [a.get("name", a.get("feature")) for a in out[key]]
        assert named and set(named) <= set(FEATURE_ORDER)


def test_score_states_names_the_fix_instead_of_raising_a_bare_arity_error(trajectory_predictor):
    predictor, _ = trajectory_predictor
    with pytest.raises(TypeError, match="score_trajectory|score_observed"):
        predictor.models[0].score_states(torch.zeros(1, len(FEATURE_ORDER)))
