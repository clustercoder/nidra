"""Integration test for NidraPredictor against a tiny model trained on
synthetic fixtures (see tests/conftest.py::trained_predictor) — verifies the
serving CONTRACT (shapes, schema validation, output fields), not real-data
forecast quality.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER, HORIZON_LENGTH


def test_forecast_output_contract(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]  # [L, F] raw
    origin_ts = datetime(2017, 7, 4, 9, 0, 0, tzinfo=timezone.utc)

    result = predictor.forecast(states, host_id="10.0.0.1", origin_ts=origin_ts)

    assert result["host_id"] == "10.0.0.1"
    assert len(result["horizons"]) == HORIZON_LENGTH
    for h in result["horizons"]:
        assert set(h.keys()) >= {"k", "ts", "p_compromise", "ci_low", "ci_high", "stage_dist", "predicted_features"}
        assert 0.0 <= h["p_compromise"] <= 1.0
        assert len(h["predicted_features"]) == len(FEATURE_ORDER)
    assert "model_version" in result
    assert "schema_ver" in result
    assert result["lead_time_s"] is None or result["lead_time_s"] >= 0
    assert len(result["top_signals"]) <= 5


def test_forecast_rejects_wrong_feature_width(trained_predictor):
    predictor, windowed = trained_predictor
    bad_states = windowed["train"].X[0][:, :-1]  # 44 features instead of 45
    with pytest.raises(ValueError):
        predictor.forecast(bad_states, host_id="h", origin_ts=datetime.now(timezone.utc))


def test_forecast_rejects_wrong_context_length(trained_predictor):
    predictor, windowed = trained_predictor
    bad_states = windowed["train"].X[0][:-1, :]  # L-1 windows instead of L
    with pytest.raises(ValueError):
        predictor.forecast(bad_states, host_id="h", origin_ts=datetime.now(timezone.utc))


def test_forecast_rejects_nan_input(trained_predictor):
    predictor, windowed = trained_predictor
    bad_states = windowed["train"].X[0].copy()
    bad_states[0, 0] = np.nan
    with pytest.raises(ValueError):
        predictor.forecast(bad_states, host_id="h", origin_ts=datetime.now(timezone.utc))


def test_counterfactual_output_contract(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    result = predictor.counterfactual(states, feature_name="new_peer_count", clamp_value=0.0)
    assert result["label"] == "model-internal what-if"
    assert len(result["risk_mean_k"]) == HORIZON_LENGTH


def test_explain_output_contract(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    result = predictor.explain(states, horizon_k=0)
    assert "current_risk_attributions" in result
    assert "predicted_stage_attributions" in result
    assert "temporal_saliency" in result
    assert len(result["current_risk_attributions"]) <= 10


def test_scaler_never_refit_by_predictor(trained_predictor):
    predictor, _ = trained_predictor
    center_before = predictor.scaler.scaler.center_.copy()
    _ = predictor.forecast(np.random.randn(CONTEXT_LENGTH, 45).astype("float32"),
                            host_id="h", origin_ts=datetime.now(timezone.utc))
    np.testing.assert_array_equal(center_before, predictor.scaler.scaler.center_)
