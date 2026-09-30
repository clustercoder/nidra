"""Integrated-gradient forecast attributions: completeness, shape, and the
deletion-based faithfulness check."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nidra.explain.forecast_attribution import (
    explain_forecast,
    faithfulness_check,
    forecast_score_fn,
    integrated_gradients,
    summarize_attributions,
)
from nidra.data.schema import FEATURE_ORDER


def test_integrated_gradients_completeness_on_a_smooth_function():
    # f(x) = sigmoid(sum of x) -> IG must sum to f(x) - f(b) up to quadrature error
    def f(x):
        return torch.sigmoid(x.sum(dim=(1, 2)) / 10.0)
    x = np.random.default_rng(0).normal(size=(4, 6)).astype("float32")
    b = np.zeros_like(x)
    attr, fx, fb = integrated_gradients(f, x, b, steps=64)
    assert attr.shape == x.shape
    assert abs(attr.sum() - (fx - fb)) < 1e-3


def test_summary_names_features_and_offsets():
    attr = np.zeros((5, len(FEATURE_ORDER)), dtype="float32")
    attr[4, FEATURE_ORDER.index("syn_ratio")] = 0.3
    attr[2, FEATURE_ORDER.index("bytes_total")] = -0.1
    x = np.zeros_like(attr)
    s = summarize_attributions(attr, x, n_features=2, n_cells=2)
    assert s["features"][0]["feature"] == "syn_ratio" and s["features"][0]["direction"] == "up"
    assert s["cells"][0] == {"window_offset": 0, "feature": "syn_ratio", "contribution": pytest.approx(0.3)}
    assert s["cells"][1]["window_offset"] == -2
    assert s["driving_window_offset"] == 0


def test_faithfulness_check_prefers_true_drivers():
    # f depends only on cell (0, 0); attributions that point there beat random deletion
    def f(x):
        return torch.sigmoid(3.0 * x[:, 0, 0])
    x = np.ones((3, 4), dtype="float32")
    b = np.zeros_like(x)
    attr = np.zeros_like(x)
    attr[0, 0] = 1.0
    res = faithfulness_check(f, x, b, attr, m=1, n_random=10, seed=0)
    assert res["drop_top_m"] > 0 and res["faithful"] and res["beats_random_fraction"] >= 0.5


def test_explain_forecast_on_trained_predictor(trained_predictor):
    predictor, windowed = trained_predictor
    scaled = predictor._validate_and_scale(windowed["train"].X[0])
    out = explain_forecast(predictor.models, scaled, predictor.scaler.zero_state_scaled(), predictor.K, steps=8, m=4, n_random=5)
    assert out["method"] == "integrated_gradients"
    assert len(out["features"]) == 10 and len(out["per_window_total"]) == scaled.shape[0]
    assert out["completeness_gap"] < 0.05
    assert "faithfulness" in out and set(out["faithfulness"]) >= {"drop_top_m", "drop_random_m_mean", "faithful"}


def test_predictor_explain_carries_forecast_attributions(trained_predictor):
    predictor, windowed = trained_predictor
    res = predictor.explain(windowed["train"].X[0], horizon_k=1)
    fa = res["forecast_attributions"]
    assert fa["features"] and "faithfulness" in fa
    assert fa["space"].startswith("scaled features")
