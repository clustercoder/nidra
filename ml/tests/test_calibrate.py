"""Tests for nidra.eval.calibrate — post-hoc Platt-scaling recalibration of
the risk head's forecast output. See the module docstring for why this
exists (the measured calibration gap: good AUC-PR ranking, near-zero recall
at the mandated 0.75 threshold)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from nidra.eval.calibrate import (
    apply_platt,
    apply_platt_by_horizon,
    fit_platt,
    fit_platt_by_horizon,
    load_calibration,
    save_calibration,
)


def _make_underconfident_data(n=2000, seed=0):
    """Simulates the exact failure mode this module targets: a probability
    that ranks correctly but is compressed toward 0.5 (never reaches 0.75
    even for true positives) — i.e. `p_compressed = sigmoid(0.3 * logit(p_true))`."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0, 3, size=n)
    labels = (rng.uniform(size=n) < 1 / (1 + np.exp(-true_logit))).astype(int)
    compressed_probs = 1 / (1 + np.exp(-0.3 * true_logit))
    return compressed_probs, labels


def test_fit_platt_sharpens_underconfident_probabilities():
    probs, labels = _make_underconfident_data()
    params = fit_platt(probs, labels)
    assert not params["degenerate"]
    assert params["a"] > 1.0  # must sharpen (undo the 0.3x compression), not flatten further

    calibrated = apply_platt(probs, params)
    # more calibrated probabilities should exceed 0.75 among true positives
    frac_ge_75_before = (probs[labels == 1] >= 0.75).mean()
    frac_ge_75_after = (calibrated[labels == 1] >= 0.75).mean()
    assert frac_ge_75_after > frac_ge_75_before


def test_apply_platt_preserves_ranking_and_auc_pr():
    """Platt scaling with a>0 is strictly monotonic — AUC-PR (rank-based)
    must be numerically identical before and after, for any fitted params."""
    probs, labels = _make_underconfident_data(seed=1)
    params = fit_platt(probs, labels)
    calibrated = apply_platt(probs, params)

    auc_before = average_precision_score(labels, probs)
    auc_after = average_precision_score(labels, calibrated)
    assert auc_before == pytest.approx(auc_after, abs=1e-9)

    order_before = np.argsort(probs)
    order_after = np.argsort(calibrated)
    assert np.array_equal(order_before, order_after)


def test_fit_platt_degenerate_single_class_falls_back_to_identity():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.zeros(4, dtype=int)  # only one class present
    params = fit_platt(probs, labels)
    assert params["degenerate"] is True
    assert params["a"] == 1.0
    assert params["b"] == 0.0
    np.testing.assert_allclose(apply_platt(probs, params), probs, atol=1e-6)


def test_fit_platt_empty_input_does_not_crash():
    params = fit_platt(np.array([]), np.array([]))
    assert params["degenerate"] is True
    assert params["n"] == 0


def test_fit_platt_by_horizon_shapes():
    rng = np.random.default_rng(2)
    N, K = 500, 6
    risk_mean_k = rng.uniform(size=(N, K))
    future_is_attack = (rng.uniform(size=(N, K)) < 0.3).astype(int)

    params_by_k = fit_platt_by_horizon(risk_mean_k, future_is_attack)
    assert len(params_by_k) == K

    calibrated = apply_platt_by_horizon(risk_mean_k, params_by_k)
    assert calibrated.shape == (N, K)
    assert np.isfinite(calibrated).all()
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_apply_platt_by_horizon_rejects_wrong_length():
    risk_mean_k = np.zeros((10, 6))
    with pytest.raises(ValueError):
        apply_platt_by_horizon(risk_mean_k, [{"a": 1.0, "b": 0.0}] * 5)  # 5 != 6


def test_save_and_load_calibration_roundtrip(tmp_path):
    params_by_k = [{"a": 1.5, "b": -0.2, "n": 100, "degenerate": False} for _ in range(6)]
    metadata = {"fit_split": "val", "n_val_samples": 100}
    path = tmp_path / "risk_calibration.json"

    save_calibration(path, params_by_k, metadata)
    loaded_params, loaded_meta = load_calibration(path)

    assert loaded_params == params_by_k
    assert loaded_meta == metadata


def test_load_calibration_returns_none_when_missing(tmp_path):
    assert load_calibration(tmp_path / "does_not_exist.json") is None
