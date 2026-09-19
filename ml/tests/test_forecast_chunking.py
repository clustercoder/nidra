"""Chunked rollout must be an exact refactor, not an approximation.

`world_model_forecast`/`ensemble_world_model_forecast` tile their whole input
batch by `n_samples` in one allocation, so a large eval split at a high sample
count gets OOM-killed by the OS with no Python traceback (see
REAL_DATA_RESULTS.md). Chunking bounds that peak, and every reduction in those
functions is per-row, so a chunked run must agree with an unchunked one.
"""

import numpy as np
import pytest
import torch

from nidra.eval.baselines import (
    _chunk_bounds,
    ensemble_world_model_forecast,
    world_model_forecast,
)
from nidra.models.world_model import WorldModel

K = 6


def _tiny_model(seed: int = 0) -> WorldModel:
    torch.manual_seed(seed)
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def _X(n_rows: int = 7) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n_rows, 30, 45)).astype(np.float32)


def test_chunk_bounds_none_is_a_single_full_width_pass():
    assert _chunk_bounds(10, None) == [(0, 10)]


def test_chunk_bounds_larger_than_input_is_a_single_pass():
    assert _chunk_bounds(10, 64) == [(0, 10)]


def test_chunk_bounds_covers_every_row_exactly_once_with_a_ragged_tail():
    bounds = _chunk_bounds(10, 4)
    assert bounds == [(0, 4), (4, 8), (8, 10)]
    covered = [row for lo, hi in bounds for row in range(lo, hi)]
    assert covered == list(range(10))


def test_chunk_bounds_rejects_nonsense_chunk_size():
    with pytest.raises(ValueError):
        _chunk_bounds(10, 0)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64, None])
def test_deterministic_forecast_is_identical_chunked_or_not(chunk_size):
    # stochastic=False removes rollout sampling noise, which is the only thing
    # chunking legitimately changes — so here the two must match exactly.
    model, X = _tiny_model(), _X()
    unchunked = world_model_forecast(X, model, K=K, n_samples=3, stochastic=False)
    chunked = world_model_forecast(X, model, K=K, n_samples=3, stochastic=False, chunk_size=chunk_size)
    for key in unchunked:
        np.testing.assert_allclose(chunked[key], unchunked[key], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 3, None])
def test_deterministic_ensemble_forecast_is_identical_chunked_or_not(chunk_size):
    models, X = [_tiny_model(0), _tiny_model(1)], _X()
    unchunked = ensemble_world_model_forecast(X, models, K=K, n_samples_per_member=2, stochastic=False)
    chunked = ensemble_world_model_forecast(X, models, K=K, n_samples_per_member=2, stochastic=False,
                                             chunk_size=chunk_size)
    for key in unchunked:
        np.testing.assert_allclose(chunked[key], unchunked[key], rtol=1e-6, atol=1e-6)


def test_chunking_preserves_row_order_under_quantile_pooling():
    # Quantile pooling is where a mis-stitched chunk loop would show up as
    # shuffled rows rather than as wrong-looking numbers.
    model, X = _tiny_model(), _X(9)
    kwargs = dict(K=K, n_samples=4, stochastic=False,
                  risk_pooling_method="quantile", risk_pooling_quantile=0.75)
    unchunked = world_model_forecast(X, model, **kwargs)
    chunked = world_model_forecast(X, model, chunk_size=2, **kwargs)
    np.testing.assert_allclose(chunked["risk_mean_k"], unchunked["risk_mean_k"], rtol=1e-6, atol=1e-6)


def test_chunked_stochastic_forecast_keeps_shapes_and_probability_bounds():
    # Sampled trajectories differ per pass (each draws its own noise), so this
    # asserts the contract, not equality: shapes, bounds, and no NaNs.
    model, X = _tiny_model(), _X(5)
    out = world_model_forecast(X, model, K=K, n_samples=4, stochastic=True, chunk_size=2)
    assert out["risk_mean_k"].shape == (5, K)
    assert out["risk_over_horizon"].shape == (5,)
    assert out["predicted_states_mean"].shape == (5, K, 45)
    assert np.isfinite(out["risk_mean_k"]).all()
    assert (out["risk_mean_k"] >= 0).all() and (out["risk_mean_k"] <= 1).all()


def test_calibration_is_applied_once_on_the_full_array_not_per_chunk():
    model, X = _tiny_model(), _X(6)
    calibration = [{"a": 2.0, "b": 0.5, "n": 100, "degenerate": False} for _ in range(K)]
    kwargs = dict(K=K, n_samples=3, stochastic=False, calibration=calibration)
    unchunked = world_model_forecast(X, model, **kwargs)
    chunked = world_model_forecast(X, model, chunk_size=2, **kwargs)
    np.testing.assert_allclose(chunked["risk_mean_k"], unchunked["risk_mean_k"], rtol=1e-6, atol=1e-6)
    # risk_over_horizon must be derived from the CALIBRATED per-k scores.
    np.testing.assert_allclose(chunked["risk_over_horizon"], chunked["risk_mean_k"].max(axis=1),
                               rtol=1e-6, atol=1e-6)


def test_calibration_passes_the_chunk_size_through(monkeypatch):
    """`forecast_chunk_size` exists because the forecast functions tile the
    whole batch by n_samples in ONE allocation, and a large split at a high
    sample count is then OOM-killed by the OS with no Python traceback.

    Every eval call site threads it through except this one, which is how a
    full production eval died 16 minutes in: 4,000 rows x 200 samples
    materialized at once during the calibration pass. An unchunked call here
    is not slower, it is fatal, so the parameter has to arrive.
    """
    seen = {}

    def spy(X, model, K, n_samples=200, **kw):
        seen["chunk_size"] = kw.get("chunk_size", "NOT PASSED")
        n = X.shape[0]
        return {"risk_mean_k": np.zeros((n, K)), "risk_ci_low_k": np.zeros((n, K)),
                "risk_ci_high_k": np.zeros((n, K)), "risk_over_horizon": np.zeros(n)}

    import nidra.eval.calibration as calmod
    monkeypatch.setattr(calmod, "world_model_forecast", spy)

    calmod.calibration_by_horizon(
        np.zeros((8, 3, 45), dtype="float32"),
        np.zeros((8, 6), dtype=int),
        model=None,
        n_samples=4,
        chunk_size=2,
    )
    assert seen["chunk_size"] == 2
