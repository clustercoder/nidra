import numpy as np
import torch

from nidra.eval.baselines import (
    baseline_lr_current_state,
    baseline_lr_flattened_history,
    baseline_oracle,
    baseline_persistence,
    world_model_forecast,
)
from nidra.models.world_model import WorldModel


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def test_lr_current_state_shapes():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 45))
    y_train = rng.integers(0, 2, 100)
    X_eval = rng.standard_normal((20, 45))
    clf, probs = baseline_lr_current_state(X_train, y_train, X_eval)
    assert probs.shape == (20,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_lr_flattened_history_shapes():
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((100, 30, 45))
    y_train = rng.integers(0, 2, 100)
    X_eval = rng.standard_normal((20, 30, 45))
    clf, probs = baseline_lr_flattened_history(X_train, y_train, X_eval)
    assert probs.shape == (20,)
    assert clf.coef_.shape[1] == 30 * 45


def test_persistence_baseline_matches_risk_head_on_last_window():
    model = _tiny_model()
    model.eval()
    X_last = np.random.randn(5, 45).astype("float32")
    probs = baseline_persistence(X_last, model)
    with torch.no_grad():
        expected, _ = model.score_states(torch.from_numpy(X_last).float())
    np.testing.assert_allclose(probs, expected.numpy(), atol=1e-6)


def test_oracle_baseline_shapes():
    model = _tiny_model()
    Y_true = np.random.randn(5, 6, 45).astype("float32")
    risk_over_horizon, risk_k, stage_k = baseline_oracle(Y_true, model)
    assert risk_over_horizon.shape == (5,)
    assert risk_k.shape == (5, 6)
    assert stage_k.shape == (5, 6, 6)


def test_world_model_forecast_shapes_and_bounds():
    model = _tiny_model()
    X = np.random.randn(4, 30, 45).astype("float32")
    out = world_model_forecast(X, model, K=6, n_samples=10)
    assert out["risk_mean_k"].shape == (4, 6)
    assert out["risk_ci_low_k"].shape == (4, 6)
    assert out["risk_ci_high_k"].shape == (4, 6)
    assert out["stage_mean_k"].shape == (4, 6, 6)
    assert out["risk_over_horizon"].shape == (4,)
    assert out["predicted_states_mean"].shape == (4, 6, 45)
    assert (out["risk_ci_low_k"] <= out["risk_ci_high_k"] + 1e-6).all()
