import numpy as np

from nidra.data.schema import FEATURE_ORDER
from nidra.explain.shap_runner import build_shap_background, explain_current_risk, explain_predicted_stage, top_signals
from nidra.models.world_model import WorldModel


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def test_build_shap_background_shape():
    rng = np.random.default_rng(0)
    benign_states = rng.standard_normal((300, 45))
    bg = build_shap_background(benign_states, n_centroids=20, seed=0)
    assert bg.shape == (20, 45)


def test_build_shap_background_falls_back_when_too_few_samples():
    benign_states = np.random.randn(1, 45)
    bg = build_shap_background(benign_states, n_centroids=100)
    assert bg.shape[0] == 1


def test_explain_current_risk_output_format():
    model = _tiny_model()
    rng = np.random.default_rng(1)
    background = build_shap_background(rng.standard_normal((50, 45)), n_centroids=10)
    state = rng.standard_normal(45)
    attributions = explain_current_risk(state, background, model, nsamples=50)
    assert len(attributions) == 45
    assert all(a["feature"] in FEATURE_ORDER for a in attributions)
    assert all(a["direction"] in ("up", "down") for a in attributions)
    # sorted by |shap_value| descending
    abs_vals = [abs(a["shap_value"]) for a in attributions]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_top_signals_truncates():
    model = _tiny_model()
    rng = np.random.default_rng(2)
    background = build_shap_background(rng.standard_normal((50, 45)), n_centroids=10)
    state = rng.standard_normal(45)
    attributions = explain_current_risk(state, background, model, nsamples=50)
    top5 = top_signals(attributions, n=5)
    assert len(top5) == 5
    assert top5 == attributions[:5]


def test_explain_predicted_stage_output_format():
    model = _tiny_model()
    rng = np.random.default_rng(3)
    background = build_shap_background(rng.standard_normal((50, 45)), n_centroids=10)
    predicted_state = rng.standard_normal(45)
    attributions = explain_predicted_stage(predicted_state, background, model, stage_idx=1, nsamples=50)
    assert len(attributions) == 45
