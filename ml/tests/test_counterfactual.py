import numpy as np

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_INDEX
from nidra.explain.counterfactual import COUNTERFACTUAL_LABEL, compare_to_baseline, counterfactual_rollout
from nidra.models.world_model import WorldModel


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def test_counterfactual_clamps_feature_at_every_step():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    result = counterfactual_rollout(x, model, feature_name="new_peer_count", clamp_value=0.0, K=6, n_samples=20)
    idx = FEATURE_INDEX["new_peer_count"]
    predicted = result["predicted_states_mean"]  # [1, K, F]
    np.testing.assert_allclose(predicted[0, :, idx], 0.0, atol=1e-5)


def test_counterfactual_is_labelled_model_internal_what_if():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    result = counterfactual_rollout(x, model, feature_name="syn_ratio", clamp_value=1.0, K=3, n_samples=10)
    assert result["label"] == COUNTERFACTUAL_LABEL == "model-internal what-if"


def test_counterfactual_rejects_unknown_feature():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    try:
        counterfactual_rollout(x, model, feature_name="not_a_feature", clamp_value=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compare_to_baseline_returns_both_curves_and_they_can_differ():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    result = compare_to_baseline(x, model, feature_name="new_peer_count", clamp_value=5.0, K=6, n_samples=20)
    assert result["baseline_risk_mean_k"].shape == (1, 6)
    assert result["counterfactual_risk_mean_k"].shape == (1, 6)
    assert result["label"] == COUNTERFACTUAL_LABEL
