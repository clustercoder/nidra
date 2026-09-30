import numpy as np

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER
from nidra.explain.saliency import temporal_saliency
from nidra.models.world_model import WorldModel


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def test_temporal_saliency_output_shape():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    result = temporal_saliency(model, x, target_feature="syn_ratio", horizon_k=0, K=6)
    assert len(result["window_importance"]) == CONTEXT_LENGTH
    assert 0 <= result["driving_window"] < CONTEXT_LENGTH
    assert all(v >= 0 for v in result["window_importance"])


def test_temporal_saliency_unknown_feature_raises():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    try:
        temporal_saliency(model, x, target_feature="not_a_real_feature")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_temporal_saliency_offset_relative_to_now():
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    result = temporal_saliency(model, x, target_feature="dst_port_entropy", horizon_k=2, K=6)
    assert result["driving_window_offset_from_now"] == result["driving_window"] - (CONTEXT_LENGTH - 1)
