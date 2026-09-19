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


def test_the_two_curves_share_their_noise_so_the_gap_between_them_is_the_clamp():
    """A what-if compares two sampled rollouts. If each draws its own noise,
    part of the difference it attributes to the clamped feature is Monte Carlo
    error — and at a step where the real effect is small, that flips the sign
    of what the user is shown. It is not only a flaky test; it is the endpoint
    being wrong about the one thing it exists to report.

    So both rollouts draw the SAME noise (common random numbers). The property
    that buys is variance reduction on the DIFFERENCE, which is what this
    measures: across seeds, the baseline-minus-counterfactual gap should move
    far less than either curve does on its own.

    With independent noise the opposite holds — the variance of a difference
    of two independent draws is the sum of their variances, so the gap would
    be roughly 1.4x MORE variable than either curve, not less. That is what
    makes this discriminating rather than decorative.
    """
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")

    baselines, gaps = [], []
    for seed in range(8):
        out = compare_to_baseline(
            x, model, feature_name="syn_ratio", clamp_value=0.0,
            K=4, n_samples=16, seed=seed,
        )
        baselines.append(out["baseline_risk_mean_k"][0])
        gaps.append(out["baseline_risk_mean_k"][0] - out["counterfactual_risk_mean_k"][0])

    curve_spread = float(np.std(np.array(baselines), axis=0).mean())
    gap_spread = float(np.std(np.array(gaps), axis=0).mean())
    assert curve_spread > 0, "a stochastic rollout that does not vary across seeds is not stochastic"
    assert gap_spread < curve_spread, (
        f"the clamp's measured effect varies more across seeds ({gap_spread:.5f}) than "
        f"the curve itself does ({curve_spread:.5f}) — the two rollouts are not sharing noise"
    )


def test_two_calls_with_the_same_inputs_give_the_same_curves():
    """A what-if a user can re-run and disagree with is not evidence of
    anything. Same states, same clamp, same numbers."""
    model = _tiny_model()
    x = np.random.randn(CONTEXT_LENGTH, 45).astype("float32")
    kwargs = dict(feature_name="new_peer_count", clamp_value=0.0, K=4, n_samples=16)
    first = compare_to_baseline(x, model, **kwargs)
    second = compare_to_baseline(x, model, **kwargs)
    np.testing.assert_allclose(
        first["counterfactual_risk_mean_k"], second["counterfactual_risk_mean_k"]
    )
    np.testing.assert_allclose(
        first["baseline_risk_mean_k"], second["baseline_risk_mean_k"]
    )
