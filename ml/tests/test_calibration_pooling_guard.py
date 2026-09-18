"""A Platt fit describes ONE pooled statistic, so it must not outlive it.

`risk_calibration.json` is fit by remapping the pooled ensemble risk score
per horizon. Change how that score is pooled out of the sampled trajectories
and the fitted (a, b) is the wrong function applied to the wrong numbers —
measured to score slightly BELOW raw scores under quantile pooling (see
REAL_DATA_RESULTS.md). These tests pin down that the mismatch is detected
rather than silently applied.
"""

from nidra.eval.calibrate import calibration_pooling_mismatch, pooling_signature


def _cfg(method="mean", quantile=0.9, head_reduction="before_pooling") -> dict:
    return {"rollout": {"risk_pooling_method": method, "risk_pooling_quantile": quantile,
                        "risk_pooling_head_reduction": head_reduction}}


def test_pooling_signature_defaults_to_the_original_behavior():
    assert pooling_signature({"rollout": {}}) == {
        "risk_pooling_method": "mean",
        "risk_pooling_quantile": 0.9,
        "risk_pooling_head_reduction": "before_pooling",
    }


def test_a_fit_matching_the_config_is_accepted():
    cfg = _cfg("quantile", 0.5)
    assert calibration_pooling_mismatch(pooling_signature(cfg), cfg) is None


def test_mean_pooled_config_accepts_a_mean_pooled_fit():
    assert calibration_pooling_mismatch({"risk_pooling_method": "mean"}, _cfg("mean")) is None


def test_a_pre_pooling_artifact_with_no_recorded_pooling_is_treated_as_mean_fit():
    # Artifacts written before pooling was configurable carry no pooling keys.
    # They were necessarily fit against mean pooling, so a quantile-pooled run
    # must reject them instead of trusting the silence.
    reason = calibration_pooling_mismatch({}, _cfg("quantile", 0.5))
    assert reason is not None
    assert "risk_pooling_method" in reason


def test_a_mean_pooled_artifact_is_rejected_by_a_quantile_pooled_run():
    reason = calibration_pooling_mismatch({"risk_pooling_method": "mean"}, _cfg("quantile", 0.5))
    assert reason is not None and "mean" in reason and "quantile" in reason


def test_a_fit_at_a_different_quantile_is_rejected():
    fitted = {"risk_pooling_method": "quantile", "risk_pooling_quantile": 0.75}
    reason = calibration_pooling_mismatch(fitted, _cfg("quantile", 0.5))
    assert reason is not None and "risk_pooling_quantile" in reason


def test_the_quantile_value_is_ignored_when_both_sides_pool_by_mean():
    # risk_pooling_quantile is dead config under mean pooling, so a stale
    # value there must not invalidate an otherwise-valid fit.
    fitted = {"risk_pooling_method": "mean", "risk_pooling_quantile": 0.99}
    assert calibration_pooling_mismatch(fitted, _cfg("mean", 0.5)) is None


def test_a_fit_under_a_different_head_reduction_order_is_rejected():
    fitted = {"risk_pooling_method": "quantile", "risk_pooling_quantile": 0.5,
              "risk_pooling_head_reduction": "before_pooling"}
    reason = calibration_pooling_mismatch(fitted, _cfg("quantile", 0.5, "after_pooling"))
    assert reason is not None and "risk_pooling_head_reduction" in reason


def test_head_reduction_is_ignored_when_both_sides_pool_by_mean():
    # The two reduction orders are mathematically identical under mean
    # pooling, so they cannot invalidate a mean-pooled fit.
    fitted = {"risk_pooling_method": "mean", "risk_pooling_head_reduction": "after_pooling"}
    assert calibration_pooling_mismatch(fitted, _cfg("mean")) is None
