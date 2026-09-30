"""Forcing a common pooling rule.

Each run selects its own pooling on its own validation, which is the protocol
and the right unit for "what would you deploy". It is the wrong unit for "does
adding CTU help the dynamics", because a row-to-row comparison then differs in
the readout as well as the training set. These tests pin that the override
changes only the readout and that it announces itself, since a forced run that
looked like a selected one in the metrics JSON would be a fabricated headline.
"""

from __future__ import annotations

import pytest

from nidra.eval.benchmark import _with_forced_pooling


def _op():
    return {"pooling": {"method": "mean", "quantile": None, "horizon_reduction": "integrated"},
            "pooling_key": "mean|q=-|integrated",
            "calibration": {"params_by_k": [{"a": 1.0, "b": 0.0}]},
            "threshold": {"f1_optimal_calibrated": 0.5019685}}


def test_only_the_pooling_changes():
    out = _with_forced_pooling(_op(), "mean|q=-|max")
    assert out["pooling"] == {"method": "mean", "quantile": None, "horizon_reduction": "max"}
    assert out["threshold"] == _op()["threshold"]          # threshold untouched
    assert out["calibration"] == _op()["calibration"]      # calibration untouched


def test_the_override_records_what_it_replaced():
    """A forced run that looked like a selected one in the metrics JSON would
    put a fabricated configuration in the headline table."""
    out = _with_forced_pooling(_op(), "mean|q=-|max")
    assert out["pooling_forced"] is True
    assert out["pooling_key"] == "mean|q=-|max"
    assert out["pooling_key_selected"] == "mean|q=-|integrated"


def test_a_quantile_pooling_key_round_trips():
    out = _with_forced_pooling(_op(), "quantile|q=0.90|max")
    assert out["pooling"]["method"] == "quantile"
    assert out["pooling"]["quantile"] == pytest.approx(0.9)


def test_a_quantile_spelled_with_one_decimal_is_rejected():
    """The canonical key writes two decimals. Accepting "q=0.9" would label the
    output with a key the run did not use."""
    with pytest.raises(ValueError, match="round-trip"):
        _with_forced_pooling(_op(), "quantile|q=0.9|max")


def test_p_above_half_round_trips():
    out = _with_forced_pooling(_op(), "p_above_half|q=-|max")
    assert out["pooling"]["method"] == "p_above_half"
    assert out["pooling"]["quantile"] is None


def test_a_key_that_does_not_round_trip_is_rejected():
    """Silently scoring under a rule other than the one named would make the
    comparison the flag exists for meaningless."""
    with pytest.raises(ValueError, match="round-trip"):
        _with_forced_pooling(_op(), "mean|0.5|max")        # quantile missing its q= prefix


def test_a_malformed_key_says_what_the_shape_should_be():
    """This is a CLI flag; a bare unpacking error would tell the user nothing."""
    with pytest.raises(ValueError, match="<method>\\|q="):
        _with_forced_pooling(_op(), "mean|max")
