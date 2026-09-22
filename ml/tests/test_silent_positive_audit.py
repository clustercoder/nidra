"""The audit states a ceiling that later sections lean on, so the counting has
to be exact: an off-by-one in the floor stratum changes a claim about what is
achievable into a claim about what the model failed to do."""

from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.silent_positive_audit import SilentAudit, audit_markdown, audit_split


@pytest.fixture
def zero():
    return np.zeros(4, dtype="float32")


def _split(rows, is_active, risk, stage, zero, name="val"):
    return audit_split(np.array(rows, dtype="float32"), np.array(is_active),
                       np.array(risk), np.array(stage, dtype=object), zero, name)


def test_counts_a_pre_onset_row_at_the_floor(zero):
    a = _split(rows=[[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]],
               is_active=[0, 0, 1], risk=[1, 0, 1], stage=["benign", "benign", "exfil"], zero=zero)
    assert a.n_positive == 2
    assert a.n_preonset == 1 and a.n_preonset_inactive == 1
    assert a.n_floor_positive == 1
    assert a.n_floor_rows == 2                      # both silent rows, either class
    assert a.floor_prevalence == pytest.approx(0.5)
    assert a.share_of_positives == pytest.approx(0.5)


def test_an_attack_window_is_never_pre_onset(zero):
    """A positive that carries attack traffic is not the case this bounds."""
    a = _split(rows=[[1, 1, 1, 1]], is_active=[1], risk=[1], stage=["c2"], zero=zero)
    assert a.n_preonset == 0 and a.n_floor_positive == 0


def test_silent_but_not_at_the_floor_is_counted_separately(zero):
    """A backward-looking delta can be non-zero while the host is silent; that
    row still carries a trace of history, so it is not in the floor stratum."""
    a = _split(rows=[[0, 0, 0, 0.7]], is_active=[0], risk=[1], stage=["benign"], zero=zero)
    assert a.n_preonset_inactive == 1
    assert a.n_floor_positive == 0


def test_tolerance_is_tight(zero):
    a = _split(rows=[[0, 0, 0, 1e-3]], is_active=[0], risk=[1], stage=["benign"], zero=zero)
    assert a.n_floor_positive == 0


def test_no_floor_rows_gives_nan_not_a_division_error(zero):
    a = _split(rows=[[5, 5, 5, 5]], is_active=[1], risk=[0], stage=["benign"], zero=zero)
    assert a.n_floor_rows == 0
    assert a.floor_prevalence != a.floor_prevalence


def test_no_positives_gives_nan_share(zero):
    a = _split(rows=[[0, 0, 0, 0]], is_active=[0], risk=[0], stage=["benign"], zero=zero)
    assert a.share_of_positives != a.share_of_positives


def test_feature_count_mismatch_raises(zero):
    with pytest.raises(ValueError, match="silent state"):
        _split(rows=[[0, 0, 0]], is_active=[0], risk=[1], stage=["benign"], zero=zero)


def test_ceiling_is_the_stratum_prevalence_not_the_split_prevalence():
    """The bound is inside the stratum. Using the split's prevalence would
    understate it wherever the floor is enriched for positives, and overstate
    it wherever it is not."""
    a = SilentAudit(split="val", n_rows=100_000, n_positive=288, n_preonset=84,
                    n_preonset_inactive=84, n_floor_positive=58, n_floor_rows=47_708)
    assert a.floor_prevalence == pytest.approx(58 / 47_708)
    assert a.floor_prevalence != pytest.approx(288 / 100_000)


def test_markdown_reports_the_ceiling_and_the_leak_check():
    a = SilentAudit(split="val", n_rows=98_876, n_positive=288, n_preonset=84,
                    n_preonset_inactive=84, n_floor_positive=58, n_floor_rows=47_708)
    md = audit_markdown([a])
    assert "0.001216" in md
    assert "20.1%" in md
    assert "leak check" in md
