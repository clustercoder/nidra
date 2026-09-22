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


def _probe(scores, y, hosts, floor=None):
    from nidra.scripts.silent_positive_audit import floor_stratum_probe
    n = len(y)
    return floor_stratum_probe(np.array(scores, dtype="float64"), np.array(y),
                               np.array(hosts, dtype=object),
                               np.ones(n, dtype=bool) if floor is None else np.array(floor))


def test_a_state_only_head_lands_exactly_on_the_ceiling():
    """One constant over the whole stratum: AP is the prevalence, ROC is 0.5."""
    p = _probe(scores=[0.3] * 10, y=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], hosts=["a"] * 5 + ["b"] * 5)
    assert p.ap == pytest.approx(p.ceiling)
    assert p.lift_over_ceiling == pytest.approx(1.0)
    assert p.roc == pytest.approx(0.5)


def test_host_identity_is_named_when_the_lift_vanishes_within_the_host():
    """All positives on one host, ranked high as a block, but unordered inside
    it — the shape the CTU floor stratum actually has."""
    # The infected host's whole block outranks everything else, but inside it
    # the positives sit BELOW its negatives — the shape the real stratum has,
    # where within-host ROC came back at 0.38.
    scores = [0.3, 0.2, 0.9, 0.8] + [0.01] * 8
    y = [1, 1, 0, 0] + [0] * 8
    hosts = ["infected"] * 4 + ["other"] * 8
    p = _probe(scores, y, hosts)
    assert p.n_positive_hosts == 1
    assert p.lift_over_ceiling > 2.0            # looks like a big win
    # 0.9, not 1.0: the host's own negatives tie with its positives under the
    # host mean, which is exactly what "the ranking is the host" looks like.
    assert p.host_mean_roc == pytest.approx(0.9)
    assert p.within_host_lift < 1.0             # and is worth less than a constant
    assert p.is_host_identity is True


def test_real_timing_signal_is_not_called_host_identity():
    scores = [0.9, 0.85, 0.1, 0.05] + [0.02] * 8
    y = [1, 1, 0, 0] + [0] * 8
    hosts = ["infected"] * 4 + ["other"] * 8
    p = _probe(scores, y, hosts)
    assert p.within_host_lift == pytest.approx(1 / 0.5, abs=0.01)   # perfect within-host ranking
    assert p.is_host_identity is False


def test_probe_respects_the_floor_mask():
    p = _probe(scores=[9.0, 0.1, 0.2, 0.3], y=[1, 1, 0, 0], hosts=["a", "a", "b", "b"],
               floor=[False, True, True, True])
    assert p.n_rows == 3 and p.n_positive == 1


def test_probe_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        _probe(scores=[0.1, 0.2], y=[0, 0], hosts=["a", "b"])


def test_probe_markdown_reports_the_verdict():
    from nidra.scripts.silent_positive_audit import probe_markdown
    p = _probe(scores=[0.3, 0.2, 0.9, 0.8] + [0.01] * 8, y=[1, 1, 0, 0] + [0] * 8,
               hosts=["infected"] * 4 + ["other"] * 8)
    md = probe_markdown({"state+hidden": p}, "val")
    assert "host identity" in md
    assert "within-host" in md


def test_a_constant_head_is_called_constant_not_signal_bearing():
    """`is_host_identity` is False for a constant head, and reporting that as
    'carries timing signal' would read as praise for emitting one number."""
    p = _probe(scores=[0.3] * 10, y=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], hosts=["a"] * 5 + ["b"] * 5)
    assert p.is_constant is True
    assert p.is_host_identity is False
    assert "ceiling" in p.verdict and "timing signal" not in p.verdict


def test_a_varying_head_is_not_called_constant():
    p = _probe(scores=[0.3, 0.2, 0.9, 0.8] + [0.01] * 8, y=[1, 1, 0, 0] + [0] * 8,
               hosts=["infected"] * 4 + ["other"] * 8)
    assert p.is_constant is False
    assert p.verdict == "**host identity**"


def test_probe_markdown_says_which_stratum_it_measured():
    from nidra.scripts.silent_positive_audit import probe_markdown
    p = _probe(scores=[0.3, 0.2, 0.9, 0.8] + [0.01] * 8, y=[1, 1, 0, 0] + [0] * 8,
               hosts=["infected"] * 4 + ["other"] * 8)
    assert "floor stratum" in probe_markdown({"h": p}, "val", stratum="floor")
    assert "whole split" in probe_markdown({"h": p}, "val", stratum="all")


def test_an_all_rows_mask_asks_the_same_question():
    """The decomposition is not specific to the floor; a split-wide result that
    rests on one infected host deserves it too."""
    p = _probe(scores=[0.9, 0.8, 0.7, 0.6] + [0.01] * 8, y=[1, 1, 0, 0] + [0] * 8,
               hosts=["infected"] * 4 + ["other"] * 8, floor=[True] * 12)
    assert p.n_rows == 12
    assert p.host_mean_roc >= 0.9
