"""The fusion screen closes a line of investigation, so its arithmetic and
especially its verdict have to be right: a rule that looks like a win because
the baseline row was misidentified would send the phase down a dead end."""

from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.head_fusion_screen import (BASELINE_RULE, FUSION_RULES, FusionRow, any_rule_wins,
                                              fusion_markdown, score_rules)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    n = 400
    y = np.zeros(n, dtype=int)
    y[:40] = 1
    risk = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.1, n), 0, 1)
    nonben = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, n), 0, 1)
    return risk, nonben, y


def test_every_rule_is_scored(scores):
    rows = score_rules(*scores, "composite")
    assert {r.rule for r in rows} == set(FUSION_RULES)
    assert all(r.label_set == "composite" and r.n_positive == 40 for r in rows)


def test_rules_are_parameter_free_and_bounded(scores):
    risk, nonben, _ = scores
    for name, fn in FUSION_RULES.items():
        out = fn(risk, nonben)
        assert out.shape == risk.shape, name
        assert np.all(out >= -1e-9) and np.all(out <= 1 + 1e-9), name


def test_rules_are_order_preserving_in_each_input():
    """A fused score must not fall when one head's score rises: a rule that
    inverts would fuse the two heads into something neither of them said."""
    a = np.array([0.1, 0.5, 0.9])
    for name, fn in FUSION_RULES.items():
        lo = fn(a, np.full(3, 0.2))
        hi = fn(a, np.full(3, 0.8))
        assert np.all(hi >= lo - 1e-9), f"{name} not monotone in the stage score"


def test_single_head_row_reproduces_its_own_head(scores):
    from sklearn.metrics import average_precision_score
    risk, nonben, y = scores
    rows = {r.rule: r for r in score_rules(risk, nonben, y, "composite")}
    assert rows[BASELINE_RULE].ap == pytest.approx(average_precision_score(y, risk))
    assert rows["stage 1-P(benign) alone"].ap == pytest.approx(average_precision_score(y, nonben))


def test_delta_is_against_the_published_head(scores):
    rows = score_rules(*scores, "composite")
    base = next(r for r in rows if r.rule == BASELINE_RULE)
    assert base.delta_ap(base) == 0.0
    other = next(r for r in rows if r.rule == "mean")
    assert other.delta_ap(base) == pytest.approx(other.ap - base.ap)


def test_a_single_head_alone_is_not_counted_as_a_fusion_win():
    """If the stage head alone happened to beat the risk head, that is a
    different finding — it is not evidence that fusing them works."""
    rows = [FusionRow(BASELINE_RULE, "composite", 40, ap=0.50, roc=0.9),
            FusionRow("stage 1-P(benign) alone", "composite", 40, ap=0.80, roc=0.9),
            FusionRow("max", "composite", 40, ap=0.40, roc=0.8),
            FusionRow("mean", "composite", 40, ap=0.45, roc=0.8),
            FusionRow("noisy-or", "composite", 40, ap=0.41, roc=0.8),
            FusionRow("geometric mean", "composite", 40, ap=0.39, roc=0.8)]
    assert any_rule_wins(rows) is False


def test_a_real_fusion_win_is_detected():
    rows = [FusionRow(BASELINE_RULE, "composite", 40, ap=0.50, roc=0.9),
            FusionRow("max", "composite", 40, ap=0.55, roc=0.91)]
    assert any_rule_wins(rows) is True


def test_missing_baseline_raises():
    with pytest.raises(ValueError, match="no baseline row"):
        any_rule_wins([FusionRow("max", "composite", 40, ap=0.55, roc=0.91)])


def test_one_class_raises(scores):
    risk, nonben, y = scores
    with pytest.raises(ValueError, match="both classes"):
        score_rules(risk, nonben, np.zeros_like(y), "composite")


def test_shape_mismatch_raises(scores):
    risk, nonben, y = scores
    with pytest.raises(ValueError, match="shape mismatch"):
        score_rules(risk[:-1], nonben, y, "composite")


def test_markdown_states_the_verdict_and_the_deltas(scores):
    rows = score_rules(*scores, "composite risk label")
    md = fusion_markdown(rows, "val")
    assert "Frozen-head fusion screen" in md
    assert "Screening only" in md
    assert ("no fusion rule beats" in md) or ("a fusion rule beats" in md)
    assert md.count("|") > 20


def test_markdown_per_stage_section_lists_every_rule(scores):
    risk, nonben, y = scores
    rows = score_rules(risk, nonben, y, "composite risk label")
    per_stage = score_rules(risk, nonben, y, "c2")
    md = fusion_markdown(rows, "val", per_stage=per_stage)
    assert "Per stage" in md
    for name in FUSION_RULES:
        assert name in md
