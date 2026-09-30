"""Paired margin intervals for the PUBLISHED arm (§36 item 21, Q12).

`benchmark.py`'s `attribution_detection` block pairs `world_model` — the
uncalibrated sampled arm — against the baselines. The number the project
publishes is `world_model_calibrated`, which has a marginal interval but no
paired one, so "is the improvement statistically supported?" could not be
answered about the system actually reported. These tests pin the offline
recomputation that closes that, from the per-row score dumps the benchmark
already writes.
"""
from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.margin_intervals import (
    BASELINE_SYSTEMS,
    PUBLISHED_SYSTEM,
    CellMargin,
    best_baseline,
    margin_for_cell,
    markdown_table,
    summarise,
    verdict,
)


def _cell(n: int = 400, seed: int = 0):
    """A cell where `world_model_calibrated` is genuinely better than every
    baseline, with clusters big enough for a cluster bootstrap to say so."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    clusters = np.array([f"c{i // 10}" for i in range(n)])
    w = np.ones(n)
    scores = {
        PUBLISHED_SYSTEM: y * 0.8 + rng.random(n) * 0.2,
        "persistence": y * 0.3 + rng.random(n) * 0.7,
        "noised_persistence": y * 0.2 + rng.random(n) * 0.8,
        "gbdt_current_state": y * 0.1 + rng.random(n) * 0.9,
        "oracle_true_future": y.astype(float),
        "world_model": y * 0.5 + rng.random(n) * 0.5,
    }
    return y, scores, w, clusters


class TestBestBaseline:
    def test_picks_the_highest_ap_baseline(self):
        y, scores, w, _ = _cell()
        assert best_baseline(y, scores, w) == "persistence"

    def test_the_oracle_is_not_a_baseline(self):
        y, scores, w, _ = _cell()
        # the oracle scores perfectly and must still never be selected
        assert "oracle_true_future" not in BASELINE_SYSTEMS
        assert best_baseline(y, scores, w) != "oracle_true_future"

    def test_neither_arm_of_the_model_is_a_baseline(self):
        for name in (PUBLISHED_SYSTEM, "world_model", "world_model_deterministic"):
            assert name not in BASELINE_SYSTEMS

    def test_none_when_the_cell_carries_no_baseline(self):
        y, scores, w, _ = _cell()
        only_model = {PUBLISHED_SYSTEM: scores[PUBLISHED_SYSTEM]}
        assert best_baseline(y, only_model, w) is None


class TestMarginForCell:
    def test_point_is_exactly_the_difference_of_the_two_aps(self):
        from nidra.eval.metrics_natural import weighted_ap

        y, scores, w, clusters = _cell()
        m = margin_for_cell("demo", "test", y, scores, w, clusters, n_resamples=40)
        assert m is not None
        expected = weighted_ap(y, scores[PUBLISHED_SYSTEM], w) - weighted_ap(y, scores[m.baseline], w)
        assert m.point == pytest.approx(expected, abs=1e-12)

    def test_the_interval_is_reproducible_under_a_fixed_seed(self):
        y, scores, w, clusters = _cell()
        a = margin_for_cell("demo", "test", y, scores, w, clusters, n_resamples=40, seed=7)
        b = margin_for_cell("demo", "test", y, scores, w, clusters, n_resamples=40, seed=7)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_a_different_seed_is_allowed_to_differ(self):
        y, scores, w, clusters = _cell()
        a = margin_for_cell("demo", "test", y, scores, w, clusters, n_resamples=40, seed=1)
        b = margin_for_cell("demo", "test", y, scores, w, clusters, n_resamples=40, seed=2)
        assert a.point == pytest.approx(b.point)  # the point estimate does not resample

    def test_a_clearly_better_system_lands_above_zero(self):
        y, scores, w, clusters = _cell()
        m = margin_for_cell("demo", "test", y, scores, w, clusters, n_resamples=200)
        assert m.ci_low > 0 and m.verdict == "above"

    def test_skipped_when_the_split_has_no_positive(self):
        y, scores, w, clusters = _cell()
        assert margin_for_cell("demo", "test", np.zeros_like(y), scores, w, clusters, n_resamples=20) is None

    def test_skipped_when_the_published_arm_is_absent(self):
        y, scores, w, clusters = _cell()
        without = {k: v for k, v in scores.items() if k != PUBLISHED_SYSTEM}
        assert margin_for_cell("demo", "test", y, without, w, clusters, n_resamples=20) is None

    def test_skipped_when_no_baseline_is_present(self):
        y, scores, w, clusters = _cell()
        only = {PUBLISHED_SYSTEM: scores[PUBLISHED_SYSTEM]}
        assert margin_for_cell("demo", "test", y, only, w, clusters, n_resamples=20) is None


class TestVerdict:
    def test_above_below_and_spanning(self):
        assert verdict(0.01, 0.20) == "above"
        assert verdict(-0.20, -0.01) == "below"
        assert verdict(-0.05, 0.05) == "spans"

    def test_an_endpoint_exactly_on_zero_does_not_count_as_support(self):
        assert verdict(0.0, 0.20) == "spans"
        assert verdict(-0.20, 0.0) == "spans"


class TestReport:
    def _margins(self):
        return [
            CellMargin("cic2cic_state+hidden", "test", 0.1178, "noised_persistence", 0.0945,
                       0.0233, 0.0005, 0.0890, 559, "above"),
            CellMargin("ctu2ctu_state", "holdout", 0.2472, "noised_persistence", 0.2313,
                       0.0159, -0.0011, 0.0451, 210, "spans"),
        ]

    def test_summarise_counts_only_intervals_that_exclude_zero(self):
        s = summarise(self._margins())
        assert s["n_cells"] == 2
        assert s["above"] == 1
        assert s["below"] == 0
        assert s["spans"] == 1

    def test_summarise_separates_validation_from_the_held_out_splits(self):
        margins = self._margins() + [
            CellMargin("cic2cic_state+hidden", "val", 0.99, "persistence", 0.98,
                       0.01, 0.001, 0.4, 300, "above"),
        ]
        s = summarise(margins)
        assert s["above"] == 2
        assert s["above_excluding_val"] == 1

    def test_markdown_has_one_row_per_cell_and_states_the_count(self):
        md = markdown_table(self._margins())
        assert md.count("| cic2cic_state+hidden | test |") == 1
        assert md.count("| ctu2ctu_state | holdout |") == 1
        assert "1 of 2" in md

    def test_markdown_marks_the_supported_cell_and_not_the_other(self):
        md = markdown_table(self._margins())
        supported = [ln for ln in md.splitlines() if "cic2cic_state+hidden" in ln][0]
        unsupported = [ln for ln in md.splitlines() if "ctu2ctu_state" in ln][0]
        assert "**" in supported
        assert "**" not in unsupported

    def test_markdown_names_the_arm_it_scored(self):
        assert PUBLISHED_SYSTEM in markdown_table(self._margins())


class TestKnifeEdgeDisclosure:
    """Three of the real cells clear zero by ~1e-05. At 300 resamples the
    percentile endpoint is an order statistic, so that is not a margin — but
    the verdict rule was fixed before the data was seen and is not rewritten
    here. It is disclosed instead.
    """

    def test_a_tiny_endpoint_is_not_printed_as_zero(self):
        from nidra.scripts.margin_intervals import fmt_endpoint

        assert fmt_endpoint(1.402550626178812e-05) not in ("+0.0000", "0.0000")
        assert fmt_endpoint(-1.6091501319931147e-05) not in ("-0.0000", "0.0000")

    def test_an_exact_zero_endpoint_still_prints_as_zero(self):
        from nidra.scripts.margin_intervals import fmt_endpoint

        assert fmt_endpoint(0.0) == "+0.0000"

    def test_an_ordinary_endpoint_keeps_four_decimals(self):
        from nidra.scripts.margin_intervals import fmt_endpoint

        assert fmt_endpoint(0.08120306604367737) == "+0.0812"

    def test_a_cell_that_barely_clears_zero_is_flagged(self):
        from nidra.scripts.margin_intervals import KNIFE_EDGE, clearance, is_knife_edge

        knife = CellMargin("cic2ctu_state", "holdout", 0.0370, "noised_persistence", 0.0340,
                           0.0030, 1.4e-05, 0.0152, 365, "above")
        clear = CellMargin("comb2cic_state+hidden", "test", 0.1654, "noised_persistence", 0.0994,
                           0.0660, 0.0235, 0.1296, 1742, "above")
        assert clearance(knife) < KNIFE_EDGE and is_knife_edge(knife)
        assert clearance(clear) > KNIFE_EDGE and not is_knife_edge(clear)

    def test_a_cell_whose_interval_spans_zero_is_never_a_knife_edge(self):
        from nidra.scripts.margin_intervals import is_knife_edge

        spanning = CellMargin("x", "test", 0.1, "persistence", 0.1, 0.0, -0.05, 0.05, 100, "spans")
        assert not is_knife_edge(spanning)

    def test_the_markdown_says_how_many_of_the_counted_cells_are_knife_edges(self):
        from nidra.scripts.margin_intervals import markdown_table

        margins = [
            CellMargin("cic2ctu_state", "holdout", 0.0370, "noised_persistence", 0.0340,
                       0.0030, 1.4e-05, 0.0152, 365, "above"),
            CellMargin("comb2cic_state+hidden", "test", 0.1654, "noised_persistence", 0.0994,
                       0.0660, 0.0235, 0.1296, 1742, "above"),
        ]
        md = markdown_table(margins)
        assert "knife edge" in md
        assert md.count("knife edge") >= 2  # the caveat sentence and the flagged row
