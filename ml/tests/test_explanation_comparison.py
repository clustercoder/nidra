"""Where in the history an explanation puts its mass."""

from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.explanation_comparison import comparison_markdown, summarise, temporal_profile


class TestTemporalProfile:
    def test_attribution_only_on_the_origin_window_is_all_recent_and_age_zero(self):
        attr = np.zeros((30, 45)); attr[-1] = 1.0
        p = temporal_profile(attr, 60)
        assert p["share_recent"] == pytest.approx(1.0)
        assert p["mean_age_min"] == pytest.approx(0.0)

    def test_attribution_on_the_oldest_window_is_the_full_age(self):
        attr = np.zeros((30, 45)); attr[0] = 1.0
        p = temporal_profile(attr, 60)
        assert p["share_recent"] == pytest.approx(0.0)
        assert p["mean_age_min"] == pytest.approx(29.0)

    def test_the_sign_of_an_attribution_does_not_move_its_mass(self):
        a = np.zeros((10, 3)); a[0] = -5.0
        b = np.zeros((10, 3)); b[0] = 5.0
        assert temporal_profile(a)["mean_age_min"] == temporal_profile(b)["mean_age_min"]

    def test_uniform_attribution_puts_mass_at_the_midpoint(self):
        p = temporal_profile(np.ones((30, 45)), 60)
        assert p["mean_age_min"] == pytest.approx(14.5)
        assert p["share_recent"] == pytest.approx(5 / 30)

    def test_the_window_shares_sum_to_one(self):
        p = temporal_profile(np.random.default_rng(0).normal(size=(30, 45)))
        assert sum(p["per_window_share"]) == pytest.approx(1.0)

    def test_an_all_zero_attribution_does_not_divide_by_zero(self):
        p = temporal_profile(np.zeros((30, 45)))
        assert np.isnan(p["share_recent"]) and p["n_windows"] == 30

    def test_the_window_length_scales_the_age(self):
        attr = np.zeros((10, 2)); attr[0] = 1.0
        assert temporal_profile(attr, 30)["mean_age_min"] == pytest.approx(4.5)


class TestSummary:
    def _rows(self):
        return [{"share_recent": 0.4, "mean_age_min": 8.0, "drop_top": 0.2, "drop_random_mean": 0.05,
                 "beats_random_fraction": 0.9, "forecast_score": 0.7},
                {"share_recent": 0.6, "mean_age_min": 6.0, "drop_top": 0.3, "drop_random_mean": 0.05,
                 "beats_random_fraction": 1.0, "forecast_score": 0.8}]

    def test_it_averages_each_statistic(self):
        s = summarise(self._rows())
        assert s["share_recent"]["mean"] == pytest.approx(0.5)
        assert s["share_recent"]["n"] == 2

    def test_a_statistic_that_is_nan_everywhere_is_reported_as_absent(self):
        rows = [{**r, "share_recent": float("nan")} for r in self._rows()]
        assert summarise(rows)["share_recent"] is None

    def test_the_table_has_one_row_per_head(self):
        md = comparison_markdown({"state": summarise(self._rows()), "hidden": summarise(self._rows())}, 40, "val")
        assert "| state |" in md and "| hidden |" in md and "val" in md
