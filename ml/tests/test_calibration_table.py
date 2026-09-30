"""The calibration table (§36 item 19).

Two things it must get right, both of which produce a plausible wrong number
rather than an error if got wrong:

1. The evaluation is natural-prevalence *stratified*, so every bin carries a
   population weight. An ECE over the raw sample counts describes the sample,
   not the population, and the two differ by more than an order of magnitude
   here because the negatives are subsampled.
2. A Brier score near zero at a prevalence near zero is not evidence of
   calibration. The table carries the constant-base-rate Brier alongside it so
   the comparison is available rather than left to the reader.
"""

from __future__ import annotations

import pytest

from nidra.scripts.report_tables import calibration_table, _ece, _base_rate


def bins(*triples):
    return [{"bin_center": c, "n": n, "weighted_n": w,
             "observed_frequency_natural": f} for c, n, w, f in triples]


class TestECE:
    def test_a_perfectly_calibrated_set_of_bins_has_zero_error(self):
        b = bins((0.05, 100, 1000.0, 0.05), (0.95, 100, 1000.0, 0.95))
        assert _ece(b) == pytest.approx(0.0)

    def test_error_is_the_weighted_mean_absolute_gap(self):
        # gaps 0.05 and 0.15, equal weights -> 0.10
        b = bins((0.10, 10, 500.0, 0.05), (0.50, 10, 500.0, 0.65))
        assert _ece(b) == pytest.approx(0.10)

    def test_it_weights_by_population_not_by_sample_count(self):
        """The bin with 1 sampled row but 99% of the population weight has to
        dominate. Weighting by `n` would give 0.055; weighting by
        `weighted_n` gives 0.0955."""
        b = bins((0.10, 999, 10.0, 0.10),      # 0 gap, tiny weight, huge n
                 (0.10, 1, 990.0, 0.0))        # 0.10 gap, almost all the weight
        assert _ece(b) == pytest.approx(0.10 * 990.0 / 1000.0)
        assert _ece(b) != pytest.approx(0.10 * 1 / 1000)

    def test_an_empty_bin_carries_a_null_frequency_and_is_skipped(self):
        """The benchmark writes `null`, not 0.0, for a bin with no rows — it
        declines to invent a frequency. Multiplying that by a zero weight
        raises before the zero can apply, which is how this was found."""
        b = bins((0.05, 0, 0.0, None), (0.95, 10, 100.0, 0.95))
        assert _ece(b) == pytest.approx(0.0)
        assert _base_rate(b) == pytest.approx(0.95)

    def test_no_bins_at_all_is_not_an_error(self):
        assert _ece([]) != _ece([])   # nan


class TestBaseRate:
    def test_it_is_the_weighted_positive_fraction(self):
        b = bins((0.05, 10, 900.0, 0.0), (0.95, 10, 100.0, 1.0))
        assert _base_rate(b) == pytest.approx(0.10)

    def test_all_bins_empty_is_nan_rather_than_zero(self):
        assert _base_rate(bins((0.05, 0, 0.0, None))) != _base_rate(bins((0.05, 0, 0.0, None)))

    def test_no_bins_is_nan_rather_than_zero(self):
        """Zero would read as 'no positives', which is a claim. nan is not."""
        assert _base_rate([]) != _base_rate([])


class TestCalibrationTable:
    @pytest.fixture
    def metrics(self):
        return {"calibration_published_label": {
            "raw": {"brier_natural": 0.0633,
                    "reliability": bins((0.05, 3918, 41874.0, 0.0079),
                                        (0.15, 5432, 68663.0, 0.0022))},
            "calibrated": {"brier_natural": 0.0056,
                           "reliability": bins((0.05, 20745, 226130.0, 0.0038),
                                               (0.15, 69, 115.0, 0.1560))}}}

    def test_both_arms_appear_with_their_brier(self, metrics):
        out = calibration_table(metrics, "test")
        assert "raw" in out and "calibrated" in out
        assert "0.06330" in out and "0.00560" in out

    def test_the_constant_base_rate_reference_is_reported(self, metrics):
        """Without it, 0.0056 reads as excellent and may be worse than
        predicting the prevalence and never moving."""
        out = calibration_table(metrics, "test")
        assert "base rate" in out.lower()

    def test_occupied_bins_are_reported_so_concentration_is_visible(self, metrics):
        out = calibration_table(metrics, "test")
        assert "occupied bins" in out.lower()

    def test_a_run_without_calibration_says_so_instead_of_rendering_dashes(self):
        assert "not present" in calibration_table({}, "test").lower()
