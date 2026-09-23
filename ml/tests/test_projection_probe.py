"""§3.41's pre-registered criterion, and the statistic it is computed from."""

import numpy as np
import pytest

from nidra.scripts.projection_probe import (
    MIN_CELLS_PER_GROUP, cloud_distance, non_degenerate, verdict,
)


class TestCloudDistance:
    def test_a_point_at_the_centre_is_at_zero_distance(self):
        ref = np.array([[0.0, 0.0], [2.0, 2.0]])
        assert cloud_distance(ref, np.array([[1.0, 1.0]])) == pytest.approx(0.0)

    def test_distance_is_in_units_of_the_clouds_own_spread(self):
        """Otherwise a wide feature dominates a narrow one for no reason."""
        # feature 0: mean 1, sd 1.  feature 1: mean 50, sd 50.
        ref = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 100.0], [2.0, 100.0]])
        narrow = np.array([[2.0, 50.0]])    # 1 sd out on the narrow feature
        wide = np.array([[1.0, 100.0]])     # 1 sd out on the wide feature
        assert cloud_distance(ref, narrow) == pytest.approx(cloud_distance(ref, wide), rel=1e-6)

    def test_degenerate_features_are_excluded_not_floored(self):
        """A constant training feature has no scale, so a deviation in it has no
        meaning in sd units. Flooring the sd produced distances of 2e6 on the
        first attempt and is what led to D145."""
        ref = np.hstack([np.random.default_rng(0).standard_normal((50, 2)), np.zeros((50, 1))])
        # The only difference between these is a large value in the constant
        # column. If it were floored rather than excluded it would dominate.
        quiet = cloud_distance(ref, np.array([[0.3, -0.2, 0.0]]))
        loud = cloud_distance(ref, np.array([[0.3, -0.2, 5.0]]))
        assert loud == pytest.approx(quiet, rel=1e-9)

    def test_non_degenerate_reports_what_it_kept(self):
        ref = np.hstack([np.random.default_rng(0).standard_normal((50, 2)), np.zeros((50, 3))])
        keep = non_degenerate(ref)
        assert keep.sum() == 2 and len(keep) == 5


class TestVerdict:
    @staticmethod
    def _cells(beating_ratios, losing_ratios):
        return ([{"beats_oracle": True, "ratio": r} for r in beating_ratios]
                + [{"beats_oracle": False, "ratio": r} for r in losing_ratios])

    def test_underpowered_below_the_minimum_per_group(self):
        v = verdict(self._cells([0.5, 0.6], [1.2, 1.3, 1.4]))
        assert v["verdict"] == "underpowered"
        assert str(MIN_CELLS_PER_GROUP) in v["reason"]

    def test_supported_when_the_groups_separate_at_one(self):
        v = verdict(self._cells([0.5, 0.6, 0.7], [1.2, 1.3, 1.4]))
        assert v["verdict"] == "supported"

    def test_one_oracle_beating_cell_at_or_above_one_refutes_it(self):
        v = verdict(self._cells([0.5, 0.6, 1.01], [1.2, 1.3, 1.4]))
        assert v["verdict"] == "refuted"
        assert "beats its oracle" in v["reason"]

    def test_one_oracle_winning_cell_below_one_refutes_it(self):
        v = verdict(self._cells([0.5, 0.6, 0.7], [1.2, 0.99, 1.4]))
        assert v["verdict"] == "refuted"

    def test_exactly_one_is_on_the_oracle_winning_side(self):
        """The criterion says 'at or above 1' for oracle-winning cells, so a
        ratio of exactly 1.0 there is consistent, not a refutation."""
        assert verdict(self._cells([0.5, 0.6, 0.7], [1.0, 1.2, 1.3]))["verdict"] == "supported"

    def test_exactly_one_refutes_on_the_oracle_beating_side(self):
        assert verdict(self._cells([0.5, 0.6, 1.0], [1.2, 1.3, 1.4]))["verdict"] == "refuted"

    def test_the_verdict_names_the_cells_that_decided_it(self):
        cells = self._cells([0.5, 0.6, 1.4], [1.2, 1.3, 1.4])
        cells[2]["cell"] = "comb2cic/test"
        assert "comb2cic/test" in verdict(cells)["reason"]


class TestThresholdRobustness:
    """§3.41 said to exclude features with degenerate training variance,
    "the degenerate ones are what produced distances of 2e6 on the first
    attempt". The code implemented that as sd > 1e-6, which does NOT implement
    it: a feature with training sd 5.3e-5 passed the threshold and contributed
    a mean z^2 of 1,009,400, giving a distance of 806 against a truth distance
    of 4.9.

    Worse, the kept-feature SET moved with the training sample size — 31
    features at cap 1500, 32 at cap 2500 — so the statistic was not stable in a
    parameter that is supposed to be irrelevant.

    The fix is not to pick a better number. It is to report the verdict across a
    range of thresholds, so the conclusion either does not depend on the choice
    or is declared not to survive it.
    """

    def test_a_near_constant_feature_cannot_dominate_at_the_default(self):
        import numpy as np
        rng = np.random.default_rng(0)
        ref = np.hstack([rng.standard_normal((200, 3)), rng.standard_normal((200, 1)) * 5.3e-5])
        probe = np.hstack([np.zeros((1, 3)), np.array([[1.0]])])   # 19000 sd out on the flat one
        assert cloud_distance(ref, probe) < 10.0

    def test_the_kept_set_is_stable_across_sample_size_at_the_default(self):
        import numpy as np
        rng = np.random.default_rng(0)
        big = np.hstack([rng.standard_normal((5000, 3)), rng.standard_normal((5000, 1)) * 5.3e-5])
        assert non_degenerate(big[:500]).sum() == non_degenerate(big).sum() == 3

    def test_sweep_reports_the_verdict_at_every_threshold(self):
        from nidra.scripts.projection_probe import THRESHOLD_SWEEP, sweep_verdicts
        cells = [{"cell": f"c{i}", "beats_oracle": i < 3,
                  "ratios": {t: (0.5 if i < 3 else 1.5) for t in THRESHOLD_SWEEP}} for i in range(6)]
        got = sweep_verdicts(cells)
        assert set(got) == set(THRESHOLD_SWEEP)
        assert all(v["verdict"] == "supported" for v in got.values())

    def test_a_verdict_that_depends_on_the_threshold_is_not_a_verdict(self):
        from nidra.scripts.projection_probe import THRESHOLD_SWEEP, sweep_summary
        lo, hi = THRESHOLD_SWEEP[0], THRESHOLD_SWEEP[-1]
        cells = [{"cell": f"c{i}", "beats_oracle": i < 3,
                  "ratios": {t: (0.5 if i < 3 else 1.5) for t in THRESHOLD_SWEEP}} for i in range(6)]
        cells[0]["ratios"][hi] = 1.4          # flips the verdict at one threshold only
        s = sweep_summary(cells)
        assert s["stable"] is False
        assert "does not survive" in s["summary"]

    def test_a_verdict_stable_across_the_sweep_is_reported_as_one(self):
        from nidra.scripts.projection_probe import THRESHOLD_SWEEP, sweep_summary
        cells = [{"cell": f"c{i}", "beats_oracle": i < 3,
                  "ratios": {t: (0.5 if i < 3 else 1.5) for t in THRESHOLD_SWEEP}} for i in range(6)]
        s = sweep_summary(cells)
        assert s["stable"] is True and s["verdict"] == "supported"
