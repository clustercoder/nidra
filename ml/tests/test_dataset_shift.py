"""Per-feature distribution distance between two datasets."""

from __future__ import annotations

import numpy as np
import pytest

from nidra.data.dataset_shift import _rank_auc, feature_shift, shift_report
from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER, regime_dropped_features


def _states(n: int, shift: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=shift, scale=1.0, size=(n, len(FEATURE_ORDER)))
    X[:, FEATURE_INDEX["is_active"]] = 1.0
    return X


class TestRankAuc:
    def test_identical_samples_score_a_half(self):
        x = np.arange(100.0)
        assert _rank_auc(x, x.copy()) == pytest.approx(0.5, abs=0.01)

    def test_disjoint_samples_score_one(self):
        assert _rank_auc(np.arange(50.0), np.arange(100.0, 150.0)) == pytest.approx(1.0)

    def test_a_constant_feature_in_both_is_not_separable(self):
        assert _rank_auc(np.zeros(50), np.zeros(80)) == pytest.approx(0.5)

    def test_direction_does_not_matter(self):
        a, b = np.arange(50.0), np.arange(25.0, 75.0)
        assert _rank_auc(a, b) == pytest.approx(_rank_auc(b, a))


class TestFeatureShift:
    def test_same_distribution_is_near_zero_distance(self):
        shifts = feature_shift(_states(3000, 0.0, 0), _states(3000, 0.0, 1))
        by = {s.feature: s for s in shifts}
        assert by["bytes_total"].wasserstein < 0.1
        assert by["bytes_total"].auc < 0.55

    def test_a_shifted_feature_is_detected(self):
        a, b = _states(3000, 0.0, 0), _states(3000, 0.0, 1)
        b[:, FEATURE_INDEX["syn_ratio"]] += 5.0
        shifts = {s.feature: s for s in feature_shift(a, b)}
        assert shifts["syn_ratio"].wasserstein > 1.5
        assert shifts["syn_ratio"].auc > 0.95
        assert shifts["ack_ratio"].auc < 0.6

    def test_only_active_rows_are_compared(self):
        a = _states(2000, 0.0, 0)
        b = _states(2000, 0.0, 1)
        silent = np.zeros((8000, len(FEATURE_ORDER)))
        b_with_silence = np.vstack([b, silent])
        with_silence = {s.feature: s for s in feature_shift(a, b_with_silence)}
        without = {s.feature: s for s in feature_shift(a, b)}
        # the silent rows are excluded, so the answer barely moves
        assert abs(with_silence["bytes_total"].auc - without["bytes_total"].auc) < 0.05

    def test_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="expected 45 features"):
            feature_shift(np.zeros((10, 44)), np.zeros((10, 44)))

    def test_report_separates_regime_kept_features(self):
        a, b = _states(1500, 0.0, 0), _states(1500, 0.0, 1)
        b[:, FEATURE_INDEX["ttl_mean"]] += 20.0     # a dropped feature
        report = shift_report(feature_shift(a, b), regime_dropped_features("cross_core"))
        assert report["n_kept_by_regime"] == 32
        assert "ttl_mean" not in [w["feature"] for w in report["worst_kept"]]
        assert report["kept_max_wasserstein"] < 1.0
