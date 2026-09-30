"""The PS 26153 benchmark — world model vs logistic regression on F1,
precision, recall and false-positive rate — with each system at ITS OWN
validation-chosen threshold.

`benchmark.py`'s `_task_table` scores every system at the world model's
operating-point threshold, which was selected for the world model's
calibrated score and for nothing else. A baseline whose scores live on a
different scale is then read at a threshold nobody chose for it, and its
false-alarm rate says more about that mismatch than about the baseline.
These tests pin the recomputation that gives every system the same rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.ps_baseline_benchmark import (
    SYSTEMS,
    markdown_report,
    paired_ap_difference,
    score_at_threshold,
    select_thresholds,
)


def _split(n: int = 2000, seed: int = 0, prevalence: float = 0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    base = y * 0.6 + rng.random(n) * 0.5
    scores = {
        "world_model_calibrated": base,
        # the same ranking on a squashed scale — a fair rule must treat it identically
        "lr_current_state": 0.001 + base * 0.01,
        "lr_flattened_history": y * 0.3 + rng.random(n) * 0.7,
    }
    return y, scores, np.ones(n), np.array([f"c{i // 20}" for i in range(n)])


class TestThresholdSelection:
    def test_each_system_gets_its_own_threshold(self):
        y, s, w, _ = _split()
        thr = select_thresholds(y, s, w, list(s))
        assert thr["world_model_calibrated"][0] != pytest.approx(thr["lr_current_state"][0])

    def test_thresholds_depend_on_validation_only(self):
        yv, sv, wv, _ = _split(seed=1)
        before = select_thresholds(yv, sv, wv, list(sv))
        # nothing about any other split can reach the selection: it takes no other split
        again = select_thresholds(yv, sv, wv, list(sv))
        assert before == again

    def test_a_monotone_rescaling_scores_identically_under_its_own_threshold(self):
        """The flaw being corrected: at a SHARED threshold the squashed copy
        would alert on nothing; at its own it must match the original exactly."""
        yv, sv, wv, _ = _split(seed=2)
        yt, st, wt, _ = _split(seed=3)
        thr = select_thresholds(yv, sv, wv, ["world_model_calibrated", "lr_current_state"])
        a = score_at_threshold(yt, st["world_model_calibrated"], wt, thr["world_model_calibrated"][0], 8.0)
        b = score_at_threshold(yt, st["lr_current_state"], wt, thr["lr_current_state"][0], 8.0)
        for k in ("precision", "recall", "f1", "fpr", "false_alarms_per_hour"):
            assert a[k] == pytest.approx(b[k], abs=1e-9), k

    def test_a_shared_threshold_would_have_misread_the_rescaled_copy(self):
        yv, sv, wv, _ = _split(seed=2)
        yt, st, wt, _ = _split(seed=3)
        thr = select_thresholds(yv, sv, wv, ["world_model_calibrated"])["world_model_calibrated"][0]
        shared = score_at_threshold(yt, st["lr_current_state"], wt, thr, 8.0)
        assert shared["recall"] == 0.0  # the squashed scale never reaches the other system's threshold


class TestScoring:
    def test_false_alarms_per_hour_is_false_positives_over_span(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.9, 0.1, 0.8, 0.95, 0.2])
        r = score_at_threshold(y, s, np.ones(5), 0.5, span_hours=2.0)
        assert r["false_alarms_per_hour"] == pytest.approx(1.0)  # 2 FP over 2 h
        assert r["precision"] == pytest.approx(1 / 3)
        assert r["recall"] == pytest.approx(0.5)

    def test_ap_is_reported_and_threshold_free(self):
        y, s, w, _ = _split(seed=4)
        lo = score_at_threshold(y, s["world_model_calibrated"], w, 0.1, 8.0)["ap"]
        hi = score_at_threshold(y, s["world_model_calibrated"], w, 0.9, 8.0)["ap"]
        assert lo == pytest.approx(hi)

    def test_span_must_be_positive(self):
        with pytest.raises(ValueError):
            score_at_threshold(np.array([0, 1]), np.array([0.2, 0.9]), np.ones(2), 0.5, span_hours=0.0)


class TestPairedAP:
    def test_point_is_the_difference_of_the_two_aps(self):
        from nidra.eval.metrics_natural import weighted_ap

        y, s, w, c = _split(seed=5)
        d = paired_ap_difference(y, s["world_model_calibrated"], s["lr_flattened_history"], w, c, n_resamples=30)
        expected = weighted_ap(y, s["world_model_calibrated"], w) - weighted_ap(y, s["lr_flattened_history"], w)
        assert d["point"] == pytest.approx(expected, abs=1e-12)
        assert d["ci_low"] <= d["point"] <= d["ci_high"]


class TestReport:
    def test_the_report_states_the_rule_and_names_both_lr_baselines(self):
        rows = {
            "test": {
                s: {"threshold": 0.5, "val_f1": 0.5, "ap": 0.1, "precision": 0.5, "recall": 0.1,
                    "f1": 0.17, "fpr": 1e-4, "false_alarms_per_hour": 1.0}
                for s in SYSTEMS
            }
        }
        paired = {"test": {s: {"point": 0.01, "ci_low": -0.01, "ci_high": 0.03} for s in SYSTEMS[1:]}}
        md = markdown_report(rows, paired, {"test": 8.0})
        assert "own validation" in md
        assert "lr_current_state" in md and "lr_flattened_history" in md
        assert md.count("| test |") == len(SYSTEMS)
