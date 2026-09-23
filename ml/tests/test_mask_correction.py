"""The D145 correction report: what the fix did to each recorded cell."""

import json

import pytest

from nidra.scripts.mask_correction import (
    RECONSIDER_DELTA, compare_cell, correction_markdown,
)


def _bench(ap_cal, ap_raw, oracle, persist, roc=0.8, n_rows=100, prev=0.01):
    return {"metrics": {"task_published_label": {
        "n_rows": n_rows, "n_pos": 1, "prevalence_natural": prev,
        "systems": {
            "world_model_calibrated": {"auc_pr": ap_cal, "roc_auc": roc},
            "world_model": {"auc_pr": ap_raw, "roc_auc": roc},
            "oracle_true_future": {"auc_pr": oracle, "roc_auc": 0.5},
            "persistence": {"auc_pr": persist, "roc_auc": 0.5},
        }}}}


class TestCompareCell:
    def test_reports_the_delta_on_both_arms(self):
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.20, 0.25, 0.05, 0.04))
        assert r["delta_calibrated"] == pytest.approx(0.10)
        assert r["delta_raw"] == pytest.approx(0.15)

    def test_a_cell_not_yet_re_scored_is_not_an_unchanged_cell(self):
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04), None)
        assert r["after_calibrated"] is None
        assert r["reconsider"] is None
        assert "not re-scored" in r["note"]

    def test_the_oracle_and_persistence_must_not_move(self):
        """Their states never carried the phantom. If they moved, the re-score
        differs in something other than the fix and the delta means nothing."""
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.20, 0.25, 0.09, 0.04))
        assert r["reconsider"] is False
        assert "oracle moved" in r["note"]

    def test_different_rows_is_not_a_correction(self):
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04, n_rows=100),
                         _bench(0.20, 0.25, 0.05, 0.04, n_rows=90))
        assert r["reconsider"] is False
        assert "different rows" in r["note"]

    def test_a_small_move_does_not_warrant_reconsidering_the_cell(self):
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.1001, 0.1001, 0.05, 0.04))
        assert r["reconsider"] is False
        assert abs(r["delta_calibrated"]) < RECONSIDER_DELTA

    def test_a_large_move_flags_the_cell(self):
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.20, 0.20, 0.05, 0.04))
        assert r["reconsider"] is True

    def test_the_correction_is_signed_not_absolute(self):
        """A fix that helps some cells and hurts others is the finding; folding
        the sign away would hide it."""
        worse = compare_cell(_bench(0.30, 0.30, 0.05, 0.04), _bench(0.20, 0.20, 0.05, 0.04))
        assert worse["delta_calibrated"] < 0

    def test_tracks_whether_the_cell_beats_its_oracle_before_and_after(self):
        r = compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.02, 0.02, 0.05, 0.04))
        assert r["beat_oracle_before"] is True
        assert r["beat_oracle_after"] is False


class TestMarkdown:
    def test_no_rescored_cell_does_not_claim_agreement(self):
        rows = [("cic2ctu_state", "test", compare_cell(_bench(0.1, 0.1, 0.05, 0.04), None))]
        assert "No cell has been re-scored yet" in correction_markdown(rows)

    def test_reports_the_direction_split_rather_than_a_mean(self):
        rows = [
            ("a", "test", compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.20, 0.20, 0.05, 0.04))),
            ("b", "test", compare_cell(_bench(0.30, 0.30, 0.05, 0.04), _bench(0.20, 0.20, 0.05, 0.04))),
        ]
        md = correction_markdown(rows)
        assert "1 up, 1 down" in md

    def test_names_every_cell_whose_oracle_verdict_flipped(self):
        rows = [("a", "test", compare_cell(_bench(0.10, 0.10, 0.05, 0.04), _bench(0.02, 0.02, 0.05, 0.04)))]
        md = correction_markdown(rows)
        assert "oracle verdict changed" in md and "a" in md

    def test_a_moved_oracle_is_surfaced_not_buried(self):
        rows = [("a", "test", compare_cell(_bench(0.1, 0.1, 0.05, 0.04), _bench(0.2, 0.2, 0.09, 0.04)))]
        assert "oracle moved" in correction_markdown(rows)
