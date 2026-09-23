"""Comparing a re-run against its recorded artifact.

Twenty cross-evaluation cells were scored before `world_model_calibrated` was
bootstrapped (§3.31) and before several other changes to the evaluation code.
Re-running them into a separate directory buys two things at once: the missing
confidence intervals, and an answer to whether the published numbers still
reproduce under the current code. The second is worth more, and it is only
worth anything if the comparison is strict about what counts as agreement.
"""

from __future__ import annotations

import json

import pytest

from nidra.scripts.reproduction_check import compare_cell, drift_markdown


def cell(ap=0.1234, roc=0.87, prevalence=0.00417, n_rows=21173, with_ci=False):
    sysrow = {"auc_pr": ap, "roc_auc": roc}
    if with_ci:
        sysrow["auc_pr_bootstrap"] = {"ci_low": ap - 0.02, "ci_high": ap + 0.02, "n_positive_clusters": 9}
    return {"metrics": {"task_published_label": {
        "n_rows": n_rows, "prevalence_natural": prevalence,
        "systems": {"world_model_calibrated": sysrow}}}}


class TestCompareCell:
    def test_identical_numbers_reproduce(self):
        r = compare_cell(cell(), cell(with_ci=True))
        assert r["reproduces"] is True
        assert r["delta_ap"] == pytest.approx(0.0)

    def test_a_tiny_difference_still_reproduces(self):
        """Bootstrap resampling and the stochastic rollout are seeded, but a
        re-run is not required to be bit-identical to be the same result."""
        r = compare_cell(cell(ap=0.1234), cell(ap=0.1236, with_ci=True))
        assert r["reproduces"] is True

    def test_a_difference_past_the_tolerance_does_not(self):
        r = compare_cell(cell(ap=0.1234), cell(ap=0.1500, with_ci=True))
        assert r["reproduces"] is False
        assert r["delta_ap"] == pytest.approx(0.0266, abs=1e-4)

    def test_different_rows_is_a_different_evaluation_not_a_drifted_one(self):
        """Same AP on a different number of rows is not agreement — it means
        the two runs did not score the same thing, which is a worse finding
        than drift and must not be reported as reproducing."""
        r = compare_cell(cell(n_rows=21173), cell(n_rows=9173, with_ci=True))
        assert r["reproduces"] is False
        assert "rows" in r["reason"]

    def test_different_prevalence_is_also_a_different_evaluation(self):
        r = compare_cell(cell(prevalence=0.00417), cell(prevalence=0.00749, with_ci=True))
        assert r["reproduces"] is False
        assert "prevalence" in r["reason"]

    def test_it_reports_whether_the_rerun_gained_the_interval(self):
        assert compare_cell(cell(), cell(with_ci=True))["gained_ci"] is True
        assert compare_cell(cell(), cell(with_ci=False))["gained_ci"] is False

    def test_a_missing_rerun_is_reported_not_skipped(self):
        r = compare_cell(cell(), None)
        assert r["reproduces"] is None and "not re-run" in r["reason"]


class TestMarkdown:
    def test_a_drifted_cell_is_marked_and_counted(self):
        rows = [("a", "test", compare_cell(cell(ap=0.1), cell(ap=0.4, with_ci=True))),
                ("b", "test", compare_cell(cell(), cell(with_ci=True)))]
        md = drift_markdown(rows)
        assert "1 of 2" in md and "**no**" in md

    def test_all_reproducing_says_so_plainly(self):
        rows = [("a", "test", compare_cell(cell(), cell(with_ci=True)))]
        md = drift_markdown(rows)
        assert "all 1" in md.lower()


def test_nothing_re_run_yet_does_not_claim_agreement():
    """"All 0 re-run cells reproduce" is a positive claim made from no
    evidence — the exact shape of error this log keeps catching elsewhere."""
    md = drift_markdown([("a", "test", compare_cell(cell(), None))])
    assert "reproduce" not in md.split("\n")[0].lower()
    assert "no cell has been re-run yet" in md.lower()
