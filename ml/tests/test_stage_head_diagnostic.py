"""Per-stage one-vs-rest scoring of the frozen heads."""

from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.stage_head_diagnostic import diagnostic_markdown, one_vs_rest


class TestOneVsRest:
    def test_a_perfect_ranking_scores_one(self):
        labels = np.array(["benign"] * 90 + ["recon"] * 10)
        r = one_vs_rest(np.concatenate([np.zeros(90), np.ones(10)]), labels, "recon")
        assert r["auc_pr"] == pytest.approx(1.0) and r["roc_auc"] == pytest.approx(1.0)

    def test_lift_is_against_that_stage_s_own_base_rate(self):
        labels = np.array(["benign"] * 99 + ["recon"])
        r = one_vs_rest(np.concatenate([np.zeros(99), np.ones(1)]), labels, "recon")
        assert r["base_rate"] == pytest.approx(0.01) and r["lift"] == pytest.approx(100.0)

    def test_other_attack_stages_count_as_negatives(self):
        """One-vs-rest, not attack-vs-benign — the question is whether the
        head separates THIS stage from everything else it sees."""
        labels = np.array(["benign"] * 50 + ["c2"] * 40 + ["recon"] * 10)
        r = one_vs_rest(np.zeros(100), labels, "recon")
        assert r["base_rate"] == pytest.approx(0.10)

    def test_a_stage_with_no_windows_claims_nothing(self):
        r = one_vs_rest(np.zeros(10), np.array(["benign"] * 10), "recon")
        assert r["n"] == 0 and np.isnan(r["auc_pr"]) and np.isnan(r["roc_auc"])

    def test_a_stage_that_is_every_row_claims_nothing(self):
        r = one_vs_rest(np.zeros(10), np.array(["recon"] * 10), "recon")
        assert np.isnan(r["roc_auc"])


class TestMarkdown:
    def _row(self, **kw):
        base = dict(stage="recon", n=17, base_rate=0.0002, auc_pr=0.5, roc_auc=0.97,
                    lift=2500.0, risk_auc_pr=0.001, risk_roc_auc=0.42)
        return {**base, **kw}

    def test_both_heads_appear_for_a_stage(self):
        md = diagnostic_markdown([self._row()], "val")
        assert "0.970" in md and "0.420" in md

    def test_a_nan_prints_a_dash(self):
        md = diagnostic_markdown([self._row(auc_pr=float("nan"), lift=float("nan"))], "val")
        assert "| — |" in md

    def test_the_split_is_named(self):
        assert "val" in diagnostic_markdown([self._row()], "val")
