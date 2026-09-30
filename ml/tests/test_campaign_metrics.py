"""Stage-transition lead time: flagged before the campaign's completing phase?

`per_episode_report` asks whether a host was warned before its FIRST
attack-labelled window, which CIC-IDS2017 cannot support (3 / 9 / 15 training
onsets within 1 / 3 / 5 min). The PS asks a narrower question — predict before
compromise completes — which a multi-phase campaign can answer: Thursday's
infiltration victim was exploited at 14:19 and scanned the LAN from 15:04.
These tests pin what counts as a campaign, which phase "completes" it, and
what counts as a flag before that phase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.eval.campaign_metrics import (
    EpisodeSpan,
    campaign_lead_report,
    completing_phase,
    episode_spans,
    group_campaigns,
)

W = 60
T0 = 1_000_000


def _ep(host: str, start_min: int, end_min: int, stage: str) -> EpisodeSpan:
    return EpisodeSpan(host=host, onset_ts=T0 + start_min * W, end_ts=T0 + end_min * W, stage=stage)


def _timeline(host: str, minutes: range, high: set[int], low: float = 0.01, high_score: float = 0.9):
    ts = np.array([T0 + m * W for m in minutes], dtype="int64")
    score = np.array([high_score if m in high else low for m in minutes], dtype="float64")
    return np.array([host] * len(ts)), ts, score


class TestGroupCampaigns:
    def test_close_episodes_on_one_host_chain(self):
        eps = [_ep("a", 0, 5, "lateral"), _ep("a", 25, 40, "lateral")]
        assert [len(c) for c in group_campaigns(eps, max_gap_minutes=60)] == [2]

    def test_a_long_gap_splits_the_campaign(self):
        eps = [_ep("a", 0, 5, "lateral"), _ep("a", 90, 100, "lateral")]
        assert [len(c) for c in group_campaigns(eps, max_gap_minutes=60)] == [1, 1]

    def test_hosts_never_share_a_campaign(self):
        eps = [_ep("a", 0, 5, "recon"), _ep("b", 10, 20, "exfil")]
        campaigns = group_campaigns(eps, max_gap_minutes=60)
        assert len(campaigns) == 2
        assert all(len({e.host for e in c}) == 1 for c in campaigns)

    def test_input_order_does_not_matter(self):
        eps = [_ep("a", 25, 40, "lateral"), _ep("a", 0, 5, "initial_access")]
        (campaign,) = group_campaigns(eps, max_gap_minutes=60)
        assert [e.onset_ts for e in campaign] == [T0, T0 + 25 * W]


class TestCompletingPhase:
    def test_most_advanced_stage_completes(self):
        c = [_ep("a", 0, 5, "initial_access"), _ep("a", 20, 30, "lateral")]
        assert completing_phase(c).stage == "lateral"

    def test_first_episode_of_the_most_advanced_stage(self):
        c = [_ep("a", 0, 5, "recon"), _ep("a", 20, 30, "exfil"), _ep("a", 40, 45, "recon"), _ep("a", 50, 55, "exfil")]
        assert completing_phase(c).onset_ts == T0 + 20 * W

    def test_single_stage_campaign_completes_at_its_last_episode(self):
        c = [_ep("a", 0, 5, "lateral"), _ep("a", 9, 20, "lateral"), _ep("a", 45, 85, "lateral")]
        assert completing_phase(c).onset_ts == T0 + 45 * W

    def test_rejects_benign_episodes(self):
        with pytest.raises(ValueError):
            completing_phase([_ep("a", 0, 5, "benign"), _ep("a", 9, 20, "lateral")])


class TestCampaignLeadReport:
    def _report(self, high: set[int], minutes=range(-30, 90), m: int = 2, eps=None):
        eps = eps or [_ep("a", 0, 5, "lateral"), _ep("a", 45, 85, "lateral")]
        host, ts, score = _timeline("a", minutes, high)
        return campaign_lead_report(eps, host, ts, score, threshold=0.5, window_seconds=W, m=m,
                                    max_gap_minutes=60, pre_onset_minutes=30)

    def test_flag_during_an_earlier_phase_is_a_lead(self):
        rep = self._report(high={10, 11})
        (c,) = rep["campaigns"]
        assert c["flagged_before_completion"] is True
        assert c["stage_transition_lead_s"] == (45 - 10) * W
        assert c["flag_before_campaign_onset"] is False
        assert rep["n_flagged_before_completion"] == 1

    def test_flag_before_the_campaign_starts_counts_and_is_marked(self):
        rep = self._report(high={-5, -4})
        (c,) = rep["campaigns"]
        assert c["flagged_before_completion"] is True
        assert c["flag_before_campaign_onset"] is True
        assert c["stage_transition_lead_s"] == (45 + 5) * W

    def test_flag_only_inside_the_completing_phase_is_not_a_lead(self):
        rep = self._report(high={50, 51})
        (c,) = rep["campaigns"]
        assert c["flagged_before_completion"] is False
        assert c["stage_transition_lead_s"] is None
        assert c["completing_phase_detected"] is True
        assert c["completing_phase_latency_min"] == 5.0

    def test_rows_missing_between_hits_break_the_run(self):
        minutes = [m for m in range(-30, 90) if m != 11]
        rep = self._report(high={10, 12}, minutes=minutes)
        (c,) = rep["campaigns"]
        assert c["flagged_before_completion"] is False

    def test_coverage_counts_the_rows_actually_scored(self):
        minutes = [m for m in range(-30, 90) if not 20 <= m < 30]
        rep = self._report(high=set(), minutes=minutes)
        (c,) = rep["campaigns"]
        assert c["expected_windows_before_completion"] == 75
        assert c["scored_windows_before_completion"] == 65

    def test_single_episode_campaigns_are_counted_not_scored(self):
        eps = [_ep("a", 0, 5, "lateral"), _ep("a", 45, 85, "lateral"), _ep("b", 0, 10, "c2")]
        rep = self._report(high=set(), eps=eps)
        assert rep["n_campaigns_scored"] == 1
        assert rep["n_single_episode_campaigns"] == 1

    def test_another_hosts_scores_never_count(self):
        eps = [_ep("a", 0, 5, "lateral"), _ep("a", 45, 85, "lateral")]
        ha, ta, sa = _timeline("a", range(-30, 90), high=set())
        hb, tb, sb = _timeline("b", range(-30, 90), high={10, 11})
        rep = campaign_lead_report(eps, np.concatenate([ha, hb]), np.concatenate([ta, tb]),
                                   np.concatenate([sa, sb]), threshold=0.5, window_seconds=W)
        assert rep["n_flagged_before_completion"] == 0

    def test_max_score_before_completion_is_threshold_free(self):
        rep = self._report(high={20})
        (c,) = rep["campaigns"]
        assert c["flagged_before_completion"] is False
        assert c["max_score_before_completion"] == pytest.approx(0.9)

    def test_rejects_misaligned_arrays(self):
        with pytest.raises(ValueError):
            campaign_lead_report([_ep("a", 0, 5, "lateral")], np.array(["a"]), np.array([T0, T0 + W]),
                                 np.array([0.1]), threshold=0.5, window_seconds=W)


class TestEpisodeSpans:
    def test_merges_short_gaps_and_takes_the_most_advanced_stage(self):
        stages = ["benign", "initial_access", "initial_access", "benign", "lateral", "benign"] + ["benign"] * 10 + ["c2"]
        df = pd.DataFrame({"host_id": "a", "window_ts": [T0 + i * W for i in range(len(stages))],
                           "stage_label": stages})
        spans = episode_spans(df, window_seconds=W, merge_gap_windows=2)
        assert [(s.onset_ts - T0) // W for s in spans] == [1, 16]
        assert [s.stage for s in spans] == ["lateral", "c2"]


class TestSplitReport:
    def _dump(self):
        host, ts, wm = _timeline("a", range(-30, 90), high={10, 11})
        _, _, pers = _timeline("a", range(-30, 90), high=set())
        return {"host_id": host, "origin_ts": ts, "world_model_calibrated": wm, "persistence": pers,
                "y_published": np.zeros(len(ts), dtype=int), "risk_k_raw": np.zeros((len(ts), 6))}

    def test_published_arm_at_threshold_others_threshold_free(self):
        from nidra.scripts.campaign_lead_time import split_report

        eps = [_ep("a", 0, 5, "lateral"), _ep("a", 45, 85, "lateral")]
        rep = split_report(eps, self._dump(), threshold=0.5, window_seconds=W, max_gap_minutes=60,
                           pre_onset_minutes=30)
        assert rep["system"] == "world_model_calibrated"
        assert rep["n_flagged_before_completion"] == 1
        assert rep["max_score_before_completion_by_system"]["persistence"] == [pytest.approx(0.01)]
        assert "y_published" not in rep["max_score_before_completion_by_system"]
        assert "risk_k_raw" not in rep["max_score_before_completion_by_system"]

    def test_missing_published_arm_fails_loudly(self):
        from nidra.scripts.campaign_lead_time import split_report

        dump = self._dump()
        del dump["world_model_calibrated"]
        with pytest.raises(KeyError):
            split_report([], dump, threshold=0.5, window_seconds=W, max_gap_minutes=60, pre_onset_minutes=30)

    def test_markdown_states_what_a_flag_is_not(self):
        from nidra.scripts.campaign_lead_time import markdown, split_report

        eps = [_ep("a", 0, 5, "lateral"), _ep("a", 45, 85, "lateral")]
        rep = split_report(eps, self._dump(), threshold=0.5, window_seconds=W, max_gap_minutes=60,
                           pre_onset_minutes=30)
        text = markdown({"holdout": rep})
        assert "not a forecast that names the completing phase" in text
        assert "1 of 1 multi-phase campaigns flagged before completion" in text
        assert "| 35 |" in text
