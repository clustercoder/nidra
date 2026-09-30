"""Stage-transition lead time: was the host flagged before its campaign's
completing phase began?

`episode_metrics.per_episode_report` asks whether a host was warned before its
FIRST attack-labelled window. On CIC-IDS2017 no system can answer that — the
training split holds 3 / 9 / 15 onsets within 1 / 3 / 5 minutes. The PS asks a
narrower question, "predict before compromise completes", and a campaign with
several phases on one host can answer it: Thursday's infiltration victim was
exploited at 14:19 and scanned the LAN from 15:04, so a flag at 14:25 is a flag
before the phase that completes the compromise.

Definitions (windows of `window_seconds`):
  campaign           one host's merged episodes, chained while the gap from one
                     episode's end to the next one's onset is <= max_gap_minutes
  completing phase   the first episode carrying the campaign's most advanced
                     stage, in `STAGE_LABELS` order (recon < initial_access <
                     lateral < c2 < exfil, the ATT&CK tactic order); when every
                     episode has the same stage, the campaign's last episode
  flagged before     `m` time-consecutive windows with score >= threshold whose
  completion         origins lie in [campaign onset - pre_onset_minutes,
                     completing onset)
  stage-transition   completing onset - first window of that run (seconds)
  lead

A single-episode campaign has no earlier phase; it is counted and not scored
(its pre-onset question is `per_episode_report`'s).

What this does NOT measure: whether the forecast NAMED the completing phase. A
flag during phase 1 is phase 1 recognised in time to act before phase 2 — the
operational meaning of the PS item, and a smaller claim than forecasting
phase 2. Report it with that wording.

Coverage: the benchmark's evaluation set caps benign strata, so windows between
episodes that sit more than `pre_onset_minutes` before the next onset may be
subsampled. Runs must be consecutive in TIME, never merely adjacent rows, so a
missing window can only hide a flag, never manufacture one; the expected and
scored window counts are reported so the reader can see how much was missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nidra.data.audit import merge_episodes
from nidra.data.schema import STAGE_INDEX


@dataclass(frozen=True)
class EpisodeSpan:
    host: str
    onset_ts: int
    end_ts: int
    stage: str


def episode_spans(table: pd.DataFrame, window_seconds: int, merge_gap_windows: int) -> list[EpisodeSpan]:
    """Merged attack episodes (the benchmark's merge rule) with the most
    advanced stage labelled inside each."""
    merged = merge_episodes(table, window_seconds, merge_gap_windows)
    if merged.empty:
        return []
    rank = table["stage_label"].map(STAGE_INDEX)
    if rank.isna().any():
        raise ValueError(f"unknown stage labels {sorted(set(table.loc[rank.isna(), 'stage_label']))}")
    host = table["host_id"].to_numpy()
    ts = table["window_ts"].to_numpy()
    rank = rank.to_numpy()
    spans = []
    for h, start, end in merged[["host_id", "start_ts", "end_ts"]].itertuples(index=False):
        inside = (host == h) & (ts >= start) & (ts <= end)
        stage_rank = int(rank[inside].max())
        spans.append(EpisodeSpan(host=str(h), onset_ts=int(start), end_ts=int(end),
                                 stage=_stage_name(stage_rank)))
    return spans


def _stage_name(rank: int) -> str:
    return next(s for s, i in STAGE_INDEX.items() if i == rank)


def group_campaigns(episodes: Sequence[EpisodeSpan], max_gap_minutes: int) -> list[list[EpisodeSpan]]:
    """Chain each host's episodes while the gap between them is short enough."""
    by_host: dict[str, list[EpisodeSpan]] = {}
    for e in episodes:
        by_host.setdefault(e.host, []).append(e)
    campaigns: list[list[EpisodeSpan]] = []
    for host in sorted(by_host):
        current: list[EpisodeSpan] = []
        for e in sorted(by_host[host], key=lambda x: x.onset_ts):
            if current and (e.onset_ts - current[-1].end_ts) > max_gap_minutes * 60:
                campaigns.append(current)
                current = []
            current = [*current, e]
        campaigns.append(current)
    return campaigns


def completing_phase(campaign: Sequence[EpisodeSpan]) -> EpisodeSpan:
    """First episode of the most advanced stage; the last episode if the
    campaign never changes stage."""
    if any(e.stage not in STAGE_INDEX or e.stage == "benign" for e in campaign):
        raise ValueError(f"campaign episodes must carry an attack stage, got {[e.stage for e in campaign]}")
    ordered = sorted(campaign, key=lambda x: x.onset_ts)
    if len({e.stage for e in ordered}) == 1:
        return ordered[-1]
    top = max(STAGE_INDEX[e.stage] for e in ordered)
    return next(e for e in ordered if STAGE_INDEX[e.stage] == top)


def _first_timed_run(ts: np.ndarray, hit: np.ndarray, m: int, window_seconds: int) -> int | None:
    """Index of the first run of `m` hits whose windows are consecutive in time."""
    run = 0
    for i in range(len(ts)):
        adjacent = i > 0 and ts[i] - ts[i - 1] == window_seconds
        run = (run + 1 if adjacent else 1) if hit[i] else 0
        if run >= m:
            return i - m + 1
    return None


def _score_campaign(campaign: list[EpisodeSpan], ts: np.ndarray, s: np.ndarray, threshold: float,
                    window_seconds: int, m: int, pre_onset_minutes: int) -> dict[str, Any]:
    done = completing_phase(campaign)
    onset = campaign[0].onset_ts
    lo = onset - pre_onset_minutes * 60
    before = (ts >= lo) & (ts < done.onset_ts)
    run = _first_timed_run(ts[before], s[before] >= threshold, m, window_seconds)
    flag_ts = int(ts[before][run]) if run is not None else None
    inside = (ts >= done.onset_ts) & (ts <= done.end_ts)
    run_in = _first_timed_run(ts[inside], s[inside] >= threshold, m, window_seconds)
    return {
        "host": campaign[0].host,
        "campaign_onset_ts": int(onset),
        "phases": [{"onset_ts": e.onset_ts, "end_ts": e.end_ts, "stage": e.stage} for e in campaign],
        "completing_onset_ts": int(done.onset_ts),
        "completing_stage": done.stage,
        "expected_windows_before_completion": int((done.onset_ts - lo) // window_seconds),
        "scored_windows_before_completion": int(before.sum()),
        "max_score_before_completion": float(s[before].max()) if before.any() else None,
        "flagged_before_completion": flag_ts is not None,
        "flag_ts": flag_ts,
        "flag_before_campaign_onset": flag_ts is not None and flag_ts < onset,
        "stage_transition_lead_s": float(done.onset_ts - flag_ts) if flag_ts is not None else None,
        "completing_phase_detected": run_in is not None,
        "completing_phase_latency_min": (float((ts[inside][run_in] - done.onset_ts) / 60.0)
                                         if run_in is not None else None),
    }


def campaign_lead_report(episodes: Sequence[EpisodeSpan], host_id: np.ndarray, origin_ts: np.ndarray,
                         score: np.ndarray, threshold: float, window_seconds: int, m: int = 2,
                         max_gap_minutes: int = 60, pre_onset_minutes: int = 30) -> dict[str, Any]:
    """Per-campaign stage-transition lead for one score at one threshold."""
    host_id, origin_ts, score = np.asarray(host_id), np.asarray(origin_ts), np.asarray(score, dtype="float64")
    if not len(host_id) == len(origin_ts) == len(score):
        raise ValueError(f"misaligned arrays: {len(host_id)} hosts, {len(origin_ts)} timestamps, {len(score)} scores")
    campaigns = group_campaigns(episodes, max_gap_minutes)
    scored = [c for c in campaigns if len(c) > 1]
    rows = []
    for c in scored:
        mask = host_id == c[0].host
        order = np.argsort(origin_ts[mask], kind="stable")
        rows.append(_score_campaign(c, origin_ts[mask][order], score[mask][order], threshold,
                                    window_seconds, m, pre_onset_minutes))
    lead = [r["stage_transition_lead_s"] for r in rows if r["stage_transition_lead_s"] is not None]
    return {
        "threshold": float(threshold), "m_consecutive": int(m), "max_gap_minutes": int(max_gap_minutes),
        "pre_onset_minutes": int(pre_onset_minutes),
        "n_campaigns_scored": len(rows),
        "n_single_episode_campaigns": len(campaigns) - len(scored),
        "n_flagged_before_completion": int(sum(r["flagged_before_completion"] for r in rows)),
        "stage_transition_lead_s": lead,
        "campaigns": rows,
    }
