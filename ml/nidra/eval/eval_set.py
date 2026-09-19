"""Stratified evaluation sets with natural-prevalence weights, and the three
task labels every system is scored on.

The published Δ=30 numbers were computed on "all positives + 2,147 random
negatives": 46% prevalence against 0.4% natural, and 90% of those negatives
were all-zero silent windows. AUC-PR is prevalence-dependent, so that sample
could only ever overstate. This module builds the set the benchmark uses
instead:

  positive         every origin with risk_label = 1 (attack in (t, t+K])
  pre_onset        every origin outside an episode with the next onset on
                   that host within `pre_onset_minutes` (the forecasting
                   rows; risk_label 0 unless within K)
  active_negative  a capped uniform sample of the remaining ACTIVE origins
  silent_negative  a capped uniform sample of the remaining silent origins

and gives each row the weight N_stratum / n_sampled_stratum. Weighted
metrics are then unbiased for the full split (a stratified estimator); the
weights are stored on the set so nothing downstream has to know how it was
sampled. Every stratum's full and sampled sizes are recorded.

Task labels (all derived from stage_label, never from features):
  A  detection      stage at t is an attack
  B  onset within H origin not inside an episode, next onset within H min
                    (evaluated only over origins outside episodes)
  C  progression    future_is_attack[k] / future_stage_idx[k] per horizon
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nidra.data.audit import merge_episodes
from nidra.data.dataset import CandidateTable, WindowedArrays, enumerate_candidates, materialize

STRATA = ("positive", "pre_onset", "active_negative", "silent_negative")
DEFAULT_CAPS = {"positive": None, "pre_onset": None, "active_negative": 6000, "silent_negative": 2000}
DEFAULT_ONSET_HORIZONS_MIN = (1, 3, 5, 10, 15, 30)


@dataclass
class EvalSet:
    arrays: WindowedArrays
    stratum: np.ndarray                 # [N] str
    weight: np.ndarray                  # [N] float, natural-prevalence weight
    cluster: np.ndarray                 # [N] str, bootstrap cluster (episode or host)
    inside_episode: np.ndarray          # [N] bool, origin inside a merged attack episode
    minutes_to_onset: np.ndarray        # [N] float, inf if no later onset on the host
    episode_key: np.ndarray             # [N] str, "" if none ("host@start_ts" of the episode the row belongs to / precedes)
    onset_labels: dict[int, np.ndarray] = field(default_factory=dict)   # H minutes -> [N] int
    stratum_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    split: str = ""
    window_seconds: int = 60
    K: int = 6
    span_hours: float = 0.0
    n_hosts: int = 0

    # --- labels ---
    @property
    def y_published(self) -> np.ndarray:
        return self.arrays.risk_label.astype(int)

    @property
    def y_detect(self) -> np.ndarray:
        return (self.arrays.stage_label != "benign").astype(int)

    @property
    def mask_forecast(self) -> np.ndarray:
        """Rows on which Task B is defined: origin outside any episode."""
        return ~self.inside_episode

    def __len__(self) -> int:
        return len(self.weight)

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_rows": int(len(self)),
            "n_hosts_full": int(self.n_hosts),
            "span_hours": float(self.span_hours),
            "strata": self.stratum_counts,
            "natural_prevalence_published": float(np.average(self.y_published, weights=self.weight)),
            "natural_prevalence_detect": float(np.average(self.y_detect, weights=self.weight)),
            "n_positive": int(self.y_published.sum()),
            "n_forecast_rows": int(self.mask_forecast.sum()),
            "n_onset_positives": {str(h): int(self.onset_labels[h][self.mask_forecast].sum()) for h in self.onset_labels},
            "n_clusters": int(len(np.unique(self.cluster))),
        }


def _episode_geometry(table: pd.DataFrame, candidates: CandidateTable, window_seconds: int,
                      merge_gap_windows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per candidate: inside a merged episode?, minutes to the next onset on
    the host (inf if none), and the key of the episode it is inside of / the
    next episode it precedes ("" if neither)."""
    episodes = merge_episodes(table, window_seconds, merge_gap_windows)
    n = len(candidates)
    inside = np.zeros(n, dtype=bool)
    to_onset = np.full(n, np.inf)
    key = np.full(n, "", dtype=object)
    if episodes.empty:
        return inside, to_onset, key
    ts = candidates.origin_ts
    hosts = candidates.host
    for host, g in episodes.groupby("host_id"):
        m = hosts == host
        if not m.any():
            continue
        idx = np.where(m)[0]
        t = ts[idx]
        for start, end in zip(g["start_ts"].to_numpy(), g["end_ts"].to_numpy()):
            ek = f"{host}@{int(start)}"
            in_ep = (t >= start) & (t <= end)
            inside[idx[in_ep]] = True
            key[idx[in_ep]] = ek
            d = (start - t) / 60.0
            before = (d > 0) & (d < to_onset[idx])
            to_onset[idx[before]] = d[before]
            key[idx[before & ~in_ep]] = ek
    return inside, to_onset, key


def build_eval_set(
    table: pd.DataFrame,
    L: int,
    K: int,
    window_seconds: int,
    split: str,
    caps: dict[str, int | None] | None = None,
    seed: int = 0,
    merge_gap_windows: int = 5,
    pre_onset_minutes: int = 30,
    onset_horizons_min: tuple[int, ...] = DEFAULT_ONSET_HORIZONS_MIN,
) -> EvalSet:
    caps = {**DEFAULT_CAPS, **(caps or {})}
    df = table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    cands = enumerate_candidates(df, L, K)
    inside, to_onset, ep_key = _episode_geometry(df, cands, window_seconds, merge_gap_windows)

    stratum = np.full(len(cands), "silent_negative", dtype=object)
    pos = cands.risk_label == 1
    pre = (~pos) & (~inside) & (~cands.origin_attack) & (to_onset <= pre_onset_minutes)
    act = (~pos) & (~pre) & cands.origin_active
    stratum[pos] = "positive"
    stratum[pre] = "pre_onset"
    stratum[act] = "active_negative"

    rng = np.random.default_rng(seed)
    select: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}
    weight_by_stratum: dict[str, float] = {}
    for name in STRATA:
        idx = np.where(stratum == name)[0]
        cap = caps.get(name)
        chosen = idx if cap is None or len(idx) <= cap else rng.choice(idx, size=cap, replace=False)
        counts[name] = {"full": int(len(idx)), "sampled": int(len(chosen))}
        weight_by_stratum[name] = (len(idx) / len(chosen)) if len(chosen) else 0.0
        select.append(np.sort(chosen))
    select_all = np.concatenate(select) if select else np.zeros(0, dtype="int64")
    select_all = np.sort(select_all)

    arrays = materialize(cands, select_all, L, K)
    stratum_sel = stratum[select_all]
    weight = np.array([weight_by_stratum[s] for s in stratum_sel], dtype="float64")
    inside_sel = inside[select_all]
    to_onset_sel = to_onset[select_all]
    key_sel = ep_key[select_all]
    cluster = np.where(key_sel != "", key_sel, cands.host[select_all]).astype(object)

    onset_labels = {}
    for h in onset_horizons_min:
        onset_labels[h] = ((~inside_sel) & (to_onset_sel <= h)).astype(int)

    ts_all = df["window_ts"].to_numpy()
    span_hours = float((ts_all.max() - ts_all.min()) / 3600.0) if len(ts_all) else 0.0
    return EvalSet(
        arrays=arrays, stratum=stratum_sel.astype(str), weight=weight, cluster=cluster.astype(str),
        inside_episode=inside_sel, minutes_to_onset=to_onset_sel, episode_key=key_sel.astype(str),
        onset_labels=onset_labels, stratum_counts=counts, split=split, window_seconds=window_seconds, K=K,
        span_hours=span_hours, n_hosts=int(df["host_id"].nunique()),
    )
