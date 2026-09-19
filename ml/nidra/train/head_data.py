"""Observed-state arrays for head training and selection.

The heads are functions of one state, S_t. Training and selecting them on
the windowed [N, L, F] subsample that the dynamics use throws away most of
the split: at L=30 the subsample keeps 500k of 2.3M training rows and 50k
of 900k validation rows, and — more importantly — the validation statistic
the head is selected on is then a WEIGHTED estimate from a 50k negative
sample, which disagreed with the benchmark by 0.3 AP on the same head
(2026-09-20: 0.72 on the subsample, 0.42 on the benchmark set, the gap
being three external hosts the benchmark sample happened to contain).

Here every row of the split becomes one sample: the scaled state, the
published risk label (recomputed for the configured K at split time), the
stage index, activity, and the onset geometry (data/onset.py). Selection
on the full validation split needs no weights — it IS the natural
prevalence. 2.3M × 45 float32 is 410 MB; the dynamics' windowed arrays
are what needs a cap, not this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nidra.data.normalize import FeatureScaler
from nidra.data.onset import (
    DEFAULT_MERGE_GAP_WINDOWS,
    DEFAULT_ONSET_HORIZONS_MIN,
    episode_geometry,
    infer_window_seconds,
    onset_targets,
)
from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER, STAGE_INDEX


@dataclass
class HeadArrays:
    states: np.ndarray            # [N, F] SCALED observed states
    risk_label: np.ndarray        # [N] int
    stage_idx: np.ndarray         # [N] int
    active: np.ndarray            # [N] bool
    inside_episode: np.ndarray    # [N] bool
    minutes_to_onset: np.ndarray  # [N] float
    host_id: np.ndarray           # [N] str
    window_ts: np.ndarray         # [N] int64

    def __len__(self) -> int:
        return len(self.risk_label)

    def onset_targets(self, horizons_min: tuple[int, ...] = DEFAULT_ONSET_HORIZONS_MIN) -> np.ndarray:
        return onset_targets(self.inside_episode, self.minutes_to_onset, horizons_min)

    def summary(self) -> dict:
        return {"n": int(len(self)), "n_pos": int(self.risk_label.sum()), "n_active": int(self.active.sum()),
                "n_inside_episode": int(self.inside_episode.sum()), "prevalence": float(self.risk_label.mean()) if len(self) else 0.0}


def build_head_arrays(table: pd.DataFrame, scaler: FeatureScaler,
                      merge_gap_windows: int = DEFAULT_MERGE_GAP_WINDOWS) -> HeadArrays:
    """Every row of a labelled state table (host_id, window_ts, stage_label,
    risk_label, FEATURE_ORDER columns) as one head sample."""
    if table.empty:
        F = len(FEATURE_ORDER)
        e = np.zeros(0)
        return HeadArrays(np.zeros((0, F), dtype="float32"), e.astype(int), e.astype(int), e.astype(bool), e.astype(bool), e,
                          np.array([], dtype=object), e.astype("int64"))
    df = table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    raw = df[FEATURE_ORDER].to_numpy(dtype="float32")
    states = scaler.transform(raw)
    hosts = df["host_id"].to_numpy().astype(object)
    ts = df["window_ts"].to_numpy(dtype="int64")
    inside, to_onset, _ = episode_geometry(df, hosts, ts, infer_window_seconds(df), merge_gap_windows)
    return HeadArrays(
        states=np.ascontiguousarray(states, dtype="float32"),
        risk_label=df["risk_label"].to_numpy(dtype="int64"),
        stage_idx=df["stage_label"].map(STAGE_INDEX).fillna(0).to_numpy(dtype="int64"),
        active=raw[:, FEATURE_INDEX["is_active"]] > 0,
        inside_episode=inside, minutes_to_onset=to_onset, host_id=hosts, window_ts=ts,
    )
