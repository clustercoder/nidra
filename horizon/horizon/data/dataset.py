"""Sample construction: labelled per-(host,window) state rows -> windowed
[L, F] history / [K, F] future arrays plus traceable metadata.

Every sample is traceable back to its source host, origin window, and
(if applicable) the attack episode it belongs to — metadata is kept
separate from the numeric tensors rather than folded into them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from horizon.data.schema import CONTEXT_LENGTH, FEATURE_ORDER, HORIZON_LENGTH
from horizon.data.splits import find_episodes


@dataclass
class WindowedArrays:
    X: np.ndarray            # [N, L, F] raw, unscaled, oldest-first
    Y: np.ndarray             # [N, K, F] raw, unscaled future ground truth
    host_id: np.ndarray       # [N] str
    origin_ts: np.ndarray     # [N] int64 — window_ts of the last observed window (t)
    episode_id: np.ndarray    # [N] int64, -1 if not inside an attack episode at t
    stage_label: np.ndarray   # [N] str — observed stage at t
    risk_label: np.ndarray    # [N] int — forward-looking risk target


def _episode_lookup(df: pd.DataFrame) -> dict:
    """host_id -> sorted list of (start_ts, end_ts, episode_id) for O(log n)
    membership checks per sample."""
    episodes = find_episodes(df)
    lookup: dict = {}
    for eid, row in episodes.reset_index(drop=True).iterrows():
        lookup.setdefault(row["host_id"], []).append((row["start_ts"], row["end_ts"], eid))
    for host in lookup:
        lookup[host].sort()
    return lookup


def _episode_id_at(lookup: dict, host: str, ts: int) -> int:
    for start, end, eid in lookup.get(host, []):
        if start <= ts <= end:
            return int(eid)
        if start > ts:
            break
    return -1


def build_windowed_arrays(
    labelled_state_table: pd.DataFrame,
    L: int = CONTEXT_LENGTH,
    K: int = HORIZON_LENGTH,
) -> WindowedArrays:
    """labelled_state_table: output of labels.attach_risk_label, sorted or
    not — this function sorts by (host_id, window_ts) itself. Each host's
    sequence is contiguous (gap-filled upstream in windowize.py), so any
    index i with i >= L-1 and i <= n-1-K is a valid sample origin."""
    df = labelled_state_table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    feature_cols = FEATURE_ORDER
    episode_lookup = _episode_lookup(df)

    X_list, Y_list = [], []
    host_list, origin_ts_list, episode_id_list, stage_list, risk_list = [], [], [], [], []

    for host, g in df.groupby("host_id", sort=False):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype="float32")
        n = len(g)
        for t in range(L - 1, n - K):
            X_list.append(feats[t - L + 1 : t + 1])
            Y_list.append(feats[t + 1 : t + 1 + K])
            host_list.append(host)
            origin_ts = int(g.loc[t, "window_ts"])
            origin_ts_list.append(origin_ts)
            episode_id_list.append(_episode_id_at(episode_lookup, host, origin_ts))
            stage_list.append(g.loc[t, "stage_label"])
            risk_list.append(int(g.loc[t, "risk_label"]))

    if not X_list:
        return WindowedArrays(
            X=np.zeros((0, L, len(feature_cols)), dtype="float32"),
            Y=np.zeros((0, K, len(feature_cols)), dtype="float32"),
            host_id=np.array([]), origin_ts=np.array([], dtype="int64"),
            episode_id=np.array([], dtype="int64"), stage_label=np.array([]),
            risk_label=np.array([], dtype="int64"),
        )

    return WindowedArrays(
        X=np.stack(X_list),
        Y=np.stack(Y_list),
        host_id=np.array(host_list),
        origin_ts=np.array(origin_ts_list, dtype="int64"),
        episode_id=np.array(episode_id_list, dtype="int64"),
        stage_label=np.array(stage_list),
        risk_label=np.array(risk_list, dtype="int64"),
    )


class WorldModelDataset(Dataset):
    """Thin tensor wrapper. Expects X/Y already scaled (see normalize.py) —
    scaling happens once over the whole split, not per __getitem__."""

    def __init__(self, arrays: WindowedArrays, X_scaled: np.ndarray | None = None, Y_scaled: np.ndarray | None = None):
        self.arrays = arrays
        self.X = X_scaled if X_scaled is not None else arrays.X
        self.Y = Y_scaled if Y_scaled is not None else arrays.Y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.from_numpy(self.X[idx]).float(),
            "y": torch.from_numpy(self.Y[idx]).float(),
            "risk_label": torch.tensor(self.arrays.risk_label[idx], dtype=torch.float32),
            "stage_label": self.arrays.stage_label[idx],
            "host_id": self.arrays.host_id[idx],
            "origin_ts": int(self.arrays.origin_ts[idx]),
            "episode_id": int(self.arrays.episode_id[idx]),
        }
