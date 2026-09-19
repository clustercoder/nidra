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

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_INDEX, FEATURE_ORDER, HORIZON_LENGTH, STAGE_INDEX
from nidra.data.splits import find_episodes


@dataclass
class WindowedArrays:
    X: np.ndarray             # [N, L, F] raw, unscaled, oldest-first
    Y: np.ndarray              # [N, K, F] raw, unscaled future ground truth
    host_id: np.ndarray        # [N] str
    origin_ts: np.ndarray      # [N] int64 — window_ts of the last observed window (t)
    episode_id: np.ndarray     # [N] int64, -1 if not inside an attack episode at t
    stage_label: np.ndarray    # [N] str — observed stage at t
    risk_label: np.ndarray     # [N] int — forward-looking risk target (any attack in (t,t+K])
    future_stage_idx: np.ndarray = None    # [N, K] int — ground truth stage index at each t+k, for horizon curves
    future_is_attack: np.ndarray = None    # [N, K] int — 1 if t+k itself is a non-benign window


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


@dataclass
class CandidateTable:
    """Pass 1 of sample construction: every valid origin as cheap scalars,
    plus the per-host columnar arrays pass 2 slices from. Nothing [L,F]-
    shaped is materialized here."""
    per_host: dict                      # host -> (feats, stage_idx, is_attack, window_ts, stage_label, risk_label)
    host: np.ndarray                    # [C] str
    t: np.ndarray                       # [C] int, row index inside the host's sequence
    risk_label: np.ndarray              # [C] int
    origin_active: np.ndarray           # [C] bool, is_active at t
    origin_attack: np.ndarray           # [C] bool, stage at t != benign
    origin_ts: np.ndarray               # [C] int64

    def __len__(self) -> int:
        return len(self.t)


def enumerate_candidates(labelled_state_table: pd.DataFrame, L: int, K: int) -> CandidateTable:
    """Every valid sample origin in the table (index i with L-1 <= i <= n-1-K
    inside each host's contiguous sequence) as per-candidate scalars."""
    feature_cols = FEATURE_ORDER
    active_idx = FEATURE_INDEX["is_active"]
    df = labelled_state_table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    per_host: dict = {}
    hosts, ts_idx, risks, actives, attacks, origin_ts = [], [], [], [], [], []
    for host, g in df.groupby("host_id", sort=False):
        g = g.reset_index(drop=True)
        # Extract every per-row field as a numpy array ONCE per host, outside
        # the sample loop — O(n_samples) scalar pandas lookups do not scale to
        # a real day's worth of windows.
        feats = g[feature_cols].to_numpy(dtype="float32")
        stage_idx = g["stage_label"].map(STAGE_INDEX).to_numpy(dtype="int64")
        is_attack = (g["stage_label"] != "benign").to_numpy(dtype="int64")
        window_ts_arr = g["window_ts"].to_numpy(dtype="int64")
        stage_label_arr = g["stage_label"].to_numpy()
        risk_label_arr = g["risk_label"].to_numpy(dtype="int64")
        per_host[host] = (feats, stage_idx, is_attack, window_ts_arr, stage_label_arr, risk_label_arr)
        n = len(g)
        if n - K < L - 1 + 1:
            continue
        t = np.arange(L - 1, n - K)
        hosts.append(np.full(len(t), host, dtype=object))
        ts_idx.append(t)
        risks.append(risk_label_arr[t])
        actives.append(feats[t, active_idx] > 0)
        attacks.append(is_attack[t] > 0)
        origin_ts.append(window_ts_arr[t])
    if not ts_idx:
        empty_i = np.zeros(0, dtype="int64")
        return CandidateTable(per_host, np.array([], dtype=object), empty_i, empty_i, np.zeros(0, bool), np.zeros(0, bool), empty_i)
    return CandidateTable(
        per_host,
        np.concatenate(hosts), np.concatenate(ts_idx), np.concatenate(risks).astype("int64"),
        np.concatenate(actives), np.concatenate(attacks), np.concatenate(origin_ts).astype("int64"),
    )


def materialize(candidates: CandidateTable, select: np.ndarray, L: int, K: int,
                episode_lookup: dict | None = None) -> WindowedArrays:
    """Pass 2: slice the [L,F] history and [K,F] future for the selected
    candidate indices only."""
    if episode_lookup is None:
        episode_lookup = {}
    X_list, Y_list, future_stage_list, future_attack_list = [], [], [], []
    host_list, origin_ts_list, episode_id_list, stage_list, risk_list = [], [], [], [], []
    for i in select:
        host, t = candidates.host[i], int(candidates.t[i])
        feats, stage_idx, is_attack, window_ts_arr, stage_label_arr, risk_label_arr = candidates.per_host[host]
        X_list.append(feats[t - L + 1 : t + 1])
        Y_list.append(feats[t + 1 : t + 1 + K])
        future_stage_list.append(stage_idx[t + 1 : t + 1 + K])
        future_attack_list.append(is_attack[t + 1 : t + 1 + K])
        host_list.append(host)
        origin_ts = int(window_ts_arr[t])
        origin_ts_list.append(origin_ts)
        episode_id_list.append(_episode_id_at(episode_lookup, host, origin_ts))
        stage_list.append(stage_label_arr[t])
        risk_list.append(int(risk_label_arr[t]))
    if not X_list:
        return _empty_arrays(L, K)
    return WindowedArrays(
        X=np.stack(X_list),
        Y=np.stack(Y_list),
        host_id=np.array(host_list),
        origin_ts=np.array(origin_ts_list, dtype="int64"),
        episode_id=np.array(episode_id_list, dtype="int64"),
        stage_label=np.array(stage_list),
        risk_label=np.array(risk_list, dtype="int64"),
        future_stage_idx=np.stack(future_stage_list),
        future_is_attack=np.stack(future_attack_list),
    )


def _empty_arrays(L: int, K: int) -> WindowedArrays:
    return WindowedArrays(
        X=np.zeros((0, L, len(FEATURE_ORDER)), dtype="float32"),
        Y=np.zeros((0, K, len(FEATURE_ORDER)), dtype="float32"),
        host_id=np.array([]), origin_ts=np.array([], dtype="int64"),
        episode_id=np.array([], dtype="int64"), stage_label=np.array([]),
        risk_label=np.array([], dtype="int64"),
        future_stage_idx=np.zeros((0, K), dtype="int64"),
        future_is_attack=np.zeros((0, K), dtype="int64"),
    )


def build_windowed_arrays(
    labelled_state_table: pd.DataFrame,
    L: int = CONTEXT_LENGTH,
    K: int = HORIZON_LENGTH,
    max_samples: int | None = None,
    seed: int = 0,
) -> WindowedArrays:
    """labelled_state_table: output of labels.attach_risk_label, sorted or
    not — this function sorts by (host_id, window_ts) itself. Each host's
    sequence is contiguous (gap-filled upstream in windowize.py), so any
    index i with i >= L-1 and i <= n-1-K is a valid sample origin.

    `max_samples`: if given and the candidate count exceeds it, applies the
    same stratified-by-risk selection as `subsample_stratified_by_risk`
    (keep every positive-risk origin, fill the remainder from negatives) —
    but BEFORE building any [L,F]/[K,F] slice, not after. This matters at
    full production scale: a real training split can have ~7M candidate
    origins, and materializing all of them as float32 [L=30,F=45] arrays
    first (~35GB) before discarding most of them is what OOM-kills a
    16GB-RAM machine, even when the caller only wanted a bounded sample of
    them. Cheap per-row scalars (host, t, risk_label) are enumerated for
    every candidate first (`enumerate_candidates`); only the selected
    subset's history/future windows are ever sliced out (`materialize`)."""
    if labelled_state_table.empty:
        return _empty_arrays(L, K)

    df = labelled_state_table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    episode_lookup = _episode_lookup(df)
    candidates = enumerate_candidates(df, L, K)
    if len(candidates) == 0:
        return _empty_arrays(L, K)

    select = np.arange(len(candidates))
    if max_samples is not None and len(candidates) > max_samples:
        risk_arr = candidates.risk_label
        rng = np.random.default_rng(seed)
        pos_idx = np.where(risk_arr == 1)[0]
        neg_idx = np.where(risk_arr == 0)[0]
        if len(pos_idx) >= max_samples:
            keep = rng.choice(pos_idx, size=max_samples, replace=False)
        else:
            n_neg = max_samples - len(pos_idx)
            neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
            keep = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(keep)
        select = keep

    return materialize(candidates, select, L, K, episode_lookup)


def subsample_stratified_by_risk(arrays: WindowedArrays, max_n: int | None, seed: int = 0) -> WindowedArrays:
    """Caps a WindowedArrays to at most `max_n` samples, stratified by
    risk_label: keeps ALL positive-risk samples and fills the remainder
    randomly from negatives. Real attack data is heavily imbalanced — a
    uniform random subsample risks dropping every positive example
    entirely, which silently turns a risk-prediction evaluation into a
    trivial all-negative one. Used both to cap training set size and to
    cap evaluation cost (rollout sampling is the expensive step, and it is
    run once per evaluation sample)."""
    if max_n is None or len(arrays.X) <= max_n:
        return arrays
    rng = np.random.default_rng(seed)
    pos_idx = np.where(arrays.risk_label == 1)[0]
    neg_idx = np.where(arrays.risk_label == 0)[0]

    if len(pos_idx) >= max_n:
        idx = rng.choice(pos_idx, size=max_n, replace=False)
    else:
        n_neg = max_n - len(pos_idx)
        neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(idx)

    return WindowedArrays(
        X=arrays.X[idx], Y=arrays.Y[idx], host_id=arrays.host_id[idx],
        origin_ts=arrays.origin_ts[idx], episode_id=arrays.episode_id[idx],
        stage_label=arrays.stage_label[idx], risk_label=arrays.risk_label[idx],
        future_stage_idx=arrays.future_stage_idx[idx] if arrays.future_stage_idx is not None else None,
        future_is_attack=arrays.future_is_attack[idx] if arrays.future_is_attack is not None else None,
    )


class WorldModelDataset(Dataset):
    """Thin tensor wrapper. Expects X/Y already scaled (see normalize.py) —
    scaling happens once over the whole split, not per __getitem__."""

    def __init__(self, arrays: WindowedArrays, X_scaled: np.ndarray | None = None, Y_scaled: np.ndarray | None = None,
                 sample_weight: np.ndarray | None = None):
        self.arrays = arrays
        self.X = X_scaled if X_scaled is not None else arrays.X
        self.Y = Y_scaled if Y_scaled is not None else arrays.Y
        self.sample_weight = sample_weight

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "x": torch.from_numpy(self.X[idx]).float(),
            "y": torch.from_numpy(self.Y[idx]).float(),
            "risk_label": torch.tensor(self.arrays.risk_label[idx], dtype=torch.float32),
            "stage_label": self.arrays.stage_label[idx],
            "host_id": self.arrays.host_id[idx],
            "origin_ts": int(self.arrays.origin_ts[idx]),
            "episode_id": int(self.arrays.episode_id[idx]),
        }
        if self.sample_weight is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weight[idx]), dtype=torch.float32)
        return item
