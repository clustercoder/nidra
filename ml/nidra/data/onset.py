"""Onset targets: "does an attack episode BEGIN within h minutes of t?"

The published risk label (`labels.risk_label_from_attack_flags`) is 1 for
any window whose K-step future contains an attack window — which includes
every window already inside an attack. The onset target is the strictly
forward-looking version the problem statement's "predict future attack
likelihood" asks for: it is defined only at origins OUTSIDE any episode,
and it is 1 when the next episode on the host starts within h minutes.

Both are legitimate supervision (they look forward to build the label);
neither is ever an input. The same geometry serves the evaluation set
(eval/eval_set.py) and the training arrays (data/dataset.py), so the two
cannot disagree about what "onset" means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nidra.data.audit import merge_episodes

DEFAULT_ONSET_HORIZONS_MIN: tuple[int, ...] = (1, 3, 5, 10, 15, 30)
DEFAULT_MERGE_GAP_WINDOWS = 5


def infer_window_seconds(table: pd.DataFrame) -> int:
    """Smallest positive within-host timestamp step — the window size of a
    gap-filled table. Raises on a table with no consecutive windows."""
    if table.empty:
        raise ValueError("cannot infer window_seconds from an empty table")
    df = table[["host_id", "window_ts"]].sort_values(["host_id", "window_ts"])
    same_host = df["host_id"].to_numpy()[1:] == df["host_id"].to_numpy()[:-1]
    diffs = np.diff(df["window_ts"].to_numpy(dtype="int64"))[same_host]
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("cannot infer window_seconds: no host has two windows")
    return int(diffs.min())


def episode_geometry(table: pd.DataFrame, hosts: np.ndarray, origin_ts: np.ndarray, window_seconds: int,
                     merge_gap_windows: int = DEFAULT_MERGE_GAP_WINDOWS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per origin (host, ts): inside a merged episode?, minutes to the next
    onset on that host (inf if none), and the key of the episode it is
    inside of or precedes ("" if neither)."""
    episodes = merge_episodes(table, window_seconds, merge_gap_windows)
    if not episodes.empty:
        episodes = episodes.sort_values(["host_id", "start_ts"])
    n = len(origin_ts)
    inside = np.zeros(n, dtype=bool)
    to_onset = np.full(n, np.inf)
    key = np.full(n, "", dtype=object)
    if episodes.empty or n == 0:
        return inside, to_onset, key
    hosts = np.asarray(hosts)
    for host, g in episodes.groupby("host_id"):
        m = hosts == host
        if not m.any():
            continue
        idx = np.where(m)[0]
        t = origin_ts[idx]
        # Episodes in start order: a row is keyed to the episode it is INSIDE,
        # else to the NEXT onset it precedes. Processing later episodes must
        # not re-key rows that sit inside an earlier one (a host that attacks
        # on two days has two episodes, not one spanning both).
        for start, end in zip(g["start_ts"].to_numpy(), g["end_ts"].to_numpy()):
            ek = f"{host}@{int(start)}"
            in_ep = (t >= start) & (t <= end)
            inside[idx[in_ep]] = True
            key[idx[in_ep]] = ek
            d = (start - t) / 60.0
            before = (d > 0) & (d < to_onset[idx])
            to_onset[idx[before]] = d[before]          # informational for rows inside an earlier episode
            key[idx[before & ~inside[idx]]] = ek       # keying never crosses an episode boundary
    return inside, to_onset, key


def onset_targets(inside: np.ndarray, minutes_to_onset: np.ndarray,
                  horizons_min: tuple[int, ...] = DEFAULT_ONSET_HORIZONS_MIN) -> np.ndarray:
    """[N, H] int: 1 where the origin is outside every episode and the next
    onset is within h minutes. Rows inside an episode are 0 at every horizon
    and must be MASKED (`~inside`) by anything that trains or scores on
    these — an ongoing attack is not a forecast."""
    out = np.zeros((len(inside), len(horizons_min)), dtype="int64")
    eligible = ~np.asarray(inside, dtype=bool)
    for j, h in enumerate(horizons_min):
        out[:, j] = (eligible & (np.asarray(minutes_to_onset) <= h)).astype("int64")
    return out


def discrete_hazard_targets(inside: np.ndarray, minutes_to_onset: np.ndarray,
                            horizons_min: tuple[int, ...] = DEFAULT_ONSET_HORIZONS_MIN,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time survival targets over the same horizon grid.

    The horizons define buckets (0, h1], (h1, h2], ... (h_{H-1}, h_H]. For
    each origin this returns

        event  [N, H] int   1 in the bucket the onset falls in, 0 elsewhere
        at_risk[N, H] bool  whether the origin is still "at risk" in that
                            bucket, i.e. no onset has happened in an earlier
                            one and the row has not been censored

    which is what a hazard head is fit against: bucket j's loss is evaluated
    only on the rows that survived to it. The resulting probabilities are
    coherent by construction — P(onset within h_m) = 1 - prod_{j<=m}(1 - p_j)
    is monotone in m, which independent per-horizon BCE is not. The Run 8
    onset head could and did report P(within 1 min) above P(within 30 min).

    Rows inside an episode are not at risk anywhere: an ongoing attack is not
    a forecast. Rows whose next onset lies beyond the last horizon (including
    `inf`, meaning the host never attacks again in this capture) are censored
    at the end — at risk in every bucket, with no event. That is the honest
    treatment: "no onset within 30 minutes" is what was observed, not "no
    onset ever".
    """
    minutes = np.asarray(minutes_to_onset, dtype="float64")
    eligible = ~np.asarray(inside, dtype=bool)
    edges = (0.0,) + tuple(float(h) for h in horizons_min)
    n, H = len(minutes), len(horizons_min)
    event = np.zeros((n, H), dtype="int64")
    at_risk = np.zeros((n, H), dtype=bool)
    for j in range(H):
        lo, hi = edges[j], edges[j + 1]
        in_bucket = eligible & (minutes > lo) & (minutes <= hi)
        event[:, j] = in_bucket.astype("int64")
        # at risk in bucket j if no onset strictly before its lower edge
        at_risk[:, j] = eligible & (minutes > lo)
    return event, at_risk


def survival_to_cumulative(hazards: np.ndarray) -> np.ndarray:
    """[N, H] per-bucket hazards -> [N, H] P(onset within h_j), monotone."""
    h = np.clip(np.asarray(hazards, dtype="float64"), 0.0, 1.0)
    return 1.0 - np.cumprod(1.0 - h, axis=-1)
