"""Temporal / episode-level / attack-type-holdout splitting.

Splitting is day-based first (train = Monday+Tuesday+Wednesday, test =
Friday, holdout = Thursday/Infiltration — never trained on), which trivially
satisfies episode integrity across days since CIC-IDS2017 attacks are
confined to a single labelled day. Within `train`, a CONTIGUOUS trailing
time block is held out for validation — never a random sample — with the
cutoff nudged earlier if it would otherwise cut through a live attack
episode.

The trailing block is taken PER TRAINING CAPTURE (`per_day=True`, the default
from the Δ=60 rebuild on): the training days are consecutive calendar days,
so a single trailing fraction of the concatenated timeline is just the end
of the last day — for CIC-IDS2017 that was the last ~3.5 hours of
Wednesday, whose only attack is one Heartbleed episode on one host. Every
checkpoint, pooling, threshold and calibration decision then rested on 54
positives of a single attack type. A per-day block gives validation an
attack from each training day (Tuesday's SSH brute force, Wednesday's
Heartbleed) at the cost of removing those episodes from training, which is
the correct trade for a split whose job is model selection. The original
concatenated-timeline behaviour is kept as `per_day=False` for
reproducing the Δ=30 artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame


def find_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Identify contiguous non-benign runs per host as (host_id, start_ts,
    end_ts) episodes, from a table carrying host_id, window_ts, stage_label.
    """
    df = df.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    is_attack = (df["stage_label"] != "benign").to_numpy()
    host = df["host_id"].to_numpy()
    ts = df["window_ts"].to_numpy()

    episodes = []
    start = None
    for i in range(len(df)):
        new_host = i == 0 or host[i] != host[i - 1]
        if new_host and start is not None:
            episodes.append((host[i - 1], ts[start], ts[i - 1]))
            start = None
        if is_attack[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                episodes.append((host[i - 1], ts[start], ts[i - 1]))
                start = None
    if start is not None:
        episodes.append((host[-1], ts[start], ts[-1]))

    return pd.DataFrame(episodes, columns=["host_id", "start_ts", "end_ts"])


def capture_day(window_ts: pd.Series) -> pd.Series:
    """Calendar day (UTC) of each window — the day-file identity of a row in
    a concatenated table. CIC-IDS2017's captures each sit inside one UTC
    day, so this is exact for that dataset; a capture spanning midnight
    would be split into two 'days' by this rule, which is still a
    contiguous, non-random partition."""
    return (window_ts.astype("int64") // 86_400).astype("int64")


def block_group(df: pd.DataFrame) -> pd.Series:
    """Which contiguous capture each row belongs to, for per-capture
    validation blocking.

    `split_group` (the config day key, attached in train.pipeline) when the
    table carries it, otherwise the UTC calendar day. The distinction only
    matters once more than one capture can share a calendar date, which
    CIC-IDS2017 never does and CTU-13 does three times over (scenarios 4, 5
    and 13 all start on 2011-08-15; 6, 7 and 8 on 08-16; 10 and 11 on
    08-18). Blocking those by date would hand a whole scenario to
    validation and leave its neighbour entirely in train — a partition by
    capture, not the trailing-time-block partition the split is supposed to
    be."""
    if "split_group" in df.columns:
        return df["split_group"].astype(str)
    return capture_day(df["window_ts"]).astype(str)


DEFAULT_PRE_ONSET_MARGIN_S = 1800  # the 30-minute pre-onset span the benchmark evaluates lead time on


def _trailing_block_cutoff(block: pd.DataFrame, val_fraction: float,
                           pre_onset_margin_s: int = DEFAULT_PRE_ONSET_MARGIN_S) -> int:
    """Cutoff timestamp for one contiguous block, nudged earlier so that no
    episode straddles it AND no episode starts within `pre_onset_margin_s`
    after it. The margin keeps an episode's run-up on the same side as the
    episode: with the cut at the onset itself, the last K training windows
    before a validation episode are genuine pre-onset windows that the
    per-split label recomputation marks negative, and the validation split
    has no pre-onset rows to measure lead time on."""
    unique_ts = np.sort(block["window_ts"].unique())
    cutoff_idx = int(len(unique_ts) * (1 - val_fraction))
    cutoff_idx = min(max(cutoff_idx, 0), len(unique_ts) - 1)
    cutoff = int(unique_ts[cutoff_idx])

    episodes = find_episodes(block)
    if not episodes.empty:
        near = episodes[(episodes["start_ts"] - pre_onset_margin_s < cutoff) & (episodes["end_ts"] >= cutoff)]
        if not near.empty:
            new_cutoff = max(int(near["start_ts"].min()) - int(pre_onset_margin_s), int(unique_ts[0]))
            logger.info("temporal_train_val_split: nudging cutoff %s -> %s (episode at %s; margin %ds)",
                        cutoff, new_cutoff, int(near["start_ts"].min()), pre_onset_margin_s)
            cutoff = new_cutoff
    return cutoff


def temporal_train_val_split(train_df: pd.DataFrame, val_fraction: float,
                             per_day: bool = True,
                             pre_onset_margin_s: int = DEFAULT_PRE_ONSET_MARGIN_S) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contiguous trailing time-block validation split with episode-boundary
    nudging: if the fraction-based cutoff falls inside a live episode, the
    cutoff moves earlier to that episode's start so no episode straddles
    the train/val boundary. With `per_day=True` the block is taken from the
    end of each capture day separately (see module docstring)."""
    if train_df.empty:
        return train_df, train_df

    if not per_day:
        cutoff = _trailing_block_cutoff(train_df, val_fraction, pre_onset_margin_s)
        train_part = train_df[train_df["window_ts"] < cutoff].reset_index(drop=True)
        val_part = train_df[train_df["window_ts"] >= cutoff].reset_index(drop=True)
        return train_part, val_part

    days = block_group(train_df)
    is_val = np.zeros(len(train_df), dtype=bool)
    for day in pd.unique(days):
        mask = (days == day).to_numpy()
        cutoff = _trailing_block_cutoff(train_df.loc[mask], val_fraction, pre_onset_margin_s)
        is_val |= mask & (train_df["window_ts"].to_numpy() >= cutoff)
    train_part = train_df.loc[~is_val].reset_index(drop=True)
    val_part = train_df.loc[is_val].reset_index(drop=True)
    return train_part, val_part


def build_splits(
    day_tables: dict[str, pd.DataFrame],
    train_days: list[str],
    test_days: list[str],
    holdout_days: list[str],
    val_fraction: float,
    val_block_per_day: bool = True,
    horizon_k: int | None = None,
    pre_onset_margin_s: int = DEFAULT_PRE_ONSET_MARGIN_S,
) -> SplitResult:
    """day_tables: mapping of config day-key -> labelled state table for that
    day (output of labels.attach_risk_label). Concatenates by role, then
    carves the validation block out of train.

    `horizon_k`, when given, recomputes `risk_label` inside each of train
    and val after the cut, so the last K training windows before a
    validation episode do not carry a label derived from validation-block
    windows (labels are attached per day before the cut, and that boundary
    leaked ≤K positives per nudged episode into train)."""
    def _concat(keys: list[str]) -> pd.DataFrame:
        parts = [day_tables[k] for k in keys if k in day_tables and not day_tables[k].empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    train_all = _concat(train_days)
    test_all = _concat(test_days)
    holdout_all = _concat(holdout_days)

    train_part, val_part = temporal_train_val_split(train_all, val_fraction, per_day=val_block_per_day,
                                                    pre_onset_margin_s=pre_onset_margin_s)
    if horizon_k is not None:
        from nidra.data.labels import reattach_risk_label
        train_part = reattach_risk_label(train_part, horizon_k) if not train_part.empty else train_part
        val_part = reattach_risk_label(val_part, horizon_k) if not val_part.empty else val_part

    return SplitResult(train=train_part, val=val_part, test=test_all, holdout=holdout_all)


def assert_no_episode_leakage(splits: SplitResult) -> None:
    """Test-facing invariant: no (host_id, window_ts) pair from a non-benign
    episode appears in more than one of {train, val, test, holdout}."""
    named = {"train": splits.train, "val": splits.val, "test": splits.test, "holdout": splits.holdout}
    seen: dict[tuple, str] = {}
    for name, df in named.items():
        if df.empty:
            continue
        attack_rows = df.loc[df["stage_label"] != "benign", ["host_id", "window_ts"]]
        for host_id, window_ts in attack_rows.itertuples(index=False):
            key = (host_id, window_ts)
            if key in seen and seen[key] != name:
                raise AssertionError(f"Episode leakage: ({host_id}, {window_ts}) appears in both {seen[key]} and {name}")
            seen[key] = name


def assert_no_temporal_overlap(splits: SplitResult) -> None:
    """Train and val must not share window timestamps (per host, per day);
    a stricter, simpler check than full episode leakage."""
    if splits.train.empty or splits.val.empty:
        return
    train_keys = set(zip(splits.train["host_id"], splits.train["window_ts"]))
    val_keys = set(zip(splits.val["host_id"], splits.val["window_ts"]))
    overlap = train_keys & val_keys
    if overlap:
        raise AssertionError(f"Temporal overlap between train and val: {len(overlap)} shared (host, window) pairs")
