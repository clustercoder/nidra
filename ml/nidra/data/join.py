"""Join packet aggregates and flow aggregates to (host_id, window_ts).

Per IMPLEMENTATION-ML.md §2.4, this deliberately avoids five-tuple
packet<->flow matching. Instead:

    1. Assign every packet to (host, window) independently.
    2. Assign every flow to (host, window) independently.
    3. Aggregate each separately (windowize.py).
    4. Join the two aggregate tables on (host_id, window_ts).

Host attribution is source-only in v1: a flow/packet contributes to the
state of its source host. The code is structured so bidirectional
(destination-host) attribution can be added later without restructuring —
see `direction` handling below — but v1 only emits source-host rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nidra.data.windowize import align_window, parse_cic_timestamp

logger = logging.getLogger(__name__)


def prepare_flows_for_windowing(flows: pd.DataFrame, window_seconds: int,
                                timestamp_is_epoch: bool = False, cic_clock_correction: bool = True) -> pd.DataFrame:
    """Attach a window_ts to each flow record. Drops rows with unparseable
    timestamps or missing src_ip/dst_ip (logged, not silently kept).

    `timestamp_is_epoch`: the `timestamp` column already holds UTC epoch
    seconds (flows assembled from a pcap by flow_assemble.py) — no parsing,
    and NO CIC-IDS2017 clock correction. `cic_clock_correction=False` reads
    a CICFlowMeter text timestamp literally, for captures that do not share
    the 2017 release's 12-hour/local-clock defects (see parse_cic_timestamp)."""
    df = flows.copy()
    required = ["src_ip", "dst_ip", "dst_port", "timestamp"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"prepare_flows_for_windowing: missing required columns {missing_cols}. "
            "This CICFlowMeter release likely lacks IP/timestamp columns — use the "
            "'TrafficLabelling' distribution, not 'MachineLearningCVE'."
        )

    if timestamp_is_epoch:
        epoch = pd.to_numeric(df["timestamp"], errors="coerce").astype("float64")
    elif cic_clock_correction:
        epoch = parse_cic_timestamp(df["timestamp"])
    else:
        epoch = parse_cic_timestamp(df["timestamp"], twelve_hour=False, utc_offset_hours=0.0)
    n_before = len(df)
    valid = epoch.notna() & df["src_ip"].notna() & df["dst_ip"].notna()
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.warning(
            "prepare_flows_for_windowing: dropping %d/%d rows with unparseable "
            "timestamp or missing IP", n_dropped, n_before,
        )
    df = df.loc[valid].copy()
    df["epoch"] = epoch.loc[valid]
    df["window_ts"] = align_window(df["epoch"], window_seconds)
    return df


def prepare_packets_for_windowing(packets: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Attach a window_ts to each packet record. Empty input passes through
    unchanged (flow-only mode is a valid input to this function)."""
    if packets is None or packets.empty:
        return pd.DataFrame()
    df = packets.copy()
    df["window_ts"] = align_window(df["frame_time_epoch"], window_seconds)
    return df


def build_day_inputs(
    flows_raw: pd.DataFrame,
    packets_raw: pd.DataFrame | None,
    window_seconds: int,
    timestamp_is_epoch: bool = False,
    cic_clock_correction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-level join entry for one day/file: returns (flows_windowed,
    packets_windowed) ready for windowize.build_state_rows."""
    flows_w = prepare_flows_for_windowing(flows_raw, window_seconds, timestamp_is_epoch=timestamp_is_epoch,
                                          cic_clock_correction=cic_clock_correction)
    packets_w = prepare_packets_for_windowing(packets_raw, window_seconds) if packets_raw is not None else pd.DataFrame()
    return flows_w, packets_w
