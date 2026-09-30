"""Windowing and state construction: flows (+ optional packets) -> the
canonical [host_id, window_ts, 45 features] state table.

Key invariants enforced here (see CLAUDE.md / IMPLEMENTATION-ML.md):
  - Window boundaries are aligned to absolute epoch multiples of
    WINDOW_SECONDS, never relative to the first observed event.
  - Empty windows are real data: a host with no source activity in a window
    still gets a row — a zero-valued vector with is_active=0 — never dropped.
  - new_peer_count / neighbour_risk_fraction are computed in one
    chronological forward pass (PeerTracker), never vectorized out of order.
  - Deltas and slopes are strictly backward-looking; sequence starts pad
    with zero, never with a future value.
  - Output columns == schema.FEATURE_ORDER, in that exact order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nidra.data.graph_features import PeerTracker, compute_window_graph, safe_div, shannon_entropy
from nidra.data.schema import FEATURE_ORDER, validate_feature_dict

logger = logging.getLogger(__name__)

_ZERO_FEATURES = {name: 0.0 for name in FEATURE_ORDER}

# The subset of features that get backward-looking delta/slope derivatives,
# per IMPLEMENTATION-ML.md §2.6 ("~12 most discriminative features").
_DYNAMICS_BASE = {
    "d_syn_ratio": "syn_ratio",
    "d_dst_port_entropy": "dst_port_entropy",
    "d_out_degree": "out_degree",
    "d_new_peer_count": "new_peer_count",
    "d_iat_var": "iat_var",
    "d_retrans_rate": "retrans_rate",
}
_SLOPE_BASE = {
    "slope3_syn_ratio": "syn_ratio",
    "slope3_dst_port_entropy": "dst_port_entropy",
    "slope3_out_degree": "out_degree",
    "slope3_iat_var": "iat_var",
}


def align_window(epoch: pd.Series, window_seconds: int) -> pd.Series:
    """Absolute epoch-multiple window alignment: floor(t / Delta) * Delta."""
    return (np.floor(epoch.astype("float64") / window_seconds) * window_seconds).astype("int64")


def _bin_entropy(values: pd.Series, bins: int = 16) -> float:
    values = values.dropna()
    if len(values) == 0:
        return 0.0
    if values.nunique() == 1:
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    return shannon_entropy(counts)


#: CIC-IDS2017 was captured at UNB in Fredericton, on Atlantic Daylight Time
#: (UTC-3) in July 2017. The published CSVs print that local wall clock; the
#: PCAPs carry true UTC epochs. Lifting the CSV clock by this many hours is
#: what puts a flow and its own packets on the same timeline.
CIC2017_UTC_OFFSET_HOURS: float = 3.0

#: The published CSVs also print a 12-hour clock and drop the AM/PM marker,
#: so an afternoon day-file writes 13:00 as "1:00". The captures run
#: 09:00-17:00 local, so the hours that actually occur are {8..12} in the
#: morning and {1..5} in the afternoon: 6 and 7 never appear and every hour
#: at or below this bound is unambiguously PM.
CIC2017_PM_HOUR_MAX: int = 7

#: Short name for the flow timebase the two constants above define. It goes
#: in the windowed-table cache key (train.pipeline.day_cache_path) because a
#: table built on a different timebase is a different table — nothing else in
#: that key would change, so a stale cache would otherwise be served silently.
CIC2017_TIMEBASE_TAG: str = "utc12h"


def parse_cic_timestamp(
    ts: pd.Series,
    *,
    twelve_hour: bool = True,
    utc_offset_hours: float = CIC2017_UTC_OFFSET_HOURS,
) -> pd.Series:
    """Parse the CICFlowMeter Timestamp column to epoch seconds (true UTC).

    CIC-IDS2017 releases are inconsistent about exact timestamp formatting
    across days; unparseable rows become NaT and are dropped by the caller
    (logged), never silently coerced to an arbitrary time.

    Two defects in the published CSVs are corrected here by default, because
    leaving either one in place is silent rather than loud — the timestamps
    stay well-formed and plausible, they just describe the wrong instant, and
    the (host_id, window_ts) join in join.py then quietly drops to a few
    percent of its rows with all packet-derived features zero-filled:

    `twelve_hour` undoes the missing AM/PM marker (see CIC2017_PM_HOUR_MAX)
    and `utc_offset_hours` undoes the local-time clock (see
    CIC2017_UTC_OFFSET_HOURS). Both are keyword arguments so a release that
    does not share these defects can be read literally, but no caller in this
    project should need to.
    """
    # CIC-IDS2017's "TrafficLabelling" release prints D/M/YYYY H:MM — MINUTE
    # resolution, no seconds field at all (every timestamp string in the
    # release is 13-14 characters). Any window shorter than 60 s therefore
    # cannot be populated at the sub-minute offsets: at Delta=30 s every :30
    # window was an artificial all-zero state. The canonical window is 60 s.
    parsed = pd.to_datetime(ts, errors="coerce", dayfirst=True)
    if twelve_hour:
        # Noon is already 12 on a 12-hour clock, so only hours at or below
        # the PM bound move; 12:xx must stay put rather than become 00:xx.
        is_pm = parsed.dt.hour <= CIC2017_PM_HOUR_MAX
        parsed = parsed + pd.to_timedelta(is_pm.astype("int64") * 12, unit="h")
    if utc_offset_hours:
        parsed = parsed + pd.Timedelta(hours=utc_offset_hours)
    epoch = pd.Series(np.nan, index=ts.index, dtype="float64")
    valid = parsed.notna()
    # `pd.to_datetime` does not always default to nanosecond resolution (it
    # has returned datetime64[us] on some pandas versions) — `.astype("int64")`
    # reinterprets the raw integer at WHATEVER resolution the array actually
    # has, so dividing by 10**9 silently produces the wrong epoch (off by
    # 1000x at microsecond resolution) unless the unit is pinned first. This
    # previously corrupted every window_ts derived from a flow timestamp,
    # collapsing all real per-host window counts far below any reasonable
    # min_windows_per_host and dropping 100% of hosts with no error raised.
    epoch.loc[valid] = parsed[valid].dt.as_unit("ns").astype("int64") // 10**9
    return epoch


def aggregate_flows(flows: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Vectorized per-(host, window) flow aggregation. `flows` must carry
    src_ip, dst_ip, dst_port, window_ts, and CICFlowMeter numeric columns.
    Host attribution is source-only in v1 (see join.py)."""
    f = flows.copy()
    f["total_packets"] = f["total_fwd_packets"].fillna(0) + f["total_bwd_packets"].fillna(0)
    f["total_bytes"] = f["total_len_fwd"].fillna(0) + f["total_len_bwd"].fillna(0)

    grouped = f.groupby(["src_ip", "window_ts"], sort=False)

    def _agg(g: pd.DataFrame) -> pd.Series:
        total_pkts = g["total_packets"].sum()
        return pd.Series({
            "syn_ratio": safe_div(g["syn_flag_count"].sum(), total_pkts),
            "ack_ratio": safe_div(g["ack_flag_count"].sum(), total_pkts),
            "rst_ratio": safe_div(g["rst_flag_count"].sum(), total_pkts),
            "fin_ratio": safe_div(g["fin_flag_count"].sum(), total_pkts),
            "psh_ratio": safe_div(g["psh_flag_count"].sum(), total_pkts),
            "urg_ratio": safe_div(g["urg_flag_count"].sum(), total_pkts),
            "bytes_total": g["total_bytes"].sum(),
            "bytes_up_down_ratio": safe_div(g["total_len_fwd"].sum(), g["total_len_bwd"].sum()),
            "pkts_per_flow_mean": g["total_packets"].mean(),
            "flow_duration_mean": g["flow_duration"].mean(),
            "flow_duration_var": g["flow_duration"].var(ddof=0) if len(g) > 1 else 0.0,
            "iat_mean": g["flow_iat_mean"].fillna(0).mean() if "flow_iat_mean" in g else 0.0,
            "iat_var": (g["flow_iat_mean"].fillna(0).var(ddof=0) if len(g) > 1 else 0.0) if "flow_iat_mean" in g else 0.0,
            "iat_max": g["flow_iat_max"].fillna(0).max() if "flow_iat_max" in g else 0.0,
            "active_flow_count": float(len(g)),
        })

    out = grouped.apply(_agg, include_groups=False).reset_index()
    out = out.rename(columns={"src_ip": "host_id"})
    return out


def aggregate_packets(packets: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Vectorized per-(host, window) packet aggregation. `packets` must carry
    ip_src, window_ts, ip_ttl, tcp_window_size_value, is_fragment,
    payload_len, is_retransmission. Returns an EMPTY-safe frame — if
    `packets` is empty (no PCAP available for this day), returns a
    zero-row frame and the caller falls back to flow-only mode."""
    cols = [
        "ttl_mean", "ttl_var", "tcp_window_mean", "tcp_window_entropy",
        "frag_flag_rate", "payload_size_mean", "payload_size_var",
        "payload_size_p95", "payload_size_entropy", "retrans_count", "retrans_rate",
    ]
    if packets.empty:
        return pd.DataFrame(columns=["host_id", "window_ts"] + cols)

    grouped = packets.groupby(["ip_src", "window_ts"], sort=False)

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        return pd.Series({
            "ttl_mean": g["ip_ttl"].mean(),
            "ttl_var": g["ip_ttl"].var(ddof=0) if n > 1 else 0.0,
            "tcp_window_mean": g["tcp_window_size_value"].mean(),
            "tcp_window_entropy": _bin_entropy(g["tcp_window_size_value"]),
            "frag_flag_rate": g["is_fragment"].mean(),
            "payload_size_mean": g["payload_len"].mean(),
            "payload_size_var": g["payload_len"].var(ddof=0) if n > 1 else 0.0,
            "payload_size_p95": g["payload_len"].quantile(0.95),
            "payload_size_entropy": _bin_entropy(g["payload_len"]),
            "retrans_count": g["is_retransmission"].sum(),
            "retrans_rate": safe_div(g["is_retransmission"].sum(), n),
        })

    out = grouped.apply(_agg, include_groups=False).reset_index()
    out = out.rename(columns={"ip_src": "host_id"})
    return out.fillna(0.0)


FLOW_FEATURES: list[str] = FEATURE_ORDER[0:15]
PACKET_FEATURES: list[str] = FEATURE_ORDER[15:26]
GRAPH_FEATURES: list[str] = FEATURE_ORDER[26:34]
assert FLOW_FEATURES[0] == "syn_ratio" and FLOW_FEATURES[-1] == "active_flow_count"
assert PACKET_FEATURES[0] == "ttl_mean" and PACKET_FEATURES[-1] == "retrans_rate"
assert GRAPH_FEATURES[0] == "out_degree" and GRAPH_FEATURES[-1] == "reciprocity"

#: Tag for the flow/packet fusion rule below; part of the windowed-table cache
#: key (train.pipeline.day_cache_path) because a table built under the old
#: left join is a different table for the same day.
FUSION_TAG: str = "fuse2"


def _fuse_flow_and_packet_windows(flow_agg: pd.DataFrame, packet_agg: pd.DataFrame) -> pd.DataFrame:
    """Join the per-(host, window) flow aggregates and packet aggregates.

    A CICFlowMeter flow is timestamped at its START, so a host with one
    long-lived connection (an SSH session, a slowloris socket, a large
    download) starts a flow in one window and then sends packets for many
    windows with no new flow start at all. The original left join from flows
    onto packets emitted every such window as an all-zero `is_active=0`
    state — on Tuesday of CIC-IDS2017 that was 24,377 host-minutes, 39% of
    the minutes in which a host was actually transmitting. Those minutes are
    real state, so the join is outer, with two restrictions that keep the
    host population and the day's extent defined by the flow records:

      - only hosts that appear as a flow SOURCE in this day file are kept
        (labels are keyed on the flow source, and a host that never initiates
        a flow — a pure responder — has no label semantics here);
      - only windows inside the day file's own flow time range are kept, so
        a day file that covers the morning does not absorb the afternoon's
        packets from the shared full-day PCAP.

    Windows with packets but no flow start carry their packet aggregates
    and zero flow aggregates (`active_flow_count` counts flow STARTS in the
    window — a window with an ongoing flow but no new one legitimately has
    zero); windows with a flow start but no packets (PCAP gap) carry zero
    packet aggregates, as before.
    """
    if packet_agg.empty or flow_agg.empty:
        return flow_agg.merge(packet_agg, on=["host_id", "window_ts"], how="left")
    flow_hosts = set(flow_agg["host_id"])
    lo, hi = flow_agg["window_ts"].min(), flow_agg["window_ts"].max()
    packet_kept = packet_agg[
        packet_agg["host_id"].isin(flow_hosts) & (packet_agg["window_ts"] >= lo) & (packet_agg["window_ts"] <= hi)
    ]
    merged = flow_agg.merge(packet_kept, on=["host_id", "window_ts"], how="outer")
    n_packet_only = len(merged) - len(flow_agg)
    logger.info(
        "_fuse_flow_and_packet_windows: %d flow-start windows + %d packet-only windows of flow-source hosts "
        "(%d packet windows outside the flow host set / time range ignored)",
        len(flow_agg), n_packet_only, len(packet_agg) - len(packet_kept),
    )
    return merged


def build_state_rows(
    flows: pd.DataFrame,
    packets: pd.DataFrame,
    window_seconds: int = 30,
    graph_degree_cap: int = 200,
) -> pd.DataFrame:
    """Full state-construction pipeline for one day/file.

    `flows` requires: src_ip, dst_ip, dst_port, window_ts, plus CICFlowMeter
    numeric columns (see flow_load.py). `packets` requires the columns
    produced by pcap_extract.py, or may be an empty DataFrame (flow-only
    mode; packet features zero-filled, logged explicitly).
    """
    if packets is None or packets.empty:
        logger.warning(
            "build_state_rows: no packet-level data supplied — running in "
            "FLOW-ONLY mode. Packet aggregate features (ttl_*, tcp_window_*, "
            "frag_flag_rate, payload_size_*, retrans_*) will be zero for "
            "every window. This must be reported, not silently treated as "
            "full-feature data."
        )

    flow_agg = aggregate_flows(flows, window_seconds)
    packet_agg = aggregate_packets(packets if packets is not None else pd.DataFrame(), window_seconds)

    merged = _fuse_flow_and_packet_windows(flow_agg, packet_agg)
    for col in PACKET_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    for col in FLOW_FEATURES:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # --- chronological pass: graph scalars + stateful peer tracker ---
    tracker = PeerTracker()
    window_order = sorted(merged["window_ts"].unique())
    flows_by_window = {w: g[["src_ip", "dst_ip", "dst_port"]] for w, g in flows.groupby("window_ts", sort=False)}
    hosts_by_window = {w: set(g["host_id"]) for w, g in merged.groupby("window_ts", sort=False)}
    empty_flows = flows.iloc[0:0][["src_ip", "dst_ip", "dst_port"]]
    graph_rows: list[dict] = []
    for w in window_order:
        w_flows = flows_by_window.get(w, empty_flows)
        graph = compute_window_graph(w_flows, degree_cap=graph_degree_cap)

        active_hosts = hosts_by_window[w]
        for h in active_hosts:
            peers = graph.out_peers.get(h, set())
            row = {
                "host_id": h,
                "window_ts": w,
                "out_degree": graph.out_degree.get(h, 0),
                "in_degree": graph.in_degree.get(h, 0),
                "dst_ip_entropy": graph.dst_ip_entropy.get(h, 0.0),
                "dst_port_entropy": graph.dst_port_entropy.get(h, 0.0),
                "local_clustering_coeff": graph.local_clustering_coeff.get(h, 0.0),
                "reciprocity": graph.reciprocity.get(h, 0.0),
                "new_peer_count": tracker.new_peer_count(h, peers),
                "neighbour_risk_fraction": tracker.neighbour_risk_fraction(peers),
            }
            graph_rows.append(row)
        tracker.commit_window(graph.out_degree)

    graph_df = pd.DataFrame(graph_rows, columns=["host_id", "window_ts"] + GRAPH_FEATURES)
    merged = merged.merge(graph_df, on=["host_id", "window_ts"], how="left")
    # Every row in `merged` is a window in which this host was observed
    # sending something — a flow start, packets of an ongoing flow, or both.
    merged["is_active"] = 1.0

    # --- fill empty windows per host across its active range ---
    filled = _fill_empty_windows(merged, window_seconds)

    # --- deltas and slopes, strictly backward-looking, per host ---
    filled = _add_dynamics(filled)

    # --- assemble canonical column order ---
    for name in FEATURE_ORDER:
        if name not in filled.columns:
            filled[name] = 0.0
    filled = filled.fillna(0.0)
    ordered = filled[["host_id", "window_ts"] + FEATURE_ORDER].sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    return ordered


def _fill_empty_windows(df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Emit zero-valued, is_active=0 rows for every window in
    [host_min, host_max] not already present. Empty windows are real data
    and must never be dropped."""
    filled_parts = []
    for host, g in df.groupby("host_id", sort=False):
        lo, hi = g["window_ts"].min(), g["window_ts"].max()
        full_index = np.arange(lo, hi + window_seconds, window_seconds)
        g_idx = g.set_index("window_ts").reindex(full_index)
        g_idx["host_id"] = host
        for name in FEATURE_ORDER:
            if name == "is_active":
                continue
            if name in g_idx.columns:
                g_idx[name] = g_idx[name].fillna(0.0)
            else:
                g_idx[name] = 0.0
        g_idx["is_active"] = g_idx["is_active"].fillna(0.0)
        g_idx.index.name = "window_ts"
        filled_parts.append(g_idx.reset_index())
    return pd.concat(filled_parts, ignore_index=True) if filled_parts else df


def _add_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """d_x[t] = x[t]-x[t-1]; slope3_x[t] = OLS slope over x[t-2..t]. Padded
    with zero at sequence start — never with a future value."""
    df = df.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    out_parts = []
    for host, g in df.groupby("host_id", sort=False):
        g = g.reset_index(drop=True).copy()
        for dname, base in _DYNAMICS_BASE.items():
            g[dname] = g[base].diff().fillna(0.0)
        for sname, base in _SLOPE_BASE.items():
            g[sname] = _rolling_ols_slope3(g[base].to_numpy())
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def _rolling_ols_slope3(x: np.ndarray) -> np.ndarray:
    """OLS slope over the trailing 3-point window x[t-2..t]. First two points
    (insufficient history) are zero-padded, never computed from future
    values."""
    n = len(x)
    slopes = np.zeros(n)
    t = np.array([0.0, 1.0, 2.0])
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum()
    for i in range(2, n):
        y = x[i - 2 : i + 1]
        y_mean = y.mean()
        slopes[i] = ((t - t_mean) * (y - y_mean)).sum() / denom if denom > 0 else 0.0
    return slopes


def windowize_day(
    flows: pd.DataFrame,
    packets: pd.DataFrame,
    window_seconds: int,
    min_windows_per_host: int,
) -> pd.DataFrame:
    """Top-level entry: builds state rows and filters hosts below the
    minimum window count. The canonical cached table is built with
    `min_windows_per_host=1` (no host filter) and the geometry-dependent
    L + K filter is applied when a split is assembled (train.pipeline),
    so one cached table serves every history/horizon configuration."""
    states = build_state_rows(flows, packets, window_seconds=window_seconds)
    if min_windows_per_host <= 1:
        return states
    counts = states.groupby("host_id")["window_ts"].transform("count")
    kept = states[counts >= min_windows_per_host].reset_index(drop=True)
    n_dropped_hosts = states["host_id"].nunique() - kept["host_id"].nunique()
    logger.info(
        "windowize_day: %d hosts kept, %d hosts dropped for <%d windows",
        kept["host_id"].nunique(), n_dropped_hosts, min_windows_per_host,
    )
    return kept
