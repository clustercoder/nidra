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

from horizon.data.graph_features import PeerTracker, compute_window_graph, safe_div, shannon_entropy
from horizon.data.schema import FEATURE_ORDER, validate_feature_dict

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


def parse_cic_timestamp(ts: pd.Series) -> pd.Series:
    """Parse the CICFlowMeter Timestamp column to epoch seconds (UTC-naive).

    CIC-IDS2017 releases are inconsistent about exact timestamp formatting
    across days; unparseable rows become NaT and are dropped by the caller
    (logged), never silently coerced to an arbitrary time.
    """
    # CIC-IDS2017's "TrafficLabelling" release uses DD/MM/YYYY HH:MM:SS.
    parsed = pd.to_datetime(ts, errors="coerce", dayfirst=True)
    epoch = pd.Series(np.nan, index=ts.index, dtype="float64")
    valid = parsed.notna()
    epoch.loc[valid] = parsed[valid].astype("int64") // 10**9
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

    merged = flow_agg.merge(packet_agg, on=["host_id", "window_ts"], how="left")
    for col in ["ttl_mean", "ttl_var", "tcp_window_mean", "tcp_window_entropy", "frag_flag_rate",
                "payload_size_mean", "payload_size_var", "payload_size_p95", "payload_size_entropy",
                "retrans_count", "retrans_rate"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    # --- chronological pass: graph scalars + stateful peer tracker ---
    tracker = PeerTracker()
    window_order = sorted(flows["window_ts"].unique())
    graph_rows: list[dict] = []
    for w in window_order:
        w_flows = flows.loc[flows["window_ts"] == w, ["src_ip", "dst_ip", "dst_port"]]
        graph = compute_window_graph(w_flows, degree_cap=graph_degree_cap)

        active_hosts = set(merged.loc[merged["window_ts"] == w, "host_id"])
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

    graph_df = pd.DataFrame(graph_rows)
    merged = merged.merge(graph_df, on=["host_id", "window_ts"], how="left")
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
    minimum window count (L + K)."""
    states = build_state_rows(flows, packets, window_seconds=window_seconds)
    counts = states.groupby("host_id")["window_ts"].transform("count")
    kept = states[counts >= min_windows_per_host].reset_index(drop=True)
    n_dropped_hosts = states["host_id"].nunique() - kept["host_id"].nunique()
    logger.info(
        "windowize_day: %d hosts kept, %d hosts dropped for <%d windows",
        kept["host_id"].nunique(), n_dropped_hosts, min_windows_per_host,
    )
    return kept
