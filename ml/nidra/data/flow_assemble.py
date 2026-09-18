"""Reconstruct CICFlowMeter-shaped flow records from raw packets.

The 15 flow features in the 45-dimensional state vector were trained on
CICFlowMeter's CSV output. A pcap uploaded by a user has no such CSV, so the
flows have to be rebuilt from packets before `windowize.build_state_rows`
can produce a state table — it left-joins packet aggregates onto flow
aggregates, so with no flows there are no rows at all.

The risk this module is written against is silent drift: if these flows
differ systematically from CICFlowMeter's, every flow feature shifts and the
model degrades with nothing raising an error. The semantics below were
calibrated against the real CIC-IDS2017 Friday capture, where both the pcap
and the CICFlowMeter CSV are available:

  * `flow_duration` and the IAT columns are in MICROSECONDS. A CSV flow
    spanning 112.74 seconds records 112,740,690.
  * Flows are bidirectional. The key is the unordered {(ip, port), (ip, port)}
    pair plus protocol, and "forward" is whichever endpoint sent first.
  * A 5-tuple recurs many times in one capture. In the calibration slice a
    single tuple carried 370 packets across 728 seconds, which CICFlowMeter
    reports as many separate flows. Packets are therefore split on TCP
    teardown (FIN/RST) and on an idle timeout, not collapsed into one
    long-lived flow — collapsing them would inflate `flow_duration_mean` and
    deflate `active_flow_count` for every window the host appears in.

Exact per-flow agreement with CICFlowMeter is not achievable from packets
alone (its splitting also depends on internal activity timers), and is not
the target. What matters is that the per-(host, window) aggregates this
feeds are distributed like the ones the model trained on; see
`scripts/validate_flow_assembly.py`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TCP flag bits, as carried in the `tcp_flags` hex string from pcap_extract.
FIN, SYN, RST, PSH, ACK, URG = "fin", "syn", "rst", "psh", "ack", "urg"
FLAG_BITS: dict[str, int] = {FIN: 0x01, SYN: 0x02, RST: 0x04, PSH: 0x08, ACK: 0x10, URG: 0x20}

#: CICFlowMeter's default idle timeout, in seconds.
DEFAULT_IDLE_TIMEOUT_S = 120.0

FLOW_COLUMNS = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp",
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_len_fwd", "total_len_bwd",
    "syn_flag_count", "ack_flag_count", "rst_flag_count",
    "fin_flag_count", "psh_flag_count", "urg_flag_count",
    "flow_iat_mean", "flow_iat_max",
]

_US = 1_000_000.0


def decode_flag_bits(flags: pd.Series) -> dict[str, np.ndarray]:
    """Hex strings like "0x0018" -> a boolean array per TCP flag.

    Missing/empty values are false across the board rather than NaN: UDP and
    ICMP packets carry no flags and must still take part in a flow.
    """
    as_int = pd.to_numeric(flags, errors="coerce").fillna(0).astype("int64")
    if as_int.eq(0).all() and flags.notna().any():
        # Values arrive as "0x0018" strings, which to_numeric cannot read.
        as_int = (flags.fillna("").astype(str).str.strip()
                  .replace("", "0").apply(lambda v: int(v, 16) if v not in ("", "0") else 0)
                  .astype("int64"))
    return {name: (as_int.to_numpy() & bit).astype(bool) for name, bit in FLAG_BITS.items()}


def _flow_key(p: pd.DataFrame) -> pd.Series:
    """Direction-independent connection key, so both halves of an exchange
    land in the same flow."""
    a = p.ip_src.astype(str) + ":" + p.src_port.fillna(0).astype("int64").astype(str)
    b = p.ip_dst.astype(str) + ":" + p.dst_port.fillna(0).astype("int64").astype(str)
    lo = np.where(a.to_numpy() <= b.to_numpy(), a.to_numpy(), b.to_numpy())
    hi = np.where(a.to_numpy() <= b.to_numpy(), b.to_numpy(), a.to_numpy())
    return pd.Series(lo, index=p.index) + "|" + pd.Series(hi, index=p.index) + "|" + p.ip_proto.fillna(0).astype("int64").astype(str)


def _segment_ids(times: np.ndarray, closed: np.ndarray, idle_timeout_s: float) -> np.ndarray:
    """Split one connection's packets into flows.

    A new flow starts when the gap since the previous packet exceeds the idle
    timeout, or when the previous packet tore the connection down (FIN/RST) —
    the next packet on a reused tuple is a new connection, not a continuation.
    """
    if len(times) == 0:
        return np.empty(0, dtype="int64")
    gap = np.diff(times, prepend=times[0])
    new_flow = gap > idle_timeout_s
    new_flow[1:] |= closed[:-1]
    new_flow[0] = False
    return np.cumsum(new_flow)


def assemble_flows(packets: pd.DataFrame,
                    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S) -> pd.DataFrame:
    """Packets (as produced by `pcap_extract`) -> one row per flow, carrying
    the columns `windowize.aggregate_flows` consumes."""
    if packets is None or len(packets) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in FLOW_COLUMNS})

    p = packets.copy()
    for col, default in (("payload_len", 0.0), ("frame_len", 0.0), ("src_port", 0), ("dst_port", 0)):
        if col not in p.columns:
            p[col] = default
    p["src_port"] = p["src_port"].fillna(0).astype("int64")
    p["dst_port"] = p["dst_port"].fillna(0).astype("int64")
    p["ip_proto"] = p.get("ip_proto", pd.Series(0, index=p.index)).fillna(0).astype("int64")

    bits = decode_flag_bits(p.get("tcp_flags", pd.Series("", index=p.index)))
    for name, arr in bits.items():
        p[f"_{name}"] = arr
    p["_closed"] = p["_fin"] | p["_rst"]

    p["_key"] = _flow_key(p)
    p = p.sort_values(["_key", "frame_time_epoch"], kind="mergesort").reset_index(drop=True)

    seg = np.concatenate([
        _segment_ids(g.frame_time_epoch.to_numpy(), g._closed.to_numpy(), idle_timeout_s)
        for _, g in p.groupby("_key", sort=False)
    ]) if len(p) else np.empty(0, dtype="int64")
    p["_flow"] = p["_key"] + "#" + pd.Series(seg, index=p.index).astype(str)

    # Forward = the direction of the flow's first packet.
    first = p.groupby("_flow", sort=False).head(1)[["_flow", "ip_src", "src_port", "ip_dst", "dst_port"]]
    first = first.rename(columns={"ip_src": "_f_src", "src_port": "_f_sport",
                                   "ip_dst": "_f_dst", "dst_port": "_f_dport"})
    p = p.merge(first, on="_flow", how="left")
    p["_is_fwd"] = (p.ip_src == p._f_src) & (p.src_port == p._f_sport)

    p["_fwd_len"] = np.where(p._is_fwd, p.payload_len.fillna(0.0), 0.0)
    p["_bwd_len"] = np.where(~p._is_fwd, p.payload_len.fillna(0.0), 0.0)

    grouped = p.groupby("_flow", sort=False)
    out = grouped.agg(
        src_ip=("_f_src", "first"), src_port=("_f_sport", "first"),
        dst_ip=("_f_dst", "first"), dst_port=("_f_dport", "first"),
        protocol=("ip_proto", "first"),
        timestamp=("frame_time_epoch", "min"),
        _t_last=("frame_time_epoch", "max"),
        total_fwd_packets=("_is_fwd", "sum"),
        _n=("_is_fwd", "size"),
        total_len_fwd=("_fwd_len", "sum"),
        total_len_bwd=("_bwd_len", "sum"),
        syn_flag_count=("_syn", "sum"), ack_flag_count=("_ack", "sum"),
        rst_flag_count=("_rst", "sum"), fin_flag_count=("_fin", "sum"),
        psh_flag_count=("_psh", "sum"), urg_flag_count=("_urg", "sum"),
    ).reset_index(drop=True)

    out["total_bwd_packets"] = out["_n"] - out["total_fwd_packets"]
    out["flow_duration"] = (out["_t_last"] - out["timestamp"]) * _US

    # IAT over the flow's packets in time order, both directions, in
    # microseconds. A single-packet flow has no interval: 0, never NaN, which
    # would otherwise propagate into the window aggregate.
    iat = grouped.frame_time_epoch.apply(
        lambda s: pd.Series({"m": 0.0, "x": 0.0}) if len(s) < 2
        else pd.Series({"m": float(np.diff(np.sort(s.to_numpy())).mean() * _US),
                        "x": float(np.diff(np.sort(s.to_numpy())).max() * _US)})
    ).unstack()
    out["flow_iat_mean"] = iat["m"].to_numpy()
    out["flow_iat_max"] = iat["x"].to_numpy()

    for c in ("total_fwd_packets", "total_bwd_packets", "syn_flag_count", "ack_flag_count",
              "rst_flag_count", "fin_flag_count", "psh_flag_count", "urg_flag_count"):
        out[c] = out[c].astype("int64")

    logger.info("assemble_flows: %d packets -> %d flows", len(p), len(out))
    return out[FLOW_COLUMNS]
