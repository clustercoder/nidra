"""Synthetic fixture generation for structural/unit tests.

Distinguished explicitly from real-data evaluation: nothing here is CIC-
IDS2017 traffic. It exists so the pipeline's mechanics (windowing, leakage
controls, rollout shapes, frozen heads, ...) can be verified deterministically
and fast, independent of whether real datasets are present on the machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nidra.data.schema import FEATURE_ORDER, WINDOW_SECONDS

RNG = np.random.default_rng(42)


def make_synthetic_flows(
    n_hosts: int = 4,
    n_windows: int = 80,
    window_seconds: int = WINDOW_SECONDS,
    start_epoch: int = 1_700_000_000,
    portscan_host_idx: int = 0,
    portscan_start_window: int = 40,
    portscan_len: int = 15,
    flows_per_host_window: int = 3,
) -> pd.DataFrame:
    """Builds a flow-level dataframe already shaped like the output of
    flow_load + join (src_ip, dst_ip, dst_port, timestamp, window_ts, label,
    and the numeric CICFlowMeter columns windowize.aggregate_flows expects).

    One host (`portscan_host_idx`) escalates into a port-scan for
    `portscan_len` windows starting at `portscan_start_window`: rising
    distinct-destination-port count, near-zero payload, SYN-heavy.
    """
    start_epoch = int(np.floor(start_epoch / window_seconds) * window_seconds)
    hosts = [f"10.0.0.{i+1}" for i in range(n_hosts)]
    rows = []

    for h_idx, host in enumerate(hosts):
        is_scanner = h_idx == portscan_host_idx
        for w in range(n_windows):
            window_ts = start_epoch + w * window_seconds
            in_scan = is_scanner and (portscan_start_window <= w < portscan_start_window + portscan_len)
            n_flows = flows_per_host_window if not in_scan else flows_per_host_window * 8
            for f in range(n_flows):
                ts_within = window_ts + RNG.uniform(0, window_seconds)
                if in_scan:
                    dst_ip = f"172.16.0.{RNG.integers(1, 250)}"
                    dst_port = int(RNG.integers(1, 65000))
                    total_fwd = 1
                    total_bwd = 0
                    len_fwd = 40.0
                    len_bwd = 0.0
                    syn, ack, rst, fin, psh, urg = 1, 0, 0, 0, 0, 0
                    duration = float(RNG.uniform(0.5, 3.0))
                    label = "PortScan"
                else:
                    dst_ip = f"172.16.0.{RNG.integers(1, 5)}"
                    dst_port = int(RNG.choice([80, 443, 22]))
                    total_fwd = int(RNG.integers(2, 10))
                    total_bwd = int(RNG.integers(2, 10))
                    len_fwd = float(RNG.uniform(100, 800))
                    len_bwd = float(RNG.uniform(100, 800))
                    syn, ack, rst, fin, psh, urg = 1, int(RNG.integers(0, 3)), 0, int(RNG.integers(0, 2)), int(RNG.integers(0, 2)), 0
                    duration = float(RNG.uniform(10, 500))
                    label = "BENIGN"

                rows.append({
                    "src_ip": host,
                    "dst_ip": dst_ip,
                    "src_port": int(RNG.integers(1024, 65000)),
                    "dst_port": dst_port,
                    "protocol": 6,
                    "timestamp": pd.Timestamp(ts_within, unit="s"),
                    "epoch": ts_within,
                    "window_ts": window_ts,
                    "flow_duration": duration,
                    "total_fwd_packets": total_fwd,
                    "total_bwd_packets": total_bwd,
                    "total_len_fwd": len_fwd,
                    "total_len_bwd": len_bwd,
                    "flow_iat_mean": float(RNG.uniform(0, 50)),
                    "flow_iat_std": float(RNG.uniform(0, 10)),
                    "flow_iat_max": float(RNG.uniform(0, 100)),
                    "syn_flag_count": syn,
                    "ack_flag_count": ack,
                    "rst_flag_count": rst,
                    "fin_flag_count": fin,
                    "psh_flag_count": psh,
                    "urg_flag_count": urg,
                    "label": label,
                })

    return pd.DataFrame(rows)


def make_synthetic_packets(flows: pd.DataFrame) -> pd.DataFrame:
    """Derives a small synthetic packet-level table consistent with the flow
    fixture, for exercising the packet-aggregate path structurally."""
    rows = []
    for _, f in flows.iterrows():
        n_pkts = max(1, int(f["total_fwd_packets"] + f["total_bwd_packets"]))
        for _ in range(min(n_pkts, 5)):
            rows.append({
                "frame_time_epoch": f["epoch"],
                "window_ts": f["window_ts"],
                "ip_src": f["src_ip"],
                "ip_dst": f["dst_ip"],
                "ip_ttl": float(RNG.integers(48, 64)),
                "tcp_window_size_value": float(RNG.integers(1000, 65000)),
                "is_fragment": 0,
                "payload_len": float(RNG.uniform(0, 100)) if f["label"] == "PortScan" else float(RNG.uniform(100, 1400)),
                "is_retransmission": int(RNG.random() < 0.02),
            })
    return pd.DataFrame(rows)
