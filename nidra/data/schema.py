"""Canonical feature and stage vocabulary for NIDRA.

This file is the single definition of feature order for the whole project — the ML
pipeline builds state vectors in this order, the services validate against it, and the
SHAP display labels from it. Feature-name drift between training and display is the bug
that eats an afternoon late in a build; importing this list is how it is avoided.

Never construct feature order from a dict iteration, never hardcode an index, never
restate the list anywhere else.
"""

from __future__ import annotations

#: Ordered feature names of a per-host, per-window state vector.
#: 15 flow aggregates + 11 packet aggregates + 8 graph scalars + 10 dynamics + activity.
FEATURE_ORDER: list[str] = [
    # --- flow aggregates (15) ---
    "syn_ratio",
    "ack_ratio",
    "rst_ratio",
    "fin_ratio",
    "psh_ratio",
    "urg_ratio",
    "bytes_total",
    "bytes_up_down_ratio",
    "pkts_per_flow_mean",
    "flow_duration_mean",
    "flow_duration_var",
    "iat_mean",
    "iat_var",
    "iat_max",
    "active_flow_count",
    # --- packet aggregates (11) ---
    "ttl_mean",
    "ttl_var",
    "tcp_window_mean",
    "tcp_window_entropy",
    "frag_flag_rate",
    "payload_size_mean",
    "payload_size_var",
    "payload_size_p95",
    "payload_size_entropy",
    "retrans_count",
    "retrans_rate",
    # --- graph scalars (8) ---
    "out_degree",
    "in_degree",
    "dst_ip_entropy",
    "dst_port_entropy",
    "new_peer_count",
    "neighbour_risk_fraction",
    "local_clustering_coeff",
    "reciprocity",
    # --- dynamics (10) ---
    "d_syn_ratio",
    "d_dst_port_entropy",
    "d_out_degree",
    "d_new_peer_count",
    "d_iat_var",
    "d_retrans_rate",
    "slope3_syn_ratio",
    "slope3_dst_port_entropy",
    "slope3_out_degree",
    "slope3_iat_var",
    # --- activity (1) ---
    "is_active",
]

assert len(FEATURE_ORDER) == 45

#: Attack-stage vocabulary. `stage_dist` on a forecast is a distribution over these.
STAGES: list[str] = ["benign", "recon", "initial_access", "lateral", "c2", "exfil"]
