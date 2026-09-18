"""Canonical NIDRA state-vector schema.

FEATURE_ORDER is the single source of truth for the 45-dimensional per-host,
per-window state vector. Every other module (windowing, normalization,
training, serving, explainability) imports this list rather than
reconstructing feature order from a dict. Never rely on dict iteration order
to determine feature position — dict insertion order is an implementation
detail, this list is the contract.
"""

from __future__ import annotations

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

assert len(FEATURE_ORDER) == 45, f"FEATURE_ORDER must have 45 entries, has {len(FEATURE_ORDER)}"
assert len(set(FEATURE_ORDER)) == 45, "FEATURE_ORDER contains duplicate feature names"

FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_ORDER)}

# Features that are strictly non-negative counts/magnitudes and heavy-tailed;
# log1p is applied to these before RobustScaler fitting (see normalize.py).
LOG1P_FEATURES: list[str] = [
    "bytes_total",
    "active_flow_count",
    "out_degree",
    "new_peer_count",
    "retrans_count",
]
for f in LOG1P_FEATURES:
    assert f in FEATURE_INDEX, f"LOG1P_FEATURES entry {f!r} not in FEATURE_ORDER"

# Windowing / rollout geometry — the other non-negotiable constants shared
# across the data pipeline, model, training, and serving code.
WINDOW_SECONDS: int = 30          # Delta
CONTEXT_LENGTH: int = 30          # L windows of history (15 min)
HORIZON_LENGTH: int = 6           # K windows of forecast (3 min)

STAGE_LABELS: list[str] = [
    "benign",
    "recon",
    "initial_access",
    "lateral",
    "c2",
    "exfil",
]
STAGE_INDEX: dict[str, int] = {name: i for i, name in enumerate(STAGE_LABELS)}

# Alias for the backend's own name for this list (nidra_common/schemas.py,
# services/inference/stub_predictor.py) — both names refer to the exact
# same six-stage taxonomy; kept as an alias rather than a rename so neither
# side of the ML/backend boundary needed to change on integration.
STAGES: list[str] = STAGE_LABELS

SCHEMA_VERSION = "1.0"


def validate_feature_dict(features: dict[str, float]) -> None:
    """Fail loudly if a feature dict does not exactly match FEATURE_ORDER.

    This is the check every producer (windowizer, feature service, test
    fixtures) must run before publishing or consuming a state vector.
    """
    got = set(features.keys())
    want = set(FEATURE_ORDER)
    if got != want:
        missing = want - got
        extra = got - want
        raise ValueError(
            f"Feature dict does not match FEATURE_ORDER. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def validate_state_array_width(width: int) -> None:
    """Fail loudly if a raw state array's feature axis does not equal 45."""
    if width != len(FEATURE_ORDER):
        raise ValueError(
            f"State array width {width} does not match FEATURE_ORDER length "
            f"{len(FEATURE_ORDER)}"
        )
