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

# Per-feature preprocessing (see normalize.py for what each kind does):
#   log1p  — non-negative, heavy-tailed counts / bytes / durations / variances
#   asinh  — SIGNED heavy-tailed quantities (backward deltas and slopes of counts)
#   zscore — unbounded or wide-range but not heavy-tailed (TTL, entropies, their deltas)
#   unit   — already on a bounded [0,1] / [-1,1] scale (rates, fractions, is_active); left as is
# Constant and exact-duplicate features are detected at fit time on the
# training data and dropped there, not declared here — the data decides.
# The choice per feature is checked, not assumed: nidra/data/preprocessing_audit.py
# measures skew and clip saturation before/after and flags any feature whose
# declared kind is not the best of {none, log1p, asinh} on the training rows.
FEATURE_TRANSFORMS: dict[str, str] = {
    # flow aggregates
    "syn_ratio": "unit", "ack_ratio": "unit", "rst_ratio": "unit", "fin_ratio": "unit",
    "psh_ratio": "unit", "urg_ratio": "unit",
    "bytes_total": "log1p", "bytes_up_down_ratio": "log1p", "pkts_per_flow_mean": "log1p",
    "flow_duration_mean": "log1p", "flow_duration_var": "log1p",
    "iat_mean": "log1p", "iat_var": "log1p", "iat_max": "log1p",
    "active_flow_count": "log1p",
    # packet aggregates
    "ttl_mean": "zscore", "ttl_var": "log1p", "tcp_window_mean": "log1p", "tcp_window_entropy": "zscore",
    "frag_flag_rate": "unit",
    "payload_size_mean": "log1p", "payload_size_var": "log1p", "payload_size_p95": "log1p",
    "payload_size_entropy": "zscore",
    "retrans_count": "log1p", "retrans_rate": "unit",
    # graph scalars
    "out_degree": "log1p", "in_degree": "log1p", "dst_ip_entropy": "zscore", "dst_port_entropy": "zscore",
    "new_peer_count": "log1p", "neighbour_risk_fraction": "unit", "local_clustering_coeff": "unit",
    "reciprocity": "unit",
    # dynamics (signed)
    "d_syn_ratio": "unit", "d_dst_port_entropy": "zscore", "d_out_degree": "asinh",
    "d_new_peer_count": "asinh", "d_iat_var": "asinh", "d_retrans_rate": "unit",
    "slope3_syn_ratio": "unit", "slope3_dst_port_entropy": "zscore", "slope3_out_degree": "asinh",
    "slope3_iat_var": "asinh",
    # activity
    "is_active": "unit",
}
assert set(FEATURE_TRANSFORMS) == set(FEATURE_ORDER), "FEATURE_TRANSFORMS must cover FEATURE_ORDER exactly"

# Kept for readers of older metadata: the Δ=30 scaler applied log1p to these
# five only (and, being an identity RobustScaler, changed nothing else).
LOG1P_FEATURES: list[str] = [f for f in FEATURE_ORDER if FEATURE_TRANSFORMS[f] == "log1p"]

# Windowing / rollout geometry — DEFAULTS only. The governing values are
# `windowing.window_seconds / context_length / horizon_length` in
# config/default.yaml, read through `nidra.train.pipeline.geometry_from_config`;
# these constants back test fixtures and functions called without a config.
#
# Δ = 60 s since the Δ=60 rebuild: the CIC-IDS2017 flow CSVs carry
# minute-resolution timestamps, so a 30 s window could only ever populate
# the :00 half — every :30 window was an artificial empty state (see
# reports/NIDRA_REEVALUATION_2026-09-19.md §4.1). Changing Δ invalidates
# every trained artifact and every recorded metric; the Δ=30 record is kept
# under the git tag `baseline-delta30-run7`.
WINDOW_SECONDS: int = 60          # Delta
CONTEXT_LENGTH: int = 30          # L windows of history (30 min at Δ=60)
HORIZON_LENGTH: int = 6           # K windows of forecast (6 min at Δ=60)

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


# ---------------------------------------------------------------------------
# Cross-dataset feature regimes
# ---------------------------------------------------------------------------
# NIDRA trains on two datasets with different flow instrumentation, and the
# 45-feature state is not equally available in both. A REGIME is the named
# answer to "which of the 45 may this experiment use", declared here once
# rather than reconstructed at each call site.
#
# FEATURE_ORDER stays 45 wide — it is a cross-service contract — and the
# excluded columns are declared `drop` in the FeatureScaler, which is the
# mechanism the pipeline already has for a feature the model must never
# read: the column is forced to 0 in scaled space and left out of
# `model_mask`, so it contributes nothing to the dynamics loss, the state
# metrics, or any head.
#
# What CTU-13's Argus binetflow cannot produce (see data/ctu_load.py):
#   - the 11 packet aggregates. CTU-13 ships PCAPs, but the public ones
#     contain ONLY the botnet's own traffic (the full captures were withheld
#     for privacy), so deriving packet features from them would hand the
#     model a perfect label: every host with packet data is the infected
#     one. They are unavailable, not merely missing.
#   - `iat_max`: Argus reports no per-packet timing, only a flow duration
#     and a packet total, so a flow's maximum inter-arrival gap does not
#     exist in the record.
#   - `d_retrans_rate`: the backward delta of a packet-level feature.
#
# What it produces differently rather than not at all: the six TCP flag
# ratios. CICFlowMeter counts how many packets carried each flag; Argus
# records which flags were seen per direction. SYN/FIN/RST/URG agree closely
# (about one per direction in a normal connection), ACK and PSH do not (a
# bulk transfer carries thousands). `cross_strict` removes all eight
# flag-derived columns; `cross_core` keeps them and the shift is measured
# rather than assumed (data/dataset_shift.py).
_PACKET_FEATURES_UNAVAILABLE_IN_CTU: list[str] = FEATURE_ORDER[15:26]
_NO_PER_PACKET_TIMING: list[str] = ["iat_max", "d_retrans_rate"]
_FLAG_SEMANTICS_DIFFER: list[str] = [
    "syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "psh_ratio", "urg_ratio",
    "d_syn_ratio", "slope3_syn_ratio",
]

FEATURE_REGIMES: dict[str, list[str]] = {
    # Everything. The within-dataset regime, and what Run 8 used.
    "full": [],
    # Every feature both datasets compute from the same underlying quantity,
    # plus the flag ratios whose estimator differs. 32 features.
    "cross_core": _PACKET_FEATURES_UNAVAILABLE_IN_CTU + _NO_PER_PACKET_TIMING,
    # Only features with identical semantics in both. 24 features.
    "cross_strict": _PACKET_FEATURES_UNAVAILABLE_IN_CTU + _NO_PER_PACKET_TIMING + _FLAG_SEMANTICS_DIFFER,
}


def regime_dropped_features(regime: str) -> list[str]:
    """Feature names a regime excludes, in FEATURE_ORDER order."""
    if regime not in FEATURE_REGIMES:
        raise ValueError(f"unknown feature regime {regime!r}; expected one of {sorted(FEATURE_REGIMES)}")
    excluded = set(FEATURE_REGIMES[regime])
    unknown = excluded - set(FEATURE_ORDER)
    if unknown:
        raise ValueError(f"feature regime {regime!r} names features that are not in FEATURE_ORDER: {sorted(unknown)}")
    return [f for f in FEATURE_ORDER if f in excluded]


def regime_kinds(regime: str) -> list[str]:
    """The per-feature transform list to construct a FeatureScaler with, so
    that this regime's excluded features are declared drops."""
    excluded = set(regime_dropped_features(regime))
    return ["drop" if f in excluded else FEATURE_TRANSFORMS[f] for f in FEATURE_ORDER]
