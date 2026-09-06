"""The 45 features of `FEATURE_ORDER`, computed from one closed window.

Everything in this module is pure. A window's accumulated counters go in, a
`dict[str, float]` keyed by `FEATURE_ORDER` comes out, and nothing here touches Redis,
the clock, or the network — which is what makes the escalating-scan test a unit test
rather than an integration test.

Three properties the rest of the service depends on:

* **Nothing after `t` is readable.** `d_x[t] = x[t] - x[t-1]` and `slope3_x[t]` is the
  OLS slope over `x[t-2..t]`. Both are strictly backward-looking and both zero-pad at the
  start of a sequence. A centred window here would be a leak that shows up as *better*
  numbers, which is why it is stated rather than inferred.
* **No NaN ever leaves.** Every ratio guards its denominator and returns `0.0`, every
  variance is clamped at zero against floating-point drift. One NaN feature becomes a
  NaN state, which becomes a NaN rollout by step three.
* **The accumulator is bounded.** A window is summarised as sums, sums of squares, and
  small value-count distributions — never a list of its events. The distributions live in
  the same Redis hash as the scalars under short field prefixes, so accumulating an event
  is one pipelined `HINCRBYFLOAT` batch and closing a window is one `HGETALL`.

`neighbour_risk_fraction` uses the heuristic prior from IMPLEMENTATION-ML.md §2.6 — the
fraction of this host's peers that had elevated fan-out in the *previous* window — and
never model output. Feeding predicted risk back into an input feature is a loop that
cannot be debugged in the time this build has.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nidra.data.schema import FEATURE_ORDER
from nidra_common.events import RawEvent

#: Hash-field prefixes for the value-count distributions carried alongside the scalars.
#: Short because they are written once per event and read once per window.
DIST_DST_IP = "di"
DIST_DST_PORT = "dp"
DIST_TCP_WINDOW = "tw"
DIST_PAYLOAD = "ps"
DIST_FLOW_IAT_MAX = "fx"

DIST_PREFIXES: frozenset[str] = frozenset(
    {DIST_DST_IP, DIST_DST_PORT, DIST_TCP_WINDOW, DIST_PAYLOAD, DIST_FLOW_IAT_MAX}
)

#: Separates a prefix from its value inside a hash field name (`dp:80`). Ports, IPs and
#: window sizes never contain it, so the split is unambiguous.
DIST_SEP = ":"

#: TCP flag bits, for packet events. Flow events carry the counts directly.
TCP_FLAG_BITS: dict[str, int] = {
    "syn": 0x02,
    "ack": 0x10,
    "rst": 0x04,
    "fin": 0x01,
    "psh": 0x08,
    "urg": 0x20,
}

#: The six values whose history feeds the `d_*` and `slope3_*` features. Kept per host in
#: `feat:prev:{tenant}:{host}` so a worker restart does not reset every delta to zero.
DYNAMIC_SOURCES: tuple[str, ...] = (
    "syn_ratio",
    "dst_port_entropy",
    "out_degree",
    "new_peer_count",
    "iat_var",
    "retrans_rate",
)

#: `d_x` -> `x`. Backward difference against the immediately preceding window.
DELTA_FEATURES: dict[str, str] = {f"d_{name}": name for name in DYNAMIC_SOURCES}

#: `slope3_x` -> `x`. OLS slope over the last three windows, current one included.
SLOPE_FEATURES: dict[str, str] = {
    "slope3_syn_ratio": "syn_ratio",
    "slope3_dst_port_entropy": "dst_port_entropy",
    "slope3_out_degree": "out_degree",
    "slope3_iat_var": "iat_var",
}

#: Points required before a slope is meaningful. Fewer and the feature is zero-padded.
SLOPE_WINDOW = 3


# --------------------------------------------------------------------------- numerics


def safe_div(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or `0.0` when the denominator is zero. Never NaN."""
    if denominator == 0:
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def variance(total: float, total_sq: float, count: float) -> float:
    """Population variance from running sums, clamped at zero.

    `E[x^2] - E[x]^2` can go slightly negative on a constant series once floating-point
    error is involved; a negative variance downstream is worse than a lost 1e-16.
    """
    if count <= 0:
        return 0.0
    return max(0.0, total_sq / count - (total / count) ** 2)


def shannon_entropy(counts: Iterable[float]) -> float:
    """Shannon entropy in nats over an empirical distribution, unnormalised.

    Natural log, as IMPLEMENTATION-ML.md §2.6 specifies. Near 0 means one destination;
    rising means the host is spreading.
    """
    values = [c for c in counts if c > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in values)


def percentile(counts: Mapping[str, float], q: float) -> float:
    """Nearest-rank `q`-quantile of a value→count distribution whose keys parse as floats."""
    pairs = sorted((float(value), count) for value, count in counts.items() if count > 0)
    total = sum(count for _, count in pairs)
    if not pairs or total <= 0:
        return 0.0
    target = q * total
    seen = 0.0
    for value, count in pairs:
        seen += count
        if seen >= target:
            return value
    return pairs[-1][0]


def ols_slope(values: list[float]) -> float:
    """Least-squares slope of `values` against evenly spaced steps, per window.

    Fewer than :data:`SLOPE_WINDOW` points returns `0.0` — the sequence start is padded
    with zeros, never with anything that has not happened yet.
    """
    n = len(values)
    if n < SLOPE_WINDOW:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denominator = sum((x - mean_x) ** 2 for x in range(n))
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
    return safe_div(numerator, denominator)


# ------------------------------------------------------------------------ accumulation


def dist_field(prefix: str, value: Any) -> str:
    """Hash field for one bucket of a value-count distribution (`dp:80`)."""
    return f"{prefix}{DIST_SEP}{value}"


def _bump(target: dict[str, float], key: str, amount: float = 1.0) -> None:
    target[key] = target.get(key, 0.0) + amount


def event_increments(event: RawEvent) -> dict[str, float]:
    """Hash-field increments contributed by one event to its source host's open window.

    A window row describes what its host *did*: only events where the host is the source
    accumulate here. What was done to it arrives through the window graph, which is where
    `in_degree` and `reciprocity` come from.
    """
    inc: dict[str, float] = {}
    if event.dst_ip:
        _bump(inc, dist_field(DIST_DST_IP, event.dst_ip))
    if event.dst_port is not None:
        _bump(inc, dist_field(DIST_DST_PORT, event.dst_port))

    fields = event.fields
    if event.kind == "flow":
        packets = fields.get("pkts_fwd", 0.0) + fields.get("pkts_bwd", 0.0)
        _bump(inc, "n_flows")
        _bump(inc, "pkt_count", packets)
        _bump(inc, "flow_pkts", packets)
        for flag in TCP_FLAG_BITS:
            count = fields.get(f"{flag}_count", 0.0)
            if count:
                _bump(inc, flag, count)
        _bump(inc, "bytes_up", fields.get("bytes_fwd", 0.0))
        _bump(inc, "bytes_down", fields.get("bytes_bwd", 0.0))
        duration = fields.get("duration", 0.0)
        _bump(inc, "dur_sum", duration)
        _bump(inc, "dur_sq", duration * duration)
        iat_mean = fields.get("iat_mean", 0.0)
        _bump(inc, "iat_sum", iat_mean)
        _bump(inc, "iat_sq", iat_mean * iat_mean)
        _bump(inc, dist_field(DIST_FLOW_IAT_MAX, round(fields.get("iat_max", 0.0), 6)))
        return inc

    _bump(inc, "n_packets")
    _bump(inc, "pkt_count")
    flags = int(fields.get("tcp_flags", 0.0))
    for flag, bit in TCP_FLAG_BITS.items():
        if flags & bit:
            _bump(inc, flag)
    frame_len = fields.get("frame_len", 0.0)
    _bump(inc, "bytes_up", frame_len)
    if "ttl" in fields:
        ttl = fields["ttl"]
        _bump(inc, "n_ttl")
        _bump(inc, "ttl_sum", ttl)
        _bump(inc, "ttl_sq", ttl * ttl)
    if "tcp_window" in fields:
        window = fields["tcp_window"]
        _bump(inc, "n_tcp_window")
        _bump(inc, "tcp_window_sum", window)
        _bump(inc, dist_field(DIST_TCP_WINDOW, int(window)))
    if fields.get("frag", 0.0):
        _bump(inc, "frag_count")
    if "tcp_len" in fields:
        payload = fields["tcp_len"]
        _bump(inc, "n_payload")
        _bump(inc, "payload_sum", payload)
        _bump(inc, "payload_sq", payload * payload)
        _bump(inc, dist_field(DIST_PAYLOAD, int(payload)))
    if fields.get("retransmission", 0.0):
        _bump(inc, "retrans_count")
    return inc


@dataclass(frozen=True, slots=True)
class WindowAccumulator:
    """One host's closed window: running scalars plus small value-count distributions."""

    scalars: dict[str, float] = field(default_factory=dict)
    dists: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_hash(cls, mapping: Mapping[str, str | float]) -> WindowAccumulator:
        """Rebuild from a `HGETALL` of `feat:win:{tenant}:{host}:{ts}`."""
        scalars: dict[str, float] = {}
        dists: dict[str, dict[str, float]] = {prefix: {} for prefix in DIST_PREFIXES}
        for raw_key, raw_value in mapping.items():
            value = float(raw_value)
            prefix, sep, bucket = raw_key.partition(DIST_SEP)
            if sep and prefix in DIST_PREFIXES:
                dists[prefix][bucket] = value
            else:
                scalars[raw_key] = value
        return cls(scalars=scalars, dists=dists)

    def get(self, name: str) -> float:
        return self.scalars.get(name, 0.0)

    def dist(self, prefix: str) -> dict[str, float]:
        return self.dists.get(prefix, {})

    @property
    def event_count(self) -> float:
        return self.get("n_flows") + self.get("n_packets")


# ---------------------------------------------------------------------- window graph


@dataclass(frozen=True, slots=True)
class WindowGraph:
    """The transient directed graph over hosts for one window.

    Built from the window's edges, used to produce four scalars, then discarded — it is
    a feature extractor, never the state itself (CLAUDE.md, "when adding features").
    """

    out_peers: dict[str, set[str]] = field(default_factory=dict)
    in_peers: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]]) -> WindowGraph:
        out_peers: dict[str, set[str]] = {}
        in_peers: dict[str, set[str]] = {}
        for src, dst in edges:
            out_peers.setdefault(src, set()).add(dst)
            in_peers.setdefault(dst, set()).add(src)
            out_peers.setdefault(dst, set())
            in_peers.setdefault(src, set())
        return cls(out_peers=out_peers, in_peers=in_peers)

    def successors(self, host: str) -> set[str]:
        return self.out_peers.get(host, set())

    def predecessors(self, host: str) -> set[str]:
        return self.in_peers.get(host, set())

    def out_degree(self, host: str) -> float:
        return float(len(self.successors(host)))

    def in_degree(self, host: str) -> float:
        return float(len(self.predecessors(host)))

    def reciprocity(self, host: str) -> float:
        """Fraction of this host's destinations that also sent to it in the same window."""
        out = self.successors(host)
        return safe_div(float(len(out & self.predecessors(host))), float(len(out)))

    def clustering(self, host: str, degree_cap: int) -> float:
        """Undirected local clustering over the host's neighbourhood, edge direction ignored.

        Dense windows make this quadratic, so a neighbourhood larger than `degree_cap` is
        sampled down to the first `degree_cap` peers in sorted order — deterministic, and
        the value is a signal rather than an exact quantity (IMPLEMENTATION-ML.md §2.7).
        """
        neighbours = sorted(self.successors(host) | self.predecessors(host))
        if len(neighbours) > degree_cap:
            neighbours = neighbours[:degree_cap]
        size = len(neighbours)
        if size < 2:
            return 0.0
        present = set(neighbours)
        links = 0
        for index, node in enumerate(neighbours):
            reachable = (self.successors(node) | self.predecessors(node)) & present
            links += sum(1 for other in neighbours[index + 1 :] if other in reachable)
        return safe_div(float(links), float(size * (size - 1) / 2))


# ------------------------------------------------------------------------- the vector


def zero_features() -> dict[str, float]:
    """An all-zero state row. A silent host still has state; `is_active` stays 0."""
    return dict.fromkeys(FEATURE_ORDER, 0.0)


def validate_features(features: Mapping[str, float]) -> None:
    """Fail loudly when the key set drifts from `FEATURE_ORDER`.

    Caught here it costs a second. Caught in a SHAP plot on demo day it costs the demo.
    """
    keys = set(features)
    expected = set(FEATURE_ORDER)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        raise ValueError(
            f"computed features must equal FEATURE_ORDER "
            f"(missing={missing}, unexpected={unexpected})"
        )


def _flow_features(acc: WindowAccumulator) -> dict[str, float]:
    packets = acc.get("pkt_count")
    flows = acc.get("n_flows")
    return {
        "syn_ratio": safe_div(acc.get("syn"), packets),
        "ack_ratio": safe_div(acc.get("ack"), packets),
        "rst_ratio": safe_div(acc.get("rst"), packets),
        "fin_ratio": safe_div(acc.get("fin"), packets),
        "psh_ratio": safe_div(acc.get("psh"), packets),
        "urg_ratio": safe_div(acc.get("urg"), packets),
        "bytes_total": acc.get("bytes_up") + acc.get("bytes_down"),
        "bytes_up_down_ratio": safe_div(acc.get("bytes_up"), acc.get("bytes_down")),
        "pkts_per_flow_mean": safe_div(acc.get("flow_pkts"), flows),
        "flow_duration_mean": safe_div(acc.get("dur_sum"), flows),
        "flow_duration_var": variance(acc.get("dur_sum"), acc.get("dur_sq"), flows),
        "iat_mean": safe_div(acc.get("iat_sum"), flows),
        "iat_var": variance(acc.get("iat_sum"), acc.get("iat_sq"), flows),
        "iat_max": percentile(acc.dist(DIST_FLOW_IAT_MAX), 1.0),
        "active_flow_count": flows,
    }


def _packet_features(acc: WindowAccumulator) -> dict[str, float]:
    observed = acc.get("n_packets")
    payloads = acc.dist(DIST_PAYLOAD)
    return {
        "ttl_mean": safe_div(acc.get("ttl_sum"), acc.get("n_ttl")),
        "ttl_var": variance(acc.get("ttl_sum"), acc.get("ttl_sq"), acc.get("n_ttl")),
        "tcp_window_mean": safe_div(acc.get("tcp_window_sum"), acc.get("n_tcp_window")),
        "tcp_window_entropy": shannon_entropy(acc.dist(DIST_TCP_WINDOW).values()),
        "frag_flag_rate": safe_div(acc.get("frag_count"), observed),
        "payload_size_mean": safe_div(acc.get("payload_sum"), acc.get("n_payload")),
        "payload_size_var": variance(
            acc.get("payload_sum"), acc.get("payload_sq"), acc.get("n_payload")
        ),
        "payload_size_p95": percentile(payloads, 0.95),
        "payload_size_entropy": shannon_entropy(payloads.values()),
        "retrans_count": acc.get("retrans_count"),
        "retrans_rate": safe_div(acc.get("retrans_count"), observed),
    }


def _graph_features(
    acc: WindowAccumulator,
    *,
    host: str,
    graph: WindowGraph,
    previous_graph: WindowGraph,
    new_peer_count: int,
    elevated_fanout: int,
    degree_cap: int,
) -> dict[str, float]:
    peers = graph.successors(host)
    elevated = sum(1 for peer in peers if previous_graph.out_degree(peer) >= elevated_fanout)
    return {
        "out_degree": graph.out_degree(host),
        "in_degree": graph.in_degree(host),
        "dst_ip_entropy": shannon_entropy(acc.dist(DIST_DST_IP).values()),
        "dst_port_entropy": shannon_entropy(acc.dist(DIST_DST_PORT).values()),
        "new_peer_count": float(new_peer_count),
        # Heuristic prior, not model output — see the module docstring.
        "neighbour_risk_fraction": safe_div(float(elevated), float(len(peers))),
        "local_clustering_coeff": graph.clustering(host, degree_cap),
        "reciprocity": graph.reciprocity(host),
    }


def _dynamics(current: Mapping[str, float], history: list[Mapping[str, float]]) -> dict[str, float]:
    """Backward differences and 3-window OLS slopes, zero-padded at the sequence start.

    `history` is the tracked values of the preceding windows, oldest first, and never
    includes the current one. Nothing here can see past `t`.
    """
    previous = history[-1] if history else None
    dynamics: dict[str, float] = {}
    for feature, source in DELTA_FEATURES.items():
        dynamics[feature] = 0.0 if previous is None else current[source] - previous.get(source, 0.0)
    tail = history[-(SLOPE_WINDOW - 1) :]
    for feature, source in SLOPE_FEATURES.items():
        series = [point.get(source, 0.0) for point in tail] + [current[source]]
        dynamics[feature] = ols_slope(series)
    return dynamics


def tracked_values(features: Mapping[str, float]) -> dict[str, float]:
    """The slice of a computed vector that the next window needs for its dynamics."""
    return {name: float(features[name]) for name in DYNAMIC_SOURCES}


def compute_features(
    acc: WindowAccumulator,
    *,
    host: str,
    graph: WindowGraph,
    previous_graph: WindowGraph,
    new_peer_count: int,
    history: list[Mapping[str, float]],
    elevated_fanout: int = 5,
    degree_cap: int = 200,
) -> dict[str, float]:
    """The 45 features of one closed window, validated against `FEATURE_ORDER`.

    `history` holds the tracked values of the windows before this one, oldest first —
    including the zero rows emitted for silent windows, because a gap in traffic is a
    real drop in the series and the deltas should say so.
    """
    features: dict[str, float] = {}
    features.update(_flow_features(acc))
    features.update(_packet_features(acc))
    features.update(
        _graph_features(
            acc,
            host=host,
            graph=graph,
            previous_graph=previous_graph,
            new_peer_count=new_peer_count,
            elevated_fanout=elevated_fanout,
            degree_cap=degree_cap,
        )
    )
    features.update(_dynamics(features, history))
    features["is_active"] = 1.0 if acc.event_count > 0 else 0.0

    validate_features(features)
    non_finite = sorted(name for name, value in features.items() if not math.isfinite(value))
    if non_finite:
        raise ValueError(f"host {host} produced non-finite features: {non_finite}")
    return features
