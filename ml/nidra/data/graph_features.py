"""Per-window communication graph scalars.

For each 30s window, a transient directed multigraph over hosts is built
from that window's flows, per-node scalar metrics are computed, and the
graph is discarded immediately after — it is a feature extractor, never a
persisted object. See IMPLEMENTATION-ML.md §2.7.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def shannon_entropy(counts) -> float:
    """Shannon entropy over an empirical count distribution. Natural log,
    unnormalized. Returns 0.0 for an empty or single-bucket distribution
    (guards against NaN from log(0))."""
    counts = np.asarray(list(counts), dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def safe_div(numerator: float, denominator: float) -> float:
    """Ratio helper: zero denominator returns 0, never NaN."""
    if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
        return 0.0
    return float(numerator) / float(denominator)


@dataclass
class PeerTracker:
    """Per-host running memory of previously contacted peers, plus the prior
    window's out-degree distribution for the neighbour_risk_fraction prior.

    Must be driven in strict chronological window order — new_peer_count is
    the one genuinely stateful feature in the pipeline and must be computed
    in a single forward pass over sorted windows, never vectorized or
    reordered.
    """

    _seen: dict = field(default_factory=lambda: defaultdict(set))
    _prev_out_degree: dict = field(default_factory=dict)
    _prev_median_out_degree: float = 0.0

    def new_peer_count(self, host: str, peers_this_window: set) -> int:
        seen = self._seen[host]
        new = peers_this_window - seen
        seen.update(peers_this_window)
        return len(new)

    def neighbour_risk_fraction(self, peers_this_window: set) -> float:
        """Heuristic prior (never model-generated risk, which would create
        an undebuggable feedback loop): fraction of this host's peers whose
        out-degree in the PREVIOUS window exceeded the previous window's
        median out-degree ("elevated fan-out")."""
        if not peers_this_window:
            return 0.0
        elevated = sum(
            1 for p in peers_this_window if self._prev_out_degree.get(p, 0.0) > self._prev_median_out_degree
        )
        return elevated / len(peers_this_window)

    def commit_window(self, out_degree_by_host: dict) -> None:
        self._prev_out_degree = dict(out_degree_by_host)
        vals = list(out_degree_by_host.values())
        self._prev_median_out_degree = float(np.median(vals)) if vals else 0.0


@dataclass
class WindowGraphResult:
    out_degree: dict
    in_degree: dict
    out_peers: dict  # host -> set of dst_ip contacted (for peer tracker)
    dst_ip_entropy: dict
    dst_port_entropy: dict
    local_clustering_coeff: dict
    reciprocity: dict


def compute_window_graph(flows: pd.DataFrame, degree_cap: int = 200, seed: int = 0) -> WindowGraphResult:
    """Build the transient directed graph for ONE window from its flows and
    compute per-host scalars. `flows` must have columns src_ip, dst_ip,
    dst_port. The graph itself is not returned or persisted."""
    out_edges: dict = defaultdict(Counter)
    in_edges: dict = defaultdict(set)
    out_ports: dict = defaultdict(Counter)

    for src, dst, port in zip(flows["src_ip"], flows["dst_ip"], flows["dst_port"]):
        out_edges[src][dst] += 1
        in_edges[dst].add(src)
        out_ports[src][port] += 1

    hosts = set(out_edges) | set(in_edges)
    out_degree = {h: len(out_edges.get(h, {})) for h in hosts}
    in_degree = {h: len(in_edges.get(h, set())) for h in hosts}
    out_peers = {h: set(out_edges.get(h, {}).keys()) for h in hosts}
    dst_ip_entropy = {h: shannon_entropy(out_edges.get(h, {}).values()) for h in hosts}
    dst_port_entropy = {h: shannon_entropy(out_ports.get(h, {}).values()) for h in hosts}

    reciprocity = {}
    for h in hosts:
        peers = out_peers[h]
        if not peers:
            reciprocity[h] = 0.0
            continue
        mutual = sum(1 for p in peers if h in in_edges.get(p, set()))
        reciprocity[h] = mutual / len(peers)

    rng = np.random.default_rng(seed)
    neighbour_sets = {h: out_peers.get(h, set()) | in_edges.get(h, set()) for h in hosts}
    local_clustering_coeff = {}
    for h in hosts:
        neighbours = list(neighbour_sets[h])
        if len(neighbours) > degree_cap:
            idx = rng.choice(len(neighbours), size=degree_cap, replace=False)
            neighbours = [neighbours[i] for i in idx]
        k = len(neighbours)
        if k < 2:
            local_clustering_coeff[h] = 0.0
            continue
        links = 0
        for i in range(k):
            ni = neighbour_sets.get(neighbours[i], set())
            for j in range(i + 1, k):
                if neighbours[j] in ni:
                    links += 1
        possible = k * (k - 1) / 2
        local_clustering_coeff[h] = links / possible if possible > 0 else 0.0

    return WindowGraphResult(
        out_degree=out_degree,
        in_degree=in_degree,
        out_peers=out_peers,
        dst_ip_entropy=dst_ip_entropy,
        dst_port_entropy=dst_port_entropy,
        local_clustering_coeff=local_clustering_coeff,
        reciprocity=reciprocity,
    )
