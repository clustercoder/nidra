"""Feature -> flow bridge: resolves top SHAP features back to the specific
flows in the driving window that produced them.

A per-host state model does not natively produce individual flagged flows
— this is a stated deliverable ("flagged flows") that requires an explicit
bridge. It is a LOOKUP over the flow table, never a second model.
`FEATURE_TO_FLOW_PREDICATE` is a registry so new mappings are additive.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

FlowPredicate = Callable[[pd.DataFrame, pd.DataFrame, str, dict], pd.DataFrame]


def _predicate_dst_port_spread(flows: pd.DataFrame, packets: pd.DataFrame, host: str, ctx: dict) -> pd.DataFrame:
    """dst_port_entropy rising -> flows to the RAREST destination ports this
    window (the ones driving the spread), not the most common ones."""
    host_flows = flows[flows["src_ip"] == host]
    if host_flows.empty:
        return host_flows
    port_counts = host_flows["dst_port"].value_counts()
    ranked = host_flows.assign(_port_rarity=host_flows["dst_port"].map(port_counts))
    return ranked.sort_values("_port_rarity").drop(columns="_port_rarity").head(ctx.get("n", 20))


def _predicate_new_peers(flows: pd.DataFrame, packets: pd.DataFrame, host: str, ctx: dict) -> pd.DataFrame:
    """new_peer_count rising -> flows to peers never contacted before this
    window, per the same running peer-set semantics as graph_features.py."""
    host_flows = flows[flows["src_ip"] == host]
    previously_seen = ctx.get("previously_seen_peers", set())
    novel = host_flows[~host_flows["dst_ip"].isin(previously_seen)]
    return novel.head(ctx.get("n", 20))


def _predicate_retransmissions(flows: pd.DataFrame, packets: pd.DataFrame, host: str, ctx: dict) -> pd.DataFrame:
    """retrans_rate rising -> flows whose packets included a retransmission.
    Requires packet-level data for this window; in flow-only mode (no PCAP
    available) this returns an empty frame rather than a fabricated guess —
    a documented limitation, not a silent wrong answer."""
    if packets is None or packets.empty:
        return flows.iloc[0:0]
    retrans_dst_ips = set(packets.loc[(packets["ip_src"] == host) & (packets["is_retransmission"] == 1), "ip_dst"])
    host_flows = flows[flows["src_ip"] == host]
    return host_flows[host_flows["dst_ip"].isin(retrans_dst_ips)].head(ctx.get("n", 20))


def _predicate_syn_heavy(flows: pd.DataFrame, packets: pd.DataFrame, host: str, ctx: dict) -> pd.DataFrame:
    """syn_ratio rising -> flows with a high SYN-flag count relative to
    total packets (single-packet SYN probes are the archetypal case)."""
    host_flows = flows[flows["src_ip"] == host].copy()
    if host_flows.empty:
        return host_flows
    total_pkts = (host_flows["total_fwd_packets"].fillna(0) + host_flows["total_bwd_packets"].fillna(0)).clip(lower=1)
    host_flows["_syn_share"] = host_flows["syn_flag_count"].fillna(0) / total_pkts
    return host_flows.sort_values("_syn_share", ascending=False).drop(columns="_syn_share").head(ctx.get("n", 20))


def _predicate_distinct_destinations(flows: pd.DataFrame, packets: pd.DataFrame, host: str, ctx: dict) -> pd.DataFrame:
    """out_degree rising -> one representative flow per distinct
    destination this window (the fan-out itself)."""
    host_flows = flows[flows["src_ip"] == host]
    return host_flows.drop_duplicates(subset="dst_ip").head(ctx.get("n", 20))


FEATURE_TO_FLOW_PREDICATE: dict[str, FlowPredicate] = {
    "dst_port_entropy": _predicate_dst_port_spread,
    "new_peer_count": _predicate_new_peers,
    "retrans_rate": _predicate_retransmissions,
    "retrans_count": _predicate_retransmissions,
    "syn_ratio": _predicate_syn_heavy,
    "out_degree": _predicate_distinct_destinations,
}


def flagged_flows(
    host: str,
    window_flows: pd.DataFrame,
    window_packets: pd.DataFrame,
    top_shap_features: list[str],
    previously_seen_peers: set | None = None,
    n: int = 20,
) -> dict[str, list[dict]]:
    """Resolves each of `top_shap_features` (typically the top-5 signed
    signals from shap_runner.py) to the flows that plausibly produced it.
    Features with no registered predicate are skipped, not errored — new
    mappings are additive, not required to cover every feature.
    """
    ctx = {"previously_seen_peers": previously_seen_peers or set(), "n": n}
    results: dict[str, list[dict]] = {}
    for feature in top_shap_features:
        predicate = FEATURE_TO_FLOW_PREDICATE.get(feature)
        if predicate is None:
            continue
        matched = predicate(window_flows, window_packets, host, ctx)
        cols = [c for c in ["src_ip", "dst_ip", "dst_port", "flow_duration", "label"] if c in matched.columns]
        results[feature] = matched[cols].to_dict(orient="records") if not matched.empty else []
    return results
