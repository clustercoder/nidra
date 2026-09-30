import pandas as pd

from nidra.explain.flow_bridge import FEATURE_TO_FLOW_PREDICATE, flagged_flows


def _synthetic_window_flows() -> pd.DataFrame:
    return pd.DataFrame([
        {"src_ip": "10.0.0.1", "dst_ip": "172.16.0.1", "dst_port": 80, "flow_duration": 100.0, "label": "BENIGN",
         "syn_flag_count": 1, "total_fwd_packets": 5, "total_bwd_packets": 5},
        {"src_ip": "10.0.0.1", "dst_ip": "172.16.0.2", "dst_port": 22222, "flow_duration": 1.0, "label": "PortScan",
         "syn_flag_count": 1, "total_fwd_packets": 1, "total_bwd_packets": 0},
        {"src_ip": "10.0.0.1", "dst_ip": "172.16.0.3", "dst_port": 33333, "flow_duration": 1.0, "label": "PortScan",
         "syn_flag_count": 1, "total_fwd_packets": 1, "total_bwd_packets": 0},
        {"src_ip": "10.0.0.1", "dst_ip": "172.16.0.1", "dst_port": 80, "flow_duration": 90.0, "label": "BENIGN",
         "syn_flag_count": 1, "total_fwd_packets": 5, "total_bwd_packets": 5},
    ])


def test_feature_to_flow_predicate_registry_has_expected_entries():
    for feature in ["dst_port_entropy", "new_peer_count", "retrans_rate", "syn_ratio", "out_degree"]:
        assert feature in FEATURE_TO_FLOW_PREDICATE


def test_dst_port_entropy_flags_rare_ports():
    flows = _synthetic_window_flows()
    result = flagged_flows("10.0.0.1", flows, pd.DataFrame(), ["dst_port_entropy"], n=2)
    ports = {r["dst_port"] for r in result["dst_port_entropy"]}
    assert 22222 in ports or 33333 in ports  # rare ports surfaced, not the repeated port-80 flow


def test_new_peer_count_flags_unseen_peers():
    flows = _synthetic_window_flows()
    previously_seen = {"172.16.0.1"}
    result = flagged_flows("10.0.0.1", flows, pd.DataFrame(), ["new_peer_count"], previously_seen_peers=previously_seen)
    dsts = {r["dst_ip"] for r in result["new_peer_count"]}
    assert "172.16.0.1" not in dsts
    assert "172.16.0.2" in dsts or "172.16.0.3" in dsts


def test_retrans_rate_returns_empty_without_packet_data_not_fabricated():
    flows = _synthetic_window_flows()
    result = flagged_flows("10.0.0.1", flows, pd.DataFrame(), ["retrans_rate"])
    assert result["retrans_rate"] == []


def test_unregistered_feature_is_skipped_not_errored():
    flows = _synthetic_window_flows()
    result = flagged_flows("10.0.0.1", flows, pd.DataFrame(), ["totally_unregistered_feature"])
    assert result == {}
