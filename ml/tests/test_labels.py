import pandas as pd

from nidra.data.labels import attach_risk_label, label_stage_table, map_label_to_stage
from nidra.data.schema import WINDOW_SECONDS
from nidra.data.windowize import build_state_rows
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def test_map_label_to_stage_known_labels():
    assert map_label_to_stage("BENIGN") == "benign"
    assert map_label_to_stage("PortScan") == "recon"
    assert map_label_to_stage("FTP-Patator") == "initial_access"
    assert map_label_to_stage("SSH-Patator") == "initial_access"
    assert map_label_to_stage("Web Attack - XSS") == "initial_access"
    assert map_label_to_stage("Infiltration") == "lateral"
    assert map_label_to_stage("Bot") == "c2"
    assert map_label_to_stage("DoS Hulk") == "exfil"
    assert map_label_to_stage("DDoS") == "exfil"


def test_map_label_to_stage_unknown_falls_back_to_benign():
    assert map_label_to_stage("Some Unseen Future Attack") == "benign"


def test_map_label_to_stage_cse_cic_ids2018_labels():
    """CSE-CIC-IDS2018 uses different raw label strings than CIC-IDS2017 for
    the same tactics — cross-referenced against the CIC's published attack
    table, not yet verified against the actual downloaded CSVs."""
    assert map_label_to_stage("FTP-BruteForce") == "initial_access"
    assert map_label_to_stage("SSH-Bruteforce") == "initial_access"
    assert map_label_to_stage("Brute Force -Web") == "initial_access"
    assert map_label_to_stage("Brute Force -XSS") == "initial_access"
    assert map_label_to_stage("SQL Injection") == "initial_access"
    # CSE-CIC-IDS2018's own misspelling (extra "e") must still map correctly.
    assert map_label_to_stage("Infilteration") == "lateral"
    assert map_label_to_stage("Bot") == "c2"
    assert map_label_to_stage("DoS attacks-GoldenEye") == "exfil"
    assert map_label_to_stage("DoS attacks-Slowloris") == "exfil"
    assert map_label_to_stage("DoS attacks-SlowHTTPTest") == "exfil"
    assert map_label_to_stage("DoS attacks-Hulk") == "exfil"
    assert map_label_to_stage("DDOS attack-LOIC-UDP") == "exfil"
    assert map_label_to_stage("DDOS attack-HOIC") == "exfil"
    assert map_label_to_stage("DDoS attacks-LOIC-HTTP") == "exfil"


def test_risk_label_is_forward_looking_only():
    flows = make_synthetic_flows(n_hosts=1, n_windows=60, portscan_start_window=30, portscan_len=5)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    labelled = attach_risk_label(states, stage_table, horizon_k=6, window_seconds=WINDOW_SECONDS)

    labelled = labelled.sort_values("window_ts").reset_index(drop=True)
    attack_window_idxs = labelled.index[labelled["stage_label"] != "benign"]
    assert len(attack_window_idxs) > 0
    first_attack_idx = attack_window_idxs.min()

    # risk_label must go to 1 strictly BEFORE the attack window itself starts
    # (it looks into (t, t+K]), and the window at/after the attack onset
    # is not required to be risk_label=1 retroactively for earlier windows
    # beyond the K horizon.
    pre_horizon_idx = first_attack_idx - 7  # outside the K=6 lookahead
    if pre_horizon_idx >= 0:
        assert labelled.loc[pre_horizon_idx, "risk_label"] == 0

    within_horizon_idx = first_attack_idx - 1
    assert labelled.loc[within_horizon_idx, "risk_label"] == 1


def test_risk_label_zero_when_no_future_attack():
    flows = make_synthetic_flows(n_hosts=1, n_windows=20)  # no portscan window range hit
    flows = flows[flows["label"] == "BENIGN"]
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    labelled = attach_risk_label(states, stage_table, horizon_k=6, window_seconds=WINDOW_SECONDS)
    assert (labelled["risk_label"] == 0).all()
