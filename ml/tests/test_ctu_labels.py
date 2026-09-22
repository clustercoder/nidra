"""CTU-13 label semantics.

CTU-13's `Label` column is a taxonomy of its own, and the one thing that
must never happen is a Background or Normal flow scoring as an attack:
`flow=Background-TCP-Attempt` contains the word "Attempt" that marks
botnet scanning, and `flow=From-Normal-V42-Grill` contains "Normal". Every
rule here is therefore gated on the `From-Botnet` prefix first.
"""

from __future__ import annotations

import pandas as pd

from nidra.data.labels import label_stage_table, map_label_to_stage


class TestCTUDialect:
    def test_background_and_normal_are_benign_even_when_they_read_like_attacks(self):
        for label in [
            "flow=Background-TCP-Attempt",
            "flow=Background-UDP-Established",
            "flow=To-Background-UDP-CVUT-DNS-Server",
            "flow=From-Normal-V42-Stribrek",
            "flow=Normal-V54-HTTP-windowsupdate",
            "flow=To-Normal-V42-UDP-NTP-server",
            "flow=Background-google-analytics10",
        ]:
            assert map_label_to_stage(label, dialect="ctu") == "benign", label

    def test_command_and_control_flows_map_to_c2(self):
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted", dialect="ctu") == "c2"
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-Established-Custom-Encryption", dialect="ctu") == "c2"

    def test_scanning_and_flood_attempts_map_to_recon(self):
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-Attempt", dialect="ctu") == "recon"
        assert map_label_to_stage("flow=From-Botnet-V42-UDP-Attempt-DNS", dialect="ctu") == "recon"

    def test_spam_and_click_fraud_map_to_the_impact_bucket(self):
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-Established-SPAM", dialect="ctu") == "exfil"
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-Established-HTTP-Ad-63", dialect="ctu") == "exfil"
        assert map_label_to_stage("flow=From-Botnet-V42-ICMP", dialect="ctu") == "exfil"

    def test_payload_retrieval_maps_to_initial_access(self):
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-Established-HTTP-Binary-Download", dialect="ctu") == "initial_access"

    def test_an_unrecognised_botnet_flow_still_counts_as_an_attack(self):
        # Default for anything labelled Botnet is c2 — never benign. A new
        # sub-label must not silently become a negative.
        assert map_label_to_stage("flow=From-Botnet-V99-Something-New", dialect="ctu") == "c2"

    def test_cic_dialect_is_unchanged_by_the_ctu_rules(self):
        assert map_label_to_stage("BENIGN") == "benign"
        assert map_label_to_stage("PortScan") == "recon"
        assert map_label_to_stage("Bot") == "c2"
        assert map_label_to_stage("DDoS") == "exfil"
        assert map_label_to_stage("Infiltration") == "lateral"

    def test_cic_dialect_would_misread_a_ctu_label_which_is_why_dialect_exists(self):
        # Guards the reason the parameter is not optional in the CTU path.
        assert map_label_to_stage("flow=Background-TCP-Attempt", dialect="cic") == "benign"
        assert map_label_to_stage("flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted", dialect="cic") == "benign"


class TestStageTableWithDialect:
    def test_stage_table_uses_the_requested_dialect(self):
        flows = pd.DataFrame({
            "src_ip": ["147.32.84.165", "147.32.84.165", "147.32.84.59"],
            "window_ts": [60, 60, 60],
            "label": ["flow=Background-TCP-Attempt", "flow=From-Botnet-V42-TCP-CC1-HTTP-Not-Encrypted",
                      "flow=Background-UDP-Established"],
        })
        table, report = label_stage_table(flows, dialect="ctu")
        got = dict(zip(table["host_id"], table["stage"]))
        assert got["147.32.84.165"] == "c2"          # most severe wins within the window
        assert got["147.32.84.59"] == "benign"
        assert report["unmapped_labels"] == []

    def test_unmapped_ctu_labels_are_reported(self):
        flows = pd.DataFrame({"src_ip": ["a"], "window_ts": [60], "label": ["something entirely else"]})
        _, report = label_stage_table(flows, dialect="ctu")
        assert report["unmapped_labels"] == ["something entirely else"]
