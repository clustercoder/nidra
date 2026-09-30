"""The ATT&CK mapping is a curated table; these tests keep it consistent
with the label rules and with the generated doc."""

from __future__ import annotations

import pytest

from nidra.data.attack_mapping import (
    ATTACK_LABELS,
    LABEL_MAPPINGS,
    STAGE_MAPPINGS,
    _doc_path,
    label_mapping,
    map_stage_distribution,
    mapping_markdown,
    progression_summary,
    stage_tactics,
    techniques_for_stage,
    validate_tables,
)
from nidra.data.schema import STAGE_LABELS


def test_tables_are_consistent_with_label_rules():
    validate_tables()


def test_every_stage_has_a_mapping_and_benign_has_no_tactic():
    assert set(STAGE_MAPPINGS) == set(STAGE_LABELS)
    assert stage_tactics("benign") == []
    for stage in STAGE_LABELS:
        if stage != "benign":
            assert stage_tactics(stage), stage


def test_all_cic2017_attack_labels_are_covered():
    expected = {"PortScan", "FTP-Patator", "SSH-Patator", "Heartbleed", "Infiltration", "Bot", "DDoS",
                "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
                "Web Attack – Brute Force", "Web Attack – XSS", "Web Attack – Sql Injection"}
    assert set(ATTACK_LABELS) == expected


def test_label_lookup_normalises_dashes_and_case():
    assert label_mapping("Web Attack - XSS").stage == "initial_access"
    assert label_mapping("web attack – xss").stage == "initial_access"
    assert label_mapping("ssh-patator").techniques[0].id == "T1110.001"
    assert label_mapping("BENIGN") is None


def test_techniques_for_stage_dedupes_across_labels():
    techs = techniques_for_stage("initial_access")
    ids = [t["id"] for t in techs]
    assert len(ids) == len(set(ids))
    assert "T1110.001" in ids and "T1190" in ids


def test_stage_distribution_mapping_ranks_and_skips_benign():
    dist = {"benign": 0.6, "recon": 0.25, "initial_access": 0.12, "lateral": 0.02, "c2": 0.01, "exfil": 0.0}
    out = map_stage_distribution(dist, min_prob=0.05)
    assert [o["stage"] for o in out] == ["recon", "initial_access"]
    assert out[0]["tactics"][0]["id"] == "TA0043"
    assert "resolution" in out[0]


def test_progression_summary_orders_tactics_by_first_horizon():
    dists = [
        {"benign": 0.9, "recon": 0.1, "initial_access": 0.0, "lateral": 0.0, "c2": 0.0, "exfil": 0.0},
        {"benign": 0.3, "recon": 0.6, "initial_access": 0.1, "lateral": 0.0, "c2": 0.0, "exfil": 0.0},
        {"benign": 0.2, "recon": 0.3, "initial_access": 0.5, "lateral": 0.0, "c2": 0.0, "exfil": 0.0},
    ]
    summary = progression_summary(dists, window_seconds=60)
    assert [s["stage"] for s in summary["steps"]] == ["benign", "recon", "initial_access"]
    assert summary["steps"][1]["t_plus_s"] == 120
    ids = [t["tactic"]["id"] for t in summary["tactics_in_order"]]
    assert ids[:2] == ["TA0043", "TA0007"]  # recon's tactics first (k=2), then initial access (k=3)
    assert "TA0001" in ids
    assert summary["label"] == "projected stage sequence (model-internal)"


def test_doc_is_in_sync_with_tables():
    path = _doc_path()
    if not path.exists():
        pytest.fail(f"{path} missing — run python -m nidra.data.attack_mapping --write")
    assert path.read_text() == mapping_markdown()


def test_exfil_bucket_declares_the_dataset_limitation():
    assert "DoS/DDoS" in STAGE_MAPPINGS["exfil"].resolution
    assert all(m.techniques[0].tactic.id == "TA0040" for m in LABEL_MAPPINGS if m.stage == "exfil")
