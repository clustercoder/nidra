"""The Δ=60 canonical-data contract: flow/packet fusion, per-day validation
blocks, K-agnostic cached labels, and geometry read from config.

Each test pins one thing the reevaluation found broken or ambiguous in the
Δ=30 pipeline (reports/NIDRA_REEVALUATION_2026-09-19.md §4, §15.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.data.labels import attach_risk_label, label_stage_table, reattach_risk_label, risk_label_from_attack_flags
from nidra.data.schema import FEATURE_INDEX, WINDOW_SECONDS
from nidra.data.splits import build_splits, capture_day, temporal_train_val_split
from nidra.data.windowize import FUSION_TAG, build_state_rows
from nidra.train.pipeline import geometry_from_config
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


# ---------------------------------------------------------------------------
# Flow / packet fusion
# ---------------------------------------------------------------------------

def _flows_and_packets():
    flows = make_synthetic_flows(n_hosts=2, n_windows=20, portscan_len=0)
    packets = make_synthetic_packets(flows)
    return flows, packets


def test_packet_only_windows_of_a_flow_host_become_active_states():
    """A host sending packets of an ongoing flow in a minute with no new
    flow start is transmitting; that minute must be an active row carrying
    its packet aggregates, not an all-zero is_active=0 gap row."""
    flows, packets = _flows_and_packets()
    host = flows["src_ip"].iloc[0]
    # remove every flow of `host` in one middle window, keep its packets there
    w = sorted(flows["window_ts"].unique())[5]
    flows_cut = flows[~((flows["src_ip"] == host) & (flows["window_ts"] == w))]
    assert (packets[(packets["ip_src"] == host) & (packets["window_ts"] == w)]).shape[0] > 0

    states = build_state_rows(flows_cut, packets, window_seconds=WINDOW_SECONDS)
    row = states[(states["host_id"] == host) & (states["window_ts"] == w)]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["is_active"] == 1.0
    assert row["active_flow_count"] == 0.0           # no flow STARTED in this window
    assert row["ttl_mean"] > 0.0                       # but the packets are real
    assert row["payload_size_mean"] > 0.0


def test_packet_only_hosts_are_not_added_to_the_state_table():
    """The host population stays defined by flow sources (labels are keyed
    on the flow source); a pure responder never gets a row."""
    flows, packets = _flows_and_packets()
    extra = packets.head(20).copy()
    extra["ip_src"] = "192.0.2.99"
    states = build_state_rows(flows, pd.concat([packets, extra], ignore_index=True), window_seconds=WINDOW_SECONDS)
    assert "192.0.2.99" not in set(states["host_id"])


def test_packets_outside_the_day_files_flow_range_are_ignored():
    """Three Friday day-files share one full-day PCAP: the morning file must
    not absorb afternoon packets, or the three tables would overlap."""
    flows, packets = _flows_and_packets()
    host = flows["src_ip"].iloc[0]
    hi = flows["window_ts"].max()
    late = packets.head(10).copy()
    late["ip_src"] = host
    late["window_ts"] = hi + 10 * WINDOW_SECONDS
    late["frame_time_epoch"] = late["window_ts"] + 1.0
    states = build_state_rows(flows, pd.concat([packets, late], ignore_index=True), window_seconds=WINDOW_SECONDS)
    assert states["window_ts"].max() == hi


def test_flow_only_mode_is_unchanged_by_fusion():
    flows, _ = _flows_and_packets()
    states = build_state_rows(flows, pd.DataFrame(), window_seconds=WINDOW_SECONDS)
    assert (states[["ttl_mean", "payload_size_mean", "retrans_count"]] == 0.0).all().all()
    assert states["is_active"].isin([0.0, 1.0]).all()


def test_fusion_tag_is_versioned():
    assert FUSION_TAG == "fuse2"


# ---------------------------------------------------------------------------
# Labels: vectorized, K-agnostic cache
# ---------------------------------------------------------------------------

def test_vectorized_risk_label_matches_the_reference_loop():
    rng = np.random.default_rng(0)

    def reference(attack, K):
        n = len(attack)
        return [int(attack[i + 1: min(n, i + 1 + K)].any()) for i in range(n)]

    for K in (1, 3, 6):
        for _ in range(50):
            a = (rng.random(rng.integers(1, 40)) < 0.2).astype(int)
            assert risk_label_from_attack_flags(a, K).tolist() == reference(a, K)


def test_risk_label_is_rederived_for_the_configured_horizon():
    flows = make_synthetic_flows(n_hosts=2, n_windows=40, portscan_start_window=20, portscan_len=5)
    states = build_state_rows(flows, pd.DataFrame(), window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    labelled_k6 = attach_risk_label(states, stage_table, horizon_k=6, window_seconds=WINDOW_SECONDS)
    labelled_k1 = reattach_risk_label(labelled_k6, horizon_k=1)
    scanner = labelled_k6[labelled_k6["host_id"] == "10.0.0.1"].sort_values("window_ts")
    scanner_k1 = labelled_k1[labelled_k1["host_id"] == "10.0.0.1"].sort_values("window_ts")
    # K=6 labels the six windows before onset, K=1 only the one before
    onset = scanner.loc[scanner["stage_label"] != "benign", "window_ts"].min()
    pre6 = scanner[(scanner["window_ts"] < onset) & (scanner["window_ts"] >= onset - 6 * WINDOW_SECONDS)]
    pre1 = scanner_k1[(scanner_k1["window_ts"] < onset) & (scanner_k1["window_ts"] >= onset - 6 * WINDOW_SECONDS)]
    assert pre6["risk_label"].sum() == 6
    assert pre1["risk_label"].sum() == 1
    assert (labelled_k1["stage_label"] == labelled_k6["stage_label"]).all()


# ---------------------------------------------------------------------------
# Per-day validation blocks
# ---------------------------------------------------------------------------

def _two_day_table():
    day1 = make_synthetic_flows(n_hosts=2, n_windows=60, start_epoch=1_700_000_000,
                                portscan_start_window=50, portscan_len=5)
    day2 = make_synthetic_flows(n_hosts=2, n_windows=60, start_epoch=1_700_000_000 + 86_400,
                                portscan_start_window=10, portscan_len=5)
    tables = []
    for flows in (day1, day2):
        states = build_state_rows(flows, pd.DataFrame(), window_seconds=WINDOW_SECONDS)
        stage_table, _ = label_stage_table(flows)
        tables.append(attach_risk_label(states, stage_table, horizon_k=3, window_seconds=WINDOW_SECONDS))
    return pd.concat(tables, ignore_index=True)


def test_capture_day_partitions_rows_by_utc_day():
    table = _two_day_table()
    assert capture_day(table["window_ts"]).nunique() == 2


def test_per_day_split_takes_a_trailing_block_from_every_day():
    table = _two_day_table()
    train, val = temporal_train_val_split(table, val_fraction=0.25, per_day=True)
    assert capture_day(val["window_ts"]).nunique() == 2
    assert capture_day(train["window_ts"]).nunique() == 2
    # within each day, every val window is later than every train window
    for day in capture_day(table["window_ts"]).unique():
        t = train[capture_day(train["window_ts"]) == day]["window_ts"]
        v = val[capture_day(val["window_ts"]) == day]["window_ts"]
        assert t.max() < v.min()
    # day 1's episode (windows 50-54 of 60) sits in the trailing quarter -> val;
    # it must not straddle the boundary
    ep_days = capture_day(val.loc[val["stage_label"] != "benign", "window_ts"]).unique()
    assert len(ep_days) == 1


def test_concatenated_split_is_still_available_for_the_historical_behaviour():
    table = _two_day_table()
    train, val = temporal_train_val_split(table, val_fraction=0.25, per_day=False)
    assert capture_day(val["window_ts"]).nunique() == 1


def test_build_splits_recomputes_labels_inside_each_split():
    """The last K train windows before the validation cut must not carry a
    label derived from validation-block windows."""
    table = _two_day_table()
    day_tables = {"d": table}
    splits = build_splits(day_tables, train_days=["d"], test_days=[], holdout_days=[],
                          val_fraction=0.25, val_block_per_day=True, horizon_k=3)
    # recomputing on the split alone is a fixed point
    for part in (splits.train, splits.val):
        again = reattach_risk_label(part, 3)
        assert (again["risk_label"].to_numpy() == part.sort_values(["host_id", "window_ts"])["risk_label"].to_numpy()).all()
    # and the train rows immediately before a val episode are NOT positive
    scanner_train = splits.train[splits.train["host_id"] == "10.0.0.1"].sort_values("window_ts")
    day1_cut = capture_day(splits.val["window_ts"]).min()
    tail = scanner_train[capture_day(scanner_train["window_ts"]) == day1_cut].tail(3)
    assert tail["risk_label"].sum() == 0


# ---------------------------------------------------------------------------
# Geometry from config
# ---------------------------------------------------------------------------

def test_geometry_from_config_reads_the_windowing_block():
    cfg = {"windowing": {"window_seconds": 60, "context_length": 15, "horizon_length": 3}}
    assert geometry_from_config(cfg) == (60, 15, 3)


def test_geometry_from_config_rejects_a_label_horizon_that_disagrees_with_k():
    cfg = {"windowing": {"window_seconds": 60, "context_length": 30, "horizon_length": 6},
           "labels": {"risk_threshold_windows": 3}}
    with pytest.raises(ValueError, match="risk_threshold_windows"):
        geometry_from_config(cfg)


def test_schema_default_window_is_the_minute_resolution_of_the_dataset():
    assert WINDOW_SECONDS == 60
    assert "is_active" in FEATURE_INDEX
