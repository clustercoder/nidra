import pandas as pd
import pytest

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import WINDOW_SECONDS
from nidra.data.splits import (
    SplitResult,
    assert_no_episode_leakage,
    assert_no_temporal_overlap,
    build_splits,
    temporal_train_val_split,
)
from nidra.data.windowize import build_state_rows
from tests.ml.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def _labelled_day(n_windows=100, portscan_start=50, portscan_len=10):
    flows = make_synthetic_flows(n_hosts=2, n_windows=n_windows, portscan_start_window=portscan_start, portscan_len=portscan_len)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    return attach_risk_label(states, stage_table, horizon_k=6, window_seconds=WINDOW_SECONDS)


def test_temporal_split_is_contiguous_not_random():
    day = _labelled_day()
    train, val = temporal_train_val_split(day, val_fraction=0.2)
    assert train["window_ts"].max() <= val["window_ts"].min()


def test_temporal_split_nudges_cutoff_to_avoid_episode_split():
    day = _labelled_day(n_windows=100, portscan_start=75, portscan_len=20)
    train, val = temporal_train_val_split(day, val_fraction=0.2)
    combined = pd.concat([train.assign(_split="train"), val.assign(_split="val")])
    scanner_attack = combined[(combined["host_id"] == "10.0.0.1") & (combined["stage_label"] != "benign")]
    assert scanner_attack["_split"].nunique() == 1, "attack episode was split across train/val"


def test_build_splits_thursday_infiltration_excluded_from_train():
    tuesday = _labelled_day(portscan_start=40, portscan_len=5).assign(_day="tuesday")
    thursday_infil = _labelled_day(portscan_start=20, portscan_len=30).assign(_day="thursday_infiltration")
    friday = _labelled_day(portscan_start=60, portscan_len=5).assign(_day="friday_morning")

    day_tables = {"tuesday": tuesday, "thursday_infiltration": thursday_infil, "friday_morning": friday}
    splits = build_splits(
        day_tables,
        train_days=["tuesday"],
        test_days=["friday_morning"],
        holdout_days=["thursday_infiltration"],
        val_fraction=0.15,
    )
    assert len(splits.holdout) == len(thursday_infil)
    assert (splits.holdout["_day"] == "thursday_infiltration").all()
    # Infiltration must never appear in train or val — day-table exclusion,
    # not a content-based filter, is what guarantees this.
    assert "thursday_infiltration" not in set(splits.train["_day"]) | set(splits.val["_day"])


def test_assert_no_temporal_overlap_passes_on_valid_split():
    day = _labelled_day()
    train, val = temporal_train_val_split(day, val_fraction=0.2)
    splits = SplitResult(train=train, val=val, test=pd.DataFrame(), holdout=pd.DataFrame())
    assert_no_temporal_overlap(splits)  # should not raise


def test_assert_no_episode_leakage_detects_synthetic_violation():
    day = _labelled_day()
    train, val = temporal_train_val_split(day, val_fraction=0.2)
    # deliberately inject one attack row from train into val to prove the
    # checker catches it
    attack_rows = train[train["stage_label"] != "benign"]
    if not attack_rows.empty:
        val_corrupted = pd.concat([val, attack_rows.iloc[[0]]], ignore_index=True)
        splits = SplitResult(train=train, val=val_corrupted, test=pd.DataFrame(), holdout=pd.DataFrame())
        with pytest.raises(AssertionError):
            assert_no_episode_leakage(splits)
