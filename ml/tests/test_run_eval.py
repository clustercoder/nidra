"""Integration test for the top-level eval CLI (nidra.eval.run_eval),
against synthetic fixtures — verifies the full baselines/ablations/
calibration/lead-time pipeline runs end to end and writes valid JSON,
not real-data numbers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.splits import SplitResult, temporal_train_val_split
from nidra.data.windowize import build_state_rows
from nidra.eval import run_eval as run_eval_mod
from nidra.explain.shap_runner import build_shap_background, save_background
from nidra.train.pipeline import build_windowed_splits, fit_scaler
from nidra.train.train_dynamics import train_one_seed
from nidra.train.train_heads import train_heads_for_seed
from tests.conftest import full_cfg_dict
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def _synthetic_split_with_test() -> SplitResult:
    def _day(seed_offset, portscan_start):
        flows = make_synthetic_flows(n_hosts=2, n_windows=200, portscan_start_window=portscan_start, portscan_len=10)
        packets = make_synthetic_packets(flows)
        states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
        stage_table, _ = label_stage_table(flows)
        return attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=WINDOW_SECONDS)

    train_day = _day(0, 150)
    test_day = _day(1, 100)
    train, val = temporal_train_val_split(train_day, val_fraction=0.2)
    return SplitResult(train=train, val=val, test=test_day, holdout=pd.DataFrame())


@pytest.fixture
def eval_ready_artifacts(tmp_path, monkeypatch):
    cfg = full_cfg_dict(tmp_path)
    scaler_dir = tmp_path / "scaler"
    scaler_dir.mkdir(parents=True, exist_ok=True)

    splits = _synthetic_split_with_test()
    windowed = build_windowed_splits(splits)
    scaler = fit_scaler(windowed["train"])
    scaler.save(scaler_dir / "robust_scaler.joblib", scaler_dir / "scaler_metadata.json")

    benign_mask = windowed["train"].stage_label == "benign"
    background = build_shap_background(scaler.transform(windowed["train"].X[benign_mask, -1, :]), n_centroids=10)
    save_background(background, scaler_dir / "shap_background.npy")

    train_one_seed(cfg, seed=0, epochs_override=1, windowed=windowed, scaler=scaler, device="cpu")
    train_heads_for_seed(cfg, seed=0, windowed=windowed, scaler=scaler, device="cpu")

    monkeypatch.setattr(run_eval_mod, "build_all_splits", lambda cfg: splits)
    return cfg, tmp_path


def test_run_eval_end_to_end_on_test_split(eval_ready_artifacts):
    cfg, tmp_path = eval_ready_artifacts
    results = run_eval_mod.run(cfg, seed=0, split_name="test", n_samples=10)

    assert "baselines" in results
    for name in ["lr_current_state", "lr_flattened_history", "persistence", "world_model", "oracle"]:
        assert name in results["baselines"]
        assert "f1" in results["baselines"][name]

    assert "ablations" in results
    assert "interpretation" in results["ablations"]["persistence"]
    assert "interpretation" in results["ablations"]["time_shuffle"]
    assert len(results["ablations"]["horizon_curve"]["auc_pr_by_k"]) == HORIZON_LENGTH

    assert "calibration" in results
    assert "lead_time" in results

    # Split-specific subdirectory (see run_eval.run): running eval for
    # multiple splits against the same metrics_dir must not have one split's
    # files silently overwrite another's.
    metrics_dir = tmp_path / "metrics" / "test"
    for name in ["baselines.json", "ablations.json", "calibration.json", "lead_time.json"]:
        content = json.loads((metrics_dir / name).read_text())
        assert content  # valid, non-empty JSON


def test_run_eval_does_not_clobber_metrics_across_splits(eval_ready_artifacts):
    """Regression test: the documented workflow runs run_eval.py once per
    split (test, then holdout) against the SAME metrics_dir. Before the
    split-subdirectory fix, the second run silently overwrote the first
    run's baselines.json/ablations.json/etc — exactly what happened running
    this by hand against real data."""
    cfg, tmp_path = eval_ready_artifacts
    run_eval_mod.run(cfg, seed=0, split_name="test", n_samples=5)
    test_baselines = json.loads((tmp_path / "metrics" / "test" / "baselines.json").read_text())

    # holdout is empty in this fixture's SplitResult, so run() returns early
    # without writing anything — the point here is only that it must not
    # touch the test split's already-written files.
    run_eval_mod.run(cfg, seed=0, split_name="holdout", n_samples=5)

    test_baselines_after = json.loads((tmp_path / "metrics" / "test" / "baselines.json").read_text())
    assert test_baselines == test_baselines_after
