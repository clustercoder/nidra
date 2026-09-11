"""Integration test for the top-level eval CLI (nidra.eval.run_eval),
against synthetic fixtures — verifies the full baselines/ablations/
calibration/lead-time pipeline runs end to end and writes valid JSON,
not real-data numbers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.splits import SplitResult, temporal_train_val_split
from nidra.data.windowize import build_state_rows
from nidra.eval import run_eval as run_eval_mod
from nidra.eval.calibrate import save_calibration
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


def test_run_eval_without_calibration_file_omits_calibrated_sections(eval_ready_artifacts):
    """Backward compatibility: a model that never ran fit_calibration must
    evaluate exactly as before — no "_calibrated"/"recalibrated" sections
    anywhere, not an error."""
    cfg, tmp_path = eval_ready_artifacts
    assert not (tmp_path / "weights" / "risk_calibration.json").exists()

    results = run_eval_mod.run(cfg, seed=0, split_name="test", n_samples=10)

    assert "world_model_calibrated" not in results["baselines"]
    assert "calibration_fit_metadata" not in results["baselines"]
    lead_time_json = json.loads((tmp_path / "metrics" / "test" / "lead_time.json").read_text())
    assert "calibrated" not in lead_time_json


def test_run_eval_applies_calibration_when_present(eval_ready_artifacts):
    """When nidra.scripts.fit_calibration has been run (risk_calibration.json
    exists next to the weights), run_eval must report a calibrated variant
    of the world-model baseline, calibration, and lead-time sections
    alongside the raw ones — never replacing them."""
    cfg, tmp_path = eval_ready_artifacts
    params_by_k = [{"a": 2.0, "b": 0.0, "n": 100, "degenerate": False} for _ in range(6)]
    save_calibration(
        tmp_path / "weights" / "risk_calibration.json", params_by_k,
        {"fit_split": "val", "n_val_samples": 100},
    )

    results = run_eval_mod.run(cfg, seed=0, split_name="test", n_samples=10)

    assert "world_model_calibrated" in results["baselines"]
    assert "f1" in results["baselines"]["world_model_calibrated"]
    assert results["baselines"]["calibration_fit_metadata"]["applied_to_single_seed_approximation"] is True

    calibration_json = json.loads((tmp_path / "metrics" / "test" / "calibration.json").read_text())
    assert "calibration_recalibrated" in calibration_json
    assert len(calibration_json["calibration_recalibrated"]["per_horizon"]) == HORIZON_LENGTH

    lead_time_json = json.loads((tmp_path / "metrics" / "test" / "lead_time.json").read_text())
    assert "raw" in lead_time_json and "calibrated" in lead_time_json


def test_run_eval_main_cli_prints_summary_without_crashing_when_calibrated(eval_ready_artifacts, monkeypatch, capsys):
    """Regression test: main()'s trailing summary print assumed every entry
    in results["baselines"] was a {f1, auc_pr, ...} metrics dict. Adding
    "calibration_fit_metadata" (a provenance dict, not a metrics row) broke
    that assumption and crashed with KeyError('f1') AFTER run() had already
    computed and written every metrics JSON correctly — caught running this
    exact CLI invocation by hand against the real trained ensemble."""
    cfg, tmp_path = eval_ready_artifacts
    params_by_k = [{"a": 1.5, "b": -0.5, "n": 50, "degenerate": False} for _ in range(HORIZON_LENGTH)]
    save_calibration(tmp_path / "weights" / "risk_calibration.json", params_by_k, {"fit_split": "val"})

    config_path = tmp_path / "config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
    monkeypatch.setattr(
        "sys.argv",
        ["run_eval", "--config", str(config_path), "--seed", "0", "--split", "test", "--n-samples", "5"],
    )
    run_eval_mod.main()
    assert "baselines:" in capsys.readouterr().out


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
