"""Shared fixtures. `trained_predictor` trains a tiny WorldModel end to end
on synthetic fixtures and wraps it in a real NidraPredictor, so both the
serving-contract tests and the latency benchmark test exercise the same
artifacts."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.splits import SplitResult, temporal_train_val_split
from nidra.data.windowize import build_state_rows
from nidra.explain.shap_runner import build_shap_background, save_background
from nidra.serve.predictor import NidraPredictor
from nidra.train.pipeline import build_windowed_splits, fit_scaler
from nidra.train.train_dynamics import train_one_seed
from nidra.train.train_heads import train_heads_for_seed
from tests.ml.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def full_cfg_dict(tmp_path):
    return {
        "ensemble": {"seeds": [0]},
        "windowing": {"window_seconds": WINDOW_SECONDS, "context_length": CONTEXT_LENGTH, "horizon_length": HORIZON_LENGTH,
                      "min_windows_per_host": CONTEXT_LENGTH + HORIZON_LENGTH},
        "model": {
            "n_features": 45,
            "encoder": {"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
            "transition": {"mlp_hidden": 32, "logvar_min": -6.0, "logvar_max": 3.0, "state_clamp": 10.0},
            "risk_head": {"hidden": 16},
            "stage_head": {"hidden": 16, "n_stages": 6},
        },
        "train_dynamics": {
            "lr": 1e-3, "weight_decay": 0.0, "batch_size": 16, "epochs": 2,
            "grad_clip": 1.0, "patience": 10, "horizon_discount": 0.85,
            "teacher_forcing": {"start_p": 1.0, "end_p": 0.3, "anneal_fraction_of_epochs": 0.6},
        },
        "train_heads": {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 16, "epochs": 2, "grad_clip": 1.0, "patience": 10},
        "rollout": {"n_samples_per_member": 10, "ci_low_quantile": 0.05, "ci_high_quantile": 0.95},
        "eval": {"risk_threshold": 0.75, "lead_time_persistence_windows": 2, "window_seconds": WINDOW_SECONDS},
        "serving": {"torch_num_threads": 2, "latency_target_ms": 300},
        "explain": {"shap_background_centroids": 10},
        "artifacts": {
            "scaler_dir": str(tmp_path / "scaler"),
            "weights_dir": str(tmp_path / "weights"),
            "processed_dir": str(tmp_path / "processed"),
            "metrics_dir": str(tmp_path / "metrics"),
        },
        "_config_hash": "test", "_git_commit": None,
    }


def synthetic_split_result() -> SplitResult:
    flows = make_synthetic_flows(n_hosts=3, n_windows=260, portscan_start_window=120, portscan_len=10)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    labelled = attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=WINDOW_SECONDS)
    train, val = temporal_train_val_split(labelled, val_fraction=0.2)
    return SplitResult(train=train, val=val, test=pd.DataFrame(), holdout=pd.DataFrame())


@pytest.fixture(scope="session")
def trained_predictor(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("predictor_artifacts")
    cfg = full_cfg_dict(tmp_path)

    scaler_dir = tmp_path / "scaler"
    scaler_dir.mkdir(parents=True, exist_ok=True)

    splits = synthetic_split_result()
    windowed = build_windowed_splits(splits)
    scaler = fit_scaler(windowed["train"])
    scaler.save(scaler_dir / "robust_scaler.joblib", scaler_dir / "scaler_metadata.json")

    benign_mask = windowed["train"].stage_label == "benign"
    benign_last_states = scaler.transform(windowed["train"].X[benign_mask, -1, :])
    background = build_shap_background(benign_last_states, n_centroids=10)
    save_background(background, scaler_dir / "shap_background.npy")

    train_one_seed(cfg, seed=0, epochs_override=2, windowed=windowed, scaler=scaler, device="cpu")
    train_heads_for_seed(cfg, seed=0, windowed=windowed, scaler=scaler, device="cpu")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))

    predictor = NidraPredictor(
        weights_dir=tmp_path / "weights",
        scaler_path=tmp_path / "scaler" / "robust_scaler.joblib",
        config_path=config_path,
        seeds=[0],
    )
    return predictor, windowed
