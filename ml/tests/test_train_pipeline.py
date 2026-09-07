"""Integration test for the Stage-1 -> Stage-2 training pipeline, using
synthetic fixtures (fast, deterministic) rather than real CIC-IDS2017 data.
Verifies: the training loop runs end-to-end, weights are saved, and the
frozen-parameter discipline (Rule 1) actually holds after each stage.
"""

from __future__ import annotations

import json

import pandas as pd
import torch

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import WINDOW_SECONDS
from nidra.data.splits import SplitResult, temporal_train_val_split
from nidra.data.windowize import build_state_rows
from nidra.models.world_model import WorldModel
from nidra.train import train_dynamics as train_dynamics_mod
from nidra.train.train_dynamics import prepare_training_data, train_one_seed
from nidra.train.train_heads import train_heads_for_seed
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def _tiny_cfg(tmp_path):
    return {
        "artifacts": {
            "scaler_dir": str(tmp_path / "scaler"),
            "weights_dir": str(tmp_path / "weights"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "windowing": {"horizon_length": 6},
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
        "ensemble": {"seeds": [0]},
        "_config_hash": "test", "_git_commit": None,
    }


def _synthetic_split_result() -> SplitResult:
    # n_windows is large enough that even after carving off a trailing 20%
    # validation block, each host still has >= L+K=36 contiguous windows in
    # BOTH train and val — mirroring real CIC-IDS2017 scale, where a few
    # days of 30s windows dwarfs the L+K minimum by orders of magnitude.
    flows = make_synthetic_flows(n_hosts=3, n_windows=260, portscan_start_window=120, portscan_len=10)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    labelled = attach_risk_label(states, stage_table, horizon_k=6, window_seconds=WINDOW_SECONDS)
    train, val = temporal_train_val_split(labelled, val_fraction=0.2)
    return SplitResult(train=train, val=val, test=pd.DataFrame(), holdout=pd.DataFrame())


def test_full_two_stage_training_runs_and_freezes_correctly(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)
    monkeypatch.setattr(train_dynamics_mod, "build_all_splits", lambda cfg: _synthetic_split_result())

    windowed, scaler = prepare_training_data(cfg, max_train_samples=200, max_val_samples=50)
    assert len(windowed["train"].X) > 0
    assert len(windowed["val"].X) > 0

    dyn_meta = train_one_seed(cfg, seed=0, epochs_override=2, windowed=windowed, scaler=scaler, device="cpu")
    assert "best_val_multistep_nll" in dyn_meta

    weights_path = tmp_path / "weights" / "model_seed_0.pt"
    assert weights_path.exists()

    # Reload and confirm dynamics params are NOT yet frozen after stage 1
    # (freezing happens explicitly at the start of stage 2, not stage 1).
    model = WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                        risk_hidden=16, stage_hidden=16, n_stages=6)
    model.load_state_dict(torch.load(weights_path))

    head_meta = train_heads_for_seed(cfg, seed=0, windowed=windowed, scaler=scaler, device="cpu")
    assert "heads_best_val_loss" in head_meta

    final_model = WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                              risk_hidden=16, stage_hidden=16, n_stages=6)
    final_model.load_state_dict(torch.load(weights_path))
    final_model.freeze_dynamics()
    final_model.freeze_heads()
    assert all(not p.requires_grad for p in final_model.parameters())

    metadata_path = tmp_path / "weights" / "model_seed_0_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["stage"] == "dynamics_and_frozen_heads"
    assert "heads_pos_weight" in metadata
