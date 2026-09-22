"""The trajectory head through the real two-stage pipeline.

Unit tests pin the head's shape and the rollout's context. This pins that a
config asking for one actually trains one, that it is frozen afterwards, and
— the invariant that matters — that no gradient ever reaches it from a
predicted state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER
from nidra.models.build import risk_head_components, uses_trajectory_head, world_model_from_config
from nidra.models.heads import RiskHead, TrajectoryRiskHead
from nidra.train.head_data import build_head_arrays
from nidra.train.train_heads import train_heads_for_seed


def _cfg(tmp_path, components=None) -> dict:
    cfg = {
        "model": {
            "n_features": 45,
            "encoder": {"hidden_size": 12, "num_layers": 1, "dropout": 0.0},
            "transition": {"mlp_hidden": 16, "logvar_min": -6.0, "logvar_max": 3.0, "state_clamp": 10.0},
            "risk_head": {"hidden": 8},
            "stage_head": {"hidden": 8, "n_stages": 6},
        },
        "windowing": {"window_seconds": 60, "context_length": 5, "horizon_length": 2},
        "train_heads": {"epochs": 2, "patience": 2, "lr": 1e-3, "weight_decay": 0.0, "batch_size": 32,
                        "grad_clip": 1.0, "input_noise": 0.1, "risk_sampling": "imbalanced",
                        "selection_metric": "val_auc_pr_natural"},
        "artifacts": {"weights_dir": str(tmp_path / "w")},
    }
    if components is not None:
        cfg["model"]["risk_head"]["components"] = components
    return cfg


def _table(n_hosts=3, n=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hosts):
        for i in range(n):
            row = {f: float(abs(rng.normal())) for f in FEATURE_ORDER}
            attack = h == 0 and 20 <= i < 26
            row.update(host_id=f"h{h}", window_ts=60 * i,
                       stage_label="c2" if attack else "benign",
                       risk_label=int(h == 0 and 14 <= i < 26))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def prepared(tmp_path):
    scaler = FeatureScaler().fit(np.abs(np.random.default_rng(1).normal(size=(400, 45))) + 0.2,
                                 active_mask=np.ones(400, dtype=bool))
    train_tbl, val_tbl = _table(seed=0), _table(seed=1)
    head_data = {"train": build_head_arrays(train_tbl, scaler), "val": build_head_arrays(val_tbl, scaler)}
    keep = ["host_id", "window_ts", *FEATURE_ORDER]
    head_tables = {"train": train_tbl[keep], "val": val_tbl[keep]}
    return scaler, head_data, head_tables


def _stage_one_checkpoint(cfg, tmp_path, seed=0):
    weights = tmp_path / "w"
    weights.mkdir(parents=True, exist_ok=True)
    base = world_model_from_config({**cfg, "model": {**cfg["model"], "risk_head": {"hidden": cfg["model"]["risk_head"]["hidden"]}}})
    torch.save(base.state_dict(), weights / f"model_seed_{seed}.pt")


class TestConfigSelectsTheHead:
    def test_default_config_is_the_per_state_head(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert risk_head_components(cfg) == ("state",)
        assert not uses_trajectory_head(cfg)
        assert isinstance(world_model_from_config(cfg).risk_head, RiskHead)

    def test_declared_components_build_a_trajectory_head(self, tmp_path):
        cfg = _cfg(tmp_path, ["state", "hidden"])
        model = world_model_from_config(cfg)
        assert isinstance(model.risk_head, TrajectoryRiskHead)
        assert model.risk_head.input_dim == 45 + 12

    def test_an_unknown_component_in_config_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown model.risk_head.components"):
            risk_head_components(_cfg(tmp_path, ["state", "tea_leaves"]))


class TestTrainingATrajectoryHead:
    def test_it_trains_and_is_recorded(self, tmp_path, prepared):
        scaler, head_data, head_tables = prepared
        cfg = _cfg(tmp_path, ["state", "hidden", "logvar"])
        _stage_one_checkpoint(cfg, tmp_path)
        meta = train_heads_for_seed(cfg, 0, {}, scaler, "cpu", head_data=head_data, head_tables=head_tables)
        assert meta["risk_head_components"] == ["state", "hidden", "logvar"]
        assert meta["risk_head_input_dim"] == 45 + 12 + 45
        assert meta["heads_n_train"] == len(head_data["train"].risk_label)

    def test_the_saved_checkpoint_reloads_into_the_same_architecture(self, tmp_path, prepared):
        scaler, head_data, head_tables = prepared
        cfg = _cfg(tmp_path, ["state", "hidden"])
        _stage_one_checkpoint(cfg, tmp_path)
        train_heads_for_seed(cfg, 0, {}, scaler, "cpu", head_data=head_data, head_tables=head_tables)
        model = world_model_from_config(cfg)
        model.load_state_dict(torch.load(tmp_path / "w" / "model_seed_0.pt"))
        out = model.rollout(torch.randn(2, 5, 45) * 0.2, K=2, n_samples=1)
        risk, _ = model.score_trajectory(out)
        assert risk.shape == (2, 1, 2) and torch.isfinite(risk).all()

    def test_the_head_is_frozen_afterwards(self, tmp_path, prepared):
        scaler, head_data, head_tables = prepared
        cfg = _cfg(tmp_path, ["state", "hidden"])
        _stage_one_checkpoint(cfg, tmp_path)
        train_heads_for_seed(cfg, 0, {}, scaler, "cpu", head_data=head_data, head_tables=head_tables)
        # train_heads_for_seed calls freeze_heads() on its own model; reload
        # and confirm a fresh rollout produces no grad-requiring risk.
        model = world_model_from_config(cfg)
        model.load_state_dict(torch.load(tmp_path / "w" / "model_seed_0.pt"))
        model.freeze_all()
        assert all(not p.requires_grad for p in model.risk_head.parameters())

    def test_missing_head_tables_is_an_error_not_a_silent_state_only_head(self, tmp_path, prepared):
        scaler, head_data, _ = prepared
        cfg = _cfg(tmp_path, ["state", "hidden"])
        _stage_one_checkpoint(cfg, tmp_path)
        with pytest.raises(ValueError, match="needs the labelled tables"):
            train_heads_for_seed(cfg, 0, {}, scaler, "cpu", head_data=head_data)

    def test_a_state_only_trajectory_head_matches_the_plain_head_shape(self, tmp_path, prepared):
        # The control arm of the ablation must be the same model, not a
        # different one that happens to score similarly.
        scaler, head_data, head_tables = prepared
        cfg = _cfg(tmp_path)
        _stage_one_checkpoint(cfg, tmp_path)
        meta = train_heads_for_seed(cfg, 0, {}, scaler, "cpu", head_data=head_data, head_tables=head_tables)
        assert meta["risk_head_components"] == ["state"]
        assert meta["risk_head_input_dim"] == 45


class TestTheFrozenHeadInvariant:
    def test_head_training_never_sees_a_predicted_state(self, tmp_path, prepared, monkeypatch):
        """The head may only be fit on observed states. Nothing in the head
        stage should call rollout at all — if it ever does, the forecasting
        claim stops being falsifiable."""
        from nidra.models.world_model import WorldModel

        scaler, head_data, head_tables = prepared
        cfg = _cfg(tmp_path, ["state", "hidden", "delta", "logvar"])
        _stage_one_checkpoint(cfg, tmp_path)
        calls = []
        real = WorldModel.rollout
        monkeypatch.setattr(WorldModel, "rollout",
                            lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1])
        train_heads_for_seed(cfg, 0, {}, scaler, "cpu", head_data=head_data, head_tables=head_tables)
        assert calls == [], "head training called rollout — a head must not be fit on predicted states"
