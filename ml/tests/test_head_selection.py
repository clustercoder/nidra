"""Head model selection must track ranking quality, and stage 2 must be
reproducible.

The risk head sees ~287 positive examples in 500,000 training rows at
production scale. Measured over 20 epochs on seed 0, the original
weighted-val-loss criterion selected epoch 0 (val AUC-PR 0.523) while epoch 4
reached 0.576 at a WORSE weighted loss (43.9 vs 37.7) — the criterion and the
objective disagree, and the head produces p_compromise, so everything
downstream inherits that choice.
"""

import numpy as np
import pytest
import torch

from nidra.train.train_heads import (
    VALID_SELECTION_METRICS,
    head_selection_score,
    train_heads_for_seed,
)

import pandas as pd

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.splits import SplitResult, temporal_train_val_split
from nidra.data.windowize import build_state_rows
from nidra.train.pipeline import build_windowed_splits, fit_scaler
from nidra.train.train_dynamics import train_one_seed
from tests.conftest import full_cfg_dict
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


@pytest.fixture
def head_ready_artifacts(tmp_path):
    """Splits, scaler and a stage-1 (dynamics-only) checkpoint — the state
    stage-2 head training expects to start from."""
    cfg = full_cfg_dict(tmp_path)
    (tmp_path / "scaler").mkdir(parents=True, exist_ok=True)

    def _day(portscan_start):
        flows = make_synthetic_flows(n_hosts=2, n_windows=200,
                                     portscan_start_window=portscan_start, portscan_len=10)
        packets = make_synthetic_packets(flows)
        states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
        stage_table, _ = label_stage_table(flows)
        return attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH,
                                 window_seconds=WINDOW_SECONDS)

    train_day = _day(150)
    train, val = temporal_train_val_split(train_day, val_fraction=0.2)
    splits = SplitResult(train=train, val=val, test=_day(100), holdout=pd.DataFrame())
    windowed = build_windowed_splits(splits)
    scaler = fit_scaler(windowed["train"])
    scaler.save(tmp_path / "scaler" / "feature_scaler.json",
                tmp_path / "scaler" / "scaler_metadata.json")
    train_one_seed(cfg, seed=0, epochs_override=1, windowed=windowed, scaler=scaler, device="cpu")
    return cfg, tmp_path, windowed, scaler



def test_auc_criterion_prefers_the_higher_auc_epoch_even_at_worse_loss():
    # The exact measured case: epoch 0 vs epoch 4 on seed 0.
    ep0 = head_selection_score(37.7, 0.523, "val_auc_pr")
    ep4 = head_selection_score(43.9, 0.576, "val_auc_pr")
    assert ep4 < ep0, "lower score is better; the higher-AUC epoch must win"


def test_loss_criterion_reproduces_the_original_choice():
    ep0 = head_selection_score(37.7, 0.523, "weighted_val_loss")
    ep4 = head_selection_score(43.9, 0.576, "weighted_val_loss")
    assert ep0 < ep4, "the original criterion selects the lower weighted loss"


def test_the_two_criteria_disagree_on_the_measured_data():
    pairs = [(37.7, 0.523), (43.9, 0.576)]
    by_auc = min(pairs, key=lambda p: head_selection_score(*p, "val_auc_pr"))
    by_loss = min(pairs, key=lambda p: head_selection_score(*p, "weighted_val_loss"))
    assert by_auc != by_loss, "if these ever agree, this whole change is a no-op"


def test_undefined_auc_never_wins_selection():
    # A single-class val split makes AUC-PR undefined; it must not be
    # selected over an epoch with a real measurement.
    assert head_selection_score(1.0, float("nan"), "val_auc_pr") == float("inf")
    assert head_selection_score(999.0, 0.1, "val_auc_pr") < head_selection_score(1.0, float("nan"), "val_auc_pr")


def test_unknown_selection_metric_raises():
    with pytest.raises(ValueError):
        head_selection_score(1.0, 0.5, "vibes")


def test_valid_metrics_are_the_documented_three():
    assert set(VALID_SELECTION_METRICS) == {"val_auc_pr", "val_auc_pr_natural", "weighted_val_loss"}


def test_natural_metric_scores_like_auc_pr():
    assert head_selection_score(1.0, 0.7, "val_auc_pr_natural") == pytest.approx(-0.7)
    assert head_selection_score(1.0, float("nan"), "val_auc_pr_natural") == float("inf")


def test_balanced_epoch_indices_repeats_positives_and_mixes_hard_negatives():
    from nidra.train.train_heads import balanced_epoch_indices
    rng = np.random.default_rng(0)
    risk = np.zeros(1000, dtype=int)
    risk[:5] = 1
    active = np.zeros(1000, dtype=bool)
    active[5:105] = True  # 100 active benign, 895 silent benign
    idx = balanced_epoch_indices(risk, active, rng, pos_repeat=4, neg_ratio=3, hard_negative_fraction=0.5)
    pos_count = int((risk[idx] == 1).sum())
    neg_count = int((risk[idx] == 0).sum())
    assert pos_count == 5 * 4
    assert neg_count == 5 * 4 * 3
    hard = int(active[idx][risk[idx] == 0].sum())
    assert hard == neg_count // 2
    # negatives outside the sampled set are untouched; the epoch is a permutation, not sorted
    assert not np.all(np.diff(idx) >= 0)


def test_balanced_epoch_indices_handles_no_active_negatives():
    from nidra.train.train_heads import balanced_epoch_indices
    rng = np.random.default_rng(0)
    risk = np.zeros(50, dtype=int)
    risk[:2] = 1
    active = np.zeros(50, dtype=bool)
    idx = balanced_epoch_indices(risk, active, rng, pos_repeat=2, neg_ratio=2, hard_negative_fraction=0.5)
    assert int((risk[idx] == 1).sum()) == 4
    assert int((risk[idx] == 0).sum()) == 8


def test_unknown_sampling_is_rejected(head_ready_artifacts):
    cfg, _, windowed, scaler = head_ready_artifacts
    bad = {**cfg, "train_heads": {**cfg["train_heads"], "risk_sampling": "whatever"}}
    with pytest.raises(ValueError, match="risk_sampling"):
        train_heads_for_seed(bad, seed=0, windowed=windowed, scaler=scaler, device="cpu")


def test_balanced_recipe_trains_and_records_recipe(head_ready_artifacts):
    cfg, _, windowed, scaler = head_ready_artifacts
    recipe = {**cfg["train_heads"], "risk_sampling": "balanced", "input_noise": 0.3, "pos_repeat": 3,
              "neg_ratio": 4, "hard_negative_fraction": 0.5, "selection_metric": "val_auc_pr_natural"}
    meta = train_heads_for_seed({**cfg, "train_heads": recipe}, seed=0, windowed=windowed, scaler=scaler, device="cpu")
    assert meta["heads_recipe"]["risk_sampling"] == "balanced"
    assert meta["heads_recipe"]["input_noise"] == pytest.approx(0.3)
    assert meta["heads_pos_weight"] is None  # balanced sampling: no pos_weight in the loss
    assert meta["heads_selection_metric"] == "val_auc_pr_natural"
    assert meta["heads_best_epoch_risk"] >= 0
    assert meta["heads_best_epoch_stage"] >= 0
    assert np.isfinite(meta["heads_best_val_auc_pr_natural"]) or np.isnan(meta["heads_best_val_auc_pr_natural"])


def _run_heads(cfg, tmp_path, windowed, scaler, metric):
    cfg = {**cfg, "train_heads": {**cfg["train_heads"], "selection_metric": metric}}
    return train_heads_for_seed(cfg, seed=0, windowed=windowed, scaler=scaler, device="cpu")


def test_stage_two_is_reproducible_when_rerun(head_ready_artifacts):
    """Re-running stage 2 must give the same heads, not continue training the
    ones already in the checkpoint — otherwise the result depends on how many
    times the script has been run."""
    cfg, tmp_path, windowed, scaler = head_ready_artifacts
    first = _run_heads(cfg, tmp_path, windowed, scaler, "val_auc_pr")
    weights_first = torch.load(tmp_path / "weights" / "model_seed_0.pt", map_location="cpu")
    risk_first = {k: v.clone() for k, v in weights_first.items() if k.startswith("risk_head")}

    second = _run_heads(cfg, tmp_path, windowed, scaler, "val_auc_pr")
    weights_second = torch.load(tmp_path / "weights" / "model_seed_0.pt", map_location="cpu")

    assert first["heads_selection_metric"] == "val_auc_pr"
    assert second["heads_best_val_loss"] == pytest.approx(first["heads_best_val_loss"], rel=1e-6)
    for key, value in risk_first.items():
        torch.testing.assert_close(weights_second[key], value)


def test_metadata_records_both_the_metric_and_the_auc(head_ready_artifacts):
    cfg, tmp_path, windowed, scaler = head_ready_artifacts
    meta = _run_heads(cfg, tmp_path, windowed, scaler, "val_auc_pr")
    assert meta["heads_selection_metric"] == "val_auc_pr"
    assert "heads_best_val_auc_pr" in meta
    assert all("val_auc_pr" in h for h in meta["heads_history"])
