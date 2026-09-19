"""The onset head trains on eligible (outside-episode) observed states only,
saves its own artifact, and scores every horizon."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nidra.models.heads import OnsetHead
from tests.test_head_selection import head_ready_artifacts  # noqa: F401  (fixture re-export)
from nidra.train.train_onset import onset_head_path, score_onset_head, train_onset_head_for_seed


def test_onset_head_round_trip(tmp_path):
    head = OnsetHead(45, 8, (1, 5))
    path = tmp_path / "h.pt"
    head.save(path)
    loaded = OnsetHead.load(path)
    x = torch.randn(3, 45)
    assert torch.allclose(head(x), loaded(x))
    assert loaded.horizons_min == (1, 5)
    assert all(not p.requires_grad for p in loaded.parameters())


def test_train_onset_head_on_fixture(head_ready_artifacts):
    cfg, _, windowed, scaler = head_ready_artifacts
    cfg = {**cfg, "onset": {"horizons_min": [1, 3, 5], "selection_horizon_min": 3, "epochs": 3, "hidden": 8,
                            "pos_repeat": 2, "neg_ratio": 2}}
    for split in ("train", "val"):
        assert windowed[split].inside_episode is not None
    meta = train_onset_head_for_seed(cfg, 0, windowed, scaler, "cpu")
    from nidra.utils.config import resolve_path
    path = onset_head_path(resolve_path(cfg, cfg["artifacts"]["weights_dir"]), 0)
    assert path.exists() and path.with_name(path.name + ".json").exists()
    assert meta["horizons_min"] == [1, 3, 5]
    assert set(meta["n_train_positives_by_horizon"]) == {"1", "3", "5"}
    X = windowed["val"].X[:, -1, :]
    from nidra.train.pipeline import scale_arrays
    Xs, _ = scale_arrays(windowed["val"], scaler)
    p, horizons = score_onset_head(path, Xs[:, -1, :])
    assert p.shape == (len(X), 3) and horizons == (1, 3, 5)
    assert np.all((p >= 0) & (p <= 1))
