"""The hazard parameterisation of the onset head, end to end."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER
from nidra.models.heads import OnsetHead
from nidra.train.train_onset import onset_head_path, score_onset_head, train_onset_head_for_seed


class _Head:
    """Minimal HeadArrays stand-in with onset geometry."""

    def __init__(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        self.states = rng.normal(size=(n, 45)).astype("float32")
        self.active = np.ones(n, dtype=bool)
        self.inside_episode = np.zeros(n, dtype=bool)
        self.inside_episode[:20] = True
        self.minutes_to_onset = np.full(n, np.inf)
        # a clean signal: feature 0 high => onset soon
        soon = rng.choice(np.arange(20, n), size=60, replace=False)
        self.minutes_to_onset[soon] = rng.uniform(0.5, 25.0, size=60)
        self.states[soon, 0] += 4.0
        self.risk_label = np.zeros(n, dtype="int64")
        self.stage_idx = np.zeros(n, dtype="int64")

    def onset_targets(self, horizons):
        from nidra.data.onset import onset_targets
        return onset_targets(self.inside_episode, self.minutes_to_onset, tuple(horizons))

    def __len__(self):
        return len(self.risk_label)


def _cfg(tmp_path, parameterisation):
    return {
        "onset": {"horizons_min": [1, 3, 5, 10, 15, 30], "selection_horizon_min": 5, "hidden": 8,
                  "epochs": 6, "patience": 6, "lr": 5e-3, "weight_decay": 0.0, "batch_size": 64,
                  "input_noise": 0.0, "pos_repeat": 10, "neg_ratio": 5, "hard_negative_fraction": 0.5,
                  "parameterisation": parameterisation},
        "artifacts": {"weights_dir": str(tmp_path / "w")},
    }


@pytest.fixture
def head_data():
    return {"train": _Head(seed=0), "val": _Head(seed=1)}


def test_the_hazard_head_trains_and_records_its_parameterisation(tmp_path, head_data):
    meta = train_onset_head_for_seed(_cfg(tmp_path, "hazard"), 0, {}, FeatureScaler(), head_data=head_data)
    assert meta["parameterisation"] == "hazard"
    assert sum(meta["n_events_by_bucket"]) > 0
    # later buckets have fewer rows still at risk
    assert meta["n_at_risk_by_bucket"] == sorted(meta["n_at_risk_by_bucket"], reverse=True)


def test_hazard_probabilities_are_monotone_in_the_horizon(tmp_path, head_data):
    train_onset_head_for_seed(_cfg(tmp_path, "hazard"), 0, {}, FeatureScaler(), head_data=head_data)
    p, horizons = score_onset_head(onset_head_path(tmp_path / "w", 0), head_data["val"].states)
    assert list(horizons) == [1, 3, 5, 10, 15, 30]
    assert (np.diff(p, axis=1) >= -1e-9).all(), "P(onset within h) must not decrease with h"


def test_the_independent_head_is_unchanged_and_may_be_non_monotone(tmp_path, head_data):
    meta = train_onset_head_for_seed(_cfg(tmp_path, "independent"), 0, {}, FeatureScaler(), head_data=head_data)
    assert meta["parameterisation"] == "independent"
    p, _ = score_onset_head(onset_head_path(tmp_path / "w", 0), head_data["val"].states)
    assert p.shape[1] == 6
    # not asserting non-monotonicity (it is a possibility, not a guarantee) —
    # only that nothing enforces monotonicity in this arm.
    assert getattr(OnsetHead.load(onset_head_path(tmp_path / "w", 0)), "parameterisation") == "independent"


def test_an_unknown_parameterisation_is_refused(tmp_path, head_data):
    with pytest.raises(ValueError, match="unknown onset.parameterisation"):
        train_onset_head_for_seed(_cfg(tmp_path, "bayesian_vibes"), 0, {}, FeatureScaler(), head_data=head_data)


def test_a_head_saved_before_the_parameterisation_existed_still_loads(tmp_path):
    head = OnsetHead(45, 8, (1, 3, 5, 10, 15, 30))
    path = tmp_path / "old.pt"
    torch.save({"state_dict": head.state_dict(), "n_features": 45, "hidden": 8,
                "horizons_min": head.horizons_min}, path)
    back = OnsetHead.load(path)
    assert back.parameterisation == "independent"
