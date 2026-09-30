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


class TestHistoryAwareOnsetHead:
    """The state-only onset head scored within noise of zero at every horizon
    on CTU-13's validation captures while the oracle reached 0.022, and warned
    0 of 8 episodes before onset. The risk-head ablation on the same data says
    the missing input is the encoder's hidden state, so the onset head is
    offered the same context — under the same discipline: observed states
    only, frozen afterwards."""

    def _cfg(self, base, components):
        return {**base, "onset": {"horizons_min": [1, 3, 5], "selection_horizon_min": 3, "epochs": 2,
                                  "hidden": 8, "pos_repeat": 2, "neg_ratio": 2, "components": components}}

    def test_declaring_context_without_a_model_is_refused(self, head_ready_artifacts):
        cfg, _, windowed, scaler = head_ready_artifacts
        with pytest.raises(ValueError, match="model"):
            train_onset_head_for_seed(self._cfg(cfg, ["state", "hidden"]), 0, windowed, scaler, "cpu")

    def _model(self, cfg):
        """The frozen stage-1 checkpoint the fixture trained."""
        import torch
        from nidra.models.build import world_model_from_config
        from nidra.utils.config import resolve_path
        m = world_model_from_config(cfg)
        weights = resolve_path(cfg, cfg["artifacts"]["weights_dir"]) / "model_seed_0.pt"
        m.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        m.eval()
        return m

    def test_it_trains_and_records_its_components(self, head_ready_artifacts):
        cfg, _, windowed, scaler = head_ready_artifacts
        model = self._model(cfg)
        meta = train_onset_head_for_seed(self._cfg(cfg, ["state", "hidden"]), 0, windowed, scaler, "cpu",
                                         model=model)
        assert meta["components"] == ["state", "hidden"]
        assert meta["input_dim"] == 45 + model.encoder.hidden_size

    def test_the_saved_head_reloads_and_scores_with_its_context(self, head_ready_artifacts):
        cfg, _, windowed, scaler = head_ready_artifacts
        model = self._model(cfg)
        train_onset_head_for_seed(self._cfg(cfg, ["state", "hidden"]), 0, windowed, scaler, "cpu", model=model)
        from nidra.utils.config import resolve_path
        path = onset_head_path(resolve_path(cfg, cfg["artifacts"]["weights_dir"]), 0)
        head = OnsetHead.load(path)
        assert head.components == ("state", "hidden")
        n = 4
        out = head(state=torch.zeros(n, 45), hidden=torch.zeros(n, model.encoder.hidden_size))
        assert out.shape == (n, 3)

    def test_the_state_only_default_is_unchanged(self, head_ready_artifacts):
        cfg, _, windowed, scaler = head_ready_artifacts
        meta = train_onset_head_for_seed(self._cfg(cfg, ["state"]), 0, windowed, scaler, "cpu")
        assert meta["components"] == ["state"]
        assert meta["input_dim"] == 45
