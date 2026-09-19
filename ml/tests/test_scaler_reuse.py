"""A scaler on disk is only reusable if it was fit on the data now in hand.

`prepare_training_data` deliberately reuses a saved scaler across every
ensemble seed — refitting per seed would give the five members five different
input spaces. The hazard is that the same code path also reuses it across a
change to the DATA, where it is silently wrong: the corrected flow timebase
(see windowize.parse_cic_timestamp) turned 11 of the 45 features from
~always-zero into populated ones, and a RobustScaler fit on the zero version
has a degenerate scale for exactly those columns. Nothing raises; the model
just trains on a mangled input space.

So reuse is conditional on a stamp, and these tests pin both halves of that:
reuse when the stamp matches, refit when it does not.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER
from nidra.data.windowize import CIC2017_TIMEBASE_TAG
from nidra.train import train_dynamics as td


class _Arrays:
    """Enough of WindowedArrays for the scaler-fitting path."""

    def __init__(self, n: int = 64):
        rng = np.random.default_rng(0)
        self.X = rng.random((n, 4, len(FEATURE_ORDER))).astype("float32")
        self.Y = rng.random((n, 2, len(FEATURE_ORDER))).astype("float32")
        self.stage_label = np.array(["benign"] * n)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Run prepare_training_data with the expensive parts stubbed out, and
    report whether the scaler was fit or loaded."""
    cfg = {
        "artifacts": {"root": str(tmp_path), "scaler_dir": str(tmp_path / "scaler")},
        "training_data": {"max_train_samples": 10, "max_val_samples": 10},
        "explain": {"shap_background_centroids": 4},
    }
    monkeypatch.setattr(td, "resolve_path", lambda _cfg, p: __import__("pathlib").Path(p))
    monkeypatch.setattr(td, "build_all_splits",
                        lambda _cfg: SimpleNamespace(train=object(), val=object()))
    monkeypatch.setattr(td, "build_windowed_arrays", lambda *a, **k: _Arrays())

    fits: list[int] = []
    real_fit = td.fit_scaler
    monkeypatch.setattr(td, "fit_scaler", lambda arrays: (fits.append(1), real_fit(arrays))[1])
    # The SHAP background is not what these tests are about.
    monkeypatch.setattr(td, "build_shap_background", lambda x, n_centroids: x[:n_centroids])
    monkeypatch.setattr(td, "save_background", lambda *a, **k: None)

    def run():
        fits.clear()
        td.prepare_training_data(cfg, None, None)
        return len(fits)

    return cfg, tmp_path / "scaler", run


def test_the_scaler_is_fit_once_and_reused_across_seeds(prepared):
    """The reason this reuse exists: five ensemble members must share one
    input space."""
    _, _, run = prepared
    assert run() == 1          # nothing on disk yet
    assert run() == 0          # second call reuses it


def test_the_saved_scaler_records_the_timebase_it_was_fit_on(prepared):
    _, scaler_dir, run = prepared
    run()
    meta = json.loads((scaler_dir / "scaler_metadata.json").read_text())
    assert meta["flow_timebase"] == CIC2017_TIMEBASE_TAG


def test_a_scaler_fit_on_a_different_timebase_is_refit_not_reused(prepared):
    """The failure this exists for. A scaler carried across the timebase fix
    scales 11 features by a spread measured when they were all zero, and
    nothing in the run says so."""
    _, scaler_dir, run = prepared
    run()
    meta_path = scaler_dir / "scaler_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["flow_timebase"] = "some-older-timebase"
    meta_path.write_text(json.dumps(meta))

    assert run() == 1
    assert json.loads(meta_path.read_text())["flow_timebase"] == CIC2017_TIMEBASE_TAG


def test_a_scaler_from_before_the_stamp_existed_is_refit(prepared):
    """Artifacts saved before this check was added carry no stamp. They were
    all fit on the uncorrected timebase, so an absent stamp is a mismatch, not
    a pass."""
    _, scaler_dir, run = prepared
    run()
    meta_path = scaler_dir / "scaler_metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["flow_timebase"]
    meta_path.write_text(json.dumps(meta))

    assert run() == 1


def test_feature_order_drift_is_still_refused_outright(tmp_path):
    """Unrelated to the timebase, and a harder failure: a scaler whose columns
    mean something else cannot be repaired by refitting the caller's data."""
    scaler = FeatureScaler()
    scaler.fit(_Arrays().X[:, -1, :])
    scaler_path, meta_path = tmp_path / "s.joblib", tmp_path / "s.json"
    scaler.save(scaler_path, meta_path)
    meta = json.loads(meta_path.read_text())
    meta["feature_order"] = list(reversed(FEATURE_ORDER))
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="feature order"):
        FeatureScaler.load(scaler_path, meta_path)
