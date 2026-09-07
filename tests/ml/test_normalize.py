import numpy as np
import pytest

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER


def test_transform_before_fit_raises():
    scaler = FeatureScaler()
    X = np.random.randn(10, 45)
    with pytest.raises(RuntimeError):
        scaler.transform(X)


def test_fit_then_transform_clips_to_range():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((1000, 45)) * 3
    scaler = FeatureScaler(clip_min=-10, clip_max=10).fit(X_train)
    X_extreme = np.full((1, 45), 1e9)
    out = scaler.transform(X_extreme)
    assert out.max() <= 10.0
    assert out.min() >= -10.0


def test_scaler_never_refit_by_val_or_test_data(tmp_path):
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((500, 45))
    X_val = rng.standard_normal((500, 45)) * 100 + 500  # wildly different distribution

    scaler = FeatureScaler().fit(X_train)
    center_before = scaler.scaler.center_.copy()
    scale_before = scaler.scaler.scale_.copy()

    _ = scaler.transform(X_val)  # transforming val data must not mutate fit stats

    np.testing.assert_array_equal(center_before, scaler.scaler.center_)
    np.testing.assert_array_equal(scale_before, scaler.scaler.scale_)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((200, 45))
    scaler = FeatureScaler().fit(X_train)
    scaler_path = tmp_path / "scaler.joblib"
    meta_path = tmp_path / "meta.json"
    scaler.save(scaler_path, meta_path, extra_metadata={"note": "test"})

    loaded = FeatureScaler.load(scaler_path, meta_path)
    X_test = rng.standard_normal((10, 45))
    np.testing.assert_allclose(scaler.transform(X_test), loaded.transform(X_test))


def test_load_rejects_schema_drift(tmp_path):
    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((100, 45))
    scaler = FeatureScaler().fit(X_train)
    scaler_path = tmp_path / "scaler.joblib"
    meta_path = tmp_path / "meta.json"
    scaler.save(scaler_path, meta_path)

    import json
    meta = json.loads(meta_path.read_text())
    meta["feature_order"] = FEATURE_ORDER[:-1] + ["renamed_feature"]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError):
        FeatureScaler.load(scaler_path, meta_path)
