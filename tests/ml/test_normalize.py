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


def test_reference_std_computed_on_fit_and_roundtrips(tmp_path):
    rng = np.random.default_rng(4)
    X_train = rng.standard_normal((300, 45))
    scaler = FeatureScaler().fit(X_train)

    assert scaler.reference_std_ is not None
    assert scaler.reference_std_.shape == (45,)
    assert (scaler.reference_std_ >= 0).all()

    scaler_path, meta_path = tmp_path / "scaler.joblib", tmp_path / "meta.json"
    scaler.save(scaler_path, meta_path)
    loaded = FeatureScaler.load(scaler_path, meta_path)
    np.testing.assert_allclose(loaded.reference_std_, scaler.reference_std_)


def test_inverse_transform_roundtrips():
    """A consumer of predicted_features (e.g. serve/predictor.py) must get
    raw units back, not the model's internal scaled+log1p representation —
    this is the round trip that guarantees that."""
    rng = np.random.default_rng(6)
    X = np.abs(rng.standard_normal((100, 45))) * 50  # positive, raw-magnitude-like
    scaler = FeatureScaler(clip_min=-1e9, clip_max=1e9).fit(X)  # disable clipping for a clean roundtrip
    X_scaled = scaler.transform(X)
    X_back = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_back, X, rtol=1e-4, atol=1e-4)


def test_inverse_transform_bounds_log1p_feature_blowup():
    """Regression test: a rollout prediction that is merely off by a modest
    amount in scaled space (state_nrmse ~5-7 is normal per REAL_DATA_RESULTS.md)
    must not invert to a physically nonsensical raw value for the 5 log1p
    features — observed directly as RMSE in the 10^11-10^14 range against the
    real trained ensemble before this fix."""
    rng = np.random.default_rng(7)
    X_train = np.abs(rng.standard_normal((500, 45))) * 100
    scaler = FeatureScaler().fit(X_train)

    # A scaled value far outside the observed training range (e.g. an
    # under-trained rollout diverging) for bytes_total (LOG1P_FEATURES[0]).
    from nidra.data.schema import FEATURE_INDEX
    bad_scaled = np.zeros((1, 45))
    bad_scaled[0, FEATURE_INDEX["bytes_total"]] = 10.0  # at the clip_max boundary

    raw = scaler.inverse_transform(bad_scaled)
    assert np.isfinite(raw).all()
    assert raw[0, FEATURE_INDEX["bytes_total"]] < 1e12, "log1p inversion must not blow up exponentially"


def test_inverse_transform_before_fit_raises():
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError):
        scaler.inverse_transform(np.random.randn(1, 45))


def test_reference_std_near_zero_for_constant_feature(tmp_path):
    """A feature that is exactly 0 for every training row (e.g. a packet
    aggregate on a flow-only day) must report a near-zero reference std, not
    NaN or an inflated value — this is what eval/metrics.state_nrmse floors
    against instead of a fragile per-eval-batch std."""
    rng = np.random.default_rng(5)
    X_train = rng.standard_normal((500, 45))
    X_train[:, 0] = 0.0  # syn_ratio held constant across the whole population
    scaler = FeatureScaler().fit(X_train)
    assert scaler.reference_std_[0] < 1e-6


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
