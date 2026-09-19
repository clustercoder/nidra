import json

import numpy as np
import pytest

from nidra.data.normalize import FeatureScaler, apply_kind, invert_kind
from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER, FEATURE_TRANSFORMS


def _raw_like(rng: np.random.Generator, n: int = 1000, silent_fraction: float = 0.5) -> np.ndarray:
    """Raw-unit-like training states: heavy-tailed positives on the log1p
    features, signed heavy tails on the asinh ones, rates in [0,1], plus a
    block of all-zero (silent) rows like a real training population."""
    X = np.zeros((n, 45))
    for name, idx in FEATURE_INDEX.items():
        kind = FEATURE_TRANSFORMS[name]
        if kind == "log1p":
            X[:, idx] = rng.lognormal(mean=4.0, sigma=2.0, size=n)
        elif kind == "asinh":
            X[:, idx] = rng.standard_cauchy(size=n) * 100
        elif kind == "zscore":
            X[:, idx] = rng.normal(loc=60, scale=20, size=n)
        else:  # unit
            X[:, idx] = rng.uniform(0, 1, size=n)
    X[:, FEATURE_INDEX["is_active"]] = 1.0
    silent = rng.random(n) < silent_fraction
    X[silent] = 0.0
    return X


def test_transform_before_fit_raises():
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.random.randn(10, 45))


def test_fit_then_transform_clips_to_range():
    rng = np.random.default_rng(0)
    scaler = FeatureScaler(clip_min=-10, clip_max=10).fit(_raw_like(rng))
    out = scaler.transform(np.full((1, 45), 1e9))
    assert out.max() <= 10.0 and out.min() >= -10.0


def test_statistics_come_from_active_rows_only():
    """The reason the v1 scaler was the identity: 98% silent rows put both
    quartiles at zero. Silent rows must not enter the fit statistics."""
    rng = np.random.default_rng(1)
    X = _raw_like(rng, n=2000, silent_fraction=0.9)
    scaler = FeatureScaler().fit(X)
    active = X[:, FEATURE_INDEX["is_active"]] > 0
    idx = FEATURE_INDEX["bytes_total"]
    expected_center = np.log1p(X[active, idx]).mean()
    assert abs(scaler.center_[idx] - expected_center) < 1e-9
    # scaled active rows are standardized; the silent row is a distinct fixed point
    Z = scaler.transform(X)
    assert abs(Z[active, idx].mean()) < 1e-6 and abs(Z[active, idx].std() - 1.0) < 1e-6
    zero = scaler.zero_state_scaled()
    assert np.allclose(Z[~active], zero)
    assert zero[idx] < -1.0                      # silence sits well below typical traffic
    assert zero[FEATURE_INDEX["syn_ratio"]] == 0.0  # unit-range features are not centred


def test_heavy_tailed_features_no_longer_saturate_the_clip():
    """Twelve features were effectively binary under the identity scaler
    (clip saturation 0.45-0.90 on active rows). After log/asinh + z-score the
    saturation rate on active rows must be negligible."""
    rng = np.random.default_rng(2)
    X = _raw_like(rng, n=5000)
    scaler = FeatureScaler().fit(X)
    Z = scaler.transform(X)
    active = X[:, FEATURE_INDEX["is_active"]] > 0
    saturated = (np.abs(Z[active]) >= 10.0).mean(axis=0)
    assert saturated.max() < 0.01


def test_constant_feature_is_dropped_and_zeroed():
    rng = np.random.default_rng(3)
    X = _raw_like(rng)
    active = X[:, FEATURE_INDEX["is_active"]] > 0
    X[active, FEATURE_INDEX["reciprocity"]] = 1.0     # constant on every active row
    scaler = FeatureScaler().fit(X)
    assert "reciprocity" in scaler.dropped_features
    assert "constant" in scaler.drop_reason_["reciprocity"]
    assert "is_active" not in scaler.dropped_features   # constant on active rows by construction, exempt
    assert not scaler.model_mask[FEATURE_INDEX["reciprocity"]]
    assert (scaler.transform(X)[:, FEATURE_INDEX["reciprocity"]] == 0.0).all()


def test_exact_duplicate_feature_is_dropped():
    rng = np.random.default_rng(4)
    X = _raw_like(rng)
    X[:, FEATURE_INDEX["flow_duration_var"]] = X[:, FEATURE_INDEX["iat_var"]]  # an exact copy
    scaler = FeatureScaler().fit(X)
    # the later feature in FEATURE_ORDER is the one dropped
    assert "iat_var" in scaler.dropped_features
    assert scaler.drop_reason_["iat_var"].startswith("duplicate of flow_duration_var")
    assert "flow_duration_var" not in scaler.dropped_features
    assert "is_active" not in scaler.dropped_features


def test_declared_unit_and_zscore_kinds_are_respected():
    rng = np.random.default_rng(5)
    X = _raw_like(rng)
    scaler = FeatureScaler().fit(X)
    assert scaler.kinds[FEATURE_INDEX["syn_ratio"]] == "unit"
    assert scaler.kinds[FEATURE_INDEX["ttl_mean"]] == "zscore"
    assert scaler.kinds[FEATURE_INDEX["d_iat_var"]] == "asinh"
    assert scaler.center_[FEATURE_INDEX["syn_ratio"]] == 0.0 and scaler.scale_[FEATURE_INDEX["syn_ratio"]] == 1.0


def test_scaler_never_refit_by_val_or_test_data():
    rng = np.random.default_rng(6)
    scaler = FeatureScaler().fit(_raw_like(rng))
    center_before, scale_before = scaler.center_.copy(), scaler.scale_.copy()
    _ = scaler.transform(rng.standard_normal((500, 45)) * 100 + 500)
    np.testing.assert_array_equal(center_before, scaler.center_)
    np.testing.assert_array_equal(scale_before, scaler.scale_)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    X = _raw_like(rng)
    scaler = FeatureScaler().fit(X)
    scaler_path, meta_path = FeatureScaler.default_paths(tmp_path)
    scaler.save(scaler_path, meta_path, extra_metadata={"note": "test"})
    loaded = FeatureScaler.load(scaler_path, meta_path)
    X_test = _raw_like(rng, n=50)
    np.testing.assert_allclose(scaler.transform(X_test), loaded.transform(X_test))
    assert loaded.kinds == scaler.kinds and loaded.dropped_features == scaler.dropped_features
    meta = json.loads(meta_path.read_text())
    assert meta["method"] == "feature_transforms+zscore_on_active_rows"
    assert meta["transforms"]["bytes_total"] == "log1p"
    assert scaler_path.suffix == ".json"   # no pickled objects


def test_reference_std_computed_on_fit_and_roundtrips(tmp_path):
    rng = np.random.default_rng(8)
    scaler = FeatureScaler().fit(_raw_like(rng))
    assert scaler.reference_std_.shape == (45,) and (scaler.reference_std_ >= 0).all()
    scaler_path, meta_path = FeatureScaler.default_paths(tmp_path)
    scaler.save(scaler_path, meta_path)
    np.testing.assert_allclose(FeatureScaler.load(scaler_path, meta_path).reference_std_, scaler.reference_std_)


def test_inverse_transform_roundtrips():
    rng = np.random.default_rng(9)
    X = _raw_like(rng)
    scaler = FeatureScaler(clip_min=-1e9, clip_max=1e9).fit(X)
    kept = scaler.model_mask
    X_back = scaler.inverse_transform(scaler.transform(X))
    np.testing.assert_allclose(X_back[:, kept], X[:, kept], rtol=1e-6, atol=1e-6)


def test_inverse_transform_bounds_log_feature_blowup():
    rng = np.random.default_rng(10)
    scaler = FeatureScaler().fit(_raw_like(rng))
    bad = np.zeros((1, 45))
    bad[0, FEATURE_INDEX["bytes_total"]] = 10.0
    raw = scaler.inverse_transform(bad)
    assert np.isfinite(raw).all()
    assert raw[0, FEATURE_INDEX["bytes_total"]] < 1e12


def test_inverse_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureScaler().inverse_transform(np.random.randn(1, 45))


def test_apply_and_invert_kind_are_inverses():
    x = np.array([-50.0, -1.0, 0.0, 1.0, 5e5])
    np.testing.assert_allclose(invert_kind(apply_kind(x, "asinh"), "asinh"), x, rtol=1e-9)
    xp = np.array([0.0, 1.0, 5e5])
    np.testing.assert_allclose(invert_kind(apply_kind(xp, "log1p"), "log1p"), xp, rtol=1e-9)
    with pytest.raises(ValueError):
        apply_kind(x, "sqrt")


def test_load_rejects_schema_drift(tmp_path):
    rng = np.random.default_rng(11)
    scaler = FeatureScaler().fit(_raw_like(rng))
    scaler_path, meta_path = FeatureScaler.default_paths(tmp_path)
    scaler.save(scaler_path, meta_path)
    meta = json.loads(meta_path.read_text())
    meta["feature_order"] = FEATURE_ORDER[:-1] + ["renamed_feature"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        FeatureScaler.load(scaler_path, meta_path)


def test_fit_requires_active_rows():
    with pytest.raises(ValueError):
        FeatureScaler().fit(np.zeros((100, 45)))
