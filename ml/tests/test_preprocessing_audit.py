import numpy as np

from nidra.data.normalize import FeatureScaler
from nidra.data.preprocessing_audit import audit_features, audit_markdown, write_audit
from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER
from tests.test_normalize import _raw_like


def test_audit_covers_every_feature_and_reports_the_fit(tmp_path):
    rng = np.random.default_rng(0)
    X = _raw_like(rng, n=3000)
    scaler = FeatureScaler().fit(X)
    report = write_audit(X, scaler, tmp_path)
    assert [r["feature"] for r in report["features"]] == FEATURE_ORDER
    assert report["n_active_rows"] == int((X[:, FEATURE_INDEX["is_active"]] > 0).sum())
    assert (tmp_path / "preprocessing_audit.json").exists() and (tmp_path / "preprocessing_audit.md").exists()
    md = audit_markdown(report)
    assert "| bytes_total | log1p |" in md


def test_audit_flags_a_transform_that_leaves_heavy_skew():
    """Declare a heavy-tailed feature as plain zscore: the audit must say
    log1p/asinh would do better, without changing the scaler."""
    rng = np.random.default_rng(1)
    X = _raw_like(rng, n=3000)
    kinds = [("zscore" if f == "bytes_total" else k) for f, k in
             zip(FEATURE_ORDER, FeatureScaler().kinds)]
    scaler = FeatureScaler(kinds=kinds).fit(X)
    report = audit_features(X, scaler)
    row = next(r for r in report["features"] if r["feature"] == "bytes_total")
    assert row["kind"] == "zscore"
    assert any("would reduce |skew|" in f for f in row["flags"])
    assert report["n_flagged"] >= 1


def test_audit_records_drops_and_correlation_partners():
    rng = np.random.default_rng(2)
    X = _raw_like(rng, n=2000)
    active = X[:, FEATURE_INDEX["is_active"]] > 0
    X[active, FEATURE_INDEX["reciprocity"]] = 1.0
    scaler = FeatureScaler().fit(X)
    report = audit_features(X, scaler)
    row = next(r for r in report["features"] if r["feature"] == "reciprocity")
    assert row["dropped"] and "constant" in row["drop_reason"]
    kept_rows = [r for r in report["features"] if not r["dropped"]]
    assert all(r["most_correlated"] is not None for r in kept_rows)
