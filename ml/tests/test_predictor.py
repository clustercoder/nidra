"""Integration test for NidraPredictor against a tiny model trained on
synthetic fixtures (see tests/conftest.py::trained_predictor) — verifies the
serving CONTRACT (shapes, schema validation, output fields), not real-data
forecast quality.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER, HORIZON_LENGTH
from nidra.eval.calibrate import save_calibration
from nidra.explain.counterfactual import COUNTERFACTUAL_LABEL
from nidra.serve.predictor import MODEL_VERSION, NidraPredictor


def test_forecast_output_contract(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]  # [L, F] raw
    origin_ts = datetime(2017, 7, 4, 9, 0, 0, tzinfo=timezone.utc)

    result = predictor.forecast(states, host_id="10.0.0.1", origin_ts=origin_ts)

    assert result["host_id"] == "10.0.0.1"
    assert len(result["horizons"]) == HORIZON_LENGTH
    for h in result["horizons"]:
        assert set(h.keys()) >= {"k", "ts", "p_compromise", "ci_low", "ci_high", "stage_dist", "predicted_features"}
        assert 0.0 <= h["p_compromise"] <= 1.0
        assert len(h["predicted_features"]) == len(FEATURE_ORDER)
    assert "model_version" in result
    assert "schema_ver" in result
    assert result["lead_time_s"] is None or result["lead_time_s"] >= 0
    assert len(result["top_signals"]) <= 5


def test_predicted_features_are_in_raw_units_not_scaled_space(trained_predictor):
    """predicted_features must be inverse-transformed back to raw units —
    a forecast consumer (dashboard, backend) never sees the model's internal
    RobustScaler+log1p space. Verified by round-tripping through the SAME
    scaler the predictor holds: re-scaling the returned raw features must
    land back in the model's valid scaled/clipped operating range."""
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    origin_ts = datetime(2017, 7, 4, 9, 0, 0, tzinfo=timezone.utc)
    result = predictor.forecast(states, host_id="10.0.0.1", origin_ts=origin_ts)

    raw = np.array([result["horizons"][0]["predicted_features"][name] for name in FEATURE_ORDER])
    rescaled = predictor.scaler.transform(raw[None, :])[0]
    assert np.isfinite(rescaled).all()
    assert (rescaled >= predictor.scaler.clip_min - 1e-6).all()
    assert (rescaled <= predictor.scaler.clip_max + 1e-6).all()


def test_forecast_rejects_wrong_feature_width(trained_predictor):
    predictor, windowed = trained_predictor
    bad_states = windowed["train"].X[0][:, :-1]  # 44 features instead of 45
    with pytest.raises(ValueError):
        predictor.forecast(bad_states, host_id="h", origin_ts=datetime.now(timezone.utc))


def test_forecast_calibration_is_opt_in_and_off_by_default(tmp_path):
    """Calibration (`nidra.scripts.fit_calibration`'s output) is OFF BY
    DEFAULT — see predictor.py's __init__ docstring/comment for why: this
    project's own evaluation found it REDUCES recall at the mandated 0.75
    threshold on the real trained ensemble (REAL_DATA_RESULTS.md), so
    applying it silently by default would quietly degrade detections.

    Three things are verified: (1) a predictor built with a
    risk_calibration.json present but apply_calibration left at its default
    (False) reports the RAW probability, unchanged; (2) apply_calibration=True
    against the same weights/calibration DOES change p_compromise, proving
    the flag actually does something rather than being dead code; (3) with
    no calibration file at all, apply_calibration=True is a harmless no-op
    (falls back to raw). Uses its own isolated tmp_path/model rather than
    the shared session `trained_predictor` fixture, so it can never
    contaminate other tests that assume no calibration file exists.
    """
    from tests.conftest import full_cfg_dict, synthetic_split_result
    from nidra.data.labels import attach_risk_label, label_stage_table
    from nidra.data.windowize import build_state_rows
    from nidra.explain.shap_runner import build_shap_background, save_background
    from nidra.train.pipeline import build_windowed_splits, fit_scaler
    from nidra.train.train_dynamics import train_one_seed
    from nidra.train.train_heads import train_heads_for_seed
    import yaml

    cfg = full_cfg_dict(tmp_path)
    scaler_dir = tmp_path / "scaler"
    scaler_dir.mkdir(parents=True, exist_ok=True)

    splits = synthetic_split_result()
    windowed = build_windowed_splits(splits)
    scaler = fit_scaler(windowed["train"])
    scaler.save(scaler_dir / "robust_scaler.joblib", scaler_dir / "scaler_metadata.json")

    benign_mask = windowed["train"].stage_label == "benign"
    background = build_shap_background(scaler.transform(windowed["train"].X[benign_mask, -1, :]), n_centroids=10)
    save_background(background, scaler_dir / "shap_background.npy")

    train_one_seed(cfg, seed=0, epochs_override=2, windowed=windowed, scaler=scaler, device="cpu")
    train_heads_for_seed(cfg, seed=0, windowed=windowed, scaler=scaler, device="cpu")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))

    weights_dir = tmp_path / "weights"
    predictor_uncalibrated = NidraPredictor(
        weights_dir=weights_dir, scaler_path=scaler_dir / "robust_scaler.joblib",
        config_path=config_path, seeds=[0],
    )
    assert predictor_uncalibrated._calibration is None

    states = windowed["train"].X[0]
    origin_ts = datetime(2017, 7, 4, 9, 0, 0, tzinfo=timezone.utc)
    # Rollout is stochastic; seed before each forecast() call so separate
    # calls are bit-for-bit reproducible and any difference between them is
    # attributable only to calibration, never to sampling noise.
    import torch as _torch
    _torch.manual_seed(0)
    raw_result = predictor_uncalibrated.forecast(states, host_id="h", origin_ts=origin_ts)

    # A large positive shift (a=1, b=+5) should push most calibrated
    # probabilities up noticeably relative to the raw ones.
    params_by_k = [{"a": 1.0, "b": 5.0, "n": 1, "degenerate": False}] * HORIZON_LENGTH
    save_calibration(weights_dir / "risk_calibration.json", params_by_k, {"fit_split": "val", "n_val_samples": 1})

    # (1) File present, apply_calibration left at default (False) -> raw,
    # unchanged output — the safety-critical default.
    predictor_file_present_not_applied = NidraPredictor(
        weights_dir=weights_dir, scaler_path=scaler_dir / "robust_scaler.joblib",
        config_path=config_path, seeds=[0],
    )
    assert predictor_file_present_not_applied._calibration is None
    _torch.manual_seed(0)
    still_raw_result = predictor_file_present_not_applied.forecast(states, host_id="h", origin_ts=origin_ts)
    for raw_h, still_raw_h in zip(raw_result["horizons"], still_raw_result["horizons"]):
        assert still_raw_h["p_compromise"] == pytest.approx(raw_h["p_compromise"], abs=1e-6)

    # (2) apply_calibration=True -> the flag actually does something.
    predictor_calibrated = NidraPredictor(
        weights_dir=weights_dir, scaler_path=scaler_dir / "robust_scaler.joblib",
        config_path=config_path, seeds=[0], apply_calibration=True,
    )
    assert predictor_calibrated._calibration is not None
    _torch.manual_seed(0)
    calibrated_result = predictor_calibrated.forecast(states, host_id="h", origin_ts=origin_ts)

    for raw_h, cal_h in zip(raw_result["horizons"], calibrated_result["horizons"]):
        assert cal_h["p_compromise"] >= raw_h["p_compromise"] - 1e-9
        assert 0.0 <= cal_h["p_compromise"] <= 1.0
    assert any(
        cal_h["p_compromise"] > raw_h["p_compromise"] + 1e-6
        for raw_h, cal_h in zip(raw_result["horizons"], calibrated_result["horizons"])
    )

    # (3) apply_calibration=True with no calibration file at all -> harmless no-op.
    (weights_dir / "risk_calibration.json").unlink()
    predictor_no_file = NidraPredictor(
        weights_dir=weights_dir, scaler_path=scaler_dir / "robust_scaler.joblib",
        config_path=config_path, seeds=[0], apply_calibration=True,
    )
    assert predictor_no_file._calibration is None


def test_forecast_rejects_wrong_context_length(trained_predictor):
    predictor, windowed = trained_predictor
    bad_states = windowed["train"].X[0][:-1, :]  # L-1 windows instead of L
    with pytest.raises(ValueError):
        predictor.forecast(bad_states, host_id="h", origin_ts=datetime.now(timezone.utc))


def test_forecast_rejects_nan_input(trained_predictor):
    predictor, windowed = trained_predictor
    bad_states = windowed["train"].X[0].copy()
    bad_states[0, 0] = np.nan
    with pytest.raises(ValueError):
        predictor.forecast(bad_states, host_id="h", origin_ts=datetime.now(timezone.utc))


def test_counterfactual_output_contract(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    result = predictor.counterfactual(states, feature_name="new_peer_count", clamp_value=0.0)
    assert result["label"] == "model-internal what-if"
    assert len(result["counterfactual"]) == HORIZON_LENGTH


def test_counterfactual_returns_the_two_curves_the_api_renders(trained_predictor):
    """The serving plane overlays a baseline curve and a clamped one, reading
    `original` and `counterfactual` as lists of {k, p_compromise, ci_low,
    ci_high}. StubPredictor returned exactly that and the API was built
    against it; NidraPredictor returned `risk_mean_k`/`risk_ci_*_k` arrays
    instead. Nothing failed — `payload.get("original", [])` is a clean miss —
    so /api/v1/counterfactual answered 200 with both curves empty.

    A Protocol is `@runtime_checkable` on method NAMES only, which is why
    swapping the implementation behind it did not catch this. The response
    shape needs its own test.
    """
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    payload = predictor.counterfactual(states, feature_name="new_peer_count", clamp_value=0.0)

    assert payload["label"] == COUNTERFACTUAL_LABEL
    assert payload["feature"] == "new_peer_count"
    assert payload["model_version"] == MODEL_VERSION
    for key in ("original", "counterfactual"):
        curve = payload[key]
        assert [p["k"] for p in curve] == list(range(1, HORIZON_LENGTH + 1)), key
        for point in curve:
            assert 0.0 <= point["ci_low"] <= point["p_compromise"] <= point["ci_high"] <= 1.0


def test_the_predictor_names_the_model_it_is(trained_predictor):
    """`/api/v1/model` reports `predictor.model_version`. Without the
    property it answered "unknown", so a running deployment could not say
    which checkpoint was serving it."""
    predictor, _ = trained_predictor
    assert predictor.model_version == MODEL_VERSION


def test_explain_output_contract(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    result = predictor.explain(states, horizon_k=1)
    assert "current_risk_attributions" in result
    assert "predicted_stage_attributions" in result
    assert "temporal_saliency" in result
    assert len(result["current_risk_attributions"]) <= 10


def test_explain_counts_horizon_steps_from_one_like_the_rest_of_the_protocol(
    trained_predictor,
):
    """`horizon_k` means the same thing here as the `k` on a HorizonPoint and
    the `k` on /api/v1/explain: step 1 is the first projected window and step
    K is the last.

    This is the bug's actual root, so it is the thing pinned. NidraPredictor
    read `horizon_k` as a 0-based tensor index while StubPredictor — which the
    HTTP layer was written against — read it as 1-based. Nothing converted at
    the boundary, so asking the real model to explain the end of its own cone
    indexed one step past the rollout and raised IndexError: a 500 on a
    correctly formed request, on every /api/v1/explain call.
    """
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]

    # Both ends of the range are valid...
    assert predictor.explain(states, horizon_k=1)["horizon_k"] == 1
    assert predictor.explain(states, horizon_k=HORIZON_LENGTH)["horizon_k"] == HORIZON_LENGTH

    # ...and neither 0 nor K+1 is. ValueError, not IndexError: the API turns
    # this one into a 422 and lets anything else become a 500.
    for bad in (0, -1, HORIZON_LENGTH + 1):
        with pytest.raises(ValueError, match="horizon_k"):
            predictor.explain(states, horizon_k=bad)


def test_scaler_never_refit_by_predictor(trained_predictor):
    predictor, _ = trained_predictor
    center_before = predictor.scaler.scaler.center_.copy()
    _ = predictor.forecast(np.random.randn(CONTEXT_LENGTH, 45).astype("float32"),
                            host_id="h", origin_ts=datetime.now(timezone.utc))
    np.testing.assert_array_equal(center_before, predictor.scaler.scaler.center_)
