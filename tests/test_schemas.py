"""P1: the shared schema package — FEATURE_ORDER, boundary validation, example payload."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.schemas import (
    SCHEMA_VERSION,
    Forecast,
    HorizonPoint,
    SignalAttribution,
    StateVector,
    horizon_k,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "web-contract" / "forecast.example.json"

ORIGIN = datetime(2017, 7, 5, 14, 32, 0, tzinfo=UTC)
WINDOW_DELTA = 30


def _features(**overrides: float) -> dict[str, float]:
    values = {name: 0.0 for name in FEATURE_ORDER}
    values.update(overrides)
    return values


def _state_vector(**overrides: Any) -> StateVector:
    kwargs: dict[str, Any] = {
        "tenant_id": "demo",
        "host_id": "192.168.10.50",
        "window_ts": ORIGIN,
        "features": _features(syn_ratio=0.66, dst_port_entropy=5.0, is_active=1.0),
    }
    kwargs.update(overrides)
    return StateVector(**kwargs)


def _horizon(k: int, **overrides: Any) -> dict[str, Any]:
    point: dict[str, Any] = {
        "k": k,
        "ts": ORIGIN + timedelta(seconds=WINDOW_DELTA * k),
        "p_compromise": 0.60,
        "ci_low": 0.50,
        "ci_high": 0.70,
        "stage_dist": {"recon": 0.7, "initial_access": 0.3},
        "predicted_features": _features(),
    }
    point.update(overrides)
    return point


def _forecast_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tenant_id": "demo",
        "host_id": "192.168.10.50",
        "origin_ts": ORIGIN,
        "horizons": [_horizon(k) for k in range(1, horizon_k() + 1)],
        "lead_time_s": 90.0,
        "observed_stage": "recon",
        "observed_risk": 0.54,
        "top_signals": [
            SignalAttribution(
                name="dst_port_entropy",
                shap_value=0.21,
                direction="up",
                display="Destination-port entropy climbing",
            )
        ],
        "driving_window": 27,
        "model_version": "nidra-0.1.0-stub",
    }
    kwargs.update(overrides)
    return kwargs


# --- FEATURE_ORDER ----------------------------------------------------------------


def test_feature_order_length_and_uniqueness() -> None:
    assert len(FEATURE_ORDER) == 45
    assert len(set(FEATURE_ORDER)) == 45


def test_stages_vocabulary() -> None:
    assert STAGES == ["benign", "recon", "initial_access", "lateral", "c2", "exfil"]


# --- StateVector ------------------------------------------------------------------


def test_state_vector_round_trips_through_json() -> None:
    original = _state_vector()
    restored = StateVector.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.schema_ver == SCHEMA_VERSION
    assert set(restored.features) == set(FEATURE_ORDER)


def test_state_vector_missing_a_feature_raises() -> None:
    features = _features()
    del features["retrans_rate"]
    with pytest.raises(ValidationError, match="retrans_rate"):
        _state_vector(features=features)


def test_state_vector_unexpected_feature_raises() -> None:
    with pytest.raises(ValidationError, match="not_a_real_feature"):
        _state_vector(features=_features() | {"not_a_real_feature": 1.0})


# --- HorizonPoint -----------------------------------------------------------------


def test_horizon_point_band_must_bracket_p_compromise() -> None:
    with pytest.raises(ValidationError, match="ci_low"):
        HorizonPoint(**_horizon(1, ci_low=0.65, p_compromise=0.60, ci_high=0.70))


def test_horizon_point_band_must_stay_within_unit_interval() -> None:
    with pytest.raises(ValidationError, match="ci_high"):
        HorizonPoint(**_horizon(1, ci_high=1.4))


def test_horizon_point_rejects_unknown_stage() -> None:
    with pytest.raises(ValidationError, match="persistence"):
        HorizonPoint(**_horizon(1, stage_dist={"recon": 0.5, "persistence": 0.5}))


# --- Forecast ---------------------------------------------------------------------


def test_forecast_accepts_a_well_formed_trajectory() -> None:
    forecast = Forecast(**_forecast_kwargs())
    assert [h.k for h in forecast.horizons] == list(range(1, horizon_k() + 1))


def test_forecast_first_horizon_must_follow_origin_ts() -> None:
    horizons = [_horizon(k) for k in range(1, horizon_k() + 1)]
    horizons[0]["ts"] = ORIGIN
    with pytest.raises(ValidationError, match="origin_ts"):
        Forecast(**_forecast_kwargs(horizons=horizons))


def test_forecast_timestamps_must_strictly_increase() -> None:
    horizons = [_horizon(k) for k in range(1, horizon_k() + 1)]
    horizons[3]["ts"] = horizons[2]["ts"]
    with pytest.raises(ValidationError, match="k=3"):
        Forecast(**_forecast_kwargs(horizons=horizons))


def test_forecast_requires_all_k_in_order() -> None:
    horizons = [_horizon(k) for k in range(1, horizon_k() + 1)]
    horizons.pop(2)
    with pytest.raises(ValidationError, match="in order"):
        Forecast(**_forecast_kwargs(horizons=horizons))

    shuffled = [_horizon(k) for k in range(1, horizon_k() + 1)]
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValidationError, match="in order"):
        Forecast(**_forecast_kwargs(horizons=shuffled))


# --- Example payload --------------------------------------------------------------


def test_example_forecast_validates() -> None:
    payload = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    forecast = Forecast.model_validate(payload)

    assert forecast.host_id == "192.168.10.50"
    assert forecast.schema_ver == SCHEMA_VERSION
    assert len(forecast.horizons) == horizon_k()
    for point in forecast.horizons:
        assert set(point.predicted_features) == set(FEATURE_ORDER)


def test_example_forecast_mirrors_the_prd_worked_example() -> None:
    forecast = Forecast.model_validate(json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")))
    risks = [h.p_compromise for h in forecast.horizons]

    assert risks[:5] == [0.61, 0.69, 0.78, 0.83, 0.85]  # PRD 7.5
    assert risks[1] < 0.75 <= risks[2]  # threshold crossed at k=3
    assert forecast.lead_time_s == 90.0  # 3 windows x 30 s

    bands = [h.ci_high - h.ci_low for h in forecast.horizons]
    assert bands == sorted(bands), "confidence band must widen with the horizon"
