"""Pydantic models shared by every NIDRA service.

One package, imported everywhere, so that a schema mismatch is caught at a service
boundary in a second rather than in a SHAP plot on demo day. Validation here is
deliberately strict and loud:

* a `StateVector` must carry exactly the keys of `FEATURE_ORDER` — no more, no fewer;
* a `HorizonPoint` band must satisfy `0 <= ci_low <= p_compromise <= ci_high <= 1`;
* a `Forecast` must carry horizons `k = 1..K` in order, strictly increasing in time and
  all strictly later than `origin_ts`.

That last rule is the causality contract: a forecast for `t+k` carries `origin_ts = t`,
and nothing at or after `origin_ts` was observed when it was produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from pydantic import BaseModel, model_validator

from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.config import get_config

SCHEMA_VERSION = "1.0"

_FEATURE_KEYS = frozenset(FEATURE_ORDER)
_STAGE_KEYS = frozenset(STAGES)


@lru_cache(maxsize=1)
def horizon_k() -> int:
    """Number of forecast steps, from `config/default.yaml`. Never hardcoded."""
    return int(get_config()["horizon_K"])


def _as_utc(value: datetime) -> datetime:
    """Comparable UTC datetime; a naive timestamp is read as UTC, as the schema states."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class StateVector(BaseModel):
    """One host's compressed state over one window. The unit the world model consumes."""

    tenant_id: str
    host_id: str
    window_ts: datetime  # window start, UTC, aligned to window_delta
    features: dict[str, float]  # keys == FEATURE_ORDER, validated below
    schema_ver: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _features_match_feature_order(self) -> StateVector:
        keys = set(self.features)
        if keys != _FEATURE_KEYS:
            missing = sorted(_FEATURE_KEYS - keys)
            unexpected = sorted(keys - _FEATURE_KEYS)
            raise ValueError(
                "features keys must equal FEATURE_ORDER "
                f"(missing={missing}, unexpected={unexpected})"
            )
        return self


class SignalAttribution(BaseModel):
    """One SHAP-attributed feature behind a forecast, with display copy for the console."""

    name: str
    shap_value: float
    direction: str  # "up" | "down"
    display: str  # "SYN ratio rising sharply"


class HorizonPoint(BaseModel):
    """The projected state at `origin_ts + k * window_delta`."""

    k: int
    ts: datetime
    p_compromise: float
    ci_low: float
    ci_high: float
    stage_dist: dict[str, float]
    predicted_features: dict[str, float]

    @model_validator(mode="after")
    def _check_band_and_stages(self) -> HorizonPoint:
        if not 0.0 <= self.ci_low <= self.p_compromise <= self.ci_high <= 1.0:
            raise ValueError(
                f"horizon k={self.k} band must satisfy 0 <= ci_low <= p_compromise "
                f"<= ci_high <= 1, got ci_low={self.ci_low}, "
                f"p_compromise={self.p_compromise}, ci_high={self.ci_high}"
            )
        unknown = sorted(set(self.stage_dist) - _STAGE_KEYS)
        if unknown:
            raise ValueError(f"horizon k={self.k} stage_dist has unknown stages: {unknown}")
        return self


class Forecast(BaseModel):
    """A K-step trajectory for one host, anchored at `origin_ts`."""

    tenant_id: str
    host_id: str
    origin_ts: datetime  # time t — nothing after this was observed
    horizons: list[HorizonPoint]
    lead_time_s: float | None  # None if the threshold is never crossed
    observed_stage: str
    observed_risk: float
    top_signals: list[SignalAttribution]
    driving_window: int  # index of the most influential past window
    model_version: str
    schema_ver: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _horizons_are_ordered_and_causal(self) -> Forecast:
        expected = list(range(1, horizon_k() + 1))
        actual = [h.k for h in self.horizons]
        if actual != expected:
            raise ValueError(
                f"horizons must be k={expected[0]}..{expected[-1]} in order, got {actual}"
            )

        previous = _as_utc(self.origin_ts)
        for point in self.horizons:
            ts = _as_utc(point.ts)
            if ts <= previous:
                raise ValueError(
                    f"horizon k={point.k} ts {point.ts.isoformat()} must be strictly later "
                    f"than {'origin_ts' if point.k == 1 else f'k={point.k - 1}'} "
                    f"{previous.isoformat()}"
                )
            previous = ts
        return self
