"""Explanation, model-internal what-if, evaluation results, and model provenance.

Four endpoints that together answer "why does the model say this, and how well does it
do it". Three of them are stub-backed today: the predictor behind `explain` and
`counterfactual` is the P7 surrogate until `nidra.serve.predictor` lands, and it says so
in `method` and `model_version` rather than presenting arithmetic as attribution.
`benchmarks` reads whatever is in `artifacts/metrics/` and reports `pending` when that is
nothing — a number the evaluation has not produced is never invented here.

Both model calls rebuild the `[L, 45]` context from the `state_vectors` table the
features worker writes (`services/features/store.py`), so an explanation is computed from
the same observed windows the forecast was, not from a live buffer that has since moved
on. Nothing after `ts` is read: the context is `window_ts <= ts` ordered backwards, which
is the same causal boundary the forecast itself obeyed.

The predictor is loaded once per process and called on a worker thread. The real one
spends up to 300 ms in torch, and the api event loop also owns the WebSocket fan-out —
blocking it here would stall every open console for the duration of one explanation.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from api.deps import CurrentTenantUUID, current_principal
from nidra.data.schema import FEATURE_ORDER
from nidra_common.config import REPO_ROOT, get_config
from nidra_common.db import get_session
from nidra_common.schemas import SCHEMA_VERSION, SignalAttribution, StateVector
from services.features.store import load_context
from services.inference.predictor_loader import Predictor
from services.inference.stub_predictor import COUNTERFACTUAL_LABEL
from services.inference.worker import states_array

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["explain"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]

#: The flow bridge (IMPLEMENTATION-ML.md §6e) resolves attributed features back to the
#: flows that produced them. It is an ML deliverable that does not exist yet, so the list
#: is empty and says what it is waiting for rather than being quietly omitted.
FLOW_BRIDGE = "flow_bridge"
FLOWS_NOTE = (
    "flagged flows resolve attributed features back to the flows in the window; "
    "the ML flow bridge that produces them has not landed"
)

#: What the counterfactual is, in the response, every time. A question about the model,
#: not about the network — CLAUDE.md invariant, and the sentence a judge should read
#: before the curves.
COUNTERFACTUAL_NOTE = (
    "what this model would predict had the feature held this value; "
    "a statement about the model, not about the network"
)

#: Starlette renamed 422 to `HTTP_422_UNPROCESSABLE_CONTENT`; the old spelling warns.
UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT

BENCHMARKS_OK = "ok"
BENCHMARKS_PENDING = "pending"
BENCHMARKS_PENDING_DETAIL = "evaluation artifacts not yet produced"


# ---------------------------------------------------------------------- config, utils


def _api_config() -> dict[str, Any]:
    return dict(get_config().get("api", {}))


def metrics_dir() -> Path:
    """Directory `/api/v1/benchmarks` serves, from `api.metrics_dir`. Repo-relative."""
    configured = Path(str(_api_config().get("metrics_dir", "artifacts/metrics")))
    return configured if configured.is_absolute() else REPO_ROOT / configured


def context_l() -> int:
    """Windows of history a forecast is computed from. The explanation reads the same L."""
    return int(get_config()["context_L"])


def window_delta() -> int:
    return int(get_config()["window_delta"])


def as_utc(value: datetime) -> datetime:
    """Comparable UTC datetime; a naive timestamp is read as UTC, as the schema states."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def get_predictor(request: Request) -> Predictor:
    """The process-wide predictor, loaded once at app construction."""
    return request.app.state.predictor  # type: ignore[no-any-return]


PredictorDep = Annotated[Predictor, Depends(get_predictor)]


async def observed_context(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    host_id: str,
    origin_ts: datetime,
) -> list[StateVector]:
    """The L observed windows ending at `origin_ts`, or a 404 if that window is not stored.

    The last row must be `origin_ts` itself. A context that merely ends *before* it would
    explain a different forecast than the one asked about, and would do it without
    saying so.
    """
    origin = as_utc(origin_ts)
    vectors = await load_context(
        session, tenant_id=tenant_id, host_id=host_id, until=origin, limit=context_l()
    )
    if not vectors or as_utc(vectors[-1].window_ts) != origin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no observed state for host {host_id} at {origin.isoformat()}",
        )
    return vectors


def context_states(vectors: list[StateVector]) -> np.ndarray:
    """`[L, 45]` in `FEATURE_ORDER`, oldest first — the array the predictor validates."""
    return states_array(vectors)


# --------------------------------------------------------------------------- schemas


class FlaggedFlows(BaseModel):
    """The flows behind an attribution, once the ML flow bridge can resolve them."""

    flows: list[dict[str, Any]] = Field(default_factory=list)
    available_after: str = FLOW_BRIDGE
    note: str = FLOWS_NOTE


class ExplainResponse(BaseModel):
    """Why the model projects this future for this host at this window."""

    host_id: str
    origin_ts: datetime
    horizon_k: int
    method: str
    model_version: str
    top_signals: list[SignalAttribution]
    #: Index into the context of the window that moved the projection most; 0 is oldest.
    driving_window: int
    #: One weight per context window, oldest first, summing to 1.
    window_importance: list[float]
    #: How many observed windows were available, against the L a full context carries.
    #: They differ at the start of a capture, and the console should say which it drew.
    context_windows: int
    context_l: int
    note: str | None = None
    flagged_flows: FlaggedFlows = Field(default_factory=FlaggedFlows)


class CounterfactualRequest(BaseModel):
    """Clamp one feature across the observed context and re-simulate."""

    host: str
    ts: datetime
    feature: str
    clamp_value: float

    @field_validator("feature")
    @classmethod
    def _feature_is_canonical(cls, value: str) -> str:
        """`FEATURE_ORDER` is the whole vocabulary. Anything else is a 422, not a guess."""
        if value not in FEATURE_ORDER:
            raise ValueError(f"unknown feature {value!r}; not in FEATURE_ORDER")
        return value


class CurvePoint(BaseModel):
    """One step of a risk curve. `ts` is `origin_ts + k * window_delta`, as in `Forecast`."""

    k: int
    ts: datetime
    p_compromise: float
    ci_low: float
    ci_high: float


class CounterfactualResponse(BaseModel):
    """Two curves and the label that says what the second one is."""

    #: CLAUDE.md invariant, judge-facing: "model-internal what-if", literally, always.
    label: str
    host_id: str
    origin_ts: datetime
    feature: str
    clamp_value: float
    model_version: str
    original: list[CurvePoint]
    counterfactual: list[CurvePoint]
    context_windows: int
    note: str = COUNTERFACTUAL_NOTE


class BenchmarksResponse(BaseModel):
    """Whatever the evaluation has produced, or an honest statement that it has not."""

    status: str
    detail: str | None = None
    #: Directory served, relative to the repo root, so the answer names its own source.
    source: str
    #: File stem → the artifact's parsed contents, verbatim. Never summarised here.
    metrics: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """The geometry every number in this API is expressed in."""

    window_delta: int
    context_L: int
    horizon_K: int
    n_features: int
    risk_threshold: float
    lead_time_m: int


class ModelInfo(BaseModel):
    """Which predictor is answering, and under what configuration."""

    impl: str
    model_version: str
    schema_ver: str
    config: ModelConfig


# ---------------------------------------------------------------------- explain routes


@router.get("/explain/{host}/{ts}", response_model=ExplainResponse)
async def explain(
    host: str,
    ts: datetime,
    tenant_id: CurrentTenantUUID,
    session: SessionDep,
    predictor: PredictorDep,
    k: Annotated[
        int | None, Query(ge=1, description="horizon step to explain; default horizon_K")
    ] = None,
) -> ExplainResponse:
    """Attribution for the forecast this tenant's host had at `ts`.

    `k` defaults to the last step of the horizon: the end of the cone is the claim being
    made, so it is the one worth explaining unless the caller says otherwise.
    """
    horizon_k = k if k is not None else int(get_config()["horizon_K"])
    vectors = await observed_context(session, tenant_id=tenant_id, host_id=host, origin_ts=ts)
    states = context_states(vectors)

    try:
        payload = await run_in_threadpool(predictor.explain, states, horizon_k)
    except ValueError as exc:
        # The predictor bounds `horizon_k` by its own K; an out-of-range step is the
        # caller's error, not a failure of the service.
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc

    return ExplainResponse(
        host_id=host,
        origin_ts=as_utc(ts),
        horizon_k=int(payload.get("horizon_k", horizon_k)),
        method=str(payload.get("method", "unknown")),
        model_version=str(payload.get("model_version", "unknown")),
        top_signals=[SignalAttribution.model_validate(s) for s in payload.get("top_signals", [])],
        driving_window=int(payload.get("driving_window", 0)),
        window_importance=[float(w) for w in payload.get("window_importance", [])],
        context_windows=len(vectors),
        context_l=context_l(),
        note=payload.get("note"),
    )


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual(
    body: CounterfactualRequest,
    tenant_id: CurrentTenantUUID,
    session: SessionDep,
    predictor: PredictorDep,
) -> CounterfactualResponse:
    """Re-simulate this host's trajectory with one feature held at `clamp_value`.

    A model-internal what-if. It answers what this predictor would have projected on a
    different input; it says nothing about what the network would have done, and the
    response carries that distinction as a label rather than leaving it to the UI.
    """
    origin = as_utc(body.ts)
    vectors = await observed_context(
        session, tenant_id=tenant_id, host_id=body.host, origin_ts=origin
    )
    states = context_states(vectors)

    payload = await run_in_threadpool(
        predictor.counterfactual, states, body.feature, body.clamp_value
    )
    label = str(payload.get("label", ""))
    if label != COUNTERFACTUAL_LABEL:
        # Fail loudly rather than serve an unlabelled what-if: the label is the claim
        # discipline, and a response missing it is the one thing this endpoint must not do.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"predictor returned label {label!r}, expected {COUNTERFACTUAL_LABEL!r}",
        )

    return CounterfactualResponse(
        label=label,
        host_id=body.host,
        origin_ts=origin,
        feature=body.feature,
        clamp_value=body.clamp_value,
        model_version=str(payload.get("model_version", "unknown")),
        original=_curve(payload.get("original", []), origin),
        counterfactual=_curve(payload.get("counterfactual", []), origin),
        context_windows=len(vectors),
    )


def _curve(points: list[dict[str, Any]], origin: datetime) -> list[CurvePoint]:
    """Predictor curve points, timestamped the way `Forecast` timestamps its horizons."""
    delta = window_delta()
    return [
        CurvePoint(
            k=int(point["k"]),
            ts=origin + timedelta(seconds=int(point["k"]) * delta),
            p_compromise=float(point["p_compromise"]),
            ci_low=float(point["ci_low"]),
            ci_high=float(point["ci_high"]),
        )
        for point in points
    ]


# ------------------------------------------------------------------- evaluation, model


# Authenticated like every other `/api/v1` route. Neither reads tenant data, but the
# console is the only client and an endpoint that answers without a token is one more
# surface to reason about at judging time than the product needs.
@router.get(
    "/benchmarks",
    response_model=BenchmarksResponse,
    tags=["evaluation"],
    dependencies=[Depends(current_principal)],
)
async def benchmarks() -> BenchmarksResponse:
    """Baselines, ablations and horizon curves — whatever `artifacts/metrics/` holds.

    Empty is a valid answer and it is `pending`, not zeros. The evaluation either ran or
    it did not, and this endpoint reports which; a placeholder number here would end up
    on a slide.
    """
    directory = metrics_dir()
    source = _relative_to_repo(directory)
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not files:
        return BenchmarksResponse(
            status=BENCHMARKS_PENDING, detail=BENCHMARKS_PENDING_DETAIL, source=source
        )

    metrics: dict[str, Any] = {}
    for path in files:
        try:
            metrics[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Serving the readable subset would present partial results as complete.
            logger.error("unreadable metrics artifact %s: %s", path, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"metrics artifact {path.name} could not be read",
            ) from exc
    return BenchmarksResponse(status=BENCHMARKS_OK, source=source, metrics=metrics)


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@router.get(
    "/model",
    response_model=ModelInfo,
    tags=["evaluation"],
    dependencies=[Depends(current_principal)],
)
async def model_info(predictor: PredictorDep) -> ModelInfo:
    """Which predictor is serving, and the configuration its numbers are expressed in.

    The config echo is not decoration: `window_delta` and `context_L` decide what a
    horizon step means, and a metric compared across two values of them is not a
    comparison. The client reads the geometry from the process that produced the numbers.
    """
    cfg = get_config()
    return ModelInfo(
        impl=str(cfg.get("predictor", {}).get("impl", "unknown")),
        model_version=str(getattr(predictor, "model_version", "unknown")),
        schema_ver=SCHEMA_VERSION,
        config=ModelConfig(
            window_delta=int(cfg["window_delta"]),
            context_L=int(cfg["context_L"]),
            horizon_K=int(cfg["horizon_K"]),
            n_features=int(cfg["n_features"]),
            risk_threshold=float(cfg["risk_threshold"]),
            lead_time_m=int(cfg["lead_time_m"]),
        ),
    )
