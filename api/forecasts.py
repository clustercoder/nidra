"""Forecast history, one forecast in full, and the current risk per host.

Three reads over one table. The persister promotes the hot fields of every `Forecast`
onto columns and keeps the whole model in JSONB alongside them, so the list and the host
view answer from indexed columns while the detail view returns the model verbatim —
including the horizon band and the attributions the console draws. Nothing here
reconstructs a `Forecast` from columns; the stored payload *is* the forecast.

Every statement carries `tenant_id = :tenant_id`, taken from the token by the dependency
and from nowhere else. There is no request parameter naming a tenant, so a cross-tenant
read is a code change rather than a crafted request — and another tenant's forecast is a
404, the same answer as one that does not exist.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import CurrentTenantUUID
from nidra_common.config import get_config
from nidra_common.db import get_session
from nidra_common.schemas import Forecast

router = APIRouter(prefix="/api/v1", tags=["forecasts"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]

#: Sortable columns of the host view, and the only way a sort key reaches the SQL. The
#: value is chosen from this mapping, never interpolated from what the client sent.
HOST_SORT_COLUMNS: dict[str, str] = {
    "risk": "max_p_compromise",
    "observed_risk": "observed_risk",
    "host": "host_id",
    "last_seen": "origin_ts",
}

HostSort = Literal["risk", "observed_risk", "host", "last_seen"]
SortOrder = Literal["desc", "asc"]

#: Promoted columns, in the order every projection below selects them.
_SUMMARY_COLUMNS = (
    "host_id, origin_ts, observed_risk, observed_stage, "
    "max_p_compromise, lead_time_s, model_version"
)

#: `CAST(... AS ...)` around each optional filter: asyncpg needs the type of a NULL bind
#: stated, and one statement with inert filters beats four assembled by string.
_LIST_FILTER = """
     WHERE tenant_id = :tenant_id
       AND (CAST(:host AS TEXT) IS NULL OR host_id = :host)
       AND (CAST(:since AS TIMESTAMPTZ) IS NULL OR origin_ts >= :since)
       AND (CAST(:until AS TIMESTAMPTZ) IS NULL OR origin_ts < :until)
"""

COUNT_FORECASTS = text(f"SELECT count(*) FROM forecasts {_LIST_FILTER}")

SELECT_FORECAST = text(
    "SELECT payload FROM forecasts "
    "WHERE tenant_id = :tenant_id AND host_id = :host_id AND origin_ts = :origin_ts"
)


def _select_forecasts(order: SortOrder) -> Any:
    """The history page. `order` is a `Literal`, so FastAPI rejects anything else at 422."""
    direction = "DESC" if order == "desc" else "ASC"
    return text(
        f"SELECT {_SUMMARY_COLUMNS} FROM forecasts {_LIST_FILTER} "
        f"ORDER BY origin_ts {direction}, host_id ASC LIMIT :limit OFFSET :offset"
    )


def _select_hosts(sort: HostSort, order: SortOrder) -> Any:
    """Latest row per host, then sorted.

    `DISTINCT ON (host_id) ... ORDER BY host_id, origin_ts DESC` walks
    `ix_forecasts_tenant_host_origin_ts` and takes one row per host; the outer sort is
    over that result, which is one row per host rather than the whole history.
    """
    column = HOST_SORT_COLUMNS[sort]
    direction = "DESC" if order == "desc" else "ASC"
    return text(
        f"SELECT {_SUMMARY_COLUMNS} FROM ("
        f"  SELECT DISTINCT ON (host_id) {_SUMMARY_COLUMNS} FROM forecasts"
        f"   WHERE tenant_id = :tenant_id"
        f"   ORDER BY host_id, origin_ts DESC"
        f") latest "
        # NULLS LAST both ways: a host whose forecasts predate the promoted column
        # sorts last rather than above every scored host.
        f"ORDER BY {column} {direction} NULLS LAST, host_id ASC LIMIT :limit"
    )


# --------------------------------------------------------------------------- schemas


class ForecastSummary(BaseModel):
    """One forecast as the history table renders it — promoted columns, no payload."""

    host_id: str
    origin_ts: datetime
    observed_risk: float | None
    observed_stage: str | None
    max_p_compromise: float | None
    lead_time_s: float | None
    model_version: str | None


class ForecastPage(BaseModel):
    """A page of history plus the total the pager needs to size itself."""

    items: list[ForecastSummary]
    total: int
    limit: int
    offset: int


class HostRisk(BaseModel):
    """A host's most recent forecast: what the console's host list is sorted by."""

    host_id: str
    origin_ts: datetime
    observed_risk: float | None
    observed_stage: str | None
    max_p_compromise: float | None
    lead_time_s: float | None
    model_version: str | None


class HostList(BaseModel):
    hosts: list[HostRisk]


# ---------------------------------------------------------------------- config, utils


def _api_config() -> dict[str, Any]:
    return dict(get_config().get("api", {}))


def page_limit() -> int:
    """Page size used when the request does not name one (`api.page_limit`)."""
    return int(_api_config().get("page_limit", 50))


def max_page_limit() -> int:
    """Ceiling on a client-requested page size (`api.max_page_limit`)."""
    return int(_api_config().get("max_page_limit", 500))


def resolved_limit(requested: int | None) -> int:
    """Clamp a requested page size to the configured ceiling. Nothing is hardcoded."""
    ceiling = max_page_limit()
    return min(requested if requested is not None else page_limit(), ceiling)


def as_utc(value: datetime) -> datetime:
    """Comparable UTC datetime; a naive timestamp is read as UTC, as the schema states."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else as_utc(value)


def _fields(row: Any) -> dict[str, Any]:
    """The promoted columns of one row, named as both response models name them."""
    return {
        "host_id": row.host_id,
        "origin_ts": row.origin_ts,
        "observed_risk": row.observed_risk,
        "observed_stage": row.observed_stage,
        "max_p_compromise": row.max_p_compromise,
        "lead_time_s": row.lead_time_s,
        "model_version": row.model_version,
    }


def _payload_dict(payload: Any) -> dict[str, Any]:
    """The JSONB column as a dict, whether the driver decoded it or handed back text."""
    return json.loads(payload) if isinstance(payload, str | bytes) else dict(payload)


# ---------------------------------------------------------------------------- routes


@router.get("/forecasts", response_model=ForecastPage)
async def list_forecasts(
    tenant_id: CurrentTenantUUID,
    session: SessionDep,
    host: Annotated[str | None, Query(description="restrict to one host")] = None,
    since: Annotated[datetime | None, Query(description="origin_ts >= since")] = None,
    until: Annotated[datetime | None, Query(description="origin_ts < until")] = None,
    limit: Annotated[
        int | None, Query(ge=1, description="page size; clamped to api.max_page_limit")
    ] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Annotated[SortOrder, Query(description="by origin_ts")] = "desc",
) -> ForecastPage:
    """This tenant's forecast history, newest first, filtered by host and time window."""
    size = resolved_limit(limit)
    params: dict[str, Any] = {
        "tenant_id": tenant_id,
        "host": host,
        "since": _optional_utc(since),
        "until": _optional_utc(until),
    }
    total = int((await session.execute(COUNT_FORECASTS, params)).scalar_one())
    rows = (
        await session.execute(_select_forecasts(order), {**params, "limit": size, "offset": offset})
    ).all()
    return ForecastPage(
        items=[ForecastSummary(**_fields(row)) for row in rows],
        total=total,
        limit=size,
        offset=offset,
    )


@router.get("/hosts", response_model=HostList)
async def list_hosts(
    tenant_id: CurrentTenantUUID,
    session: SessionDep,
    sort: Annotated[HostSort, Query(description="column to sort by")] = "risk",
    order: Annotated[SortOrder, Query()] = "desc",
    limit: Annotated[int | None, Query(ge=1)] = None,
) -> HostList:
    """Every host this tenant has a forecast for, at its most recent forecast."""
    size = resolved_limit(limit)
    rows = (
        await session.execute(_select_hosts(sort, order), {"tenant_id": tenant_id, "limit": size})
    ).all()
    return HostList(hosts=[HostRisk(**_fields(row)) for row in rows])


@router.get("/forecasts/{host}/{ts}", response_model=Forecast)
async def get_forecast(
    host: str,
    ts: datetime,
    tenant_id: CurrentTenantUUID,
    session: SessionDep,
) -> Forecast:
    """One forecast in full, from the stored JSONB. 404 for another tenant's.

    Validated on the way out: the payload was written by the persister and the console
    draws a horizon band from it, so a row that no longer matches the schema is a loud
    500 here rather than an empty chart there.
    """
    row = (
        await session.execute(
            SELECT_FORECAST,
            {"tenant_id": tenant_id, "host_id": host, "origin_ts": as_utc(ts)},
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="forecast not found")
    return Forecast.model_validate(_payload_dict(row.payload))
