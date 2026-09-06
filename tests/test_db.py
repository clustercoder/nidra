"""P2: the migrated schema is really there, and it really is idempotent-writable.

Runs against the Postgres from `docker compose up -d postgres` with `alembic upgrade
head` applied (`make up` does both). No mocks — a mocked unique constraint would tell
us nothing about whether the persister's ON CONFLICT path works.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nidra_common.db import dispose_engine, get_sessionmaker

EXPECTED_TABLES = {
    "tenants",
    "users",
    "ingest_jobs",
    "forecasts",
    "episodes",
    "benchmark_runs",
}

INSERT_FORECAST = text("""
    INSERT INTO forecasts
        (tenant_id, host_id, origin_ts, observed_risk, observed_stage,
         max_p_compromise, lead_time_s, payload, model_version)
    VALUES
        (:tenant_id, :host_id, :origin_ts, :observed_risk, :observed_stage,
         :max_p_compromise, :lead_time_s, CAST(:payload AS JSONB), :model_version)
    """)


def _forecast_row(tenant_id: uuid.UUID, host_id: str, origin_ts: datetime) -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "host_id": host_id,
        "origin_ts": origin_ts,
        "observed_risk": 0.42,
        "observed_stage": "recon",
        "max_p_compromise": 0.86,
        "lead_time_s": 90.0,
        "payload": json.dumps({"schema_version": "1.0", "host_id": host_id}),
        "model_version": "nidra-0.1.0-stub",
    }


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory, with the engine disposed after each test.

    Each test case gets its own event loop, and a pooled connection bound to a closed
    loop fails in a way that reads like a database problem. Disposing is cheaper than
    debugging that twice.
    """
    factory = get_sessionmaker()
    try:
        yield factory
    finally:
        await dispose_engine()


async def test_all_six_tables_exist(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with sessions() as session:
        result = await session.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        tables = {row[0] for row in result}
    assert EXPECTED_TABLES <= tables, f"missing tables: {sorted(EXPECTED_TABLES - tables)}"


async def test_citext_extension_installed(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with sessions() as session:
        result = await session.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'citext'"))
        assert result.scalar() == 1


async def test_forecast_alert_partial_index_exists(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    async with sessions() as session:
        result = await session.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'forecasts' AND indexname = :name"
            ),
            {"name": "ix_forecasts_tenant_max_p_compromise"},
        )
        indexdef = result.scalar_one()
    assert "max_p_compromise > 0.5" in indexdef.replace("(", "").replace(")", "")


async def test_duplicate_origin_ts_violates_uniqueness(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """At-least-once delivery will redeliver a forecast. The DB must refuse the second."""
    tenant_id = uuid.uuid4()
    origin_ts = datetime(2017, 7, 5, 9, 30, tzinfo=UTC)
    row = _forecast_row(tenant_id, "192.168.10.50", origin_ts)

    async with sessions() as session:
        await session.execute(INSERT_FORECAST, row)
        await session.flush()
        with pytest.raises(IntegrityError):
            await session.execute(INSERT_FORECAST, row)
        # Rolled back, so the test leaves no rows behind.
        await session.rollback()


async def test_same_host_different_origin_ts_is_allowed(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    tenant_id = uuid.uuid4()
    base = datetime(2017, 7, 5, 9, 30, tzinfo=UTC)

    async with sessions() as session:
        await session.execute(INSERT_FORECAST, _forecast_row(tenant_id, "192.168.10.50", base))
        await session.execute(
            INSERT_FORECAST,
            _forecast_row(tenant_id, "192.168.10.50", base.replace(minute=31)),
        )
        result = await session.execute(
            text("SELECT count(*) FROM forecasts WHERE tenant_id = :t"),
            {"t": tenant_id},
        )
        assert result.scalar_one() == 2
        await session.rollback()
