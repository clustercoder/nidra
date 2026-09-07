"""Durable observed state vectors — the writer, and the read the explain endpoint makes.

The forecast the console draws is produced from an `[L, 45]` context, and explaining it
afterwards means handing the predictor that same context again. Two places could hold
it, and only one of them is safe:

* the inference worker's Redis sequence buffer (`seq:{tenant}:{host}`) is a rolling
  window of the last L vectors, under a one-hour TTL, on a Redis deliberately started
  with `--appendonly no --save ""`. It can answer "explain the newest forecast" and
  nothing older, and it answers nothing at all after a restart.
* a row per `(tenant, host, window)` in Postgres survives a restart and can be read at
  any origin the history page offers.

So the features worker writes here **before** it publishes to `state_vectors`: every
vector that reaches inference is already durable, which makes "every forecast in the
database can be explained" a property of the ordering rather than a hope about timing.

Both halves live in one module because they are the same table read in two directions —
`save` writes the features dict, `load_context` projects it back through `FEATURE_ORDER`.
Splitting the reader into `api/` would put the column names in two files.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nidra_common.schemas import StateVector

logger = logging.getLogger(__name__)

#: `ON CONFLICT DO NOTHING`, for the same reason the forecasts insert has one: a
#: reclaimed `raw_events` message replays the window that produced this vector, and the
#: second write of an identical row must be a no-op rather than a duplicate or an error.
INSERT_STATE_VECTOR = text("""
    INSERT INTO state_vectors (tenant_id, host_id, window_ts, features, schema_ver)
    VALUES (:tenant_id, :host_id, :window_ts, CAST(:features AS JSONB), :schema_ver)
    ON CONFLICT (tenant_id, host_id, window_ts) DO NOTHING
    RETURNING id
    """)

#: The L windows ending at `until`, oldest first. The inner query walks
#: `uq_state_vectors_tenant_host_window_ts` backwards and stops after `limit` rows; the
#: outer sort puts them back in the order the encoder reads them.
SELECT_CONTEXT = text("""
    SELECT host_id, window_ts, features, schema_ver FROM (
        SELECT host_id, window_ts, features, schema_ver
          FROM state_vectors
         WHERE tenant_id = :tenant_id AND host_id = :host_id AND window_ts <= :until
         ORDER BY window_ts DESC
         LIMIT :limit
    ) recent
    ORDER BY window_ts ASC
    """)


def as_utc(value: datetime) -> datetime:
    """Comparable UTC datetime; a naive timestamp is read as UTC, as the schema states."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def tenant_uuid(tenant_id: str) -> uuid.UUID:
    """The vector's tenant as a UUID. Anything else did not come from our own tokens."""
    try:
        return uuid.UUID(tenant_id)
    except ValueError as exc:
        raise ValueError(f"state vector carries a non-UUID tenant_id: {tenant_id!r}") from exc


class StateVectorStore:
    """Writes observed vectors to Postgres. Owned by the features worker, one row each."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def save(self, vector: StateVector) -> bool:
        """Store one vector. True when the row is new, False when it was already there."""
        async with self.sessions() as session:
            inserted = await self.save_in(session, vector)
            await session.commit()
        return inserted

    @staticmethod
    async def save_in(session: AsyncSession, vector: StateVector) -> bool:
        """The insert itself, inside a caller-owned transaction. Never commits."""
        result = await session.execute(
            INSERT_STATE_VECTOR,
            {
                "tenant_id": tenant_uuid(vector.tenant_id),
                "host_id": vector.host_id,
                "window_ts": as_utc(vector.window_ts),
                "features": json.dumps(vector.features),
                "schema_ver": vector.schema_ver,
            },
        )
        return result.first() is not None


async def load_context(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    host_id: str,
    until: datetime,
    limit: int,
) -> list[StateVector]:
    """Up to `limit` observed vectors ending at `until` inclusive, oldest first.

    Tenant-filtered like every other query in this codebase, and validated on the way out
    against `FEATURE_ORDER` by `StateVector` itself — a row written under an older
    feature set is a loud failure here rather than a silently mis-labelled attribution.
    """
    rows = (
        await session.execute(
            SELECT_CONTEXT,
            {
                "tenant_id": tenant_id,
                "host_id": host_id,
                "until": as_utc(until),
                "limit": limit,
            },
        )
    ).all()
    return [
        StateVector(
            tenant_id=str(tenant_id),
            host_id=row.host_id,
            window_ts=row.window_ts,
            features=_payload_dict(row.features),
            **({"schema_ver": row.schema_ver} if row.schema_ver else {}),
        )
        for row in rows
    ]


def _payload_dict(features: Any) -> dict[str, float]:
    """The JSONB column as a dict, whether the driver decoded it or handed back text."""
    decoded = json.loads(features) if isinstance(features, str | bytes) else dict(features)
    return {key: float(value) for key, value in decoded.items()}
