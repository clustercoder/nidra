"""The `forecasts` consumer: durable rows, and the episodes built from them.

Two things happen to every forecast that reaches this worker, and both of them have to
survive being told the same thing twice. At-least-once delivery is not an edge case here
— a reclaimed message after a worker crash, a redelivery after a network stall — so the
row write is `ON CONFLICT (tenant_id, host_id, origin_ts) DO NOTHING` and the episode
lifecycle is gated on a `last_ts` watermark in Redis. Neither can double-count.

The causality contract is re-asserted at this boundary even though `Forecast` already
validates it. This is the last gate before the number becomes durable and gets shown to a
judge as "warned 90 s before onset"; a horizon at or before its own `origin_ts` would be
a forecast that saw the future, and the right response is to raise and leave the message
pending, never to coerce the timestamp into shape.

The full `Forecast` is stored as JSONB with the hot fields promoted to columns, so the
console's queries stay indexed while schema evolution costs nothing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nidra_common.config import get_config
from nidra_common.schemas import Forecast
from services.persister.episodes import (
    EpisodeRules,
    ForecastSummary,
    Transition,
    advance,
    read_state,
    write_state,
)

logger = logging.getLogger(__name__)

INSERT_FORECAST = text("""
    INSERT INTO forecasts
        (tenant_id, host_id, origin_ts, observed_risk, observed_stage,
         max_p_compromise, lead_time_s, payload, model_version)
    VALUES
        (:tenant_id, :host_id, :origin_ts, :observed_risk, :observed_stage,
         :max_p_compromise, :lead_time_s, CAST(:payload AS JSONB), :model_version)
    ON CONFLICT (tenant_id, host_id, origin_ts) DO NOTHING
    RETURNING id
    """)

#: `stages` is `TEXT[]`; binding it needs the array type stated, not inferred.
_STAGES_PARAM = sa.bindparam("stages", type_=postgresql.ARRAY(sa.Text()))

INSERT_EPISODE = text("""
    INSERT INTO episodes
        (id, tenant_id, host_id, started_at, ended_at, peak_risk, stages, first_alert_at)
    VALUES
        (:id, :tenant_id, :host_id, :started_at, :ended_at, :peak_risk, :stages,
         :first_alert_at)
    """).bindparams(_STAGES_PARAM)

#: One statement for both the in-flight update and the close; `ended_at` stays NULL while
#: the episode is open. Tenant-filtered like every other query in this codebase.
UPDATE_EPISODE = text("""
    UPDATE episodes
       SET ended_at = :ended_at,
           peak_risk = :peak_risk,
           stages = :stages,
           first_alert_at = :first_alert_at
     WHERE id = :id AND tenant_id = :tenant_id
    """).bindparams(_STAGES_PARAM)


def as_utc(value: datetime) -> datetime:
    """Comparable UTC datetime; a naive timestamp is read as UTC, as the schema states."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def assert_causal(forecast: Forecast) -> None:
    """Every horizon must be strictly later than `origin_ts`. Raise, never coerce.

    A forecast for `t+k` carries `origin_ts = t` and nothing at or after `t` was observed
    when it was produced. A horizon at or before `t` is not a forecast, and writing it
    would put a number in the database that the whole project's claim rests on being
    impossible.
    """
    origin = as_utc(forecast.origin_ts)
    for point in forecast.horizons:
        if as_utc(point.ts) <= origin:
            raise ValueError(
                f"origin_ts causality violated for host {forecast.host_id}: horizon "
                f"k={point.k} at {point.ts.isoformat()} is not after origin_ts "
                f"{forecast.origin_ts.isoformat()}"
            )


def max_p_compromise(forecast: Forecast) -> float:
    """The peak of the projected trajectory — the value the alert query is indexed on."""
    return max(point.p_compromise for point in forecast.horizons)


def tenant_uuid(tenant_id: str) -> uuid.UUID:
    """The forecast's tenant as a UUID. Anything else did not come from our own tokens."""
    try:
        return uuid.UUID(tenant_id)
    except ValueError as exc:
        raise ValueError(f"forecast carries a non-UUID tenant_id: {tenant_id!r}") from exc


@dataclass(frozen=True, slots=True)
class PersistOutcome:
    """What one forecast did: a new row or a duplicate, and any episode movement.

    Returned so a test can drive the worker directly instead of polling the database.
    """

    inserted: bool
    transition: Transition | None


class PersisterWorker:
    """One `forecasts` consumer. Rows in Postgres, open-episode state in Redis."""

    def __init__(
        self,
        redis: Redis,
        sessions: async_sessionmaker[AsyncSession],
        *,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.redis = redis
        self.sessions = sessions
        self.cfg = cfg if cfg is not None else get_config()

    # ------------------------------------------------------------------ configuration

    @property
    def rules(self) -> EpisodeRules:
        """Episode semantics, from `config/default.yaml`. Nothing here is hardcoded."""
        return EpisodeRules(
            risk_threshold=float(self.cfg["risk_threshold"]),
            close_after=int(self.cfg["episode_close_after"]),
            lead_time_m=int(self.cfg["lead_time_m"]),
        )

    # ------------------------------------------------------------------------ handler

    async def handle(self, payload: bytes) -> None:
        """Bus entry point: one `forecasts` message.

        Raising leaves the message unacked and pending for a later `XAUTOCLAIM`. Both
        writes downstream are idempotent, so the retry is harmless.
        """
        await self.process(Forecast.model_validate_json(payload))

    async def process(self, forecast: Forecast) -> PersistOutcome:
        """Write one forecast and advance its host's episode, in that order."""
        assert_causal(forecast)
        tenant = tenant_uuid(forecast.tenant_id)
        peak = max_p_compromise(forecast)

        state = await read_state(self.redis, forecast.tenant_id, forecast.host_id)
        transition = advance(
            state,
            ForecastSummary(
                origin_ts=as_utc(forecast.origin_ts),
                max_p_compromise=peak,
                observed_stage=forecast.observed_stage,
            ),
            self.rules,
        )

        async with self.sessions() as session:
            inserted = await self._insert_forecast(session, forecast, tenant, peak)
            if transition is not None:
                await self._write_episode(session, forecast, tenant, transition)
            await session.commit()

        # Only after the commit: a crash in between replays the message, the insert
        # conflicts, and the episode advances on the retry because `last_ts` never moved.
        if transition is not None:
            await write_state(self.redis, forecast.tenant_id, forecast.host_id, transition.state)

        if not inserted:
            logger.debug(
                "tenant %s host %s: forecast at origin %s already stored",
                forecast.tenant_id,
                forecast.host_id,
                forecast.origin_ts.isoformat(),
            )
        return PersistOutcome(inserted=inserted, transition=transition)

    # --------------------------------------------------------------------- sql writes

    async def _insert_forecast(
        self, session: AsyncSession, forecast: Forecast, tenant: uuid.UUID, peak: float
    ) -> bool:
        """Insert the row, or do nothing because it is already there. True when it is new."""
        result = await session.execute(
            INSERT_FORECAST,
            {
                "tenant_id": tenant,
                "host_id": forecast.host_id,
                "origin_ts": as_utc(forecast.origin_ts),
                "observed_risk": forecast.observed_risk,
                "observed_stage": forecast.observed_stage,
                "max_p_compromise": peak,
                "lead_time_s": forecast.lead_time_s,
                "payload": forecast.model_dump_json(),
                "model_version": forecast.model_version,
            },
        )
        return result.first() is not None

    async def _write_episode(
        self,
        session: AsyncSession,
        forecast: Forecast,
        tenant: uuid.UUID,
        transition: Transition,
    ) -> None:
        """Open a new episode row, or refresh the open one — closing it if this ended it."""
        episode = transition.episode
        if episode is None:
            return

        params = {
            "id": episode.id,
            "tenant_id": tenant,
            "ended_at": transition.ended_at,
            "peak_risk": episode.peak_risk,
            "stages": list(episode.stages),
            "first_alert_at": episode.first_alert_at,
        }
        if transition.opened:
            await session.execute(
                INSERT_EPISODE,
                {**params, "host_id": forecast.host_id, "started_at": episode.started_at},
            )
            logger.info(
                "tenant %s host %s: episode %s opened at %s (peak %.3f)",
                forecast.tenant_id,
                forecast.host_id,
                episode.id,
                episode.started_at.isoformat(),
                episode.peak_risk,
            )
            return

        await session.execute(UPDATE_EPISODE, params)
        if transition.closed and transition.ended_at is not None:
            logger.info(
                "tenant %s host %s: episode %s closed at %s (peak %.3f, stages %s)",
                forecast.tenant_id,
                forecast.host_id,
                episode.id,
                transition.ended_at.isoformat(),
                episode.peak_risk,
                ",".join(episode.stages) or "-",
            )
