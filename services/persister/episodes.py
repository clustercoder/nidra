"""Open-episode state, in Redis.

An episode is the product-facing form of lead time: the interval from the first forecast
whose trajectory crossed `risk_threshold` to the forecast at which the system declared
the risk over. `ground_truth_onset` minus `first_alert_at` is then a fact about *that*
episode — "warned 90 s before onset" — rather than an average in a notebook.

Deciding where an episode is open requires memory across messages, and a persister that
kept it in process memory would lose every open episode on restart: the rows would sit in
Postgres with a `NULL` `ended_at` forever, and the next above-threshold forecast would
open a second episode alongside them. So the aggregate lives in one Redis hash per
`(tenant, host)` and the process holds nothing.

```
key                 type    contents                                          TTL
ep:{tenant}:{host}  hash    last processed origin_ts + the open aggregate      24 h
```

The hash outlives the episode it described. `last_ts` is what makes the episode logic
idempotent under at-least-once redelivery — a forecast at or before the last one
processed for this host is skipped — and dropping the key at close would let a
redelivered above-threshold forecast open a duplicate episode.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

#: Long enough that a paused replay does not lose an open episode, short enough that an
#: abandoned tenant's host state does not outlive the demo.
EPISODE_TTL_S = 24 * 3600

LAST_TS_FIELD = "last_ts"

#: Hash fields describing the open aggregate; removed when the episode closes.
OPEN_FIELDS = ("id", "started_at", "peak_risk", "stages", "first_alert_at", "above", "below")


def episode_key(tenant_id: str, host_id: str) -> str:
    """The one piece of per-host persister state, and it is in Redis, not in this worker."""
    return f"ep:{tenant_id}:{host_id}"


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_ts(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class OpenEpisode:
    """The running aggregate of an episode that has not closed yet."""

    id: uuid.UUID
    started_at: datetime
    peak_risk: float
    stages: tuple[str, ...]
    first_alert_at: datetime | None
    above: int  # consecutive forecasts at or above the threshold
    below: int  # consecutive forecasts below it — `episode_close_after` of these close it


@dataclass(frozen=True, slots=True)
class HostEpisodeState:
    """Everything the persister remembers about one host: how far it got, and any episode."""

    last_ts: datetime | None = None
    episode: OpenEpisode | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    """What one forecast did to a host's episode state.

    `opened` and `closed` pick the SQL write; `episode` is the aggregate to store on the
    row, and is `None` only when the forecast touched no episode at all — the quiet case,
    and by far the common one.
    """

    state: HostEpisodeState
    episode: OpenEpisode | None = None
    opened: bool = False
    closed: bool = False
    ended_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EpisodeRules:
    """The three config knobs the lifecycle reads. Nothing here is hardcoded upstream."""

    risk_threshold: float
    close_after: int
    lead_time_m: int


@dataclass(frozen=True, slots=True)
class ForecastSummary:
    """The three values of a `Forecast` the episode lifecycle actually depends on."""

    origin_ts: datetime
    max_p_compromise: float
    observed_stage: str = ""


def _with_stage(stages: tuple[str, ...], stage: str) -> tuple[str, ...]:
    """Append a stage the first time it appears. The arc, in the order it happened."""
    if not stage or stage in stages:
        return stages
    return (*stages, stage)


def advance(
    state: HostEpisodeState, forecast: ForecastSummary, rules: EpisodeRules
) -> Transition | None:
    """Fold one forecast into a host's episode state.

    Returns `None` when the forecast is a redelivery or arrives out of order — the caller
    then leaves Redis untouched, which is what makes the lifecycle idempotent under the
    duplicates at-least-once delivery guarantees will produce.
    """
    if state.last_ts is not None and forecast.origin_ts <= state.last_ts:
        return None

    seen = replace(state, last_ts=forecast.origin_ts)
    above = forecast.max_p_compromise >= rules.risk_threshold
    episode = state.episode

    if episode is None:
        if not above:
            return Transition(state=seen)
        opened = OpenEpisode(
            id=uuid.uuid4(),
            started_at=forecast.origin_ts,
            peak_risk=forecast.max_p_compromise,
            stages=_with_stage((), forecast.observed_stage),
            # A single window over the line is a spike, not an alert; the alert is raised
            # once `lead_time_m` consecutive windows have been above it.
            first_alert_at=forecast.origin_ts if rules.lead_time_m <= 1 else None,
            above=1,
            below=0,
        )
        return Transition(
            state=replace(seen, episode=opened),
            episode=opened,
            opened=True,
        )

    # The episode spans the forecast that closes it, so every forecast delivered while it
    # is open contributes to the arc — including the below-threshold tail.
    updated = replace(
        episode,
        peak_risk=max(episode.peak_risk, forecast.max_p_compromise),
        stages=_with_stage(episode.stages, forecast.observed_stage),
        above=episode.above + 1 if above else 0,
        below=0 if above else episode.below + 1,
    )
    if updated.first_alert_at is None and updated.above >= rules.lead_time_m:
        updated = replace(updated, first_alert_at=forecast.origin_ts)

    if not above and updated.below >= rules.close_after:
        return Transition(
            state=replace(seen, episode=None),
            episode=updated,
            closed=True,
            ended_at=forecast.origin_ts,
        )
    return Transition(state=replace(seen, episode=updated), episode=updated)


# ------------------------------------------------------------------------ redis access


async def read_state(redis: Redis, tenant_id: str, host_id: str) -> HostEpisodeState:
    """Load one host's episode state. An absent key is a host nothing has happened to."""
    raw = await redis.hgetall(episode_key(tenant_id, host_id))  # type: ignore[misc]
    if not raw:
        return HostEpisodeState()
    fields = {_decode(key): _decode(value) for key, value in raw.items()}

    last_raw = fields.get(LAST_TS_FIELD)
    last_ts = _parse_ts(last_raw) if last_raw else None
    if "id" not in fields:
        return HostEpisodeState(last_ts=last_ts)

    first_alert = fields.get("first_alert_at")
    return HostEpisodeState(
        last_ts=last_ts,
        episode=OpenEpisode(
            id=uuid.UUID(fields["id"]),
            started_at=_parse_ts(fields["started_at"]),
            peak_risk=float(fields["peak_risk"]),
            stages=tuple(json.loads(fields["stages"])),
            first_alert_at=_parse_ts(first_alert) if first_alert else None,
            above=int(fields["above"]),
            below=int(fields["below"]),
        ),
    )


async def write_state(redis: Redis, tenant_id: str, host_id: str, state: HostEpisodeState) -> None:
    """Persist the state after the database write committed, never before.

    A crash between the two replays the forecast: the insert conflicts and `last_ts` is
    unchanged, so the episode lifecycle sees the message for the first time and completes.
    """
    key = episode_key(tenant_id, host_id)
    mapping: dict[str, str] = {}
    if state.last_ts is not None:
        mapping[LAST_TS_FIELD] = state.last_ts.isoformat()

    episode = state.episode
    if episode is not None:
        mapping.update(
            {
                "id": str(episode.id),
                "started_at": episode.started_at.isoformat(),
                "peak_risk": repr(episode.peak_risk),
                "stages": json.dumps(list(episode.stages)),
                "above": str(episode.above),
                "below": str(episode.below),
            }
        )
        if episode.first_alert_at is not None:
            mapping["first_alert_at"] = episode.first_alert_at.isoformat()

    pipe = redis.pipeline(transaction=True)
    if episode is None:
        # Closed: drop the aggregate, keep `last_ts` so a redelivery cannot reopen it.
        pipe.hdel(key, *OPEN_FIELDS)
    elif episode.first_alert_at is None:
        pipe.hdel(key, "first_alert_at")
    if mapping:
        pipe.hset(key, mapping=mapping)
    pipe.expire(key, EPISODE_TTL_S)
    await pipe.execute()
