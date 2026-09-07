"""Stream lag, and the minimal Prometheus surface `/metrics` renders.

Deliberately small: four series that answer the questions asked of a live demo — is the
api serving, how many consoles are attached, and is any stage of the pipeline falling
behind. A client library and a registry would add a dependency and a scrape endpoint
that says nothing more.

Stream lag is the pending count of the group that *reads* each stream — the messages
delivered but not yet acked. It is the number that grows when a worker dies, and the one
that stays flat while the pipeline keeps up.

`BacklogMonitor` is the §7 backpressure signal made actionable. A single high pending
count means nothing — a burst of thirty messages is a burst. A count that has risen at
every sample for half a minute is a stage falling behind the arrival rate, and it is worth
a log line before the demo notices. `BacklogWatch` polls it on a timer inside the api
process, so the warning does not depend on anyone loading `/health`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config

logger = logging.getLogger(__name__)

#: `(label, stream key, consumer group)` in pipeline order. Lag is a property of a group
#: rather than of a stream, so `forecasts` appears twice: once for the durable writer and
#: once for this process's own socket fan-out, which can fall behind independently.
STREAM_CONSUMERS: tuple[tuple[str, str, str], ...] = (
    ("ingest_jobs", "ingest_jobs", "ingest"),
    ("raw_events", "raw_events", "features"),
    ("state_vectors", "state_vectors", "inference"),
    ("forecasts", "forecasts", "persister"),
    ("forecasts:api", "forecasts", "api"),
)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class RequestCounter:
    """Requests served, by method and status. One process, one counter, no registry."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, int]] = Counter()

    def observe(self, method: str, status_code: int) -> None:
        self._counts[(method.upper(), int(status_code))] += 1

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def items(self) -> list[tuple[str, int, int]]:
        """`(method, status, count)`, ordered so successive scrapes read the same way."""
        return [(method, status, count) for (method, status), count in sorted(self._counts.items())]


async def stream_lag(redis: Redis, cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Pending count per stream. A group that does not exist yet has no backlog.

    Never raises: `/health` reporting a degraded pipeline is useful, and `/health` itself
    failing because Redis blinked is not.
    """
    lag: dict[str, int] = {}
    for label, key, group in STREAM_CONSUMERS:
        try:
            bus = Bus(redis, stream_name(key, cfg), group=group, consumer="metrics")
            lag[label] = await bus.pending_count()
        except RedisError as exc:
            # NOGROUP (nothing consumed yet) and a dropped connection land here alike.
            logger.debug("no pending count for %s/%s: %s", key, group, exc)
            lag[label] = 0
    return lag


def render(
    *,
    requests: RequestCounter,
    ws_connections: int,
    ws_dropped: int,
    lag: dict[str, int],
) -> str:
    """The exposition text. Prometheus wants a trailing newline; give it one."""
    lines = [
        "# HELP nidra_http_requests_total HTTP requests served by this process.",
        "# TYPE nidra_http_requests_total counter",
    ]
    lines.extend(
        f'nidra_http_requests_total{{method="{method}",status="{status}"}} {count}'
        for method, status, count in requests.items()
    )
    lines += [
        "# HELP nidra_websocket_connections Open forecast-stream sockets.",
        "# TYPE nidra_websocket_connections gauge",
        f"nidra_websocket_connections {ws_connections}",
        "# HELP nidra_websocket_dropped_total Forecasts dropped by the drop-oldest policy.",
        "# TYPE nidra_websocket_dropped_total counter",
        f"nidra_websocket_dropped_total {ws_dropped}",
        "# HELP nidra_stream_pending Unacknowledged messages in each stream's consumer group.",
        "# TYPE nidra_stream_pending gauge",
    ]
    lines.extend(
        f'nidra_stream_pending{{stream="{stream}"}} {count}' for stream, count in lag.items()
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- backpressure


@dataclass(slots=True)
class _Streak:
    """One consumer group's current run of strictly increasing pending counts."""

    started_at: float
    last_value: int
    warned_at: float | None = None


class BacklogMonitor:
    """Flags a consumer group whose pending count has only grown for `warn_after_s`.

    Monotone growth is the discriminating signal, not the absolute number: a stage that
    processes a burst has a high count that comes back down, while a stage that cannot
    keep up has a count that never does. Any sample that fails to increase — including a
    drop to zero — ends the streak, so a recovered pipeline stops warning by itself.
    """

    def __init__(self, warn_after_s: float) -> None:
        self.warn_after_s = warn_after_s
        self._streaks: dict[str, _Streak] = {}

    @property
    def growing(self) -> list[str]:
        """Groups currently over the warning threshold — what `/health` reports."""
        return sorted(
            stream for stream, streak in self._streaks.items() if streak.warned_at is not None
        )

    def observe(self, lag: Mapping[str, int], now: float) -> list[str]:
        """Fold one sample in; returns the groups that warned on *this* sample.

        `now` is a parameter rather than a clock read so a test can drive thirty seconds
        of backpressure without waiting thirty seconds.
        """
        warned: list[str] = []
        for stream, value in lag.items():
            streak = self._streaks.get(stream)
            if streak is None or value <= streak.last_value:
                # First sample, or the count held or fell: this is where growth restarts.
                self._streaks[stream] = _Streak(started_at=now, last_value=int(value))
                continue
            streak.last_value = int(value)
            if now - streak.started_at < self.warn_after_s:
                continue
            if streak.warned_at is not None and now - streak.warned_at < self.warn_after_s:
                continue  # already warning; say so once per interval, not once per sample
            streak.warned_at = now
            warned.append(stream)
            logger.warning(
                "backpressure: %s pending has grown for %.0fs, now %d unacked — "
                "the stage reading it is behind the arrival rate",
                stream,
                now - streak.started_at,
                value,
            )
        return warned


def backlog_settings(cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    """`(poll interval, warn-after)` in seconds, from `api.backlog_*`."""
    api = dict((cfg if cfg is not None else get_config()).get("api", {}))
    return float(api.get("backlog_poll_s", 5)), float(api.get("backlog_warn_after_s", 30))


class BacklogWatch:
    """Samples stream lag on a timer and feeds `BacklogMonitor`.

    Owned by the api lifespan, like the socket fan-out. A warning that only fires when
    someone loads `/health` is a warning nobody sees during a replay.
    """

    def __init__(
        self,
        monitor: BacklogMonitor,
        *,
        interval_s: float,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.monitor = monitor
        self.interval_s = interval_s
        self.cfg = cfg
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def _run(self) -> None:
        redis = create_redis(self.cfg)
        loop = asyncio.get_running_loop()
        try:
            while not self._stop.is_set():
                try:
                    self.monitor.observe(await stream_lag(redis, self.cfg), loop.time())
                except RedisError as exc:  # pragma: no cover — stream_lag swallows its own
                    logger.warning("backlog sample failed: %s", exc)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                except TimeoutError:
                    continue
        finally:
            await redis.aclose()

    async def start(self) -> None:
        if self._task is not None:  # pragma: no cover — lifespan runs once
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
