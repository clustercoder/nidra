"""Stream lag, and the minimal Prometheus surface `/metrics` renders.

Deliberately small: four series that answer the questions asked of a live demo — is the
api serving, how many consoles are attached, and is any stage of the pipeline falling
behind. A client library and a registry would add a dependency and a scrape endpoint
that says nothing more.

Stream lag is the pending count of the group that *reads* each stream — the messages
delivered but not yet acked. It is the number that grows when a worker dies, and the one
that stays flat while the pipeline keeps up.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from nidra_common.bus import Bus, stream_name

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
