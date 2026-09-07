"""Window state, in Redis.

Everything the features service accumulates between two events lives here and nowhere
else. That is not tidiness: a worker that keeps the open window in process memory loses
it on restart, and the loss is silent — the stream keeps flowing and the features are
merely wrong for a while. Restartability is worth the round trips.

```
key                              type    contents                                TTL
feat:open:{tenant}               hash    open window ts + wall time of last event  1 h
feat:hosts:{tenant}:{ts}         set     source hosts seen in that window          1 h
feat:win:{tenant}:{host}:{ts}    hash    scalar counters + value distributions     1 h
feat:graph:{tenant}:{ts}         hash    "src>dst" -> event count                  1 h
feat:peers:{tenant}:{host}       set     previously contacted peers               24 h
feat:prev:{tenant}:{host}        hash    last emitted ts + last 3 windows tracked  24 h
```

Window boundaries are absolute epoch multiples of `window_delta`, never relative to the
first packet — otherwise two files that overlap in time produce windows that do not join.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

from nidra_common.events import RawEvent
from services.features.compute import event_increments

#: Long enough to survive a worker restart mid-capture, short enough that an abandoned
#: tenant's window state does not outlive the demo.
WINDOW_TTL_S = 3600

#: The peer set is the one genuinely long-lived piece of per-host memory: `new_peer_count`
#: means nothing if it resets every hour.
PEERS_TTL_S = 24 * 3600

#: Separates the endpoints of an edge inside the graph hash. IPs never contain it.
EDGE_SEP = ">"

OPEN_WINDOW_FIELD = "window_ts"
OPEN_SEEN_FIELD = "last_seen"

PREV_TS_FIELD = "last_ts"
PREV_HISTORY_FIELD = "history"

#: How many windows of tracked values to carry per host — enough for `slope3_*`, which
#: reads two past windows plus the current one.
HISTORY_LEN = 3


def decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def align(ts: datetime, window_delta: int) -> int:
    """Epoch seconds of the window containing `ts`, floored to an absolute multiple."""
    at = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    return int(at.timestamp()) // window_delta * window_delta


def window_start(window_ts: int) -> datetime:
    """The aligned epoch second as the UTC datetime a `StateVector` carries."""
    return datetime.fromtimestamp(window_ts, tz=UTC)


# ------------------------------------------------------------------------------- keys


def open_key(tenant_id: str) -> str:
    return f"feat:open:{tenant_id}"


def hosts_key(tenant_id: str, window_ts: int) -> str:
    return f"feat:hosts:{tenant_id}:{window_ts}"


def window_key(tenant_id: str, host: str, window_ts: int) -> str:
    return f"feat:win:{tenant_id}:{host}:{window_ts}"


def graph_key(tenant_id: str, window_ts: int) -> str:
    return f"feat:graph:{tenant_id}:{window_ts}"


def peers_key(tenant_id: str, host: str) -> str:
    return f"feat:peers:{tenant_id}:{host}"


def prev_key(tenant_id: str, host: str) -> str:
    return f"feat:prev:{tenant_id}:{host}"


def edge_field(src: str, dst: str) -> str:
    return f"{src}{EDGE_SEP}{dst}"


def parse_edge(field: str) -> tuple[str, str] | None:
    src, sep, dst = field.partition(EDGE_SEP)
    return (src, dst) if sep and src and dst else None


# ------------------------------------------------------------------------ accumulation


async def accumulate(redis: Redis, event: RawEvent, window_ts: int, *, now: float) -> None:
    """Fold one event into its host's open window. One pipeline, one round trip.

    The graph edge is recorded at tenant scope because `in_degree` and `reciprocity` are
    properties of the window's whole host graph, not of any one row.
    """
    tenant, host = event.tenant_id, event.src_ip
    win = window_key(tenant, host, window_ts)
    graph = graph_key(tenant, window_ts)

    pipe = redis.pipeline(transaction=False)
    for field, amount in event_increments(event).items():
        pipe.hincrbyfloat(win, field, amount)
    pipe.expire(win, WINDOW_TTL_S)
    pipe.sadd(hosts_key(tenant, window_ts), host)
    pipe.expire(hosts_key(tenant, window_ts), WINDOW_TTL_S)
    if event.dst_ip:
        pipe.hincrbyfloat(graph, edge_field(host, event.dst_ip), 1.0)
        pipe.expire(graph, WINDOW_TTL_S)
    pipe.hset(
        open_key(tenant),
        mapping={OPEN_WINDOW_FIELD: str(window_ts), OPEN_SEEN_FIELD: repr(now)},
    )
    pipe.expire(open_key(tenant), WINDOW_TTL_S)
    await pipe.execute()


async def read_open(redis: Redis, tenant_id: str) -> tuple[int, float] | None:
    """`(window_ts, wall time of the last event)` for the tenant's open window, if any."""
    raw = await redis.hgetall(open_key(tenant_id))  # type: ignore[misc]
    if not raw:
        return None
    fields = {decode(k): decode(v) for k, v in raw.items()}
    if OPEN_WINDOW_FIELD not in fields:
        return None
    return int(fields[OPEN_WINDOW_FIELD]), float(fields.get(OPEN_SEEN_FIELD, 0.0))


async def open_tenants(redis: Redis) -> list[str]:
    """Every tenant with an open window, for the watchdog sweep."""
    prefix = open_key("")
    tenants: list[str] = []
    async for key in redis.scan_iter(match=f"{prefix}*"):
        tenants.append(decode(key)[len(prefix) :])
    return sorted(tenants)


async def clear_open(redis: Redis, tenant_id: str, window_ts: int) -> bool:
    """Drop the open marker, unless a newer window has already replaced it.

    Returns False when the marker moved on — the caller raced the event path and the
    window it was closing is no longer the open one.
    """
    current = await read_open(redis, tenant_id)
    if current is None or current[0] != window_ts:
        return False
    await redis.delete(open_key(tenant_id))
    return True


# --------------------------------------------------------------------------- reading


async def read_hosts(redis: Redis, tenant_id: str, window_ts: int) -> list[str]:
    """Source hosts with data in the window, sorted so emission order is deterministic."""
    raw = await redis.smembers(hosts_key(tenant_id, window_ts))  # type: ignore[misc]
    return sorted(decode(host) for host in raw or [])


async def read_window(redis: Redis, tenant_id: str, host: str, window_ts: int) -> dict[str, str]:
    raw = await redis.hgetall(window_key(tenant_id, host, window_ts))  # type: ignore[misc]
    return {decode(k): decode(v) for k, v in (raw or {}).items()}


async def read_edges(redis: Redis, tenant_id: str, window_ts: int) -> list[tuple[str, str]]:
    """The window's host graph as edges. The graph itself is never persisted."""
    raw = await redis.hgetall(graph_key(tenant_id, window_ts))  # type: ignore[misc]
    edges = (parse_edge(decode(field)) for field in (raw or {}))
    return [edge for edge in edges if edge is not None]


async def read_history(redis: Redis, tenant_id: str, host: str) -> tuple[int | None, list[dict]]:
    """`(last emitted window ts, tracked values of the last 3 windows)` for one host."""
    raw = await redis.hgetall(prev_key(tenant_id, host))  # type: ignore[misc]
    fields = {decode(k): decode(v) for k, v in (raw or {}).items()}
    last_ts = int(fields[PREV_TS_FIELD]) if fields.get(PREV_TS_FIELD) else None
    history = json.loads(fields.get(PREV_HISTORY_FIELD) or "[]")
    return last_ts, list(history)


async def write_history(
    redis: Redis, tenant_id: str, host: str, *, last_ts: int, history: list[dict]
) -> None:
    """Persist the host's rolling history, trimmed to the windows the dynamics need."""
    key = prev_key(tenant_id, host)
    await redis.hset(  # type: ignore[misc]
        key,
        mapping={
            PREV_TS_FIELD: str(last_ts),
            PREV_HISTORY_FIELD: json.dumps(history[-HISTORY_LEN:]),
        },
    )
    await redis.expire(key, PEERS_TTL_S)


async def record_peers(redis: Redis, tenant_id: str, host: str, peers: set[str]) -> int:
    """Add this window's destinations to the host's peer set; return how many were new.

    Read before write, in that order: `new_peer_count` is the whole point and an
    unconditional `SADD` first would always report zero.
    """
    key = peers_key(tenant_id, host)
    if not peers:
        await redis.expire(key, PEERS_TTL_S)
        return 0
    known = {decode(peer) for peer in await redis.smembers(key) or []}  # type: ignore[misc]
    new = peers - known
    await redis.sadd(key, *peers)  # type: ignore[misc]
    await redis.expire(key, PEERS_TTL_S)
    return len(new)


async def discard_window(redis: Redis, tenant_id: str, window_ts: int, hosts: list[str]) -> None:
    """Delete the accumulators for a closed window. The graph outlives it by one window.

    `neighbour_risk_fraction` at `t` reads the graph at `t-1`, so the graph key is dropped
    only once the window after it has closed.
    """
    keys = [window_key(tenant_id, host, window_ts) for host in hosts]
    keys.append(hosts_key(tenant_id, window_ts))
    await redis.delete(*keys)


async def discard_graph(redis: Redis, tenant_id: str, window_ts: int) -> None:
    await redis.delete(graph_key(tenant_id, window_ts))
