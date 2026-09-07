"""The live forecast stream, and the fan-out that feeds it.

`IMPLEMENTATION-Backend.md` §8 gives each socket its own consumer name inside one Redis
consumer group. That is exactly backwards: consumers *within* a group split the stream
between them, so two browser tabs would each receive roughly half the forecasts and both
would look like the pipeline was dropping messages. Groups fan out; consumers inside a
group load-balance.

So the api process runs **one** consumer in the `api` group and does its own fan-out:

```
forecasts stream ──> ForecastFanout (group "api", one consumer per process)
                        └─> ConnectionManager.broadcast
                              ├─> Subscription queue ──> socket task (tab 1)
                              └─> Subscription queue ──> socket task (tab 2)
```

Each socket owns a bounded queue and drains it itself, so one stalled tab cannot slow
the fan-out: a full queue drops its **oldest** entry, because the newest forecast is the
one the chart is about to draw.

The doc's filter callback is also broken — it rebinds `hosts` inside a nested function
with no `nonlocal`, so the assignment lands on a local and the filter never changes. The
fix here is structural rather than a keyword: the filter lives on the `Subscription`
object both tasks hold, so there is no binding to get wrong. The update is acknowledged
back to the client, which is also what lets a test know when it has taken effect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket as _socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis

from api.deps import verify_ws_token
from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.schemas import Forecast
from nidra_common.worker import run_consumer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

#: The one consumer group the api process reads `forecasts` under. `persister` reads the
#: same stream under its own group and both see every message.
CONSUMER_GROUP = "api"

#: Client → server message types. Anything else is ignored rather than fatal: a socket
#: is not the place to fail a connection over an unknown key.
FILTER_MESSAGE = "filter"
PONG_MESSAGE = "pong"
PING_MESSAGE = "ping"

#: Close code for a peer that stopped answering heartbeats (RFC 6455 "going away").
WS_GOING_AWAY = 1001

#: Close code for a handshake we refused. Sent before `accept()`, which the ASGI server
#: turns into an HTTP 403 — the connection is refused, never opened.
WS_POLICY_VIOLATION = 1008


# ----------------------------------------------------------------------- connections


@dataclass(eq=False, slots=True)
class Subscription:
    """One socket's view of the fan-out: its tenant, its filter, and its own queue.

    Identity-hashed (`eq=False`) because two sockets on the same tenant with the same
    filter are still two connections.
    """

    tenant_id: str
    queue: asyncio.Queue[tuple[str | None, bytes]]
    hosts: set[str] | None = None  # None = every host of this tenant
    dropped: int = 0

    def wants(self, host_id: str | None) -> bool:
        """Applied at drain time, so a filter set mid-stream governs what is still queued.

        `host_id is None` marks a control frame the server generated for this socket; it
        is never subject to the host filter.
        """
        return host_id is None or self.hosts is None or host_id in self.hosts

    def set_hosts(self, hosts: Iterable[str] | None) -> None:
        """Replace the host filter. An empty list means 'no filter', as `None` does."""
        selected = None if hosts is None else {str(host) for host in hosts}
        self.hosts = selected or None

    def offer(self, host_id: str | None, payload: bytes) -> bool:
        """Enqueue for this socket, dropping the oldest entry when the queue is full.

        Returns False when something had to be dropped. Never blocks and never awaits:
        the fan-out consumer must not be slowed by the slowest reader on the socket.
        """
        dropped = False
        while self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover — drained concurrently
                break
            dropped = True
            self.dropped += 1
        self.queue.put_nowait((host_id, payload))
        return not dropped


class ConnectionManager:
    """Every open socket on this process, and the one place a forecast is fanned out."""

    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self._subscriptions: set[Subscription] = set()

    @property
    def connection_count(self) -> int:
        """Open sockets — the gauge `/metrics` reports."""
        return len(self._subscriptions)

    @property
    def dropped_total(self) -> int:
        """Messages dropped by the overflow policy, across every open socket."""
        return sum(sub.dropped for sub in self._subscriptions)

    def subscribe(self, tenant_id: str, hosts: Iterable[str] | None = None) -> Subscription:
        """Register a socket. Called before `accept()`, so nothing published is missed."""
        sub = Subscription(tenant_id=tenant_id, queue=asyncio.Queue(maxsize=self.queue_size))
        sub.set_hosts(hosts)
        self._subscriptions.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        """Deregister a socket. Idempotent — the close path may run twice."""
        self._subscriptions.discard(sub)

    def broadcast(self, tenant_id: str, host_id: str, payload: bytes) -> int:
        """Offer one forecast to every socket of `tenant_id`. Returns how many got it.

        Tenant filtering happens here rather than at the socket: a queue that never holds
        another tenant's bytes cannot leak them through a filtering bug downstream.
        """
        delivered = 0
        for sub in self._subscriptions:
            if sub.tenant_id != tenant_id:
                continue
            if not sub.offer(host_id, payload):
                logger.warning("socket queue full for tenant %s; dropped oldest", tenant_id)
            delivered += 1
        return delivered


# -------------------------------------------------------------------------- fan-out


def consumer_name() -> str:
    """Identifies this api process within the `api` group. One consumer per process."""
    return f"{_socket.gethostname()}-{os.getpid()}"


class ForecastFanout:
    """The single `forecasts` consumer behind every socket on this process.

    Owns its Redis client and its task. `start`/`stop` are driven by the app lifespan;
    a test can point it at a private stream and a short block interval instead.
    """

    def __init__(
        self,
        manager: ConnectionManager,
        *,
        stream: str | None = None,
        cfg: dict[str, Any] | None = None,
        block_ms: int | None = None,
    ) -> None:
        self.manager = manager
        self.cfg = cfg if cfg is not None else get_config()
        self.stream = stream if stream is not None else stream_name("forecasts", self.cfg)
        self.block_ms = block_ms
        self._redis: Redis | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def handle(self, payload: bytes) -> None:
        """One forecast: validate it, hand the original bytes to every matching socket.

        The bytes are forwarded verbatim rather than re-serialised, so what the browser
        parses is exactly what the inference worker produced.
        """
        forecast = Forecast.model_validate_json(payload)
        self.manager.broadcast(forecast.tenant_id, forecast.host_id, payload)

    async def start(self) -> None:
        """Begin consuming. Signal handlers stay with uvicorn, which owns the process."""
        if self._task is not None:  # pragma: no cover — lifespan runs once
            return
        self._redis = create_redis(self.cfg)
        bus = Bus(self._redis, self.stream, group=CONSUMER_GROUP, consumer=consumer_name())
        kwargs: dict[str, Any] = {"stop": self._stop, "install_signals": False}
        if self.block_ms is not None:
            kwargs["block_ms"] = self.block_ms
        self._task = asyncio.create_task(run_consumer(bus, self.handle, **kwargs))
        logger.info("websocket fan-out consuming %s as %s", self.stream, CONSUMER_GROUP)

    async def stop(self) -> None:
        """Stop consuming and release the client. Waits for the in-flight read to return."""
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


# ------------------------------------------------------------------------- heartbeat


@dataclass(slots=True)
class WsSettings:
    """Socket tunables, from `config/default.yaml`. Nothing here is hardcoded."""

    queue_size: int = 512
    ping_interval_s: float = 30.0
    max_missed_pongs: int = 2


def ws_settings(cfg: dict[str, Any] | None = None) -> WsSettings:
    """Read the `api.ws_*` block."""
    api = dict((cfg if cfg is not None else get_config()).get("api", {}))
    return WsSettings(
        queue_size=int(api.get("ws_queue_size", 512)),
        ping_interval_s=float(api.get("ws_ping_interval_s", 30.0)),
        max_missed_pongs=int(api.get("ws_max_missed_pongs", 2)),
    )


@dataclass(slots=True)
class Heartbeat:
    """Unanswered pings. Shared by the two socket tasks — an object, not a closure."""

    missed: int = 0


# ---------------------------------------------------------------------------- socket


def parse_hosts(hosts: list[str] | None) -> set[str] | None:
    """`?hosts=a&hosts=b` or `?hosts=a,b`; absent or empty means every host."""
    if not hosts:
        return None
    selected = {part.strip() for value in hosts for part in value.split(",") if part.strip()}
    return selected or None


async def _receive(ws: WebSocket, sub: Subscription, beat: Heartbeat) -> None:
    """Client → server: filter updates and pongs, until the socket closes.

    Returns rather than raises on disconnect; the sending task watches this task and
    stops when it finishes.
    """
    while True:
        try:
            message = await ws.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # not JSON: ignore the frame, keep the socket
        if not isinstance(message, dict):
            continue
        kind = message.get("type")
        if kind == PONG_MESSAGE:
            beat.missed = 0
        elif kind == FILTER_MESSAGE:
            sub.set_hosts(message.get("hosts"))
            # Acknowledged through the queue, not sent from here: one task owns sending.
            sub.offer(None, _control(FILTER_MESSAGE, hosts=sorted(sub.hosts or [])))


def _control(kind: str, **fields: Any) -> bytes:
    """A server control frame, encoded like every other frame on the socket."""
    return json.dumps({"type": kind, **fields}).encode()


async def _serve(ws: WebSocket, sub: Subscription, settings: WsSettings) -> None:
    """Server → client: forecasts as they arrive, a ping when they do not.

    One task sends, so no two coroutines ever write to the socket at once. It waits on
    the queue and on the receiving task together: a client that disconnects is noticed
    immediately rather than at the next heartbeat.
    """
    beat = Heartbeat()
    receiver = asyncio.create_task(_receive(ws, sub, beat))
    pending: asyncio.Task[tuple[str | None, bytes]] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(sub.queue.get())
            done, _ = await asyncio.wait(
                {pending, receiver},
                timeout=settings.ping_interval_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receiver in done:
                return
            if pending in done:
                host_id, payload = pending.result()
                pending = None
                if sub.wants(host_id):
                    await ws.send_text(payload.decode())
                continue
            # Nothing to send for a whole interval: heartbeat, or give up on the peer.
            if beat.missed >= settings.max_missed_pongs:
                logger.info("closing socket after %d unanswered pings", beat.missed)
                await ws.close(code=WS_GOING_AWAY)
                return
            beat.missed += 1
            await ws.send_text(_control(PING_MESSAGE).decode())
    except WebSocketDisconnect:
        return
    finally:
        receiver.cancel()
        if pending is not None:
            pending.cancel()


@router.websocket("/api/v1/stream/forecast")
async def stream_forecast(
    ws: WebSocket,
    token: Annotated[str, Query(description="access token; browsers cannot set WS headers")],
    hosts: Annotated[list[str] | None, Query(description="restrict to these hosts")] = None,
) -> None:
    """Live forecasts for the token's tenant, filterable by host at any time."""
    try:
        principal = verify_ws_token(token)
    except HTTPException:
        # Before `accept()`: the handshake is refused with 403, never opened and closed.
        await ws.close(code=WS_POLICY_VIOLATION)
        return

    manager: ConnectionManager = ws.app.state.connections
    settings: WsSettings = ws.app.state.ws_settings
    sub = manager.subscribe(principal.tenant_id, parse_hosts(hosts))
    await ws.accept()
    logger.info("socket open for tenant %s (%d total)", sub.tenant_id, manager.connection_count)
    try:
        await _serve(ws, sub, settings)
    finally:
        manager.unsubscribe(sub)
        logger.info("socket closed for tenant %s", sub.tenant_id)
