"""The consumer loop every service worker runs.

One shape, four services: ensure the group exists, reclaim whatever a dead worker
abandoned, then read and dispatch. The only interesting rule is where the ack happens —
after `handler` returns, never before. A handler that raises leaves its message pending
so a later :meth:`Bus.reclaim` retries it; ack-then-process would silently drop it.

Shutdown is cooperative: SIGTERM (compose stop, Kubernetes drain) sets an event, the
current handler finishes, and the loop exits without acking anything it did not process.
Worst-case latency to exit is one `block_ms`.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable

from nidra_common.bus import DEFAULT_BLOCK_MS, DEFAULT_COUNT, DEFAULT_MIN_IDLE_MS, Bus

logger = logging.getLogger(__name__)

#: Called with the raw JSON payload of one message. Returning normally means "acked".
Handler = Callable[[bytes], Awaitable[None]]

#: How often to sweep the pending list for messages abandoned by another consumer.
DEFAULT_RECLAIM_INTERVAL_S = 30.0

SHUTDOWN_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def install_shutdown_handlers(stop: asyncio.Event) -> None:
    """Wire SIGTERM/SIGINT to `stop`. Falls back to `signal.signal` off the main thread."""
    loop = asyncio.get_running_loop()
    for sig in SHUTDOWN_SIGNALS:
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, ValueError, RuntimeError):
            signal.signal(sig, lambda *_: stop.set())


async def _dispatch(bus: Bus, handler: Handler, msg_id: str, payload: bytes) -> None:
    """Process one message and ack it — or log and leave it pending for the next reclaim."""
    try:
        await handler(payload)
    except Exception:
        logger.exception(
            "handler failed on %s id=%s; leaving unacked for reclaim", bus.stream, msg_id
        )
        return
    await bus.ack(msg_id)


async def run_consumer(
    bus: Bus,
    handler: Handler,
    *,
    count: int = DEFAULT_COUNT,
    block_ms: int = DEFAULT_BLOCK_MS,
    min_idle_ms: int = DEFAULT_MIN_IDLE_MS,
    reclaim_interval_s: float = DEFAULT_RECLAIM_INTERVAL_S,
    stop: asyncio.Event | None = None,
    install_signals: bool = True,
) -> None:
    """Consume `bus` until `stop` is set, dispatching each payload to `handler`.

    Pass `stop` to drive the loop from a test or a larger application; leave it `None`
    in a `__main__` and the process shuts down on SIGTERM.
    """
    if stop is None:
        stop = asyncio.Event()
    if install_signals:
        install_shutdown_handlers(stop)

    await bus.ensure_group()
    loop = asyncio.get_running_loop()
    next_reclaim = loop.time()
    logger.info("consuming %s as %s/%s", bus.stream, bus.group, bus.consumer)

    while not stop.is_set():
        if loop.time() >= next_reclaim:
            async for msg_id, payload in bus.reclaim(min_idle_ms=min_idle_ms, count=count):
                await _dispatch(bus, handler, msg_id, payload)
                if stop.is_set():
                    break
            next_reclaim = loop.time() + reclaim_interval_s
            if stop.is_set():
                break

        for msg_id, payload in await bus.read(count=count, block_ms=block_ms):
            await _dispatch(bus, handler, msg_id, payload)
            if stop.is_set():
                break

    logger.info("stopped consuming %s as %s/%s", bus.stream, bus.group, bus.consumer)
