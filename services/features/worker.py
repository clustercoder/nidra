"""The `raw_events` consumer: accumulate, close windows, publish `StateVector`s.

Two things close a window. Usually it is an event that belongs to a later one — the
stream is in timestamp order, so a later window arriving means the earlier one is done.
The other is the watchdog, and it is the one that matters for the demo: the final window
of a capture has no successor event, so without a timer it never closes and the last
forecast — the one the reality overlay lands on — never fires.

Windows are closed at **tenant** scope rather than per host. Four of the eight graph
scalars are properties of the window's whole host graph, and closing one host's window
while another host is still writing edges into the same window would compute `in_degree`
against a graph that is half-built. The per-host accumulators are still keyed per host,
exactly as the backend doc specifies; only the decision to close is shared.

Silence is data. When a host reappears after a gap, the windows it missed are emitted
first as zero rows with `is_active=0`, so the time axis stays uniform and lead time
downstream is measured in windows that all exist. Past `features.max_gap_windows` the
gap stops being silence and starts being a restart: the history is reset instead, and
the host's next vector is zero-padded like a fresh sequence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis

from nidra_common.bus import Bus
from nidra_common.config import get_config
from nidra_common.events import RawEvent
from nidra_common.schemas import StateVector
from services.features.compute import (
    DIST_DST_IP,
    DYNAMIC_SOURCES,
    WindowAccumulator,
    WindowGraph,
    compute_features,
    tracked_values,
    validate_features,
    zero_features,
)
from services.features.state import (
    HISTORY_LEN,
    accumulate,
    align,
    clear_open,
    discard_graph,
    discard_window,
    open_tenants,
    read_edges,
    read_history,
    read_hosts,
    read_open,
    read_window,
    record_peers,
    window_start,
    write_history,
)
from services.features.store import StateVectorStore

logger = logging.getLogger(__name__)

#: Fraction of the watchdog grace to sweep at. Small enough that the final window closes
#: promptly, large enough that an idle service is not scanning Redis in a tight loop.
WATCHDOG_SWEEP_FRACTION = 0.25

MIN_SWEEP_INTERVAL_S = 0.25


class FeaturesWorker:
    """Turns `raw_events` into schema-valid `StateVector`s, one window at a time.

    `now` is injectable so the watchdog can be driven deterministically in a test —
    the alternative is a test that spends the grace period waiting for it.

    `store` is the durable copy of every vector published, which is what makes a forecast
    explainable after its Redis context has expired. It is optional so the worker still
    runs against Redis alone; the process entry point always wires one in.
    """

    def __init__(
        self,
        redis: Redis,
        bus: Bus,
        *,
        cfg: dict[str, Any] | None = None,
        now: Callable[[], float] = time.time,
        store: StateVectorStore | None = None,
    ) -> None:
        self.redis = redis
        self.bus = bus
        self.cfg = cfg if cfg is not None else get_config()
        self._now = now
        self.store = store
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ configuration

    @property
    def window_delta(self) -> int:
        return int(self.cfg["window_delta"])

    @property
    def _features_cfg(self) -> dict[str, Any]:
        return dict(self.cfg.get("features", {}))

    @property
    def max_gap_windows(self) -> int:
        return int(self._features_cfg.get("max_gap_windows", 40))

    @property
    def elevated_fanout(self) -> int:
        return int(self._features_cfg.get("elevated_fanout", 5))

    @property
    def degree_cap(self) -> int:
        return int(self._features_cfg.get("clustering_degree_cap", 200))

    @property
    def grace_s(self) -> float:
        """Watchdog grace: two windows of replayed time, never under the configured floor.

        At 60× a 30 s window arrives every half second, so two windows is one second —
        below the floor, and a floor is what stops a brief stall from closing a window
        that still has events coming.
        """
        floor = float(self._features_cfg.get("watchdog_floor_s", 2.0))
        speed = float(self.cfg.get("replay", {}).get("default_speed", 1.0))
        if speed <= 0:
            return floor
        return max(floor, 2 * self.window_delta / speed)

    def _lock(self, tenant_id: str) -> asyncio.Lock:
        """One lock per tenant: the event path and the watchdog must not both close."""
        lock = self._locks.get(tenant_id)
        if lock is None:
            lock = self._locks[tenant_id] = asyncio.Lock()
        return lock

    # ------------------------------------------------------------------------ handler

    async def handle(self, payload: bytes) -> None:
        """Bus entry point: one `raw_events` message."""
        await self.ingest_event(RawEvent.model_validate_json(payload))

    async def ingest_event(self, event: RawEvent) -> list[StateVector]:
        """Fold one event in, closing the previous window first if this one is later.

        Returns whatever closing emitted, which is `[]` for most events — the return
        value exists so a test can drive the pipeline without polling the stream.
        """
        if not event.src_ip:
            logger.warning("dropping %s event with no source host", event.kind)
            return []

        window_ts = align(event.ts, self.window_delta)
        emitted: list[StateVector] = []
        async with self._lock(event.tenant_id):
            current = await read_open(self.redis, event.tenant_id)
            if current is not None and window_ts < current[0]:
                # Its window has already closed; folding it in now would change a vector
                # that has already been published.
                logger.warning(
                    "tenant %s: dropping late event at window %d, open window is %d",
                    event.tenant_id,
                    window_ts,
                    current[0],
                )
                return []
            if current is not None and window_ts > current[0]:
                emitted = await self._close_window(event.tenant_id, current[0])
            await accumulate(self.redis, event, window_ts, now=self._now())
        return emitted

    # ----------------------------------------------------------------------- watchdog

    async def watchdog_tick(self) -> list[StateVector]:
        """Close every open window whose grace has elapsed since its last event."""
        emitted: list[StateVector] = []
        for tenant_id in await open_tenants(self.redis):
            async with self._lock(tenant_id):
                current = await read_open(self.redis, tenant_id)
                if current is None:
                    continue
                window_ts, last_seen = current
                if self._now() - last_seen < self.grace_s:
                    continue
                logger.info(
                    "tenant %s: watchdog closing window %d after %.1fs of silence",
                    tenant_id,
                    window_ts,
                    self._now() - last_seen,
                )
                emitted.extend(await self._close_window(tenant_id, window_ts))
        return emitted

    async def run_watchdog(self, stop: asyncio.Event) -> None:
        """Sweep for stalled windows until `stop` is set."""
        interval = max(MIN_SWEEP_INTERVAL_S, self.grace_s * WATCHDOG_SWEEP_FRACTION)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
            if stop.is_set():
                return
            try:
                await self.watchdog_tick()
            except Exception:
                # A failed sweep must not take the consumer down with it; the window
                # stays open and the next sweep tries again.
                logger.exception("features watchdog sweep failed")

    # ------------------------------------------------------------------- window close

    async def _close_window(self, tenant_id: str, window_ts: int) -> list[StateVector]:
        """Compute and publish every host's vector for one closed window."""
        hosts = await read_hosts(self.redis, tenant_id, window_ts)
        graph = WindowGraph.from_edges(await read_edges(self.redis, tenant_id, window_ts))
        previous = WindowGraph.from_edges(
            await read_edges(self.redis, tenant_id, window_ts - self.window_delta)
        )

        vectors: list[StateVector] = []
        for host in hosts:
            vectors.extend(await self._emit_host(tenant_id, host, window_ts, graph, previous))

        await discard_window(self.redis, tenant_id, window_ts, hosts)
        # The graph of `window_ts` is still needed: it is the "previous window" the next
        # close reads for `neighbour_risk_fraction`.
        await discard_graph(self.redis, tenant_id, window_ts - self.window_delta)
        await clear_open(self.redis, tenant_id, window_ts)
        logger.debug(
            "tenant %s: closed window %d, emitted %d vector(s)", tenant_id, window_ts, len(vectors)
        )
        return vectors

    async def _emit_host(
        self,
        tenant_id: str,
        host: str,
        window_ts: int,
        graph: WindowGraph,
        previous: WindowGraph,
    ) -> list[StateVector]:
        """Silent windows since this host was last seen, then its vector for `window_ts`."""
        last_ts, history = await read_history(self.redis, tenant_id, host)
        vectors: list[StateVector] = []

        if last_ts is not None:
            history, silent = await self._fill_gap(tenant_id, host, last_ts, window_ts, history)
            vectors.extend(silent)

        accumulator = WindowAccumulator.from_hash(
            await read_window(self.redis, tenant_id, host, window_ts)
        )
        peers = set(accumulator.dist(DIST_DST_IP))
        new_peer_count = await record_peers(self.redis, tenant_id, host, peers)

        features = compute_features(
            accumulator,
            host=host,
            graph=graph,
            previous_graph=previous,
            new_peer_count=new_peer_count,
            history=history,
            elevated_fanout=self.elevated_fanout,
            degree_cap=self.degree_cap,
        )
        vectors.append(await self._publish(tenant_id, host, window_ts, features))

        history.append(tracked_values(features))
        await write_history(
            self.redis, tenant_id, host, last_ts=window_ts, history=history[-HISTORY_LEN:]
        )
        return vectors

    async def _fill_gap(
        self,
        tenant_id: str,
        host: str,
        last_ts: int,
        window_ts: int,
        history: list[dict],
    ) -> tuple[list[dict], list[StateVector]]:
        """Emit an `is_active=0` row for each window this host sat out."""
        gap = (window_ts - last_ts) // self.window_delta - 1
        if gap <= 0:
            return history, []
        if gap > self.max_gap_windows:
            logger.info(
                "tenant %s host %s: %d silent windows exceeds max_gap_windows=%d; "
                "restarting the sequence instead of backfilling",
                tenant_id,
                host,
                gap,
                self.max_gap_windows,
            )
            return [], []

        silent_tracked = dict.fromkeys(DYNAMIC_SOURCES, 0.0)
        vectors: list[StateVector] = []
        for step in range(1, gap + 1):
            silent_ts = last_ts + step * self.window_delta
            vectors.append(await self._publish(tenant_id, host, silent_ts, zero_features()))
            history = [*history, dict(silent_tracked)][-HISTORY_LEN:]
        return history, vectors

    async def _publish(
        self, tenant_id: str, host: str, window_ts: int, features: dict[str, float]
    ) -> StateVector:
        """Validate against `FEATURE_ORDER` one last time, store it, then put it on the bus.

        Stored before published, not after: a vector that reaches inference is then
        already durable, so every forecast in the database has the context that produced
        it sitting behind it. The other order leaves a window where a forecast exists and
        the state it was computed from does not.
        """
        validate_features(features)
        vector = StateVector(
            tenant_id=tenant_id,
            host_id=host,
            window_ts=window_start(window_ts),
            features=features,
        )
        if self.store is not None:
            await self.store.save(vector)
        await self.bus.publish(vector)
        return vector
