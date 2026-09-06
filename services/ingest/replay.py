"""Capture time → wall-clock time, at a speed multiplier.

A capture replayed at full speed arrives in a burst: every window closes in the same
instant, the cone never opens on screen, and the thing the demo is meant to show —
a trajectory bending over minutes — is over before anyone looks up. `ReplayClock` maps
each event's capture timestamp onto the wall clock so 60× means one 30 s window every
half second.

Both the clock and the sleep are injectable so the speed arithmetic can be tested
without spending the time it describes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

#: Returns a monotonically increasing seconds count. `time.monotonic` in production.
Monotonic = Callable[[], float]

#: Sleeps for the given number of seconds. `asyncio.sleep` in production.
Sleeper = Callable[[float], Awaitable[None]]


def as_utc(value: datetime) -> datetime:
    """Comparable UTC datetime; a naive timestamp is read as UTC, as the schemas state."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ReplayClock:
    """Paces a stream of timestamped events against wall-clock time.

    `speed` is a multiplier on capture time: 60 replays a minute of capture per second.
    `speed <= 0` disables pacing entirely — every `wait_until` returns immediately, which
    is what the tests and the offline backfill path want.
    """

    def __init__(
        self,
        first_ts: datetime,
        speed: float = 60.0,
        *,
        monotonic: Monotonic = time.monotonic,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        self.first_ts = as_utc(first_ts)
        self.speed = float(speed)
        self._monotonic = monotonic
        self._sleep = sleep
        self.start = monotonic()

    @property
    def paced(self) -> bool:
        """False when the clock is a no-op (`speed <= 0`)."""
        return self.speed > 0

    def delay_for(self, event_ts: datetime) -> float:
        """Seconds to wait before publishing `event_ts`. Negative means already late."""
        if not self.paced:
            return 0.0
        target = (as_utc(event_ts) - self.first_ts).total_seconds() / self.speed
        return target - (self._monotonic() - self.start)

    async def wait_until(self, event_ts: datetime) -> float:
        """Sleep until `event_ts` is due; returns the seconds actually slept.

        Running behind is normal and is not corrected for: the clock never sleeps a
        negative amount and never tries to catch up by publishing early.
        """
        delay = self.delay_for(event_ts)
        if delay <= 0:
            return 0.0
        await self._sleep(delay)
        return delay
