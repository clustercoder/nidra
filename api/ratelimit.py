"""Rate limiting: a Redis sliding window in front of every request.

`IMPLEMENTATION-Backend.md` §8 asks for 100 req/min in general and 5/hour on ingest. Both
ceilings, and the window each is measured over, are config rather than literals, and both
are multiplied by `api.rate_limit.dev_multiplier` when `NIDRA_ENV=dev` — the limiter still
runs on a development machine, so a bug in it surfaces there, but the ceiling is out of
reach of the test suite and of a console being clicked through.

The window is a Redis sorted set per `(scope, caller)`: entries scored by arrival time,
everything older than the window trimmed on each hit, and the cardinality compared with
the limit. A fixed counter per minute would let 2×limit through across a boundary; this
does not. A refused request removes its own entry, so being throttled cannot extend the
throttle.

The caller is its tenant when the request carries a valid token, and its address before
one exists — otherwise every unauthenticated login attempt on the internet would share one
bucket. The tenant is read from the token exactly as `api/deps.py` reads it; there is no
header or parameter that names a caller.

`/health` and `/metrics` are exempt. A compose healthcheck polling every few seconds must
never be throttled, and a throttled health endpoint would report a dead process.

Redis being unreachable fails **open**, with a warning. A limiter that refuses every
request when its datastore blinks is a worse outage than the one it prevents.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.auth import ACCESS_TOKEN_TYPE, decode_token
from nidra_common.bus import create_redis
from nidra_common.config import get_config, is_dev

logger = logging.getLogger(__name__)

#: Paths the limiter never counts: the liveness probe and the scrape endpoint.
EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/metrics"})

#: The upload route, which carries its own much tighter ceiling in addition to the
#: general one. Matched exactly; nothing else under `/api/v1/ingest` is an upload.
INGEST_PATH = "/api/v1/ingest"
INGEST_METHOD = "POST"

GENERAL_SCOPE = "general"
INGEST_SCOPE = "ingest"

RETRY_AFTER_HEADER = "Retry-After"

#: Key prefix in Redis. `rl:{scope}:{caller}`, one sorted set each.
KEY_PREFIX = "rl"

TOO_MANY = status.HTTP_429_TOO_MANY_REQUESTS

MILLISECONDS = 1000


def now_ms() -> int:
    """Wall-clock milliseconds — the score every window entry is stamped with."""
    return int(time.time() * MILLISECONDS)


# ---------------------------------------------------------------------------- settings


@dataclass(frozen=True, slots=True)
class Limit:
    """One ceiling: `limit` requests per `window_s` seconds."""

    scope: str
    limit: int
    window_s: float

    @property
    def window_ms(self) -> int:
        return max(int(self.window_s * MILLISECONDS), 1)


@dataclass(frozen=True, slots=True)
class RateLimitSettings:
    """Both ceilings, already adjusted for the environment."""

    enabled: bool
    general: Limit
    ingest: Limit


def _limit(block: dict[str, Any], scope: str, multiplier: int) -> Limit:
    return Limit(
        scope=scope,
        limit=int(block.get("limit", 0)) * multiplier,
        window_s=float(block.get("window_s", 60)),
    )


def rate_limit_settings(cfg: dict[str, Any] | None = None) -> RateLimitSettings:
    """Read `api.rate_limit`, applying the dev multiplier when this is a dev process."""
    config = cfg if cfg is not None else get_config()
    block = dict(dict(config.get("api", {})).get("rate_limit", {}))
    multiplier = int(block.get("dev_multiplier", 1)) if is_dev(config) else 1
    return RateLimitSettings(
        enabled=bool(block.get("enabled", True)),
        general=_limit(dict(block.get("general", {})), GENERAL_SCOPE, multiplier),
        ingest=_limit(dict(block.get("ingest", {})), INGEST_SCOPE, multiplier),
    )


# ------------------------------------------------------------------------- the window


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one hit against one ceiling."""

    allowed: bool
    limit: Limit
    remaining: int
    retry_after: int = 0


def caller_key(request: Request) -> str:
    """`tenant:{uuid}` for an authenticated caller, `ip:{address}` before that.

    The tenant comes from the verified token and from nowhere else, so two tenants can
    never share a bucket and no caller can choose which bucket it lands in.
    """
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() == "bearer" and token:
        try:
            claims = decode_token(token.strip(), expected_type=ACCESS_TOKEN_TYPE)
        except HTTPException:
            pass  # invalid or expired: limited as an anonymous caller, and 401 downstream
        else:
            return f"tenant:{claims['tenant_id']}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


async def hit(redis: Redis, caller: str, limit: Limit, at_ms: int) -> Decision:
    """Record one request against `limit` and say whether it is allowed.

    Over the ceiling, the entry just added is removed again: a caller hammering a closed
    window would otherwise keep pushing its own retry time out.
    """
    key = f"{KEY_PREFIX}:{limit.scope}:{caller}"
    member = f"{at_ms}-{uuid.uuid4().hex}"

    pipe = redis.pipeline(transaction=True)
    pipe.zremrangebyscore(key, 0, at_ms - limit.window_ms)
    pipe.zadd(key, {member: at_ms})
    pipe.zcard(key)
    pipe.pexpire(key, limit.window_ms)
    used = int((await pipe.execute())[2])

    if used <= limit.limit:
        return Decision(allowed=True, limit=limit, remaining=limit.limit - used)

    await redis.zrem(key, member)
    oldest = await redis.zrange(key, 0, 0, withscores=True)
    if oldest:
        free_at_ms = int(oldest[0][1]) + limit.window_ms
        retry_after = max(1, math.ceil((free_at_ms - at_ms) / MILLISECONDS))
    else:  # pragma: no cover — the window emptied between the count and this read
        retry_after = max(1, math.ceil(limit.window_s))
    return Decision(allowed=False, limit=limit, remaining=0, retry_after=retry_after)


def limits_for(request: Request, settings: RateLimitSettings) -> tuple[Limit, ...]:
    """Which ceilings this request is measured against, tightest last."""
    if request.method == INGEST_METHOD and request.url.path.rstrip("/") == INGEST_PATH:
        return (settings.general, settings.ingest)
    return (settings.general,)


def too_many(decision: Decision) -> JSONResponse:
    """The 429, carrying the `Retry-After` a client is expected to honour."""
    return JSONResponse(
        status_code=TOO_MANY,
        content={
            "detail": (
                f"rate limit exceeded: {decision.limit.limit} requests per "
                f"{int(decision.limit.window_s)}s on {decision.limit.scope}"
            )
        },
        headers={RETRY_AFTER_HEADER: str(decision.retry_after)},
    )


class RateLimiter:
    """The middleware. One Redis client per request, as everywhere else in `api/`.

    A pool binds to the event loop it is first used on, and the api's own tests build a
    fresh app per test; a per-request client costs a connection and removes that failure
    mode entirely (DECISIONS.md D30).
    """

    def __init__(self, settings: RateLimitSettings, cfg: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.cfg = cfg

    def applies_to(self, request: Request) -> bool:
        return self.settings.enabled and request.url.path not in EXEMPT_PATHS

    async def check(self, request: Request, at_ms: int) -> Decision | None:
        """The first ceiling this request exceeds, or `None` when it is under all of them."""
        caller = caller_key(request)
        redis = create_redis(self.cfg)
        try:
            for limit in limits_for(request, self.settings):
                decision = await hit(redis, caller, limit, at_ms)
                if not decision.allowed:
                    return decision
        except RedisError as exc:
            # Fail open: an unreachable Redis must not take the API down with it.
            logger.warning("rate limiter unavailable, allowing request: %s", exc)
        finally:
            await redis.aclose()
        return None

    async def __call__(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.applies_to(request):
            return await call_next(request)
        decision = await self.check(request, now_ms())
        if decision is not None:
            logger.info(
                "rate limited %s %s on %s; retry after %ss",
                request.method,
                request.url.path,
                decision.limit.scope,
                decision.retry_after,
            )
            return too_many(decision)
        return await call_next(request)
