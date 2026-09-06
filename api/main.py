"""FastAPI application factory.

Run with `uvicorn api.main:app`. The factory exists so tests can build an isolated app
(and mount their own probe routes) without importing a module-level singleton that has
already been configured.

The lifespan owns one background consumer: the `forecasts` fan-out behind the WebSocket
(`api/ws.py`). It is built here rather than inside the socket route so a test can swap in
a private stream before the app starts, and so a process with no sockets open still keeps
its place in the `api` consumer group.

`/health` answers with per-stream pending counts — liveness plus the one number that says
whether a stage of the pipeline is falling behind — and stays 200 with a `degraded`
status when Redis is unreachable, because a health endpoint that fails to answer is
indistinguishable from a process that is gone.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from redis.exceptions import RedisError

from api import auth, forecasts, ingest, ws
from api.metrics import CONTENT_TYPE, RequestCounter, render, stream_lag
from nidra_common.bus import create_redis
from nidra_common.db import dispose_engine

logger = logging.getLogger(__name__)

TITLE = "NIDRA"
DESCRIPTION = "Predictive network world model — serving plane."

HEALTH_OK = "ok"
HEALTH_DEGRADED = "degraded"


def _package_version() -> str:
    try:
        return version("nidra-backend")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+unknown"


class HealthResponse(BaseModel):
    """Liveness, plus the pending count of every consumer group in the pipeline."""

    status: str
    streams: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await app.state.fanout.start()
    try:
        yield
    finally:
        await app.state.fanout.stop()
        await dispose_engine()


async def _lag(app: FastAPI) -> tuple[str, dict[str, int]]:
    """Stream lag and the status it implies. One client per call; these are rare."""
    redis = create_redis()
    try:
        return HEALTH_OK, await stream_lag(redis)
    except RedisError as exc:
        logger.warning("stream lag unavailable: %s", exc)
        return HEALTH_DEGRADED, {}
    finally:
        await redis.aclose()


def create_app() -> FastAPI:
    """Build the API application."""
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=_package_version(),
        lifespan=lifespan,
    )
    app.state.requests = RequestCounter()
    app.state.ws_settings = ws.ws_settings()
    app.state.connections = ws.ConnectionManager(app.state.ws_settings.queue_size)
    app.state.fanout = ws.ForecastFanout(app.state.connections)

    app.include_router(auth.router)
    app.include_router(ingest.router)
    app.include_router(forecasts.router)
    app.include_router(ws.router)

    @app.middleware("http")
    async def count_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """The only thing standing between `/metrics` and a request-count client library."""
        response = await call_next(request)
        app.state.requests.observe(request.method, response.status_code)
        return response

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        status_value, lag = await _lag(app)
        return HealthResponse(status=status_value, streams=lag)

    @app.get("/metrics", response_class=Response, tags=["ops"])
    async def metrics() -> Response:
        _, lag = await _lag(app)
        body = render(
            requests=app.state.requests,
            ws_connections=app.state.connections.connection_count,
            ws_dropped=app.state.connections.dropped_total,
            lag=lag,
        )
        return Response(content=body, media_type=CONTENT_TYPE)

    return app


app = create_app()
