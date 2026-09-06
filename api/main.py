"""FastAPI application factory.

Run with `uvicorn api.main:app`. The factory exists so tests can build an isolated app
(and mount their own probe routes) without importing a module-level singleton that has
already been configured.

`/health` reports stream lag once the bus exists (P4); until then `streams` is empty
rather than absent, so the response shape does not change under the frontend later.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from pydantic import BaseModel

from api import auth
from nidra_common.db import dispose_engine

logger = logging.getLogger(__name__)

TITLE = "NIDRA"
DESCRIPTION = "Predictive network world model — serving plane."


def _package_version() -> str:
    try:
        return version("nidra-backend")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+unknown"


class HealthResponse(BaseModel):
    """Liveness, plus per-stream pending counts once the bus is wired in."""

    status: str
    streams: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    """Build the API application."""
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=_package_version(),
        lifespan=lifespan,
    )
    app.include_router(auth.router)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", streams={})

    return app


app = create_app()
