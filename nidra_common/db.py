"""Async SQLAlchemy engine and session factory.

The Postgres URL comes from `config/default.yaml` (overridable with
``NIDRA_POSTGRES_URL``) — never from a literal in a service module. Both the engine
and the session factory are process-wide singletons; :func:`dispose_engine` tears them
down, which matters in tests where each case may run on its own event loop.

Nothing here creates tables. Schema changes are Alembic migrations, always.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nidra_common.config import get_config


def database_url(cfg: dict[str, Any] | None = None) -> str:
    """Async Postgres URL for this process."""
    config = cfg if cfg is not None else get_config()
    url = config.get("postgres", {}).get("url")
    if not url:
        raise ValueError("postgres.url is missing from the NIDRA config")
    return str(url)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Process-wide async engine."""
    return create_async_engine(database_url(), pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Process-wide session factory bound to :func:`get_engine`."""
    return async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI-style dependency yielding one session per request."""
    async with get_sessionmaker()() as session:
        yield session


async def dispose_engine() -> None:
    """Close pooled connections and drop the cached engine/session factory."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
    get_sessionmaker.cache_clear()
    get_engine.cache_clear()
