"""Suite-wide setup: the tests get their own Redis logical database.

Since P11 removed the `full` profile, `docker compose up -d` runs the workers as well as
the datastores. The test suite is a *second* deployment of the same plane — it builds the
api in-process and drives the same workers — so with one shared bus the containerised
ingest worker consumes a job the suite has just queued, fails it (the upload it names is a
host path no container can see), and the suite watches its own job turn to `error`.

Separating the two at the bus is the smallest fix that keeps both real: the suite writes
to `redis.test_db`, the compose stack keeps database 0, and neither reads the other's
streams. Applied here as the `NIDRA_REDIS_URL` override, before any module has read the
config. An override already in the environment wins — a runner that has pointed the suite
somewhere means it.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from nidra_common.config import ENV_PREFIX, load_config

REDIS_URL_ENV = f"{ENV_PREFIX}REDIS_URL"


def redis_url_on_db(url: str, db: int) -> str:
    """The same Redis URL, pointed at database `db`."""
    return urlunsplit(urlsplit(url)._replace(path=f"/{db}"))


def isolate_redis() -> None:
    """Point the suite at `redis.test_db` unless the environment already chose a URL."""
    if os.environ.get(REDIS_URL_ENV):
        return
    redis_cfg = dict(load_config().get("redis", {}))
    test_db = redis_cfg.get("test_db")
    if test_db is None:  # pragma: no cover — config ships one
        return
    os.environ[REDIS_URL_ENV] = redis_url_on_db(str(redis_cfg["url"]), int(test_db))


isolate_redis()
