"""Configuration loader.

`config/default.yaml` is the single source of truth for every tunable in the serving
plane. Nothing is hardcoded in a script. A small set of deployment-specific values —
the Redis and Postgres URLs and the JWT secret — may be overridden by `NIDRA_`-prefixed
environment variables so that the same image runs under docker compose and locally.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"

ENV_PREFIX = "NIDRA_"

#: Development-only fallback. Never used when NIDRA_SECRET_KEY is set; every load
#: without it logs a warning at WARNING level so it cannot slip into a deployment
#: unnoticed.
DEV_SECRET_KEY = "dev-insecure-secret-change-me"

#: env var -> path into the config tree.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    f"{ENV_PREFIX}REDIS_URL": ("redis", "url"),
    f"{ENV_PREFIX}POSTGRES_URL": ("postgres", "url"),
    f"{ENV_PREFIX}SECRET_KEY": ("auth", "secret_key"),
    f"{ENV_PREFIX}UPLOAD_DIR": ("ingest", "upload_dir"),
    f"{ENV_PREFIX}ENV": ("env",),
    f"{ENV_PREFIX}PREDICTOR_IMPL": ("predictor", "impl"),
}

#: Comma-separated list, not a single scalar, so it is handled separately from
#: ENV_OVERRIDES: the frontend's deployed origin (e.g. a Vercel URL) is only known at
#: deploy time and must not require editing config/default.yaml per environment.
CORS_ORIGINS_ENV = f"{ENV_PREFIX}CORS_ORIGINS"

#: The environment name that relaxes the API rate limits. Everything else — including an
#: unset `NIDRA_ENV` under a config that does not say `dev` — gets the real ceilings.
DEV_ENV = "dev"


def _set_in(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set `value` at `path` inside a nested dict, creating intermediate dicts."""
    node = tree
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = value


def config_path() -> Path:
    """Path of the YAML config actually loaded (``NIDRA_CONFIG_PATH`` wins)."""
    override = os.environ.get(f"{ENV_PREFIX}CONFIG_PATH")
    return Path(override).resolve() if override else DEFAULT_CONFIG_PATH


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the YAML config and apply ``NIDRA_`` environment overrides.

    Uncached: callers that want a process-wide singleton use :func:`get_config`.
    """
    resolved = Path(path).resolve() if path is not None else config_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"NIDRA config not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"NIDRA config must be a YAML mapping: {resolved}")

    for env_name, target in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            _set_in(cfg, target, value)

    cors_origins = os.environ.get(CORS_ORIGINS_ENV)
    if cors_origins:
        origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
        _set_in(cfg, ("api", "cors_origins"), origins)

    auth = cfg.setdefault("auth", {})
    if not auth.get("secret_key"):
        logger.warning(
            "%sSECRET_KEY is not set — falling back to the insecure development key. "
            "Set it before running anything that is not a local demo.",
            ENV_PREFIX,
        )
        auth["secret_key"] = DEV_SECRET_KEY

    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Process-wide cached config. Use in service startup paths."""
    return load_config()


def environment(cfg: dict[str, Any] | None = None) -> str:
    """Deployment environment name (`NIDRA_ENV` overrides `env:` in the YAML)."""
    config = cfg if cfg is not None else get_config()
    return str(config.get("env") or "").strip().lower()


def is_dev(cfg: dict[str, Any] | None = None) -> bool:
    """True when this process is running as development, which relaxes rate limits."""
    return environment(cfg) == DEV_ENV
