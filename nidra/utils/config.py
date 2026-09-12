"""Configuration loading. config/default.yaml is the single source of truth
for hyperparameters; nothing important should be hardcoded in scripts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ml/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(path)
    cfg["_git_commit"] = git_commit_sha()
    return cfg


def config_hash(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def resolve_path(cfg: dict[str, Any], relative: str) -> Path:
    """Resolve a config-relative artifact path against the project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
