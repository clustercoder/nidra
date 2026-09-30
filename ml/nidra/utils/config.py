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


#: Marker a child mapping sets to replace the parent's mapping outright
#: instead of merging into it. `dataset.days` needs it: a CTU-only config
#: whose day list merged with CIC-IDS2017's would silently train on both.
REPLACE_MARKER = "_replace"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; lists and scalars REPLACE rather than combine.

    A list that merged would be impossible to shorten — a child config could
    never say "train on fewer days than the parent", which is the main thing
    the CTU-13 variants do. A mapping merges unless it sets `_replace: true`.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get(REPLACE_MARKER):
            out[key] = {k: v for k, v in value.items() if k != REPLACE_MARKER}
        elif key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_with_parents(path: Path, seen: tuple[Path, ...] = ()) -> tuple[dict, list[Path]]:
    """Resolve one config and its `extends:` chain, parents first."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"config `extends` cycle: {chain}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent_ref = cfg.pop("extends", None)
    if parent_ref is None:
        return cfg, [path]
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    if not parent_path.exists():
        raise FileNotFoundError(f"{path}: `extends: {parent_ref}` points at {parent_path}, which does not exist")
    parent_cfg, lineage = _read_with_parents(parent_path, seen=(*seen, path))
    return _deep_merge(parent_cfg, cfg), [*lineage, path]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config, resolving an `extends:` chain by deep merge.

    A child names only what differs from its parent; everything else is
    inherited, so the CTU-13 / transfer / combined variants cannot silently
    drift from the baseline in a hyperparameter nobody meant to change.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    cfg, lineage = _read_with_parents(Path(path))
    cfg["_config_path"] = str(path)
    cfg["_config_lineage"] = [str(p) for p in lineage]
    # The hash covers the WHOLE chain: a run is not reproducible from a child
    # config whose parent has since changed.
    cfg["_config_hash"] = config_hash(*lineage)
    cfg["_git_commit"] = git_commit_sha()
    return cfg


def config_hash(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()[:16]


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
