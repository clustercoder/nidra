"""Experiment provenance: the record every training / evaluation run must
carry so that a number can be traced back to exactly what produced it.

One function builds the record (`experiment_record`); `file_digest` and
`dataset_version` describe the inputs. Nothing here decides anything — it
only observes and writes down. Every field in the record is one the audit
found missing or ambiguous at least once (a committed F1 from a 20-rollout
smoke run looked identical to the published 200-rollout number; the shipped
pooling quantile was stamped "mean"; the scaler that scaled the corrected
features had been fit before the fix). The record makes those mistakes
visible instead of silent.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from nidra.data.schema import FEATURE_ORDER, SCHEMA_VERSION


def file_digest(path: str | Path, algorithm: str = "sha256") -> str | None:
    """Content hash of one file, or None if it does not exist. Large files
    (the multi-hundred-MB packet parquet) are streamed, never read whole."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: str | Path) -> dict[str, Any] | None:
    """Size and mtime — enough to notice a swapped input without hashing a
    multi-GB CSV on every run."""
    p = Path(path)
    if not p.exists():
        return None
    st = p.stat()
    return {"path": str(p), "bytes": int(st.st_size), "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}


def dataset_version(cfg: dict, hash_processed: bool = True) -> dict[str, Any]:
    """Describe the dataset a run saw: the declared day files (size/mtime),
    the declared packet parquet, and the content hash of every cached
    windowed table actually used (those are small and are the thing the
    model reads)."""
    from nidra.train.pipeline import day_cache_path, declared_packets_tag
    from nidra.utils.config import resolve_path

    dataset_cfg = cfg["dataset"]
    flow_dir = Path(dataset_cfg["cic2017_flow_dir"]).expanduser()
    packets_dir = Path(dataset_cfg.get("packets_dir") or dataset_cfg["cic2017_flow_dir"]).expanduser()
    processed_dir = resolve_path(cfg, cfg["artifacts"]["processed_dir"])
    row_cap = dataset_cfg.get("mvp_row_cap_per_day")

    days: dict[str, Any] = {}
    for day_key, day_meta in dataset_cfg["days"].items():
        entry: dict[str, Any] = {
            "role": day_meta.get("role"),
            "flow_csv": file_stat(flow_dir / day_meta["file"]),
            "packets_parquet": file_stat(packets_dir / day_meta["packets"]) if day_meta.get("packets") else None,
        }
        cache = day_cache_path(processed_dir, day_key, cfg["windowing"], row_cap, declared_packets_tag(day_meta))
        entry["processed_table"] = file_stat(cache)
        if hash_processed and cache.exists():
            entry["processed_table_sha256"] = file_digest(cache)
        days[day_key] = entry
    return {
        "name": "CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet)",
        "flow_timebase": _flow_timebase_tag(),
        "days": days,
    }


def _flow_timebase_tag() -> str:
    from nidra.data.windowize import CIC2017_TIMEBASE_TAG
    return CIC2017_TIMEBASE_TAG


def geometry(cfg: dict) -> dict[str, Any]:
    w = cfg["windowing"]
    return {
        "window_seconds": int(w["window_seconds"]),
        "context_length_L": int(w["context_length"]),
        "horizon_length_K": int(w["horizon_length"]),
        "history_minutes": w["window_seconds"] * w["context_length"] / 60.0,
        "horizon_minutes": w["window_seconds"] * w["horizon_length"] / 60.0,
        "min_windows_per_host": int(w["min_windows_per_host"]),
    }


def checkpoint_digests(cfg: dict, seeds: list[int] | None = None) -> dict[str, Any]:
    from nidra.utils.config import resolve_path

    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    scaler_dir = resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    seeds = seeds if seeds is not None else cfg["ensemble"]["seeds"]
    out: dict[str, Any] = {"weights": {}, "scaler": {}}
    for s in seeds:
        out["weights"][f"seed_{s}"] = {
            "path": str(weights_dir / f"model_seed_{s}.pt"),
            "sha256": file_digest(weights_dir / f"model_seed_{s}.pt"),
        }
    for name in ("robust_scaler.joblib", "feature_scaler.json", "scaler_metadata.json", "shap_background.npy"):
        out["scaler"][name] = file_digest(scaler_dir / name)
    calib = weights_dir / "risk_calibration.json"
    out["calibration"] = {"path": str(calib), "sha256": file_digest(calib)}
    return out


def experiment_record(
    cfg: dict,
    *,
    stage: str,
    split: str | None = None,
    seed: int | None = None,
    seeds: list[int] | None = None,
    threshold: float | None = None,
    pooling: dict | None = None,
    calibration: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
    hash_processed: bool = False,
) -> dict[str, Any]:
    """The provenance block. Every experiment JSON written from here on
    starts with this. `hash_processed=True` also content-hashes the cached
    windowed tables (a few seconds); the baseline manifest does this, routine
    per-epoch logging does not."""
    record: dict[str, Any] = {
        "stage": stage,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": cfg.get("_git_commit"),
        "config_path": cfg.get("_config_path"),
        "config_hash": cfg.get("_config_hash"),
        "schema_version": SCHEMA_VERSION,
        "n_features": len(FEATURE_ORDER),
        "geometry": geometry(cfg),
        "dataset": dataset_version(cfg, hash_processed=hash_processed),
        "seed": seed,
        "ensemble_seeds": seeds if seeds is not None else cfg.get("ensemble", {}).get("seeds"),
        "split": split,
        "threshold": threshold if threshold is not None else cfg.get("eval", {}).get("risk_threshold"),
        "pooling": pooling,
        "calibration": calibration,
        "checkpoints": checkpoint_digests(cfg, seeds),
        "platform": {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system()},
        "metrics": metrics,
    }
    if extra:
        record.update(extra)
    return record


def write_json(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return p


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)
