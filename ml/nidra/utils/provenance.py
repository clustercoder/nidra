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
    from nidra.data.ctu_load import CTU_INTERNAL_CIDRS
    from nidra.train.pipeline import (_flow_dirs, day_cache_path, day_format, declared_packets_tag,
                                       host_scope_tag)
    from nidra.utils.config import resolve_path

    dataset_cfg = cfg.get("dataset")
    if not dataset_cfg or "days" not in dataset_cfg:
        return {"name": "unspecified (no dataset block in config)", "flow_timebase": _flow_timebase_tag(), "days": {}}
    flow_dirs = _flow_dirs(dataset_cfg)
    packets_dir = Path(dataset_cfg.get("packets_dir") or dataset_cfg.get("cic2017_flow_dir") or ".").expanduser()
    processed_dir = resolve_path(cfg, cfg.get("artifacts", {}).get("processed_dir", "artifacts/processed"))
    row_cap = dataset_cfg.get("mvp_row_cap_per_day")
    internal_cidrs = tuple(dataset_cfg.get("ctu13_internal_cidrs", CTU_INTERNAL_CIDRS) or ())
    host_scope = host_scope_tag(internal_cidrs)

    # `dataset.days[].role` is a hand-written annotation; the split a day is
    # ACTUALLY in comes from cfg["splits"], and the two had drifted apart —
    # ctu_4 and ctu_6 are annotated "test" and are the validation captures.
    # A provenance record that misstates which split a day landed in is worse
    # than no record, so the effective role is derived and the annotation is
    # kept beside it only when it disagrees.
    effective = _effective_roles(cfg)

    days: dict[str, Any] = {}
    sources: set[str] = set()
    for day_key, day_meta in dataset_cfg["days"].items():
        fmt = day_format(day_meta)
        sources.add(fmt)
        entry: dict[str, Any] = {
            "role": effective.get(day_key, "unused"),
            "format": fmt,
            "flow_file": file_stat(flow_dirs[fmt] / day_meta["file"]),
            "packets_parquet": file_stat(packets_dir / day_meta["packets"]) if day_meta.get("packets") else None,
        }
        annotated = day_meta.get("role")
        if annotated is not None and annotated != entry["role"]:
            entry["role_annotated_in_config"] = annotated
        if fmt == "ctu_binetflow":
            entry["family"] = day_meta.get("family")
            entry["scenario"] = day_meta.get("scenario")
        cache = day_cache_path(processed_dir, day_key, cfg["windowing"], row_cap, declared_packets_tag(day_meta),
                                flow_format=fmt, host_scope=host_scope)
        entry["processed_table"] = file_stat(cache)
        if hash_processed and cache.exists():
            entry["processed_table_sha256"] = file_digest(cache)
        days[day_key] = entry
    names = {
        "cicflowmeter": "CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet)",
        "ctu_binetflow": "CTU-13 (Argus binetflow, flow-only)",
    }
    return {
        "name": " + ".join(names[s] for s in sorted(sources)) or "unspecified",
        "flow_timebase": _flow_timebase_tag(),
        "feature_regime": cfg.get("features", {}).get("regime", "full"),
        "ctu13_internal_cidrs": list(internal_cidrs),
        "days": days,
    }


def _effective_roles(cfg: dict) -> dict[str, str]:
    """Which split each day actually lands in, read from cfg["splits"].

    A day named in `val_days` is validation; one only in `train_days` is
    training, unless it is also in `val_carve_train_days`, in which case its
    tail is carved into validation and it is both. A day in no list is unused
    by this run, which is worth recording as such rather than as whatever the
    config happened to annotate.
    """
    splits = cfg.get("splits") or {}
    roles: dict[str, str] = {}
    for key, role in (("train_days", "train"), ("test_days", "test"),
                      ("holdout_days", "holdout"), ("val_days", "val")):
        for day in splits.get(key) or []:
            roles[day] = role
    for day in splits.get("val_carve_train_days") or []:
        if roles.get(day) == "train":
            roles[day] = "train+val_carve"
    return roles


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
