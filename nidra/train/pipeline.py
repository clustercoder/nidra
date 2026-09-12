"""Shared end-to-end pipeline: raw day files -> labelled state tables ->
splits -> scaler -> windowed tensors. Used by both train_dynamics.py and
train_heads.py so the two stages always see identically constructed data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nidra.data.dataset import WindowedArrays, build_windowed_arrays
from nidra.data.flow_load import load_cicflowmeter_csv
from nidra.data.join import build_day_inputs
from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH
from nidra.data.splits import SplitResult, assert_no_episode_leakage, assert_no_temporal_overlap, build_splits
from nidra.data.windowize import windowize_day
from nidra.utils.config import resolve_path

logger = logging.getLogger(__name__)


def load_and_label_day(flow_csv_path: str | Path, window_seconds: int, min_windows_per_host: int,
                        packets_parquet_path: str | Path | None = None, row_cap: int | None = None,
                        cache_path: str | Path | None = None) -> pd.DataFrame:
    """Full per-day pipeline: load CSV -> join/window -> windowize -> label.
    `packets_parquet_path` is optional; when absent, the day runs in
    flow-only mode (packet features zero-filled, logged). `row_cap` bounds
    how many input rows are read from this day's CSV — for a day file too
    large to fully load; `None` reads the whole file (what both
    config/default.yaml and config/mvp_2017.yaml currently use).

    `cache_path`, when given, short-circuits the whole CSV-load -> join ->
    windowize -> label pass if a matching parquet already exists there —
    this stage (real CIC-IDS2017 day files, up to ~700k flow rows and tens
    of millions of packets, with several per-host Python loops in
    windowize.py) is the dominant cost of every train/eval invocation, and
    without caching it is repeated from scratch by train_dynamics,
    train_heads, and every run_eval.py call (test split, holdout split) even
    though the underlying labelled table never changes for a fixed
    (day, window_seconds, min_windows_per_host, packet-availability) tuple.
    The caller is responsible for making `cache_path` reflect exactly that
    tuple (see `build_all_splits`) so a config change invalidates it rather
    than silently serving a stale table."""
    if cache_path is not None and Path(cache_path).exists():
        logger.info("load_and_label_day: using cached labelled table %s", cache_path)
        return pd.read_parquet(cache_path)

    raw, report = load_cicflowmeter_csv(flow_csv_path, row_cap=row_cap)
    packets_raw = pd.read_parquet(packets_parquet_path) if packets_parquet_path else pd.DataFrame()

    flows_w, packets_w = build_day_inputs(raw, packets_raw, window_seconds)
    states = windowize_day(flows_w, packets_w, window_seconds, min_windows_per_host)
    stage_table, label_report = label_stage_table(flows_w)
    if label_report["unmapped_labels"]:
        logger.warning("load_and_label_day: unmapped raw labels in %s: %s", flow_csv_path, label_report["unmapped_labels"])
    labelled = attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=window_seconds)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        labelled.to_parquet(cache_path)
        logger.info("load_and_label_day: cached labelled table to %s (%d rows)", cache_path, len(labelled))
    return labelled


def build_all_splits(cfg: dict) -> SplitResult:
    dataset_cfg = cfg["dataset"]
    windowing_cfg = cfg["windowing"]
    raw_dir = dataset_cfg.get("cic2017_flow_dir")
    if raw_dir is None:
        raise KeyError("cfg['dataset'] must set 'cic2017_flow_dir'")
    flow_dir = Path(raw_dir).expanduser()
    row_cap = dataset_cfg.get("mvp_row_cap_per_day")
    processed_dir = cfg.get("artifacts", {}).get("processed_dir")
    processed_dir = resolve_path(cfg, processed_dir) if processed_dir else None

    day_tables: dict[str, pd.DataFrame] = {}
    for day_key, day_meta in dataset_cfg["days"].items():
        csv_path = flow_dir / day_meta["file"]
        if not csv_path.exists():
            logger.warning("build_all_splits: %s not found, skipping day %s", csv_path, day_key)
            continue
        # Optional per-day packet parquet (from pcap_extract.py) — a day
        # without one runs in flow-only mode, logged loudly downstream in
        # windowize.build_state_rows, never silently treated as full-feature
        # data. `packets` in config is a path relative to `packets_dir` (or
        # `flow_dir` if `packets_dir` is unset); an absolute path is used
        # as-is.
        packets_parquet_path = None
        if day_meta.get("packets"):
            packets_dir = Path(dataset_cfg.get("packets_dir") or raw_dir).expanduser()
            candidate = Path(day_meta["packets"])
            packets_parquet_path = candidate if candidate.is_absolute() else packets_dir / candidate
            if not packets_parquet_path.exists():
                logger.warning("build_all_splits: packet parquet %s not found for day %s, running flow-only",
                                packets_parquet_path, day_key)
                packets_parquet_path = None
        logger.info("processing day %s (%s)%s", day_key, csv_path.name,
                    " with packet-level features" if packets_parquet_path else " (flow-only)")
        cache_path = None
        if processed_dir is not None:
            # Cache key encodes everything that changes the resulting table:
            # window geometry, host-count filter, packet availability (a day
            # gains real packet features the moment its PCAP is extracted —
            # the cache must not silently keep serving the old flow-only
            # table), and the row cap.
            packets_tag = Path(packets_parquet_path).name if packets_parquet_path else "flowonly"
            cache_path = processed_dir / (
                f"{day_key}__w{windowing_cfg['window_seconds']}"
                f"__m{windowing_cfg['min_windows_per_host']}"
                f"__{packets_tag}__cap{row_cap}.parquet"
            )
        day_tables[day_key] = load_and_label_day(
            csv_path,
            window_seconds=windowing_cfg["window_seconds"],
            min_windows_per_host=windowing_cfg["min_windows_per_host"],
            packets_parquet_path=packets_parquet_path,
            row_cap=row_cap,
            cache_path=cache_path,
        )

    splits_cfg = cfg["splits"]
    splits = build_splits(
        day_tables,
        train_days=splits_cfg["train_days"],
        test_days=splits_cfg["test_days"],
        holdout_days=splits_cfg["holdout_days"],
        val_fraction=splits_cfg["val_fraction_of_train_time"],
    )
    assert_no_temporal_overlap(splits)
    assert_no_episode_leakage(splits)
    return splits


def fit_scaler(train_arrays: WindowedArrays) -> FeatureScaler:
    """Fit RobustScaler on TRAIN split's raw states only — both the history
    (X) and future targets (Y) come from the same underlying state table, so
    fitting on the union of both is still "training period only"; it is
    never fit on val/test/holdout arrays."""
    all_train_states = np.concatenate([train_arrays.X.reshape(-1, train_arrays.X.shape[-1]),
                                        train_arrays.Y.reshape(-1, train_arrays.Y.shape[-1])], axis=0)
    return FeatureScaler().fit(all_train_states)


def scale_arrays(arrays: WindowedArrays, scaler: FeatureScaler) -> tuple[np.ndarray, np.ndarray]:
    return scaler.transform(arrays.X), scaler.transform(arrays.Y)


def build_windowed_splits(splits: SplitResult) -> dict[str, WindowedArrays]:
    return {
        name: build_windowed_arrays(df, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
        for name, df in [("train", splits.train), ("val", splits.val), ("test", splits.test), ("holdout", splits.holdout)]
    }
