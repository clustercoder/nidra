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

logger = logging.getLogger(__name__)


def load_and_label_day(flow_csv_path: str | Path, window_seconds: int, min_windows_per_host: int,
                        packets_parquet_path: str | Path | None = None, row_cap: int | None = None) -> pd.DataFrame:
    """Full per-day pipeline: load CSV -> join/window -> windowize -> label.
    `packets_parquet_path` is optional; when absent, the day runs in
    flow-only mode (packet features zero-filled, logged). `row_cap` bounds
    how many input rows are read from this day's CSV — used for MVP-scale
    runs against a full dataset (e.g. CSE-CIC-IDS2018) where reading every
    row of every day is not the goal; `None` reads the whole file, matching
    prior behavior."""
    raw, report = load_cicflowmeter_csv(flow_csv_path, row_cap=row_cap)
    packets_raw = pd.read_parquet(packets_parquet_path) if packets_parquet_path else pd.DataFrame()

    flows_w, packets_w = build_day_inputs(raw, packets_raw, window_seconds)
    states = windowize_day(flows_w, packets_w, window_seconds, min_windows_per_host)
    stage_table, label_report = label_stage_table(flows_w)
    if label_report["unmapped_labels"]:
        logger.warning("load_and_label_day: unmapped raw labels in %s: %s", flow_csv_path, label_report["unmapped_labels"])
    labelled = attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=window_seconds)
    return labelled


def build_all_splits(cfg: dict) -> SplitResult:
    dataset_cfg = cfg["dataset"]
    windowing_cfg = cfg["windowing"]
    # `flow_dir` is the dataset-agnostic key (used by config/mvp_2018.yaml);
    # `cic2017_flow_dir` is kept for backward compatibility with
    # config/default.yaml and config/real_smoke.yaml, which predate the
    # CSE-CIC-IDS2018 switch.
    raw_dir = dataset_cfg.get("flow_dir") or dataset_cfg.get("cic2017_flow_dir")
    if raw_dir is None:
        raise KeyError("cfg['dataset'] must set either 'flow_dir' or 'cic2017_flow_dir'")
    flow_dir = Path(raw_dir).expanduser()
    row_cap = dataset_cfg.get("mvp_row_cap_per_day")

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
        day_tables[day_key] = load_and_label_day(
            csv_path,
            window_seconds=windowing_cfg["window_seconds"],
            min_windows_per_host=windowing_cfg["min_windows_per_host"],
            packets_parquet_path=packets_parquet_path,
            row_cap=row_cap,
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
