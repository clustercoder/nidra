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
from nidra.data.ctu_load import CTU_INTERNAL_CIDRS, load_binetflow
from nidra.data.flow_load import load_cicflowmeter_csv
from nidra.data.join import build_day_inputs
from nidra.data.labels import attach_risk_label, label_stage_table, reattach_risk_label
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, FEATURE_INDEX, HORIZON_LENGTH, WINDOW_SECONDS, regime_kinds
from nidra.data.splits import SplitResult, assert_no_episode_leakage, assert_no_temporal_overlap, build_splits
from nidra.data.windowize import CIC2017_TIMEBASE_TAG, FUSION_TAG, windowize_day
from nidra.utils.config import resolve_path

logger = logging.getLogger(__name__)


#: Flow formats this pipeline can read. `cicflowmeter` is CIC-IDS2017's
#: published CSV; `ctu_binetflow` is CTU-13's Argus bidirectional NetFlow
#: (nidra/data/ctu_load.py). Both land on the same internal columns, so
#: everything after the loader is shared.
FLOW_FORMATS = ("cicflowmeter", "ctu_binetflow")

#: Label taxonomy that goes with each format by default (labels.py).
_DEFAULT_DIALECT = {"cicflowmeter": "cic", "ctu_binetflow": "ctu"}


def day_format(day_meta: dict) -> str:
    """The flow format of one config day entry, defaulting to the CIC-IDS2017
    CSV so every existing config keeps working untouched."""
    fmt = str(day_meta.get("format", "cicflowmeter"))
    if fmt not in FLOW_FORMATS:
        raise ValueError(f"unknown flow format {fmt!r}; expected one of {FLOW_FORMATS}")
    return fmt


def _load_raw_flows(path: Path, flow_format: str, row_cap: int | None,
                    internal_cidrs: tuple[str, ...] | None):
    if flow_format == "cicflowmeter":
        return load_cicflowmeter_csv(path, row_cap=row_cap)
    if flow_format == "ctu_binetflow":
        return load_binetflow(path, row_cap=row_cap, internal_cidrs=internal_cidrs)
    raise ValueError(f"unknown flow format {flow_format!r}; expected one of {FLOW_FORMATS}")


def load_and_label_day(flow_csv_path: str | Path, window_seconds: int, min_windows_per_host: int,
                        packets_parquet_path: str | Path | None = None, row_cap: int | None = None,
                        cache_path: str | Path | None = None, horizon_k: int = HORIZON_LENGTH,
                        flow_format: str = "cicflowmeter", label_dialect: str | None = None,
                        internal_cidrs: tuple[str, ...] | None = CTU_INTERNAL_CIDRS) -> pd.DataFrame:
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

    if flow_format not in FLOW_FORMATS:
        raise ValueError(f"unknown flow format {flow_format!r}; expected one of {FLOW_FORMATS}")
    dialect = label_dialect or _DEFAULT_DIALECT[flow_format]
    raw, report = _load_raw_flows(Path(flow_csv_path), flow_format, row_cap, internal_cidrs)
    packets_raw = pd.read_parquet(packets_parquet_path) if packets_parquet_path else pd.DataFrame()

    # CTU-13's Argus records already carry a true epoch and none of the
    # CIC-IDS2017 CSV clock defects, so neither correction is applied to them.
    flows_w, packets_w = build_day_inputs(
        raw, packets_raw, window_seconds,
        timestamp_is_epoch=(flow_format == "ctu_binetflow"),
    )
    states = windowize_day(flows_w, packets_w, window_seconds, min_windows_per_host)
    stage_table, label_report = label_stage_table(flows_w, dialect=dialect)
    if label_report["unmapped_labels"]:
        logger.warning("load_and_label_day: unmapped raw labels in %s: %s", flow_csv_path, label_report["unmapped_labels"])
    labelled = attach_risk_label(states, stage_table, horizon_k=horizon_k, window_seconds=window_seconds)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        labelled.to_parquet(cache_path)
        logger.info("load_and_label_day: cached labelled table to %s (%d rows)", cache_path, len(labelled))
    return labelled


FLOW_ONLY_TAG = "flowonly"


def declared_packets_tag(day_meta: dict) -> str:
    """Cache-key tag for a day, derived from the config-DECLARED packet file
    and deliberately NOT from whatever happens to sit on the local disk.

    A clone of this repo carries the committed windowed parquet under
    artifacts/processed/ but neither the ~50GB raw CIC-IDS2017 release nor
    the tshark-extracted packet parquet. Keying on local presence there
    would compute a `flowonly` tag, miss every committed file, and skip the
    day — so the tag states what the cached table is supposed to contain.
    """
    packets = day_meta.get("packets")
    return Path(packets).name if packets else FLOW_ONLY_TAG


def host_scope_tag(internal_cidrs: tuple[str, ...] | None) -> str:
    """Cache-key tag for the monitored-network host filter. A table built
    over the whole Internet-facing source population is a different table
    from one built over 147.32.0.0/16, with the same day key."""
    if not internal_cidrs:
        return "allhosts"
    return "_".join(str(c).replace("/", "-") for c in internal_cidrs)


def day_cache_path(processed_dir: str | Path, day_key: str, windowing_cfg: dict,
                    row_cap: int | None, packets_tag: str,
                    flow_format: str = "cicflowmeter", host_scope: str | None = None) -> Path:
    """Cache key encodes everything that changes the resulting table: the
    window length, packet availability (a day gains real packet features the
    moment its PCAP is extracted — the cache must not keep serving the old
    flow-only table), the row cap, the flow timebase (CIC2017_TIMEBASE_TAG —
    the CSV clock corrections in windowize.parse_cic_timestamp decide which
    packets a flow window meets at all) and the flow/packet fusion rule
    (windowize.FUSION_TAG).

    Deliberately NOT in the key: the history/horizon geometry (L, K) and the
    per-host window-count filter. The cached table is the unfiltered state
    table with the K-independent `stage_label`; `risk_label` is re-derived
    for the configured K and hosts are filtered to L + K windows when a
    split is assembled (`build_all_splits`), so one table per day serves
    every geometry. The Δ=30 tables (`__m36__...utc12h.parquet`, no fusion
    tag) are kept under the old key as the historical record; this loader
    does not read them."""
    if flow_format == "cicflowmeter":
        # Unchanged key: the Δ=60 CIC tables already on disk must keep hitting.
        return Path(processed_dir) / (
            f"{day_key}__w{windowing_cfg['window_seconds']}"
            f"__{packets_tag}__cap{row_cap}__{CIC2017_TIMEBASE_TAG}__{FUSION_TAG}.parquet"
        )
    return Path(processed_dir) / (
        f"{day_key}__{flow_format}__w{windowing_cfg['window_seconds']}"
        f"__{packets_tag}__cap{row_cap}__{host_scope or 'allhosts'}__{FUSION_TAG}.parquet"
    )


def _resolve_packets_path(dataset_cfg: dict, day_meta: dict, day_key: str) -> Path | None:
    """Locate this day's packet parquet, or None to run flow-only. `packets`
    in config is a path relative to `packets_dir` (or the flow dir if unset);
    an absolute path is used as-is. A day without one runs in flow-only mode,
    logged loudly downstream in windowize.build_state_rows, never silently
    treated as full-feature data."""
    if not day_meta.get("packets"):
        return None
    packets_dir = Path(dataset_cfg.get("packets_dir") or dataset_cfg["cic2017_flow_dir"]).expanduser()
    candidate = Path(day_meta["packets"])
    path = candidate if candidate.is_absolute() else packets_dir / candidate
    if not path.exists():
        logger.warning("build_all_splits: packet parquet %s not found for day %s, running flow-only",
                        path, day_key)
        return None
    return path


#: Which dataset a format belongs to, for the `source_dataset` column that
#: lets a combined CIC+CTU split still be reported per dataset.
_DATASET_OF_FORMAT = {"cicflowmeter": "cic2017", "ctu_binetflow": "ctu13"}


def _flow_dirs(dataset_cfg: dict) -> dict[str, Path]:
    """Base directory for each flow format present in the config. A format
    that no day uses need not be configured."""
    dirs: dict[str, Path] = {}
    if dataset_cfg.get("cic2017_flow_dir"):
        dirs["cicflowmeter"] = Path(dataset_cfg["cic2017_flow_dir"]).expanduser()
    if dataset_cfg.get("ctu13_dir"):
        dirs["ctu_binetflow"] = Path(dataset_cfg["ctu13_dir"]).expanduser()
    needed = {day_format(m) for m in dataset_cfg.get("days", {}).values()}
    missing = needed - set(dirs)
    if missing:
        keys = {"cicflowmeter": "cic2017_flow_dir", "ctu_binetflow": "ctu13_dir"}
        raise KeyError(f"cfg['dataset'] must set {[keys[m] for m in sorted(missing)]} for the configured days")
    return dirs


def geometry_from_config(cfg: dict) -> tuple[int, int, int]:
    """(window_seconds, L, K) — the one place the config's geometry is read
    for windowing, labelling and sample construction. `labels.
    risk_threshold_windows`, if present, must agree with K: the risk label
    is "attack in (t, t+K]" and a horizon that differs from the rollout's
    would make the label and the forecast describe different questions."""
    w = cfg.get("windowing", {})
    window_seconds = int(w.get("window_seconds", WINDOW_SECONDS))
    L = int(w.get("context_length", CONTEXT_LENGTH))
    K = int(w.get("horizon_length", HORIZON_LENGTH))
    declared = cfg.get("labels", {}).get("risk_threshold_windows")
    if declared is not None and int(declared) != K:
        raise ValueError(
            f"labels.risk_threshold_windows={declared} disagrees with windowing.horizon_length={K}; "
            "the risk label horizon and the rollout horizon must be the same K"
        )
    return window_seconds, L, K


def _filter_min_windows(table: pd.DataFrame, min_windows_per_host: int) -> pd.DataFrame:
    if table.empty or min_windows_per_host <= 1:
        return table
    counts = table.groupby("host_id")["window_ts"].transform("count")
    return table[counts >= min_windows_per_host].reset_index(drop=True)


def build_all_splits(cfg: dict) -> SplitResult:
    dataset_cfg = cfg["dataset"]
    windowing_cfg = cfg["windowing"]
    window_seconds, L, K = geometry_from_config(cfg)
    min_windows = int(windowing_cfg.get("min_windows_per_host", L + K))
    if min_windows < L + K:
        raise ValueError(f"windowing.min_windows_per_host={min_windows} is below L+K={L + K}; no sample could be built")
    flow_dirs = _flow_dirs(dataset_cfg)
    internal_cidrs = tuple(dataset_cfg.get("ctu13_internal_cidrs", CTU_INTERNAL_CIDRS) or ())
    host_scope = host_scope_tag(internal_cidrs)
    row_cap = dataset_cfg.get("mvp_row_cap_per_day")
    processed_dir = cfg.get("artifacts", {}).get("processed_dir")
    processed_dir = resolve_path(cfg, processed_dir) if processed_dir else None

    day_tables: dict[str, pd.DataFrame] = {}
    for day_key, day_meta in dataset_cfg["days"].items():
        fmt = day_format(day_meta)
        csv_path = flow_dirs[fmt] / day_meta["file"]
        # Resolution order matters. A cache written under the DECLARED packet
        # tag is authoritative and needs no raw inputs at all — this is what
        # lets a clone run evaluation with only the committed
        # artifacts/processed/ tables. Only on a cache miss do we fall back
        # to recomputing, which does require the raw CSV.
        declared_cache = (
            day_cache_path(processed_dir, day_key, windowing_cfg, row_cap, declared_packets_tag(day_meta),
                           flow_format=fmt, host_scope=host_scope)
            if processed_dir is not None else None
        )
        if declared_cache is not None and declared_cache.exists():
            day_tables[day_key] = load_and_label_day(
                csv_path,
                window_seconds=window_seconds,
                min_windows_per_host=1,
                cache_path=declared_cache,
                horizon_k=K,
                flow_format=fmt,
            )
            continue

        if not csv_path.exists():
            logger.warning("build_all_splits: skipping day %s — no cached table at %s and no raw CSV at %s",
                            day_key, declared_cache, csv_path)
            continue

        packets_parquet_path = _resolve_packets_path(dataset_cfg, day_meta, day_key)
        logger.info("processing day %s (%s)%s", day_key, csv_path.name,
                    " with packet-level features" if packets_parquet_path else " (flow-only)")
        # Recompute writes under the tag actually used, which differs from
        # the declared tag when the packet parquet is missing locally — a
        # degraded flow-only table must never land under a full-feature key.
        actual_tag = Path(packets_parquet_path).name if packets_parquet_path else FLOW_ONLY_TAG
        cache_path = (day_cache_path(processed_dir, day_key, windowing_cfg, row_cap, actual_tag,
                                     flow_format=fmt, host_scope=host_scope)
                      if processed_dir is not None else None)
        day_tables[day_key] = load_and_label_day(
            csv_path,
            window_seconds=window_seconds,
            min_windows_per_host=1,
            packets_parquet_path=packets_parquet_path,
            row_cap=row_cap,
            cache_path=cache_path,
            horizon_k=K,
            flow_format=fmt,
            internal_cidrs=internal_cidrs or None,
        )

    # The cached table is unfiltered and K-agnostic: apply this config's
    # geometry now — hosts need at least L+K windows to yield one sample,
    # and risk_label is re-derived for this K regardless of the K the
    # table was cached under.
    for day_key in list(day_tables):
        table = _filter_min_windows(day_tables[day_key], min_windows)
        if not table.empty:
            table = reattach_risk_label(table, K)
            # Which capture each row came from, so the validation block is
            # carved per capture rather than per calendar date (several CTU
            # scenarios share a date) and so per-source reporting is possible
            # once two datasets are concatenated.
            table = table.assign(
                split_group=day_key,
                source_dataset=str(dataset_cfg["days"][day_key].get("dataset", _DATASET_OF_FORMAT[day_format(dataset_cfg["days"][day_key])])),
            )
        day_tables[day_key] = table

    splits_cfg = cfg["splits"]
    splits = build_splits(
        day_tables,
        train_days=splits_cfg["train_days"],
        test_days=splits_cfg["test_days"],
        holdout_days=splits_cfg["holdout_days"],
        val_fraction=splits_cfg["val_fraction_of_train_time"],
        val_block_per_day=bool(splits_cfg.get("val_block_per_day", True)),
        horizon_k=K,
        pre_onset_margin_s=int(splits_cfg.get("pre_onset_margin_minutes", 30)) * 60,
        val_days=splits_cfg.get("val_days"),
    )
    assert_no_temporal_overlap(splits)
    assert_no_episode_leakage(splits)
    return splits


def training_dataset_tag(cfg: dict) -> str:
    """Which datasets this config's TRAIN days come from, e.g. `cic2017`,
    `ctu13`, or `cic2017+ctu13`. Part of the scaler's identity: normalizing
    a combined training population with a single-dataset scaler is a silent
    error, not a loud one."""
    days = cfg.get("dataset", {}).get("days", {})
    train_keys = cfg.get("splits", {}).get("train_days", [])
    tags = sorted({
        str(days[k].get("dataset", _DATASET_OF_FORMAT[day_format(days[k])]))
        for k in train_keys if k in days
    })
    return "+".join(tags) if tags else "unknown"


def feature_regime(cfg: dict) -> str:
    """Which feature regime this run uses (schema.FEATURE_REGIMES). `full`
    unless the config says otherwise, so every existing config is unchanged."""
    return str(cfg.get("features", {}).get("regime", "full"))


def fit_scaler(train_arrays: WindowedArrays, regime: str = "full") -> FeatureScaler:
    """Fit the FeatureScaler on the TRAIN split's raw states only — both the
    history (X) and future targets (Y) come from the same underlying state
    table, so fitting on the union of both is still "training period only";
    it is never fit on val/test/holdout arrays. Statistics are computed on
    active windows (see normalize.py); the sampled training windows contain
    ~98% silent rows and a scaler fit on those was the identity.

    `regime` declares the cross-dataset feature subset (schema.regime_kinds).
    Declaring it here rather than at each model call site means the choice is
    serialized with the scaler and cannot be lost between training and
    serving."""
    all_train_states = np.concatenate([train_arrays.X.reshape(-1, train_arrays.X.shape[-1]),
                                        train_arrays.Y.reshape(-1, train_arrays.Y.shape[-1])], axis=0)
    active = all_train_states[:, FEATURE_INDEX["is_active"]] > 0
    return FeatureScaler(kinds=regime_kinds(regime)).fit(all_train_states, active_mask=active)


def scale_arrays(arrays: WindowedArrays, scaler: FeatureScaler) -> tuple[np.ndarray, np.ndarray]:
    return scaler.transform(arrays.X), scaler.transform(arrays.Y)


def build_windowed_splits(splits: SplitResult, L: int = CONTEXT_LENGTH, K: int = HORIZON_LENGTH) -> dict[str, WindowedArrays]:
    return {
        name: build_windowed_arrays(df, L=L, K=K)
        for name, df in [("train", splits.train), ("val", splits.val), ("test", splits.test), ("holdout", splits.holdout)]
    }
