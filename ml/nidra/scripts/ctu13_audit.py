"""CTU-13 dataset audit: inventory, label semantics, windowed geometry.

    python -m nidra.scripts.ctu13_audit --config config/ctu13.yaml \
        --out artifacts/metrics/ctu13_audit.json

Runs the same per-scenario pipeline training will run (so the numbers it
reports are the numbers a model would see, not an approximation of them) and
writes one machine-readable record per scenario:

  * the raw file, its size and SHA-256, and the loader's own drop report;
  * flow counts by CTU-13 label class and the full label vocabulary;
  * temporal coverage and whether the clock supports the Delta=60 geometry;
  * the monitored-host population, and how much the CIDR filter removed;
  * the WINDOWED state table: rows, active rows, attack rows, prevalence;
  * episodes: count, hosts, duration distribution, and the pre-onset
    inventory at each lead time — the quantity that decides whether advance
    warning is measurable on this dataset at all.

The windowed tables are cached under artifacts/processed/ on the way past, so
the audit doubles as the data-build step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nidra.data.ctu_load import CTU_INTERNAL_CIDRS, load_binetflow
from nidra.data.labels import map_label_to_stage
from nidra.data.onset import episode_geometry, infer_window_seconds
from nidra.data.schema import FEATURE_ORDER
from nidra.data.splits import find_episodes
from nidra.utils.config import load_config, resolve_path
from nidra.train.pipeline import (
    day_cache_path,
    day_format,
    geometry_from_config,
    host_scope_tag,
    load_and_label_day,
)

logger = logging.getLogger(__name__)

ONSET_HORIZONS_MIN = (1, 3, 5, 10, 15, 30, 60)


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _label_inventory(labels: pd.Series, dialect: str = "ctu") -> dict:
    counts = labels.value_counts()
    by_class = {"botnet": 0, "normal": 0, "background": 0}
    for value, n in counts.items():
        text = str(value).lower()
        if "botnet" in text:
            by_class["botnet"] += int(n)
        elif "normal" in text:
            by_class["normal"] += int(n)
        else:
            by_class["background"] += int(n)
    stages: dict[str, int] = {}
    for value, n in counts.items():
        stages[map_label_to_stage(str(value), dialect=dialect)] = stages.get(map_label_to_stage(str(value), dialect=dialect), 0) + int(n)
    return {
        "n_distinct_labels": int(len(counts)),
        "flows_by_class": by_class,
        "flows_by_stage": stages,
        "top_labels": {str(k): int(v) for k, v in counts.head(15).items()},
    }


def _timestamp_resolution(ts: pd.Series) -> dict:
    """Does the clock actually resolve below the window length, or is it
    printed at minute granularity like CIC-IDS2017's CSVs?"""
    frac = np.asarray(ts, dtype="float64") % 60.0
    return {
        "distinct_sub_minute_offsets": int(pd.Series(np.round(frac, 3)).nunique()),
        "sub_second_precision": bool(np.any(np.abs(frac - np.round(frac)) > 1e-6)),
    }


def _pre_onset_inventory(table: pd.DataFrame, window_seconds: int) -> dict:
    """How many rows sit h minutes before an episode ONSET on the same host,
    and how many independent episodes contribute at each h.

    This is the quantity that decides whether advance warning is measurable.
    An episode whose host has no observed window before its onset offers
    nothing to forecast from, however long the episode is.
    """
    if table.empty:
        return {"n_episodes": 0, "by_horizon": {}}
    hosts = table["host_id"].to_numpy().astype(object)
    ts = table["window_ts"].to_numpy(dtype="int64")
    inside, to_onset, _ = episode_geometry(table, hosts, ts, window_seconds, merge_gap_windows=5)
    eligible = ~inside & np.isfinite(to_onset)
    by_horizon = {}
    for h in ONSET_HORIZONS_MIN:
        sel = eligible & (to_onset > 0) & (to_onset <= h)
        # Independent episodes = distinct (host, next onset time) pairs.
        if sel.any():
            onset_ts = ts[sel] + (to_onset[sel] * 60.0).astype("int64")
            keys = {(hosts[i], int(o)) for i, o in zip(np.flatnonzero(sel), onset_ts)}
        else:
            keys = set()
        by_horizon[str(h)] = {"n_rows": int(sel.sum()), "n_episodes": len(keys)}
    return {"by_horizon": by_horizon}


def audit_scenario(day_key: str, day_meta: dict, cfg: dict) -> dict:
    dataset_cfg = cfg["dataset"]
    window_seconds, L, K = geometry_from_config(cfg)
    ctu_dir = Path(dataset_cfg["ctu13_dir"]).expanduser()
    path = ctu_dir / day_meta["file"]
    internal = tuple(dataset_cfg.get("ctu13_internal_cidrs", CTU_INTERNAL_CIDRS) or ())

    t0 = time.time()
    record: dict = {
        "day_key": day_key,
        "scenario": day_meta.get("scenario"),
        "family": day_meta.get("family"),
        "behaviour": day_meta.get("behaviour"),
        "role": day_meta.get("role"),
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "file_sha256": file_digest(path),
    }

    # --- raw flows, both with and without the monitored-network filter ---
    flows_all, report_all = load_binetflow(path, internal_cidrs=None)
    record["raw"] = {
        "input_rows": report_all.input_rows,
        "accepted_rows": report_all.accepted_rows,
        "drop_reasons": report_all.drop_reasons,
        "distinct_source_hosts": int(flows_all["src_ip"].nunique()),
        "distinct_destination_hosts": int(flows_all["dst_ip"].nunique()),
        "start_utc": pd.to_datetime(flows_all["timestamp"].min(), unit="s").isoformat(),
        "end_utc": pd.to_datetime(flows_all["timestamp"].max(), unit="s").isoformat(),
        "duration_hours": round(float(flows_all["timestamp"].max() - flows_all["timestamp"].min()) / 3600, 3),
        "protocols": {str(k): int(v) for k, v in flows_all["protocol"].value_counts().head(10).items()},
        "timestamp_resolution": _timestamp_resolution(flows_all["timestamp"]),
    }
    record["labels"] = _label_inventory(flows_all["label"])
    botnet = flows_all[flows_all["label"].str.contains("Botnet", case=False, na=False)]
    record["botnet_source_hosts"] = {str(k): int(v) for k, v in botnet["src_ip"].value_counts().items()}

    internal_mask = flows_all["src_ip"].str.startswith("147.32.") if internal else pd.Series(True, index=flows_all.index)
    record["host_filter"] = {
        "cidrs": list(internal),
        "flows_kept": int(internal_mask.sum()),
        "flows_dropped": int((~internal_mask).sum()),
        "hosts_kept": int(flows_all.loc[internal_mask, "src_ip"].nunique()),
        "botnet_flows_dropped": int((~internal_mask & flows_all["label"].str.contains("Botnet", case=False, na=False)).sum()),
    }
    del flows_all, botnet

    # --- windowed state table (cached; this is also the data build) ---
    processed_dir = resolve_path(cfg, cfg["artifacts"]["processed_dir"])
    cache = day_cache_path(processed_dir, day_key, cfg["windowing"], None, "flowonly",
                           flow_format=day_format(day_meta), host_scope=host_scope_tag(internal))
    table = load_and_label_day(path, window_seconds=window_seconds, min_windows_per_host=1,
                               cache_path=cache, horizon_k=K, flow_format=day_format(day_meta),
                               internal_cidrs=internal or None)

    attack = table["stage_label"] != "benign"
    episodes = find_episodes(table)
    ep_len = ((episodes["end_ts"] - episodes["start_ts"]) // window_seconds + 1) if not episodes.empty else pd.Series(dtype=float)
    per_host_windows = table.groupby("host_id")["window_ts"].count()
    record["windowed"] = {
        "cache_path": str(cache),
        "window_seconds": window_seconds,
        "n_rows": int(len(table)),
        "n_hosts": int(table["host_id"].nunique()),
        "n_active_rows": int((table["is_active"] > 0).sum()),
        "active_fraction": round(float((table["is_active"] > 0).mean()), 5),
        "n_attack_rows": int(attack.sum()),
        "attack_prevalence": round(float(attack.mean()), 6),
        "n_risk_positive_rows": int(table["risk_label"].sum()),
        "risk_prevalence": round(float(table["risk_label"].mean()), 6),
        "stage_counts": {str(k): int(v) for k, v in table["stage_label"].value_counts().items()},
        "n_hosts_with_at_least_L_plus_K_windows": int((per_host_windows >= L + K).sum()),
        "n_rows_on_those_hosts": int(per_host_windows[per_host_windows >= L + K].sum()),
    }
    record["episodes"] = {
        "n_episodes": int(len(episodes)),
        "n_hosts_with_episodes": int(episodes["host_id"].nunique()) if not episodes.empty else 0,
        "length_windows": {
            "min": int(ep_len.min()) if len(ep_len) else 0,
            "median": float(ep_len.median()) if len(ep_len) else 0.0,
            "max": int(ep_len.max()) if len(ep_len) else 0,
            "total": int(ep_len.sum()) if len(ep_len) else 0,
        },
    }
    record["pre_onset"] = _pre_onset_inventory(table, window_seconds)
    record["feature_coverage"] = {
        name: round(float((table[name] != 0).mean()), 5) for name in FEATURE_ORDER
    }
    record["seconds"] = round(time.time() - t0, 1)
    logger.info("audited %s in %.0fs: %d rows, %d hosts, %d episodes",
                day_key, record["seconds"], record["windowed"]["n_rows"],
                record["windowed"]["n_hosts"], record["episodes"]["n_episodes"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/ctu13.yaml")
    parser.add_argument("--out", default="artifacts/metrics/ctu13_audit.json")
    parser.add_argument("--only", default=None, help="comma-separated day keys")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    days = cfg["dataset"]["days"]
    keys = args.only.split(",") if args.only else list(days)
    keys = sorted(keys, key=lambda k: days[k].get("scenario", 0))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {"scenarios": {}}

    for key in keys:
        existing["scenarios"][key] = audit_scenario(key, days[key], cfg)
        existing["config"] = {
            "config_path": cfg["_config_path"], "config_hash": cfg["_config_hash"],
            "git_commit": cfg["_git_commit"], "internal_cidrs": cfg["dataset"].get("ctu13_internal_cidrs"),
            "feature_regime": cfg.get("features", {}).get("regime", "full"),
        }
        out_path.write_text(json.dumps(existing, indent=2))
        logger.info("wrote %s (%d scenarios)", out_path, len(existing["scenarios"]))


if __name__ == "__main__":
    main()
