"""Data audit for the canonical windowed tables: the checks the first-
principles reevaluation had to run by hand, made repeatable.

    python -m nidra.data.audit --config config/default.yaml --out artifacts/metadata/data_audit.json

For every split it reports window alignment (no sub-window offsets — the
Δ=30 defect), activity rate, positive counts under the published label and
under the detection / pre-onset decompositions, attack episodes (merged
across short gaps) with their lengths and hosts, and the label composition
"inside a running episode" vs "before onset". These numbers are what the
model card quotes; they are measured here, not copied.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from nidra.data.schema import FEATURE_ORDER
from nidra.data.splits import SplitResult, find_episodes

logger = logging.getLogger(__name__)


def merge_episodes(df: pd.DataFrame, window_seconds: int, merge_gap_windows: int) -> pd.DataFrame:
    """Attack episodes per host with runs separated by at most
    `merge_gap_windows` benign windows merged into one. `find_episodes`
    splits on every benign window; a brute-force run that pauses for one
    minute is one episode for lead-time / per-episode purposes, not two."""
    raw = find_episodes(df)
    if raw.empty:
        return raw.assign(n_windows=pd.Series(dtype="int64"))
    merged: list[tuple] = []
    for host, g in raw.sort_values(["host_id", "start_ts"]).groupby("host_id", sort=False):
        start, end = None, None
        for s, e in zip(g["start_ts"], g["end_ts"]):
            if start is None:
                start, end = s, e
            elif (s - end) // window_seconds - 1 <= merge_gap_windows:
                end = max(end, e)
            else:
                merged.append((host, start, end))
                start, end = s, e
        merged.append((host, start, end))
    out = pd.DataFrame(merged, columns=["host_id", "start_ts", "end_ts"])
    out["n_windows"] = (out["end_ts"] - out["start_ts"]) // window_seconds + 1
    return out


def label_composition(df: pd.DataFrame, episodes: pd.DataFrame, window_seconds: int, horizon_k: int) -> dict[str, Any]:
    """Where the risk_label=1 windows sit relative to merged episodes."""
    if df.empty or episodes.empty:
        return {"n_positive": int(df.get("risk_label", pd.Series(dtype=int)).sum()) if not df.empty else 0}
    df = df.sort_values(["host_id", "window_ts"])
    ts = df["window_ts"].to_numpy()
    hosts = df["host_id"].to_numpy()
    inside = np.zeros(len(df), dtype=bool)
    dist_to_onset = np.full(len(df), np.inf)
    for host, start, end in zip(episodes["host_id"], episodes["start_ts"], episodes["end_ts"]):
        m = hosts == host
        inside |= m & (ts >= start) & (ts <= end)
        d = (start - ts) / window_seconds
        before = m & (d > 0)
        dist_to_onset[before] = np.minimum(dist_to_onset[before], d[before])
    pos = df["risk_label"].to_numpy() == 1
    n_pos = int(pos.sum())
    pre = pos & ~inside
    return {
        "n_positive": n_pos,
        "inside_running_episode": int((pos & inside).sum()),
        "inside_running_episode_fraction": float((pos & inside).sum() / n_pos) if n_pos else None,
        "pre_onset_within_K": int(pre.sum()),
        "n_true_onsets": int(len(episodes)),
        "pre_onset_windows_available": {
            f"<={h}min": int((~inside & (dist_to_onset <= h * 60 / window_seconds)).sum())
            for h in (1, 3, 5, 10, 15, 30)
        },
    }


def audit_split(name: str, df: pd.DataFrame, window_seconds: int, horizon_k: int, merge_gap_windows: int) -> dict[str, Any]:
    if df.empty:
        return {"name": name, "rows": 0}
    ts = df["window_ts"].to_numpy()
    active = df["is_active"].to_numpy() > 0
    attack = (df["stage_label"] != "benign").to_numpy()
    episodes = merge_episodes(df, window_seconds, merge_gap_windows)
    feat = df[FEATURE_ORDER].to_numpy(dtype="float64")
    zero_rows = (np.abs(feat).sum(axis=1) == 0)
    out: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "hosts": int(df["host_id"].nunique()),
        "window_seconds": window_seconds,
        "window_ts_all_aligned": bool(((ts % window_seconds) == 0).all()),
        "sub_window_offsets_present": sorted({int(x) for x in np.unique(ts % window_seconds)}),
        "time_span_hours": float((ts.max() - ts.min()) / 3600.0),
        "active_rate": float(active.mean()),
        "all_zero_row_rate": float(zero_rows.mean()),
        "attack_windows": int(attack.sum()),
        "attack_windows_active": int((attack & active).sum()),
        "attacked_hosts": sorted(df.loc[attack, "host_id"].unique().tolist()),
        "stage_counts": {k: int(v) for k, v in df["stage_label"].value_counts().items()},
        "risk_positive": int(df["risk_label"].sum()),
        "natural_prevalence": float(df["risk_label"].mean()),
        "episodes": {
            "merge_gap_windows": merge_gap_windows,
            "n": int(len(episodes)),
            "per_host": {h: int(n) for h, n in episodes.groupby("host_id").size().items()} if not episodes.empty else {},
            "lengths_windows": sorted(episodes["n_windows"].astype(int).tolist()) if not episodes.empty else [],
            "raw_contiguous_runs": int(len(find_episodes(df))),
        },
        "label_composition": label_composition(df, episodes, window_seconds, horizon_k),
        "packet_features_populated_rate_on_active": float(
            (df.loc[active, ["ttl_mean", "tcp_window_mean", "payload_size_mean"]].abs().sum(axis=1) > 0).mean()
        ) if active.any() else None,
        "flow_features_populated_rate_on_active": float(
            (df.loc[active, ["active_flow_count"]].to_numpy() > 0).mean()
        ) if active.any() else None,
    }
    return out


def audit_splits(splits: SplitResult, window_seconds: int, horizon_k: int, merge_gap_windows: int = 5) -> dict[str, Any]:
    report = {
        name: audit_split(name, getattr(splits, name), window_seconds, horizon_k, merge_gap_windows)
        for name in ("train", "val", "test", "holdout")
    }
    # cross-split host overlap (informational — the same enterprise network is captured every day)
    hosts = {n: set(getattr(splits, n)["host_id"]) if not getattr(splits, n).empty else set() for n in report}
    report["host_overlap"] = {
        "train_test": len(hosts["train"] & hosts["test"]),
        "train_holdout": len(hosts["train"] & hosts["holdout"]),
        "train_val": len(hosts["train"] & hosts["val"]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--merge-gap-windows", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.utils.config import load_config
    from nidra.utils.provenance import experiment_record, write_json

    cfg = load_config(args.config)
    window_seconds, L, K = geometry_from_config(cfg)
    gap = args.merge_gap_windows if args.merge_gap_windows is not None else int(cfg.get("eval", {}).get("episode_merge_gap_windows", 5))
    splits = build_all_splits(cfg)
    report = audit_splits(splits, window_seconds, K, merge_gap_windows=gap)
    record = experiment_record(cfg, stage="data_audit", metrics=report, hash_processed=True)
    write_json(args.out, record)
    for name in ("train", "val", "test", "holdout"):
        r = report[name]
        if r.get("rows"):
            print(f"{name:8s} rows={r['rows']:8d} hosts={r['hosts']:5d} active={r['active_rate']:.3f} "
                  f"aligned={r['window_ts_all_aligned']} attack_win={r['attack_windows']:5d} pos={r['risk_positive']:5d} "
                  f"prev={r['natural_prevalence']:.5f} episodes={r['episodes']['n']} lens={r['episodes']['lengths_windows']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
