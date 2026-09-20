"""Offline forecasting from a capture file — no services, no network.

    python -m nidra.cli.forecast --pcap capture.pcap --out out/
    python -m nidra.cli.forecast --csv flows.csv [--pcap capture.pcap] --out out/

Pipeline (the same modules the training data went through):

    pcap  --tshark-->  packets parquet  --flow_assemble-->  flow records
    csv   --flow_load-->  CICFlowMeter flow records
    flows + packets  --join/windowize-->  per-host state table at Δ (gap-filled,
                                          silent minutes are real states)
    state table  --NidraPredictor.forecast_batch-->  risk curve per (host, minute)

Outputs, under --out:
    forecasts.csv     one row per (host, origin minute): p_within_horizon,
                      p_attack_at_k for every k, band, above_threshold
    alerts.json       every above-threshold origin with its full forecast()
                      output (risk curve, projected stages, ATT&CK candidates,
                      top signals) — capped by --max-alerts
    summary.json      counts, thresholds, provenance, and — when the CSV
                      carried a Label column — per-episode lead time against
                      the ground truth, computed exactly as the benchmark does

What this is NOT: it is not a detector over the capture. Every number is
the model's forecast for the minutes AFTER the origin, scored with the
operating point frozen on validation. A CSV without packets runs in
flow-only mode (11 packet features zero) and the summary says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nidra.data.flow_assemble import assemble_flows
from nidra.data.flow_load import load_cicflowmeter_csv
from nidra.data.join import build_day_inputs
from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import FEATURE_ORDER, STAGE_LABELS
from nidra.data.windowize import build_state_rows
from nidra.serve.predictor import NidraPredictor
from nidra.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def extract_packets(pcap: Path, work_dir: Path) -> pd.DataFrame:
    from nidra.data.pcap_extract import extract_pcap_to_parquet
    out = work_dir / (pcap.stem + ".packets.parquet")
    if not out.exists():
        n = extract_pcap_to_parquet(pcap, out)
        logger.info("extracted %d packets from %s", n, pcap)
    return pd.read_parquet(out)


def build_state_table(flows_raw: pd.DataFrame | None, packets_raw: pd.DataFrame | None, window_seconds: int,
                      timestamp_is_epoch: bool, cic_clock_correction: bool, horizon_k: int) -> tuple[pd.DataFrame, dict]:
    """State rows for the capture; labelled when the flows carry a label."""
    if flows_raw is None:
        if packets_raw is None or packets_raw.empty:
            raise ValueError("no flows and no packets — nothing to forecast")
        flows_raw = assemble_flows(packets_raw)
        timestamp_is_epoch = True
        logger.info("assembled %d flows from packets", len(flows_raw))
    flows_w, packets_w = build_day_inputs(flows_raw, packets_raw, window_seconds,
                                          timestamp_is_epoch=timestamp_is_epoch, cic_clock_correction=cic_clock_correction)
    states = build_state_rows(flows_w, packets_w, window_seconds=window_seconds)
    report: dict[str, Any] = {"n_flows": int(len(flows_w)), "n_packets": int(len(packets_w)) if packets_w is not None else 0,
                              "flow_only": packets_w is None or packets_w.empty, "labelled": False}
    if "label" in flows_w.columns:
        stage_table, label_report = label_stage_table(flows_w)
        states = attach_risk_label(states, stage_table, horizon_k=horizon_k, window_seconds=window_seconds)
        report["labelled"] = True
        report["unmapped_labels"] = label_report.get("unmapped_labels", [])
    else:
        states["stage_label"] = "benign"
        states["risk_label"] = 0
    return states.sort_values(["host_id", "window_ts"]).reset_index(drop=True), report


def enumerate_contexts(states: pd.DataFrame, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every origin with L windows of history: X [N, L, F], host [N], origin_ts [N]."""
    feats = states[FEATURE_ORDER].to_numpy(dtype="float32")
    hosts_col = states["host_id"].to_numpy()
    ts_col = states["window_ts"].to_numpy(dtype="int64")
    X, hosts, ts = [], [], []
    start = 0
    for host, g in states.groupby("host_id", sort=False):
        n = len(g)
        if n >= L:
            idx = np.arange(L - 1, n)
            win = np.lib.stride_tricks.sliding_window_view(feats[start:start + n], (L, feats.shape[1]))[:, 0]
            X.append(win)
            hosts.append(np.full(len(idx), host, dtype=object))
            ts.append(ts_col[start + idx])
        start += n
    if not X:
        return np.zeros((0, L, len(FEATURE_ORDER)), dtype="float32"), np.array([], dtype=object), np.array([], dtype="int64")
    return np.concatenate(X), np.concatenate(hosts), np.concatenate(ts)


def lead_time_report(states: pd.DataFrame, hosts: np.ndarray, ts: np.ndarray, above: np.ndarray,
                     window_seconds: int, m: int = 2) -> dict[str, Any]:
    """Per labelled episode: first alert relative to onset (benchmark
    definition: `m` consecutive above-threshold origins)."""
    from nidra.data.audit import merge_episodes
    episodes = merge_episodes(states, window_seconds, merge_gap_windows=5)
    out = []
    for _, ep in episodes.iterrows():
        h = ep["host_id"]
        sel = hosts == h
        t = ts[sel]
        a = above[sel]
        order = np.argsort(t)
        t, a = t[order], a[order]
        run, first_alert = 0, None
        for i in range(len(t)):
            run = run + 1 if a[i] else 0
            if run >= m and t[i] <= ep["end_ts"]:
                first_alert = int(t[i - m + 1])
                break
        entry = {"host_id": h, "onset_ts": int(ep["start_ts"]), "end_ts": int(ep["end_ts"]),
                 "n_windows": int(ep["n_windows"]), "first_alert_ts": first_alert}
        if first_alert is not None:
            entry["lead_time_s"] = int(ep["start_ts"] - first_alert)
            entry["warned_before_onset"] = bool(first_alert < ep["start_ts"])
        out.append(entry)
    return {"n_episodes": len(out), "n_warned_before_onset": int(sum(1 for e in out if e.get("warned_before_onset"))),
            "n_alerted_within_episode": int(sum(1 for e in out if e.get("first_alert_ts") is not None and not e.get("warned_before_onset"))),
            "episodes": out}


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    cfg = load_config(args.config)
    weights_dir = Path(args.weights_dir) if args.weights_dir else resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    scaler_dir = Path(args.scaler_dir) if args.scaler_dir else resolve_path(cfg, cfg["artifacts"]["scaler_dir"])
    predictor = NidraPredictor(weights_dir=weights_dir, scaler_path=scaler_dir / "feature_scaler.json",
                               config_path=args.config, seeds=args.seeds)
    window_seconds = int(cfg["windowing"]["window_seconds"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    packets_raw = extract_packets(Path(args.pcap), work_dir) if args.pcap else None
    flows_raw = None
    if args.csv:
        flows_raw, load_report = load_cicflowmeter_csv(args.csv)
        logger.info("loaded %d flow rows from %s", len(flows_raw), args.csv)
    states, report = build_state_table(flows_raw, packets_raw, window_seconds, timestamp_is_epoch=False,
                                       cic_clock_correction=not args.no_cic_clock_correction, horizon_k=predictor.K)
    logger.info("state table: %d rows, %d hosts (%.0fs)", len(states), states["host_id"].nunique(), time.time() - t0)

    X, hosts, ts = enumerate_contexts(states, predictor.L)
    if len(X) == 0:
        raise SystemExit(f"no host has {predictor.L} windows of history at Δ={window_seconds}s — nothing to forecast")
    if args.max_origins and len(X) > args.max_origins:
        keep = np.sort(np.random.default_rng(0).choice(len(X), size=args.max_origins, replace=False))
        X, hosts, ts = X[keep], hosts[keep], ts[keep]
    logger.info("forecasting %d origins across %d hosts", len(X), len(set(hosts.tolist())))
    fb = predictor.forecast_batch(X, chunk=args.chunk, n_samples_per_member=args.samples)

    rows = pd.DataFrame({"host_id": hosts, "origin_ts": ts,
                         "origin_iso": [datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat() for t in ts],
                         "p_within_horizon": fb["p_within_horizon"], "above_threshold": fb["above_threshold"]})
    for k in range(predictor.K):
        rows[f"p_attack_at_k{k + 1}"] = fb["p_attack_at_k"][:, k]
        rows[f"band_low_k{k + 1}"] = fb["band_low"][:, k]
        rows[f"band_high_k{k + 1}"] = fb["band_high"][:, k]
        rows[f"stage_k{k + 1}"] = [STAGE_LABELS[i] for i in fb["stage_mean_k"][:, k, :].argmax(axis=1)]
    if report["labelled"]:
        lab = states.set_index(["host_id", "window_ts"])
        rows["risk_label"] = lab.loc[list(zip(hosts, ts)), "risk_label"].to_numpy()
        rows["stage_label_at_origin"] = lab.loc[list(zip(hosts, ts)), "stage_label"].to_numpy()
    rows.to_csv(out_dir / "forecasts.csv", index=False)

    alerts = []
    flagged = np.where(fb["above_threshold"])[0]
    order = flagged[np.argsort(-fb["p_within_horizon"][flagged])][: args.max_alerts]
    for i in order:
        origin = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc)
        alerts.append(predictor.forecast(X[i], host_id=str(hosts[i]), origin_ts=origin))
    (out_dir / "alerts.json").write_text(json.dumps(alerts, indent=2, default=float))

    span_h = float((ts.max() - ts.min()) / 3600.0) if len(ts) > 1 else 0.0
    summary: dict[str, Any] = {
        "input": {"pcap": args.pcap, "csv": args.csv, **report},
        "geometry": {"window_seconds": window_seconds, "L": predictor.L, "K": predictor.K},
        "operating_point": predictor._operating_point_summary(),
        "n_hosts": int(states["host_id"].nunique()), "n_state_rows": int(len(states)),
        "n_origins_forecast": int(len(X)), "n_above_threshold": int(fb["above_threshold"].sum()),
        "alerts_per_hour_of_capture": float(fb["above_threshold"].sum() / span_h) if span_h > 0 else None,
        "n_trajectories_per_origin": fb["n_trajectories"], "calibrated": fb["calibrated"],
        "model_version": predictor.model_version, "wall_seconds": time.time() - t0,
    }
    if report["labelled"]:
        summary["ground_truth"] = lead_time_report(states, hosts, ts, fb["above_threshold"], window_seconds)
        y = rows["risk_label"].to_numpy()
        if y.sum() > 0 and (y == 0).sum() > 0:
            from sklearn.metrics import average_precision_score, roc_auc_score
            summary["published_label_metrics"] = {"auc_pr": float(average_precision_score(y, rows["p_within_horizon"])),
                                                  "roc_auc": float(roc_auc_score(y, rows["p_within_horizon"])),
                                                  "n_pos": int(y.sum()), "n": int(len(y))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pcap", default=None, help="raw capture (needs tshark on PATH)")
    parser.add_argument("--csv", default=None, help="CICFlowMeter CSV (TrafficLabelling layout)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--scaler-dir", default=None)
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--samples", type=int, default=None, help="rollout samples per ensemble member")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--max-origins", type=int, default=None, help="random cap on origins scored (smoke runs)")
    parser.add_argument("--max-alerts", type=int, default=25, help="how many above-threshold origins get a full explained forecast")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--no-cic-clock-correction", action="store_true",
                        help="read CSV timestamps literally (not a CIC-IDS2017 TrafficLabelling file)")
    args = parser.parse_args()
    if not args.pcap and not args.csv:
        parser.error("give --pcap and/or --csv")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("shap").setLevel(logging.WARNING)  # the shap package logs every kernel weight vector at INFO
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "ground_truth"}, indent=2, default=float))
    if "ground_truth" in summary:
        gt = summary["ground_truth"]
        print(f"ground truth: {gt['n_episodes']} episodes, {gt['n_warned_before_onset']} warned before onset, "
              f"{gt['n_alerted_within_episode']} alerted only after onset")


if __name__ == "__main__":
    main()
