"""The offline CLI takes a CICFlowMeter-layout CSV (and optionally packets),
builds the state table the training data went through, forecasts every
origin with L windows of history, and writes forecasts/alerts/summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nidra.cli.forecast import build_state_table, enumerate_contexts, run
from nidra.data.schema import FEATURE_ORDER
from tests.fixtures.synth import make_synthetic_flows

_INV = {
    "src_ip": "Source IP", "dst_ip": "Destination IP", "src_port": "Source Port", "dst_port": "Destination Port",
    "protocol": "Protocol", "timestamp": "Timestamp", "flow_duration": "Flow Duration",
    "total_fwd_packets": "Total Fwd Packets", "total_bwd_packets": "Total Backward Packets",
    "total_len_fwd": "Total Length of Fwd Packets", "total_len_bwd": "Total Length of Bwd Packets",
    "syn_flag_count": "SYN Flag Count", "ack_flag_count": "ACK Flag Count", "rst_flag_count": "RST Flag Count",
    "fin_flag_count": "FIN Flag Count", "psh_flag_count": "PSH Flag Count", "urg_flag_count": "URG Flag Count",
    "flow_iat_mean": "Flow IAT Mean", "flow_iat_max": "Flow IAT Max", "label": "Label",
}


def _write_cic_csv(path: Path, flows: pd.DataFrame) -> Path:
    df = flows[[c for c in _INV if c in flows.columns]].copy()
    # CICFlowMeter TrafficLabelling layout: D/M/YYYY H:MM local 12-hour clock.
    # Emit literal timestamps and let the CLI read them literally.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%d/%m/%Y %H:%M:%S")
    df = df.rename(columns=_INV)
    df.to_csv(path, index=False)
    return path


def test_build_state_table_from_csv_is_labelled_and_gap_filled(tmp_path):
    flows = make_synthetic_flows(n_hosts=2, n_windows=40, window_seconds=60, portscan_start_window=25, portscan_len=8)
    csv = _write_cic_csv(tmp_path / "flows.csv", flows)
    from nidra.data.flow_load import load_cicflowmeter_csv
    raw, _ = load_cicflowmeter_csv(csv)
    states, report = build_state_table(raw, None, 60, timestamp_is_epoch=False, cic_clock_correction=False, horizon_k=3)
    assert report["labelled"] and report["flow_only"]
    assert set(FEATURE_ORDER) <= set(states.columns)
    assert (states["stage_label"] == "recon").sum() > 0
    assert states["risk_label"].sum() > 0
    per_host = states.groupby("host_id")["window_ts"].agg(lambda s: np.all(np.diff(np.sort(s)) == 60))
    assert per_host.all()


def test_enumerate_contexts_gives_one_origin_per_eligible_window():
    states = pd.DataFrame({"host_id": ["a"] * 10 + ["b"] * 3, "window_ts": list(range(0, 600, 60)) + [0, 60, 120]})
    for f in FEATURE_ORDER:
        states[f] = 0.0
    X, hosts, ts = enumerate_contexts(states, L=5)
    assert X.shape == (6, 5, len(FEATURE_ORDER))  # host a: origins 4..9; host b too short
    assert set(hosts) == {"a"} and ts.tolist() == [240, 300, 360, 420, 480, 540]


def test_cli_end_to_end_on_csv(trained_predictor, tmp_path):
    predictor, _ = trained_predictor
    flows = make_synthetic_flows(n_hosts=2, n_windows=60, window_seconds=predictor.cfg["windowing"]["window_seconds"],
                                 portscan_start_window=45, portscan_len=8)
    csv = _write_cic_csv(tmp_path / "flows.csv", flows)
    args = argparse.Namespace(
        pcap=None, csv=str(csv), out=str(tmp_path / "out"), config=predictor.cfg.get("_config_path"),
        weights_dir=str(predictor._weights_dir), scaler_dir=str(predictor._scaler_path.parent), seeds=[0],
        samples=4, chunk=16, max_origins=None, max_alerts=3, work_dir=None, no_cic_clock_correction=True,
    )
    summary = run(args)
    out = tmp_path / "out"
    rows = pd.read_csv(out / "forecasts.csv")
    assert len(rows) == summary["n_origins_forecast"] > 0
    assert {"host_id", "origin_ts", "p_within_horizon", "above_threshold", "p_attack_at_k1", "risk_label"} <= set(rows.columns)
    assert summary["input"]["labelled"] and summary["input"]["flow_only"]
    assert "ground_truth" in summary and summary["ground_truth"]["n_episodes"] >= 1
    alerts = json.loads((out / "alerts.json").read_text())
    assert len(alerts) <= 3
    for a in alerts:
        assert "risk_curve" in a and "progression" in a


def test_html_report_renders_from_a_run(trained_predictor, tmp_path):
    from nidra.cli.report_html import render
    predictor, _ = trained_predictor
    flows = make_synthetic_flows(n_hosts=2, n_windows=60, window_seconds=predictor.cfg["windowing"]["window_seconds"],
                                 portscan_start_window=45, portscan_len=8)
    csv = _write_cic_csv(tmp_path / "flows.csv", flows)
    args = argparse.Namespace(
        pcap=None, csv=str(csv), out=str(tmp_path / "out"), config=predictor.cfg.get("_config_path"),
        weights_dir=str(predictor._weights_dir), scaler_dir=str(predictor._scaler_path.parent), seeds=[0],
        samples=4, chunk=16, max_origins=None, max_alerts=2, work_dir=None, no_cic_clock_correction=True,
    )
    run(args)
    page = render(tmp_path / "out")
    assert "<svg" in page and "Projected stage sequence" in page or "Alerts (0" in page
    assert "model-internal" in page
