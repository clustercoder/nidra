"""Static HTML report for an offline forecast run — the demo surface that
needs nothing but a browser.

    python -m nidra.cli.report_html --run out/ [--out out/report.html]

Reads forecasts.csv, alerts.json and summary.json written by
nidra.cli.forecast and renders: the run summary and operating point, a
per-host timeline of the within-horizon forecast (SVG, no JavaScript
libraries), and one card per alert with the multi-horizon risk curve and
its trajectory band, the projected stage sequence with its ATT&CK tactics
(labelled model-internal), and the signed forecast attributions. Colours
never carry the risk level alone; the number is always printed.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

_CSS = """
body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1b1b1b;background:#fff}
h1,h2,h3{font-weight:600} h1{font-size:22px} h2{font-size:17px;margin-top:28px} h3{font-size:15px;margin:0 0 6px}
table{border-collapse:collapse;margin:8px 0} td,th{border:1px solid #ddd;padding:4px 8px;text-align:left;font-size:13px}
.card{border:1px solid #ddd;border-radius:6px;padding:12px 14px;margin:12px 0}
.muted{color:#666} .mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.tag{display:inline-block;border:1px solid #999;border-radius:3px;padding:0 6px;margin-right:6px;font-size:12px}
svg text{font-size:10px;fill:#333}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def risk_curve_svg(curve: dict, width: int = 420, height: int = 150) -> str:
    p = curve["p_attack_at_k"]
    lo, hi = curve["band_low"], curve["band_high"]
    secs = curve["horizon_seconds"]
    thr = curve["threshold"]
    n = len(p)
    ml, mr, mt, mb = 34, 10, 10, 22
    W, H = width - ml - mr, height - mt - mb

    def X(i):
        return ml + (W * i / max(n - 1, 1))

    def Y(v):
        return mt + H * (1.0 - max(0.0, min(1.0, v)))

    band = " ".join(f"{X(i):.1f},{Y(hi[i]):.1f}" for i in range(n)) + " " + " ".join(f"{X(i):.1f},{Y(lo[i]):.1f}" for i in reversed(range(n)))
    line = " ".join(f"{X(i):.1f},{Y(p[i]):.1f}" for i in range(n))
    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="risk curve">']
    for g in (0.0, 0.5, 1.0):
        parts.append(f'<line x1="{ml}" y1="{Y(g):.1f}" x2="{ml + W}" y2="{Y(g):.1f}" stroke="#eee"/>'
                     f'<text x="2" y="{Y(g) + 3:.1f}">{g:.1f}</text>')
    parts.append(f'<line x1="{ml}" y1="{Y(thr):.1f}" x2="{ml + W}" y2="{Y(thr):.1f}" stroke="#b00" stroke-dasharray="4 3"/>'
                 f'<text x="{ml + W - 60}" y="{Y(thr) - 3:.1f}">thr {thr:.2f}</text>')
    parts.append(f'<polygon points="{band}" fill="#9ec5fe" fill-opacity="0.35" stroke="none"/>')
    parts.append(f'<polyline points="{line}" fill="none" stroke="#1f5fbf" stroke-width="2"/>')
    for i in range(n):
        parts.append(f'<circle cx="{X(i):.1f}" cy="{Y(p[i]):.1f}" r="3" fill="#1f5fbf"/>'
                     f'<text x="{X(i) - 10:.1f}" y="{height - 6}">+{secs[i] // 60}m</text>')
    parts.append("</svg>")
    return "".join(parts)


def host_timeline_svg(rows: pd.DataFrame, threshold: float, width: int = 900, height: int = 90) -> str:
    rows = rows.sort_values("origin_ts")
    t = rows["origin_ts"].to_numpy()
    p = rows["p_within_horizon"].to_numpy()
    if len(t) == 0:
        return ""
    ml, mr, mt, mb = 34, 10, 8, 18
    W, H = width - ml - mr, height - mt - mb
    t0, t1 = float(t.min()), float(max(t.max(), t.min() + 1))

    def X(v):
        return ml + W * (v - t0) / (t1 - t0)

    def Y(v):
        return mt + H * (1.0 - max(0.0, min(1.0, float(v))))

    line = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t, p))
    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<line x1="{ml}" y1="{Y(threshold):.1f}" x2="{ml + W}" y2="{Y(threshold):.1f}" stroke="#b00" stroke-dasharray="4 3"/>',
             f'<polyline points="{line}" fill="none" stroke="#1f5fbf" stroke-width="1.5"/>',
             f'<text x="2" y="{Y(1) + 3:.1f}">1.0</text><text x="2" y="{Y(0) + 3:.1f}">0.0</text>']
    if "risk_label" in rows.columns:
        for a, y in zip(t, rows["risk_label"].to_numpy()):
            if y == 1:
                parts.append(f'<rect x="{X(a) - 1:.1f}" y="{mt}" width="2" height="{H}" fill="#f0a" fill-opacity="0.25"/>')
    parts.append(f'<text x="{ml}" y="{height - 4}">{pd.to_datetime(t0, unit="s", utc=True):%Y-%m-%d %H:%M}</text>'
                 f'<text x="{ml + W - 110}" y="{height - 4}">{pd.to_datetime(t1, unit="s", utc=True):%Y-%m-%d %H:%M} UTC</text></svg>')
    return "".join(parts)


def render(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text())
    alerts = json.loads((run_dir / "alerts.json").read_text())
    rows = pd.read_csv(run_dir / "forecasts.csv")
    thr = float(summary["operating_point"]["threshold"])
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>NIDRA offline forecast</title><style>{_CSS}</style></head><body>",
           "<h1>NIDRA offline forecast report</h1>",
           "<p class='muted'>Every number is a forecast for the minutes after each origin, from the learned dynamics rolled forward and "
           "scored by frozen heads. The projected stage sequence and its ATT&amp;CK tactics are model-internal objects, not attribution.</p>"]
    g = summary["geometry"]
    op = summary["operating_point"]
    out.append("<h2>Run</h2><table>")
    for k, v in [("input", summary["input"].get("pcap") or summary["input"].get("csv")),
                 ("flow-only (packet features zero)", summary["input"].get("flow_only")),
                 ("labelled ground truth present", summary["input"].get("labelled")),
                 ("window / history / horizon", f"{g['window_seconds']} s / {g['L']} windows / {g['K']} windows"),
                 ("hosts / state rows / origins forecast", f"{summary['n_hosts']} / {summary['n_state_rows']} / {summary['n_origins_forecast']}"),
                 ("operating point", f"{op.get('pooling_key')} — threshold {op['threshold']:.3f} ({op.get('source')})"),
                 ("above threshold", f"{summary['n_above_threshold']} origins; {summary.get('alerts_per_hour_of_capture')} per capture hour"),
                 ("trajectories per origin", summary["n_trajectories_per_origin"]), ("model version", summary["model_version"])]:
        out.append(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>")
    out.append("</table>")
    if "ground_truth" in summary:
        gt = summary["ground_truth"]
        out.append(f"<h2>Ground truth (labels in the input)</h2><p>{gt['n_episodes']} episodes; warned before onset: {gt['n_warned_before_onset']}; "
                   f"alerted only after onset: {gt['n_alerted_within_episode']}.</p><table><tr><th>host</th><th>onset (UTC)</th><th>windows</th><th>first alert</th><th>lead time</th></tr>")
        for e in gt["episodes"]:
            onset = pd.to_datetime(e["onset_ts"], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
            fa = pd.to_datetime(e["first_alert_ts"], unit="s", utc=True).strftime("%H:%M") if e.get("first_alert_ts") else "—"
            lead = f"{e['lead_time_s'] // 60:+d} min" if "lead_time_s" in e else "—"
            out.append(f"<tr><td>{_esc(e['host_id'])}</td><td>{onset}</td><td>{e['n_windows']}</td><td>{fa}</td><td>{lead}</td></tr>")
        out.append("</table>")
        if "published_label_metrics" in summary:
            m = summary["published_label_metrics"]
            out.append(f"<p class='muted'>Published-label ranking on this capture: AUC-PR {m['auc_pr']:.3f}, ROC-AUC {m['roc_auc']:.3f} "
                       f"({m['n_pos']} positives of {m['n']} origins — one capture, no confidence interval).</p>")
    out.append("<h2>Per-host forecast timelines</h2>")
    top_hosts = rows.groupby("host_id")["p_within_horizon"].max().sort_values(ascending=False).head(12).index
    for h in top_hosts:
        sub = rows[rows["host_id"] == h]
        out.append(f"<div class='card'><h3>{_esc(h)} <span class='muted'>max {sub['p_within_horizon'].max():.3f}, "
                   f"{int(sub['above_threshold'].sum())} above threshold of {len(sub)}</span></h3>{host_timeline_svg(sub, thr)}</div>")
    out.append(f"<h2>Alerts ({len(alerts)} shown, highest forecast first)</h2>")
    for a in alerts:
        c = a["risk_curve"]
        out.append(f"<div class='card'><h3>{_esc(a['host_id'])} at {_esc(a['origin_ts'])}</h3>"
                   f"<p>P(attack within horizon) = <b>{c['p_attack_within_horizon']:.3f}</b> (threshold {c['threshold']:.3f}; "
                   f"{'calibrated' if c['calibrated'] else 'raw'}); observed risk now {a['observed_risk']:.3f}; observed stage {_esc(a['observed_stage'])}; "
                   f"lead time {a['lead_time_s'] if a['lead_time_s'] is not None else '—'} s</p>{risk_curve_svg(c)}")
        steps = a["progression"]["steps"]
        out.append("<p><b>Projected stage sequence</b> <span class='muted'>(" + _esc(a["progression"]["label"]) + ")</span>: " +
                   " → ".join(f"+{s['t_plus_s'] // 60}m {_esc(s['stage'])} ({s['probability']:.2f})" for s in steps) + "</p>")
        tactics = a["progression"]["tactics_in_order"]
        if tactics:
            out.append("<p>" + " ".join(f"<span class='tag'>{_esc(t['tactic']['id'])} {_esc(t['tactic']['name'])} from +{t['first_t_plus_s'] // 60}m</span>" for t in tactics) + "</p>")
        sig = a.get("top_signals", [])
        if sig:
            out.append("<p><b>Top signals (current state, SHAP)</b>: " + "; ".join(_esc(s.get("text") or s.get("name") or s) for s in sig[:5]) + "</p>")
        out.append("</div>")
    out.append("</body></html>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="directory written by nidra.cli.forecast")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "report.html"
    out.write_text(render(run_dir))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
