"""One-command demo: load the shipped ensemble and forecast a real window.

This is the entry point for anyone who has cloned the repo and does NOT have
the ~50GB raw CIC-IDS2017 release. It reads a committed windowed table from
artifacts/processed/, picks a genuine [L, F] context window, and runs the
same NidraPredictor the serving plane uses — no dataset download, no
retraining, no network access.

    python -m nidra.scripts.demo_forecast

Everything it prints comes from the committed checkpoints and the committed
data; nothing here is synthetic or hard-coded.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER
from nidra.serve.predictor import NidraPredictor
from nidra.train.pipeline import day_cache_path, declared_packets_tag
from nidra.utils.config import load_config, resolve_path
from nidra.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Default demo day: a test-split day that contains a real labelled attack
# episode, so `--want-risk 1` has something to find. The FILENAME is derived
# from the config through the same function that writes the cache, never
# written out here — a spelled-out name is a second source for the cache key
# that goes stale silently the next time the key changes (it already has: the
# flow-timebase tag was added to it).
DEFAULT_DAY_KEY = "friday_ddos"

REQUIRED_COLUMNS = ("host_id", "window_ts", "risk_label", "stage_label")


@dataclass(frozen=True)
class DemoWindow:
    """One host's contiguous [L, F] context window plus its ground truth."""

    host_id: str
    states: np.ndarray
    origin_ts: datetime
    risk_label: int
    stage_label: str


def pick_window(df: pd.DataFrame, L: int, window_seconds: int, want_risk: int = 1) -> DemoWindow:
    """Earliest [L, F] window that is L rows of ONE host, contiguous in
    `window_seconds` steps, ending on a row whose risk_label == want_risk.

    Contiguity is enforced rather than assumed: a window spanning a gap in a
    host's timeline would hand the model a history that never happened, and
    the resulting risk curve would be a demo artifact rather than a
    prediction. Hosts with gaps are skipped, not patched.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")

    for host_id, host_df in df.sort_values(["host_id", "window_ts"]).groupby("host_id", sort=True):
        if len(host_df) < L:
            continue
        ts = host_df["window_ts"].to_numpy()
        labels = host_df["risk_label"].to_numpy()
        for end in range(L - 1, len(host_df)):
            if labels[end] != want_risk:
                continue
            start = end - L + 1
            span = ts[start:end + 1]
            if not np.array_equal(np.diff(span), np.full(L - 1, window_seconds)):
                continue  # time gap — skip rather than fabricate history
            rows = host_df.iloc[start:end + 1]
            return DemoWindow(
                host_id=str(host_id),
                states=rows[list(FEATURE_ORDER)].to_numpy(dtype=np.float64),
                origin_ts=datetime.fromtimestamp(float(ts[end]), tz=timezone.utc),
                risk_label=int(labels[end]),
                stage_label=str(rows["stage_label"].iloc[-1]),
            )

    raise ValueError(
        f"no host has {L} contiguous windows ending on risk_label=={want_risk} in this table"
    )


def _summarize(forecast: dict, window: DemoWindow, threshold: float) -> str:
    lines = [
        "",
        "=" * 72,
        "NIDRA forecast — shipped ensemble, real CIC-IDS2017 window",
        "=" * 72,
        f"host                {window.host_id}",
        f"window origin       {forecast['origin_ts']}",
        f"ground truth        risk_label={window.risk_label}  stage={window.stage_label}",
        f"model               {forecast['model_version']}  (schema {forecast['schema_ver']})",
        f"trajectories        {forecast['n_trajectories']}",
        "",
        f"observed risk now   {forecast['observed_risk']:.4f}",
        "",
        f"forecast risk by horizon (alert threshold {threshold}):",
    ]
    for h in forecast["horizons"]:
        flag = "  <-- ALERT" if h["p_compromise"] >= threshold else ""
        lines.append(
            f"  t+{h['k']}  p_compromise={h['p_compromise']:.4f}"
            f"  [{h['ci_low']:.4f}, {h['ci_high']:.4f}]{flag}"
        )
    lead = forecast["lead_time_s"]
    lines += [
        "",
        f"lead time           {'no alert in horizon' if lead is None else f'{lead:.0f}s'}",
        "",
        "top contributing signals (SHAP):",
    ]
    for s in forecast["top_signals"]:
        lines.append(f"  {s['direction']:>4}  {s['name']:<28} {s['shap_value']:+.5f}")
    lines += ["", "=" * 72, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast one real CIC-IDS2017 window using the shipped ensemble."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--day", type=str, default=None,
                        help="filename under artifacts/processed/ to draw the window from; "
                             f"defaults to the cached table for '{DEFAULT_DAY_KEY}'")
    parser.add_argument("--want-risk", type=int, default=1, choices=[0, 1],
                        help="1 = a window labelled as leading to an attack, 0 = a benign window")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="alert threshold used only for the printed ALERT markers")
    parser.add_argument("--json", action="store_true", help="print the raw forecast dict as JSON")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the rollout sampler; the forecast is a Monte Carlo "
                             "estimate over sampled trajectories, so this is what makes two "
                             "runs of the demo agree")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # The risk curve is pooled over sampled rollout trajectories. Unseeded,
    # a reviewer running this twice sees different numbers for the same
    # window and has no way to tell sampling noise from a real difference.
    set_seed(args.seed)
    cfg = load_config(args.config)
    artifacts = cfg["artifacts"]

    processed_dir = resolve_path(cfg, artifacts["processed_dir"])
    if args.day:
        table_path = processed_dir / args.day
    else:
        day_meta = cfg["dataset"]["days"][DEFAULT_DAY_KEY]
        table_path = day_cache_path(processed_dir, DEFAULT_DAY_KEY, cfg["windowing"],
                                     cfg["dataset"].get("mvp_row_cap_per_day"),
                                     declared_packets_tag(day_meta))
    if not table_path.exists():
        available = sorted(p.name for p in resolve_path(cfg, artifacts["processed_dir"]).glob("*.parquet"))
        raise SystemExit(
            f"windowed table not found: {table_path}\n"
            f"available tables: {available or '(none — artifacts/processed/ is empty)'}"
        )

    df = pd.read_parquet(table_path)
    window = pick_window(df, L=CONTEXT_LENGTH,
                         window_seconds=cfg["windowing"]["window_seconds"],
                         want_risk=args.want_risk)

    predictor = NidraPredictor(
        weights_dir=resolve_path(cfg, artifacts["weights_dir"]),
        scaler_path=resolve_path(cfg, artifacts["scaler_dir"]) / "robust_scaler.joblib",
        config_path=args.config,
    )
    forecast = predictor.forecast(window.states, host_id=window.host_id, origin_ts=window.origin_ts)

    if args.json:
        print(json.dumps(forecast, indent=2))
    else:
        print(_summarize(forecast, window, args.threshold))


if __name__ == "__main__":
    main()
