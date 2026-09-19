"""Build the /demo replay from REAL CIC-IDS2017 traffic and the REAL world model.

This replaces scripts/make_demo_fixture.py, which synthesised per-window driver
curves, pushed them through StubPredictor, and then scripted a stage narrative on
top because the stub refuses to claim late stages. That script existed because the
serving predictor was the stub and the trained ensemble lived on another branch.
Both of those things are no longer true.

Every number this writes is the trained 5-seed ensemble's own output, pooled the
way `NidraPredictor` pools it in production (quantile 0.85 over 1,000 sampled
trajectories), over real windowed CIC-IDS2017 telemetry committed under
ml/artifacts/processed/. Nothing is synthesised, scripted or hand-picked: the
hosts are real hosts, the attack is the real Friday-morning Botnet episode, and
where the model is wrong the fixture shows it being wrong.

The predictor is obtained through `services.inference.predictor_loader.load_predictor`
— the same call the inference worker makes — so the console cannot be showing a
configuration the serving plane would not use.

Run from the repo root:  .venv/bin/python scripts/make_demo_replay.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER, HORIZON_LENGTH  # noqa: E402
from nidra.train.pipeline import day_cache_path, declared_packets_tag  # noqa: E402
from nidra.utils.config import load_config as load_ml_config  # noqa: E402
from nidra.utils.seed import set_seed  # noqa: E402
from nidra_common.config import get_config  # noqa: E402
from nidra_common.schemas import SCHEMA_VERSION, Forecast  # noqa: E402
from services.inference.predictor_loader import load_predictor  # noqa: E402

logger = logging.getLogger("make_demo_replay")

OUT = REPO / "web" / "src" / "fixtures" / "demo-replay.json"

#: Friday morning's Botnet (ARES) episode. Chosen over the Friday afternoon DDoS
#: and PortScan because it runs long enough for a host to be watched crossing from
#: quiet into compromise inside one 48-window view, and because several other
#: hosts on the same segment stay benign throughout — which is what makes the
#: quiet rows meaningful rather than decorative.
DAY_KEY = "friday_morning"
VICTIM = "192.168.10.9"
N_WINDOWS = 48
#: Display windows before the victim's first attack-labelled window. The console
#: opens on quiet traffic so the rise is visible rather than already underway.
LEAD_IN = 12
TENANT = "demo"
WINDOW_SECONDS = 30


def contiguous(ts: np.ndarray) -> bool:
    return len(ts) > 1 and bool(np.all(np.diff(ts) == WINDOW_SECONDS))


def host_frame(df: pd.DataFrame, host: str) -> pd.DataFrame:
    return df[df.host_id == host].sort_values("window_ts").reset_index(drop=True)


def pick_origins(victim_df: pd.DataFrame) -> tuple[int, int]:
    """Origin index range [start, start+N) for the replay, anchored so the
    victim's first attack-labelled window lands LEAD_IN windows in."""
    risk = victim_df.risk_label.to_numpy()
    if not risk.any():
        raise SystemExit(f"{VICTIM} has no attack-labelled window in {DAY}")
    first = int(np.argmax(risk == 1))
    start = first - LEAD_IN
    if start < CONTEXT_LENGTH - 1:
        raise SystemExit(f"{VICTIM}'s episode starts too early for {CONTEXT_LENGTH} windows of context")
    if start + N_WINDOWS > len(victim_df):
        raise SystemExit(f"{VICTIM} has too few windows after the episode starts")
    return start, first


def find_quiet_hosts(df: pd.DataFrame, ts_lo: int, ts_hi: int, need: int) -> list[str]:
    """Hosts that are contiguous across the whole displayed span AND carry no
    attack-labelled window in it. A host that merely has no label in the visible
    range but is attacked just outside it would be a misleading 'quiet' row."""
    eligible: list[str] = []
    for host, g in df.groupby("host_id", sort=True):
        if host == VICTIM:
            continue
        g = g.sort_values("window_ts")
        ts = g.window_ts.to_numpy()
        if ts[0] > ts_lo - WINDOW_SECONDS * (CONTEXT_LENGTH - 1) or ts[-1] < ts_hi:
            continue
        if not contiguous(ts):
            continue
        if g.risk_label.to_numpy().any():
            continue
        eligible.append(str(host))

    # Prefer workstations on the victim's own /24 over the external CDN and
    # service addresses that also appear in the capture. A SOC console compares
    # a compromised host against its peers; three Cloudflare edge IPs sitting
    # quietly next to it would be true but meaningless as a comparison.
    subnet = VICTIM.rsplit(".", 1)[0] + "."
    local = [h for h in eligible if h.startswith(subnet)]
    return (local + [h for h in eligible if h not in local])[:need]


def forecast_series(predictor, g: pd.DataFrame, start: int, n: int) -> list[dict]:
    """One forecast per displayed window, each from that window's own [L, F]
    history. Seeded per window so regenerating the fixture is reproducible —
    the risk curve is a Monte Carlo estimate over sampled rollouts."""
    feats = g[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(g.window_ts.to_numpy(), unit="s", utc=True)
    out = []
    for i in range(n):
        end = start + i
        set_seed(1000 + i)
        states = feats[end - CONTEXT_LENGTH + 1: end + 1]
        f = predictor.forecast(states, host_id=str(g.host_id.iloc[0]), origin_ts=ts[end].to_pydatetime())
        f["tenant_id"] = TENANT
        out.append(f)
    return out


def strip_predicted_features(forecast: dict) -> dict:
    """Validated in full, written without the 45-float predicted_features block
    per horizon — the console never reads it and it would multiply the fixture's
    size by roughly thirty."""
    slim = dict(forecast)
    slim["horizons"] = [{k: v for k, v in h.items() if k != "predicted_features"} for h in forecast["horizons"]]
    return slim


def measure(victim_df: pd.DataFrame, start: int, n: int, series: list[dict], threshold: float) -> dict:
    """Score the displayed episode against its own ground-truth labels.

    A demo that shows one host is showing one sample, and a single host is
    noisier than the split-wide numbers in the README. Rather than let the
    console imply otherwise, the fixture carries what this specific episode
    actually scored, so the page can state it.
    """
    labels = victim_df.risk_label.to_numpy()[start:start + n]
    peak = np.array([max(h["p_compromise"] for h in f["horizons"]) for f in series])
    tp = int(((peak >= threshold) & (labels == 1)).sum())
    fp = int(((peak >= threshold) & (labels == 0)).sum())
    fn = int(((peak < threshold) & (labels == 1)).sum())
    return {
        "attack_labelled_windows": int(labels.sum()),
        "windows": int(n),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
    }


def build(limit: int | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_config()
    if cfg["predictor"]["impl"] != "nidra":
        raise SystemExit("predictor.impl must be 'nidra' — this script exists to show the real model")

    predictor = load_predictor()
    # The cached table's filename is derived through the same function that
    # writes it, never spelled out here: a second copy of the cache key goes
    # stale silently the next time the key changes.
    ml_cfg = load_ml_config(str(REPO / "ml" / "config" / "default.yaml"))
    day_meta = ml_cfg["dataset"]["days"][DAY_KEY]
    day_file = day_cache_path(REPO / "ml" / "artifacts" / "processed", DAY_KEY,
                               ml_cfg["windowing"], ml_cfg["dataset"].get("mvp_row_cap_per_day"),
                               declared_packets_tag(day_meta))
    df = pd.read_parquet(day_file)

    victim_df = host_frame(df, VICTIM)
    if not contiguous(victim_df.window_ts.to_numpy()):
        raise SystemExit(f"{VICTIM}'s window sequence has gaps; a replay over it would skip time")
    n = limit or N_WINDOWS
    start, first_attack = pick_origins(victim_df)
    ts_lo = int(victim_df.window_ts.iloc[start])
    ts_hi = int(victim_df.window_ts.iloc[start + n - 1])

    quiet = find_quiet_hosts(df, ts_lo, ts_hi, need=3)
    if len(quiet) < 3:
        raise SystemExit(f"only found {len(quiet)} quiet hosts spanning the window")
    hosts = [VICTIM, *quiet]
    print(f"day        : {DAY_KEY}  ({day_file.name})")
    print(f"victim     : {VICTIM} (first attack-labelled window at display index {LEAD_IN})")
    print(f"quiet      : {', '.join(quiet)}")
    print(f"windows    : {n} x {len(hosts)} hosts = {n * len(hosts)} forecasts")

    forecasts: list[dict] = []
    explanations: dict[str, dict] = {}
    for host in hosts:
        g = host_frame(df, host)
        # Align every host to the victim's absolute timeline: ranking by index
        # across hosts is only meaningful if the same index is the same instant.
        offset = int(np.searchsorted(g.window_ts.to_numpy(), ts_lo))
        series = forecast_series(predictor, g, offset, n)
        for f in series:
            Forecast.model_validate(f)          # the API's own validator
            forecasts.append(strip_predicted_features(f))
        if host == VICTIM:
            feats = g[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
            for i, f in enumerate(series):
                end = offset + i
                # horizon_k counts from 1, so the last step of the cone is K.
                e = predictor.explain(feats[end - CONTEXT_LENGTH + 1: end + 1], horizon_k=HORIZON_LENGTH)
                sal = e.get("temporal_saliency", {})
                first_feat = next(iter(sal.values()), {}) if isinstance(sal, dict) else {}
                explanations[f"{host}@{f['origin_ts']}"] = {
                    "method": "kernel-shap + input-gradient temporal saliency",
                    "model_version": f["model_version"],
                    "driving_window": int(f["driving_window"]),
                    "window_importance": [float(x) for x in first_feat.get("window_importance", [])],
                    "context_windows": CONTEXT_LENGTH,
                    "context_l": CONTEXT_LENGTH,
                }
        print(f"  {host:16s} {len(series)} forecasts")

    victim_series = [f for f in forecasts if f["host_id"] == VICTIM]
    crossed = [i for i, f in enumerate(victim_series) if f["observed_risk"] >= cfg["risk_threshold"]]
    warned = [i for i, f in enumerate(victim_series) if f["lead_time_s"] is not None]
    quiet_max = max(f["observed_risk"] for f in forecasts if f["host_id"] != VICTIM)
    measured = measure(victim_df, start, n, victim_series, float(cfg["risk_threshold"]))

    replay = {
        "source": {
            "kind": "real",
            "generator": "scripts/make_demo_replay.py",
            "summary": (
                "Real CIC-IDS2017 Friday-morning traffic (Botnet ARES) windowed into the "
                "committed 45-feature state tables, scored by the trained 5-seed NIDRA "
                "ensemble through the same predictor the inference worker loads. Risk is "
                "pooled at the 85th percentile of 1,000 sampled rollout trajectories. "
                "Every value — risk, confidence bands, horizons, lead times, SHAP signals, "
                "stage distributions — is the model's own output on real traffic. Nothing "
                "is synthesised or scripted, including where the model is wrong."
            ),
            "episode": {
                "day": DAY_KEY,
                "attack": "Botnet ARES",
                "stage": "c2",
                "stage_in_training_data": False,
                "note": (
                    "This is one host over 24 minutes, not the evaluation set: a single "
                    "episode is noisier than the split-wide figures in the README. It is "
                    "also the harder kind — `c2` is one of three attack stages that appear "
                    "only in the evaluation days, so the model is forecasting a stage it "
                    "has zero training examples of. Training contains `initial_access` and "
                    "`exfil` and nothing else."
                ),
                "measured_on_this_episode": measured,
            },
        },
        "model": {
            "impl": cfg["predictor"]["impl"],
            "model_version": forecasts[0]["model_version"],
            "schema_ver": SCHEMA_VERSION,
            "config": {
                "window_delta": WINDOW_SECONDS,
                "context_L": CONTEXT_LENGTH,
                "horizon_K": HORIZON_LENGTH,
                "n_features": len(FEATURE_ORDER),
                "risk_threshold": float(cfg["risk_threshold"]),
                "lead_time_m": int(cfg["lead_time_m"]),
            },
        },
        "tenant_id": TENANT,
        "hosts": hosts,
        "forecasts": forecasts,
        "explanations": explanations,
    }
    OUT.write_text(json.dumps(replay, indent=1) + "\n")

    print(f"\nwrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"victim observed risk crosses {cfg['risk_threshold']} at display windows: {crossed[:6]}"
          f"{' ...' if len(crossed) > 6 else ''}")
    print(f"victim windows that raise an advance warning: {len(warned)} of {n}")
    print(f"quiet hosts' highest observed risk: {quiet_max:.4f}")
    print(f"this episode, forecast vs labels: {measured}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="fewer windows, for a quick check")
    build(ap.parse_args().limit)
