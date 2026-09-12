"""Build the /demo replay fixture — synthesised telemetry, real pipeline arithmetic.

Why this script exists (PROMPTBOOK-FRONTEND F2, source path 2): the preferred path is
capturing the live WebSocket, but that path cannot produce the fixture F2 specifies even
with the stack up. The serving predictor is the stub (`predictor.impl: stub`), and the
stub *by design* never claims the late stages — `stage_distribution` caps its mass at
`initial_access` and holds a 0.01 sliver on lateral/c2/exfil — while the bundled replay
capture (`tests/fixtures/replay.csv`) is labelled BENIGN and PortScan only. A capture
therefore cannot show `benign → recon → initial_access → lateral`.

So this script synthesises exactly one thing at the telemetry level — per-window driver
curves for four hosts, one of which runs a scan-to-intrusion arc — and feeds them through
`services.inference.stub_predictor.StubPredictor`, the same code the pipeline serves.
Risk, confidence bands, horizons, lead times, top signals, driving windows and predicted
features are all the stub's own arithmetic over that telemetry, not hand-picked numbers.

The one post-processing step is the stage story for the escalating host: its
`observed_stage` / `stage_dist` are replaced by a scripted progression that continues
into `lateral`, because the demo's narrative requires the stage the stub refuses to
claim. Quiet hosts keep the stub's stages verbatim. All of this is restated for the
reader in web/fixtures/README.md, which is what the /demo banner cites.

Every forecast is validated through `nidra_common.schemas.Forecast.model_validate` —
the API's own validator — and what is written to disk is the validator's canonical
JSON dump. Deterministic: same script, same bytes.

Run from the repo root:  .venv/bin/python scripts/make_demo_fixture.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nidra.data.schema import FEATURE_ORDER, STAGES  # noqa: E402
from nidra_common.config import get_config  # noqa: E402
from nidra_common.schemas import SCHEMA_VERSION, Forecast  # noqa: E402
from services.inference.stub_predictor import StubPredictor  # noqa: E402

FIXTURES = REPO / "web" / "fixtures"
TENANT = "demo"

#: CIC-IDS2017 Wednesday. The arc is timed so the threshold crossing lands mid-file.
BASE_TS = datetime(2017, 7, 5, 14, 20, 0, tzinfo=UTC)
N_WINDOWS = 48

VICTIM = "192.168.10.50"
QUIET = ["192.168.10.9", "192.168.10.17", "192.168.10.25"]
HOSTS = [VICTIM, *QUIET]

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_ORDER)}

# --------------------------------------------------------------------- driver curves
#
# Piecewise-linear keyframes per driver, (window, value). The shape encodes the story:
# quiet baseline -> port-scan ramp (recon) -> sharp intrusion burst -> sustained
# lateral-movement phase with drivers still climbing slowly. The stub's surrogate risk
# reads levels *and* backward slopes of these four, so the burst is what carries the
# observed risk over the threshold — exactly the dynamics-over-levels behaviour the
# real model is being built to learn.

VICTIM_KEYFRAMES: dict[str, list[tuple[int, float]]] = {
    "syn_ratio": [(0, 0.07), (13, 0.08), (18, 0.18), (22, 0.30), (25, 0.55), (28, 0.95), (34, 0.96), (47, 0.93)],
    "dst_port_entropy": [(0, 0.85), (13, 0.90), (18, 1.60), (22, 2.40), (25, 3.70), (28, 6.20), (34, 6.60), (47, 6.90)],
    "new_peer_count": [(0, 0.3), (13, 0.5), (18, 3.5), (22, 7.0), (25, 15.0), (28, 34.0), (34, 37.0), (47, 40.0)],
    "out_degree": [(0, 3.0), (13, 3.5), (18, 9.0), (22, 16.0), (25, 32.0), (28, 80.0), (34, 96.0), (47, 110.0)],
}

QUIET_BASE: dict[str, float] = {
    "syn_ratio": 0.06,
    "dst_port_entropy": 0.8,
    "new_peer_count": 0.3,
    "out_degree": 3.0,
}

#: Benign values for every non-driver feature — held near-constant with seeded noise.
BENIGN_BASE: dict[str, float] = {
    "ack_ratio": 0.82, "rst_ratio": 0.02, "fin_ratio": 0.05, "psh_ratio": 0.22,
    "urg_ratio": 0.0, "bytes_total": 48_000.0, "bytes_up_down_ratio": 0.35,
    "pkts_per_flow_mean": 14.0, "flow_duration_mean": 1.6e6, "flow_duration_var": 4.0e11,
    "iat_mean": 2.4e5, "iat_var": 9.0e9, "iat_max": 1.2e6, "active_flow_count": 9.0,
    "ttl_mean": 62.0, "ttl_var": 14.0, "tcp_window_mean": 8100.0,
    "tcp_window_entropy": 1.4, "frag_flag_rate": 0.0, "payload_size_mean": 420.0,
    "payload_size_var": 90_000.0, "payload_size_p95": 1360.0, "payload_size_entropy": 4.1,
    "retrans_count": 1.0, "retrans_rate": 0.01, "in_degree": 2.0, "dst_ip_entropy": 0.9,
    "neighbour_risk_fraction": 0.0, "local_clustering_coeff": 0.35, "reciprocity": 0.8,
}

#: How the intrusion arc pulls selected non-driver features, (window, value) keyframes.
#: Cosmetic realism only — the stub does not read these — but predicted_features are
#: shown in the console later and a scan that changes nothing but four numbers would
#: look synthetic in the worst way.
VICTIM_SIDE_EFFECTS: dict[str, list[tuple[int, float]]] = {
    "rst_ratio": [(0, 0.02), (18, 0.10), (27, 0.38), (47, 0.30)],
    "ack_ratio": [(0, 0.82), (18, 0.70), (27, 0.34), (47, 0.40)],
    "dst_ip_entropy": [(0, 0.9), (20, 1.6), (27, 3.4), (47, 3.6)],
    "active_flow_count": [(0, 9.0), (20, 30.0), (27, 140.0), (47, 120.0)],
    "bytes_total": [(0, 48_000.0), (24, 90_000.0), (27, 420_000.0), (47, 520_000.0)],
    "payload_size_mean": [(0, 420.0), (26, 240.0), (30, 90.0), (47, 130.0)],
    "flow_duration_mean": [(0, 1.6e6), (24, 8.0e5), (28, 9.0e4), (47, 1.5e5)],
    "neighbour_risk_fraction": [(0, 0.0), (30, 0.0), (34, 0.25), (47, 0.25)],
    "reciprocity": [(0, 0.8), (22, 0.55), (28, 0.18), (47, 0.22)],
}

# ---------------------------------------------------------------------- stage script
#
# Continuous stage progress s(w) for the victim: 0=benign, 1=recon, 2=initial_access,
# 3=lateral. Horizons extrapolate along the current per-window rate, so a forecast made
# during late recon puts forward mass on initial_access — the stage strip predicts the
# *next* stage, matching how the arc actually unfolds.

STAGE_KEYFRAMES: list[tuple[int, float]] = [
    (0, 0.0), (13, 0.0), (17, 0.5), (21, 1.0), (24, 1.4), (26, 1.8),
    (28, 2.1), (32, 2.5), (36, 3.0), (47, 3.3),
]
STAGE_KERNEL_WIDTH = 1.15
STAGE_FLOOR = 0.004
STAGE_MAX = 3.4  # a sliver of forward mass on c2 late in the arc; never exfil


def interp(keyframes: list[tuple[int, float]], w: int) -> float:
    xs = [k[0] for k in keyframes]
    ys = [k[1] for k in keyframes]
    return float(np.interp(w, xs, ys))


def stage_progress(w: float) -> float:
    return min(interp(STAGE_KEYFRAMES, int(w)) if float(w).is_integer()
               else float(np.interp(w, [k[0] for k in STAGE_KEYFRAMES], [k[1] for k in STAGE_KEYFRAMES])),
               STAGE_MAX)


def scripted_stage_dist(s: float) -> dict[str, float]:
    """Triangular kernel over the stage axis, floored and normalised."""
    s = min(s, STAGE_MAX)
    raw = [max(0.0, 1.0 - abs(i - s) / STAGE_KERNEL_WIDTH) + STAGE_FLOOR for i in range(len(STAGES))]
    total = sum(raw)
    dist = {stage: round(v / total, 6) for stage, v in zip(STAGES, raw)}
    # rounding drift: pin the sum to 1 exactly on the largest entry
    drift = round(1.0 - sum(dist.values()), 6)
    top = max(dist, key=lambda k: dist[k])
    dist[top] = round(dist[top] + drift, 6)
    return dist


def argmax_stage(dist: dict[str, float]) -> str:
    return max(STAGES, key=lambda stage: dist.get(stage, 0.0))


# ------------------------------------------------------------------- state synthesis


def smooth(series: np.ndarray) -> np.ndarray:
    """Two passes of a 3-tap kernel: rounds keyframe corners so the 3-window
    backward slope the stub fits varies smoothly instead of jumping."""
    kernel = np.array([0.25, 0.5, 0.25])
    out = series
    for _ in range(2):
        padded = np.concatenate([[out[0]], out, [out[-1]]])
        out = np.convolve(padded, kernel, mode="valid")
    return out


VICTIM_DRIVER_NOISE = {
    "syn_ratio": 0.004, "dst_port_entropy": 0.025,
    "new_peer_count": 0.15, "out_degree": 0.25,
}

VICTIM_DRIVER_LEVELS: dict[str, np.ndarray] = {
    name: smooth(np.array([interp(kf, w) for w in range(N_WINDOWS)]))
    for name, kf in VICTIM_KEYFRAMES.items()
}


def victim_driver(name: str, w: int, rng: np.random.Generator) -> float:
    level = float(VICTIM_DRIVER_LEVELS[name][w])
    return max(0.0, level + float(rng.normal(0.0, VICTIM_DRIVER_NOISE[name])))


def quiet_driver(name: str, w: int, rng: np.random.Generator) -> float:
    base = QUIET_BASE[name]
    wobble = 0.04 * base * float(np.sin(w / 5.0 + hash(name) % 7))
    noise = 0.03 * base * float(rng.normal())
    return max(0.0, base + wobble + noise)


def synth_states(host: str, rng: np.random.Generator) -> np.ndarray:
    """[N_WINDOWS, 45] observed feature rows, dynamics features derived, not invented."""
    rows = np.zeros((N_WINDOWS, len(FEATURE_ORDER)), dtype=float)
    is_victim = host == VICTIM

    for w in range(N_WINDOWS):
        row = {}
        for name, base in BENIGN_BASE.items():
            jitter = 0.02 * base * float(rng.normal()) if base else 0.0
            row[name] = max(0.0, base + jitter)
        if is_victim:
            for name, keyframes in VICTIM_SIDE_EFFECTS.items():
                base = interp(keyframes, w)
                row[name] = max(0.0, base * (1.0 + 0.02 * float(rng.normal())))
        for name in ("syn_ratio", "dst_port_entropy", "new_peer_count", "out_degree"):
            row[name] = victim_driver(name, w, rng) if is_victim else quiet_driver(name, w, rng)
        row["syn_ratio"] = min(1.0, row["syn_ratio"])
        row["is_active"] = 1.0
        # dynamics: deltas and 3-window slopes of the synthesised series themselves
        for name in FEATURE_ORDER:
            if name.startswith(("d_", "slope3_")):
                row.setdefault(name, 0.0)
        for name, value in row.items():
            rows[w, FEATURE_INDEX[name]] = value

        def prev(col: str, back: int = 1) -> float:
            return float(rows[w - back, FEATURE_INDEX[col]]) if w >= back else float(rows[w, FEATURE_INDEX[col]])

        for delta_name, src in [
            ("d_syn_ratio", "syn_ratio"), ("d_dst_port_entropy", "dst_port_entropy"),
            ("d_out_degree", "out_degree"), ("d_new_peer_count", "new_peer_count"),
            ("d_iat_var", "iat_var"), ("d_retrans_rate", "retrans_rate"),
        ]:
            rows[w, FEATURE_INDEX[delta_name]] = rows[w, FEATURE_INDEX[src]] - prev(src)
        for slope_name, src in [
            ("slope3_syn_ratio", "syn_ratio"), ("slope3_dst_port_entropy", "dst_port_entropy"),
            ("slope3_out_degree", "out_degree"), ("slope3_iat_var", "iat_var"),
        ]:
            if w >= 2:
                tail = rows[w - 2 : w + 1, FEATURE_INDEX[src]]
                x = np.arange(3.0) - 1.0
                rows[w, FEATURE_INDEX[slope_name]] = float((x * (tail - tail.mean())).sum() / (x**2).sum())
    return rows


# ------------------------------------------------------------------------- assembly


def window_ts(w: int) -> datetime:
    return BASE_TS + timedelta(seconds=w * WINDOW_DELTA)


def build() -> None:
    predictor = StubPredictor()
    forecasts: list[dict] = []
    explanations: dict[str, dict] = {}
    victim_observed: list[float] = []

    for host in HOSTS:
        rng = np.random.default_rng(abs(hash(host)) % (2**32))
        states = synth_states(host, rng)
        for w in range(N_WINDOWS):
            context = states[max(0, w + 1 - CONTEXT_L) : w + 1]
            raw = predictor.forecast(context, host_id=host, origin_ts=window_ts(w))
            raw["tenant_id"] = TENANT

            if host == VICTIM:
                s_now = stage_progress(w)
                rate = max(0.0, s_now - stage_progress(max(0, w - 1)))
                raw["observed_stage"] = argmax_stage(scripted_stage_dist(s_now))
                for h in raw["horizons"]:
                    h["stage_dist"] = scripted_stage_dist(s_now + h["k"] * rate)
                victim_observed.append(raw["observed_risk"])

            validated = Forecast.model_validate(raw)
            dumped = validated.model_dump(mode="json")
            for h in dumped["horizons"]:
                keys = set(h["predicted_features"])
                assert keys == set(FEATURE_ORDER), "predicted_features must be the 45 keys"
                assert set(h["stage_dist"]) == set(STAGES), "stage_dist must cover 6 stages"
            forecasts.append(dumped)

            # /demo has no network, so the ExplanationPanel's inputs — the
            # stub's own explain() output — ride along in the fixture, keyed
            # by host@origin_ts exactly as the payload spells them.
            exp = predictor.explain(context, horizon_k=1)
            explanations[f"{host}@{dumped['origin_ts']}"] = {
                "method": exp["method"],
                "model_version": exp["model_version"],
                "driving_window": exp["driving_window"],
                "window_importance": exp["window_importance"],
                "context_windows": int(context.shape[0]),
                "context_l": CONTEXT_L,
                "note": exp["note"],
            }

    # ---------------- story invariants: regeneration cannot silently drift ----------
    by_host: dict[str, list[dict]] = {h: [] for h in HOSTS}
    for f in forecasts:
        by_host[f["host_id"]].append(f)

    crossing_w = next(
        (w for w, r in enumerate(victim_observed) if r >= RISK_THRESHOLD), None
    )
    assert crossing_w is not None, "victim observed risk never crosses the threshold"
    assert 20 <= crossing_w <= 36, f"crossing at w={crossing_w}, not partway through"

    stages_seen = []
    for f in by_host[VICTIM]:
        if f["observed_stage"] not in stages_seen:
            stages_seen.append(f["observed_stage"])
    assert stages_seen == ["benign", "recon", "initial_access", "lateral"], stages_seen

    def drawn_crossing_k(f: dict) -> int | None:
        ps = [h["p_compromise"] for h in f["horizons"]]
        for i in range(len(ps) - LEAD_M + 1):
            if all(p >= RISK_THRESHOLD for p in ps[i : i + LEAD_M]):
                return i + 1
        return None

    ks = {drawn_crossing_k(f) for f in by_host[VICTIM]}
    assert 3 in ks, "no victim forecast crosses at k=3"
    assert 1 in ks, "no victim forecast crosses at k=1"
    for host in QUIET:
        assert all(f["lead_time_s"] is None for f in by_host[host]), f"{host} crossed"
        assert max(f["observed_risk"] for f in by_host[host]) < 0.2, f"{host} not quiet"

    # ------------------------------------------------------------- counterfactual --
    # At the first k=3-crossing origin, clamp dst_port_entropy to its benign level and
    # re-simulate — the stub's own counterfactual, ts added per the API's CurvePoint.
    cf_index = next(i for i, f in enumerate(by_host[VICTIM]) if drawn_crossing_k(f) == 3)
    rng = np.random.default_rng(abs(hash(VICTIM)) % (2**32))
    victim_states = synth_states(VICTIM, rng)
    cf_context = victim_states[max(0, cf_index + 1 - CONTEXT_L) : cf_index + 1]
    cf = predictor.counterfactual(cf_context, "dst_port_entropy", QUIET_BASE["dst_port_entropy"])
    cf_origin = window_ts(cf_index)

    def with_ts(points: list[dict]) -> list[dict]:
        return [
            {**p, "ts": (cf_origin + timedelta(seconds=p["k"] * WINDOW_DELTA)).isoformat().replace("+00:00", "Z")}
            for p in points
        ]

    counterfactual = {
        "label": cf["label"],
        "host_id": VICTIM,
        "origin_ts": cf_origin.isoformat().replace("+00:00", "Z"),
        "feature": cf["feature"],
        "clamp_value": cf["clamp_value"],
        "model_version": cf["model_version"],
        "original": with_ts(cf["original"]),
        "counterfactual": with_ts(cf["counterfactual"]),
        "context_windows": int(cf_context.shape[0]),
        "note": (
            "what this model would predict had the feature held this value; "
            "a statement about the model, not about the network"
        ),
    }
    assert counterfactual["label"] == "model-internal what-if"

    # --------------------------------------------------------------------- write ----
    FIXTURES.mkdir(parents=True, exist_ok=True)

    replay = {
        "source": {
            "kind": "synthesised",
            "generator": "scripts/make_demo_fixture.py",
            "summary": (
                "Synthesised CIC-IDS2017-Wednesday-shaped telemetry driven through the "
                "backend's own StubPredictor; stage narrative for the escalating host "
                "is scripted (the stub never claims late stages). Every forecast is "
                "validated by nidra_common.schemas.Forecast. Values illustrative of "
                "the pipeline's shape, not measurements."
            ),
        },
        "model": {
            "impl": "stub",
            "model_version": predictor.model_version,
            "schema_ver": SCHEMA_VERSION,
            "config": {
                "window_delta": WINDOW_DELTA,
                "context_L": CONTEXT_L,
                "horizon_K": int(CFG["horizon_K"]),
                "n_features": int(CFG["n_features"]),
                "risk_threshold": RISK_THRESHOLD,
                "lead_time_m": LEAD_M,
            },
        },
        "tenant_id": TENANT,
        "hosts": HOSTS,
        "forecasts": forecasts,
        "explanations": explanations,
    }
    (FIXTURES / "demo-replay.json").write_text(json.dumps(replay, indent=1) + "\n")
    (FIXTURES / "counterfactual.single.json").write_text(json.dumps(counterfactual, indent=2) + "\n")

    # model.json mirrors GET /api/v1/model for the serving stub.
    (FIXTURES / "model.json").write_text(json.dumps(
        {k: replay["model"][k] for k in ("impl", "model_version", "schema_ver", "config")},
        indent=2,
    ) + "\n")

    # forecast.single.json: the backend's published example, seeded verbatim after
    # revalidating it through the same validator.
    example = json.loads((REPO / "web-contract" / "forecast.example.json").read_text())
    Forecast.model_validate(example)
    shutil.copyfile(REPO / "web-contract" / "forecast.example.json", FIXTURES / "forecast.single.json")

    # ------------------------------------------------------------------ report -----
    print(f"forecasts: {len(forecasts)} ({len(HOSTS)} hosts x {N_WINDOWS} windows)")
    print(f"victim observed-risk crossing at w={crossing_w} ({window_ts(crossing_w).time()} UTC)")
    print(f"victim stages, in order of first appearance: {stages_seen}")
    print(f"drawn crossing ks present across victim forecasts: {sorted(k for k in ks if k)}")
    firsts = {k: next(i for i, f in enumerate(by_host[VICTIM]) if drawn_crossing_k(f) == k)
              for k in (3, 1)}
    print(f"first k=3 crossing at w={firsts[3]}, first k=1 crossing at w={firsts[1]}")
    print(f"counterfactual origin w={cf_index}: original k6="
          f"{counterfactual['original'][-1]['p_compromise']}, "
          f"clamped k6={counterfactual['counterfactual'][-1]['p_compromise']}")
    quiet_max = max(f["observed_risk"] for h in QUIET for f in by_host[h])
    print(f"quiet hosts max observed risk: {quiet_max}")

    # Overlay coverage: for each victim origin, how many of the six later-observed
    # risks fall inside that forecast's band (Appendix A semantics). The specimen's
    # "proof shot" state picks its origin off this table.
    obs_by_w = {w: f["observed_risk"] for w, f in enumerate(by_host[VICTIM])}
    print("overlay coverage (w: inside/available):")
    for w in range(16, 34):
        f = by_host[VICTIM][w]
        inside = avail = 0
        for h in f["horizons"]:
            actual = obs_by_w.get(w + h["k"])
            if actual is None:
                continue
            avail += 1
            if h["ci_low"] <= actual <= h["ci_high"]:
                inside += 1
        print(f"  w={w} lead={f['lead_time_s']}: {inside}/{avail}")


CFG = get_config()
WINDOW_DELTA = int(CFG["window_delta"])
CONTEXT_L = int(CFG["context_L"])
RISK_THRESHOLD = float(CFG["risk_threshold"])
LEAD_M = int(CFG["lead_time_m"])


if __name__ == "__main__":
    build()
