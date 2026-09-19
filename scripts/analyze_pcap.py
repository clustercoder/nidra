"""Score an arbitrary pcap with the trained model and emit a /demo replay.

This is the upload path behind the dashboard's "analyse a capture" panel. It
takes a capture the model has never seen, runs it through exactly the stages
the CIC-IDS2017 replay goes through, and writes the same JSON shape
`scripts/make_demo_replay.py` writes — so the console renders an uploaded
capture with the same components, and no code path exists that could show a
real capture through a different model than the demo does:

    pcap -> tshark (nidra.data.pcap_extract)
         -> flows  (nidra.data.flow_assemble)
         -> 45-feature per-(host, window) states (nidra.data.windowize)
         -> NidraPredictor (services.inference.predictor_loader.load_predictor)

Two things are honestly different from the replay and are stated in the
output rather than hidden, because a viewer cannot tell by looking:

  * There are no labels. An uploaded capture has no ground truth, so the
    replay's `measured_on_this_episode` block is absent and the page must not
    claim precision or recall for it.
  * The flow features are reconstructed rather than read from a CICFlowMeter
    CSV. `scripts/validate_flow_assembly.py` quantifies what that costs; the
    known residuals travel with the output in `source.fidelity`.

Run from the repo root:

    .venv/bin/python scripts/analyze_pcap.py capture.pcap --out replay.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nidra.data.flow_assemble import assemble_flows  # noqa: E402
from nidra.data.join import prepare_packets_for_windowing  # noqa: E402
from nidra.data.pcap_extract import extract_pcap_to_parquet  # noqa: E402
from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER, HORIZON_LENGTH  # noqa: E402
from nidra.data.windowize import align_window, build_state_rows  # noqa: E402
from nidra.utils.seed import set_seed  # noqa: E402

from nidra_common.config import get_config  # noqa: E402
from nidra_common.schemas import SCHEMA_VERSION, Forecast  # noqa: E402
from services.inference.predictor_loader import load_predictor  # noqa: E402

logger = logging.getLogger("analyze_pcap")

WINDOW_SECONDS = 30
TENANT = "upload"

#: Hosts shown, and windows per host. The console ranks hosts against each
#: other at the same instant, so every host shown must span the same windows.
DEFAULT_MAX_HOSTS = 6
DEFAULT_MAX_WINDOWS = 96

#: A host needs a full context window before the model can be asked anything,
#: plus at least one window to forecast from.
MIN_WINDOWS_PER_HOST = CONTEXT_LENGTH + 1

#: Carried into the output so the console can caption an uploaded capture
#: differently from the labelled replay. Each line is a measured residual from
#: scripts/validate_flow_assembly.py, not a disclaimer written by hand.
FIDELITY_NOTES = [
    "Flow features here are reconstructed from packets, not read from a "
    "CICFlowMeter CSV. On the one capture where both exist (CIC-IDS2017 "
    "Friday), comparing the same (host, 30s window) computed both ways: the "
    "volume and timing features track the reference closely — Spearman 0.92 "
    "for bytes_total, 0.87 flow_duration_mean, 0.86 iat_mean/iat_max/"
    "active_flow_count, 0.85 pkts_per_flow_mean.",
    "The TCP flag ratios do not reconstruct. syn_ratio, fin_ratio and "
    "rst_ratio have essentially no rank correlation with CICFlowMeter's "
    "(Spearman -0.08, 0.03, 0.00), whose flag columns record presence rather "
    "than a count and fire far less often than the packets warrant. Treat "
    "stage calls that rest on flag structure as unsupported on an upload.",
    "urg_ratio is always 0 here. CICFlowMeter reports it nonzero on 9.5% of "
    "flows from a capture containing ~90 URG packets in 9.9 million, so the "
    "reference value is an artifact of its accounting, not a property of the "
    "traffic, and is not recoverable from a pcap.",
    "Fewer hosts appear than CICFlowMeter would list — 515 against 1,904 on "
    "the calibration hour — because a conversation is attributed to whoever "
    "opened it. The hosts that do appear are the right ones: 512 of 515 are "
    "also in the reference.",
    "There are no attack labels on an uploaded capture, so nothing on this "
    "page is scored against ground truth.",
]


class AnalysisError(RuntimeError):
    """The capture cannot be analysed, with a reason meant for the uploader."""


def packets_from_pcap(pcap_path: Path, workdir: Path) -> pd.DataFrame:
    """tshark-extract a capture to the packet frame the rest of the pipeline
    consumes. Raises AnalysisError rather than returning an empty frame, so
    the caller never scores a capture it failed to read."""
    parquet = workdir / "packets.parquet"
    n_rows = extract_pcap_to_parquet(pcap_path, parquet)
    if n_rows == 0:
        raise AnalysisError(
            "No packets could be read from this capture. tshark accepted the "
            "file but produced no IP packets — a capture of non-IP traffic, or "
            "an empty one."
        )
    return pd.read_parquet(parquet)


def states_from_packets(packets: pd.DataFrame) -> pd.DataFrame:
    """Packets -> the 45-feature per-(host, window) state table.

    The flows are assembled rather than loaded, which is the whole reason
    `nidra.data.flow_assemble` exists; see this module's docstring for what
    that costs.
    """
    flows = assemble_flows(packets)
    if len(flows) == 0:
        raise AnalysisError(
            "No flows could be assembled from this capture. Every conversation "
            "in it is a single packet, which CICFlowMeter would not publish "
            "either."
        )
    flows = flows.assign(window_ts=align_window(flows["timestamp"], WINDOW_SECONDS))
    packets_w = prepare_packets_for_windowing(packets, WINDOW_SECONDS)
    return build_state_rows(flows, packets_w, window_seconds=WINDOW_SECONDS)


#: Candidate timelines tried when choosing what to show. Each candidate is one
#: host's own span, and every other host has to be checked against it, so this
#: bounds the search on a capture with thousands of hosts.
TIMELINE_CANDIDATES = 12


def _span(states: pd.DataFrame, host: str, max_windows: int) -> np.ndarray:
    """One host's windows, capped from the end so the newest traffic is kept.

    The cap allows for the context run-up as well as the displayed windows,
    since `select_hosts` splits the span into those two parts.
    """
    stamps = np.sort(states.loc[states.host_id == host, "window_ts"].unique())
    keep = max_windows + CONTEXT_LENGTH - 1
    return stamps[-keep:] if len(stamps) > keep else stamps


def select_hosts(
    states: pd.DataFrame, max_hosts: int, max_windows: int
) -> tuple[list[str], np.ndarray]:
    """Pick the hosts to show and the common window timeline to show them on.

    Every host shown must produce a forecast for every displayed window: the
    console ranks hosts against each other at a single index, and that
    comparison is meaningless if the index is a different instant per host, or
    missing for some of them.

    That takes more than covering the displayed windows. Each one is forecast
    from the CONTEXT_LENGTH windows before it, so a host whose data merely
    STARTS at the first displayed window can forecast none of the early ones.
    A candidate span is therefore split into a context run-up and the displayed
    remainder, and a host has to hold the whole span to qualify. (This was
    found end-to-end, not in review: on a real 1.49M-packet capture every
    selected host covered the displayed windows, produced different numbers of
    forecasts, and the run aborted.)

    Which span is a real choice. Taking the busiest host's is the obvious rule
    and the wrong one — on that same capture, 145 hosts were present and
    exactly one covered the busiest host's, leaving the console a single row
    with nothing to rank it against. So several candidates are tried and the
    one held by the most hosts wins, longer span breaking a tie. A shorter
    window several hosts share says more about a capture than a long one only
    its busiest host survives.
    """
    counts = states.groupby("host_id").size().sort_values(ascending=False)
    eligible = [str(h) for h in counts[counts >= MIN_WINDOWS_PER_HOST].index]
    if not eligible:
        raise AnalysisError(
            f"No host in this capture is active for {MIN_WINDOWS_PER_HOST} "
            f"windows "
            f"({MIN_WINDOWS_PER_HOST * WINDOW_SECONDS / 60:.0f} minutes). "
            "The model reads 15 minutes of history before it will forecast, so "
            "a capture shorter than that cannot be scored."
        )

    windows_by_host = {
        host: set(int(t) for t in states.loc[states.host_id == host, "window_ts"])
        for host in eligible
    }

    best: tuple[int, int, list[str], np.ndarray] | None = None
    for definer in eligible[:TIMELINE_CANDIDATES]:
        span = _span(states, definer, max_windows)
        if len(span) < MIN_WINDOWS_PER_HOST:
            continue
        # The run-up is consumed as history; what is left is displayable.
        displayed = span[CONTEXT_LENGTH - 1:]
        wanted = set(int(t) for t in span)
        covering = [h for h in eligible if wanted.issubset(windows_by_host[h])]
        # Hosts past `max_hosts` are never rendered, so a span that gains one
        # is not worth the windows it costs. Counting them would trade real
        # timeline for a row nobody sees.
        score = (min(len(covering), max_hosts), len(displayed))
        if best is None or score > (best[0], best[1]):
            best = (len(covering), len(displayed), covering, displayed)

    if best is None:
        raise AnalysisError(
            "No host in this capture holds a continuous stretch long enough to "
            "forecast over."
        )
    _, _, covering, displayed = best
    return covering[:max_hosts], displayed


def forecast_host(predictor, g: pd.DataFrame, timeline: np.ndarray) -> list[dict]:
    """One forecast per window on the shared timeline, each from that window's
    own [L, F] history. Seeded per window so the same capture analysed twice
    gives the same page — the risk figure is a Monte Carlo estimate over
    sampled rollouts, and an unseeded one would wobble between runs with no
    change in the traffic."""
    g = g.sort_values("window_ts").reset_index(drop=True)
    feats = g[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
    ts_index = {int(t): i for i, t in enumerate(g.window_ts.to_numpy())}
    stamps = pd.to_datetime(g.window_ts.to_numpy(), unit="s", utc=True)
    host_id = str(g.host_id.iloc[0])

    out: list[dict] = []
    for i, window_ts in enumerate(timeline):
        end = ts_index[int(window_ts)]
        # select_hosts guarantees the run-up; a gap here means the two have
        # drifted apart, and skipping would silently shorten this host's series
        # so it no longer lines up with the others.
        assert end >= CONTEXT_LENGTH - 1, (
            f"{host_id} has no context before window {int(window_ts)}"
        )
        set_seed(1000 + i)
        f = predictor.forecast(
            feats[end - CONTEXT_LENGTH + 1: end + 1],
            host_id=host_id,
            origin_ts=stamps[end].to_pydatetime(),
        )
        f["tenant_id"] = TENANT
        Forecast.model_validate(f)
        # The console never reads the 45-float per-horizon state, and keeping
        # it multiplies the payload by roughly thirty.
        f["horizons"] = [
            {k: v for k, v in h.items() if k != "predicted_features"}
            for h in f["horizons"]
        ]
        out.append(f)
    return out


def analyze(pcap_path: Path, max_hosts: int, max_windows: int) -> dict:
    cfg = get_config()
    if cfg["predictor"]["impl"] != "nidra":
        raise AnalysisError(
            "predictor.impl is not 'nidra'. This endpoint exists to run the real "
            "model; it will not present the stub's output as an analysis."
        )

    with tempfile.TemporaryDirectory(prefix="nidra-pcap-") as tmp:
        packets = packets_from_pcap(pcap_path, Path(tmp))
    logger.info("analyze: %d packets extracted", len(packets))

    states = states_from_packets(packets)
    hosts, timeline = select_hosts(states, max_hosts, max_windows)
    if not hosts:
        raise AnalysisError(
            "No host in this capture spans a single continuous stretch long "
            "enough to forecast over."
        )

    predictor = load_predictor()
    forecasts: list[dict] = []
    kept: list[str] = []
    for host in hosts:
        series = forecast_host(predictor, states[states.host_id == host], timeline)
        if not series:
            continue
        kept.append(host)
        forecasts.extend(series)
    if not forecasts:
        raise AnalysisError("The capture produced no scoreable windows.")

    lengths = {len([f for f in forecasts if f["host_id"] == h]) for h in kept}
    if len(lengths) != 1:
        raise AnalysisError(
            "Hosts came out on different timelines; refusing to rank them "
            "against each other."
        )

    span = pd.to_datetime([timeline[0], timeline[-1]], unit="s", utc=True)
    return {
        "source": {
            "kind": "upload",
            "generator": "scripts/analyze_pcap.py",
            "summary": (
                f"{pcap_path.name}: {len(packets):,} packets over "
                f"{(timeline[-1] - timeline[0]) / 60:.0f} minutes "
                f"({span[0]:%Y-%m-%d %H:%M} to {span[1]:%H:%M} UTC), windowed into "
                f"{len(FEATURE_ORDER)}-feature per-host states and scored by the "
                "trained 5-seed NIDRA ensemble through the same predictor the "
                "inference worker loads. Flow features are reconstructed from the "
                "packets themselves — see the fidelity notes."
            ),
            "fidelity": FIDELITY_NOTES,
            "capture": {
                "filename": pcap_path.name,
                "packets": int(len(packets)),
                "hosts_in_capture": int(states.host_id.nunique()),
                "hosts_shown": len(kept),
                "window_count": next(iter(lengths)),
                "first_window_utc": span[0].isoformat(),
                "last_window_utc": span[1].isoformat(),
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
        "hosts": kept,
        "forecasts": forecasts,
        "explanations": {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a pcap with the trained NIDRA ensemble.")
    ap.add_argument("pcap", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write JSON here instead of stdout")
    ap.add_argument("--max-hosts", type=int, default=DEFAULT_MAX_HOSTS)
    ap.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if not args.pcap.exists():
        print(json.dumps({"error": f"no such file: {args.pcap}"}), file=sys.stdout)
        return 2
    try:
        replay = analyze(args.pcap, args.max_hosts, args.max_windows)
    except AnalysisError as exc:
        # A reason the uploader can act on, on stdout as JSON: the caller is a
        # route handler that shows this text, not a person reading a traceback.
        print(json.dumps({"error": str(exc)}))
        return 1
    payload = json.dumps(replay)
    if args.out:
        args.out.write_text(payload + "\n")
        print(
            f"wrote {args.out} ({len(payload) / 1024:.0f} KB, "
            f"{len(replay['forecasts'])} forecasts)",
            file=sys.stderr,
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
