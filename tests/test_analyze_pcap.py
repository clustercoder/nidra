"""The uploaded-capture path: packets -> flows -> states -> forecasts.

The failures this pins are the ones a user would otherwise see as a page of
plausible-looking numbers. An uploaded capture has no labels and no
CICFlowMeter CSV, so nothing downstream can tell that the analysis was run on
too little traffic, on hosts that do not share a timeline, or through the stub
predictor. Each of those has to fail loudly here instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import analyze_pcap as ap  # noqa: E402
from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER  # noqa: E402


def _states(hosts: dict[str, list[int]]) -> pd.DataFrame:
    """A minimal state table: {host: [window_ts, ...]}, features all 1.0."""
    rows = []
    for host, stamps in hosts.items():
        for ts in stamps:
            row = {f: 1.0 for f in FEATURE_ORDER}
            row.update(host_id=host, window_ts=ts, is_active=1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _packets(rows):
    cols = ["frame_time_epoch", "ip_src", "src_port", "ip_dst", "dst_port",
            "ip_proto", "payload_len", "frame_len", "tcp_flags"]
    return pd.DataFrame(rows, columns=cols)


def test_a_capture_shorter_than_the_context_window_is_refused_not_scored():
    """The model reads 15 minutes of history before it will forecast. A
    shorter capture has to be refused with that reason: silently scoring it
    from a padded context would produce a risk number with nothing behind it.
    """
    short = _states({"10.0.0.1": [30 * i for i in range(CONTEXT_LENGTH - 5)]})
    with pytest.raises(ap.AnalysisError, match="15 minutes of history"):
        ap.select_hosts(short, max_hosts=6, max_windows=96)


def test_only_hosts_spanning_the_whole_timeline_are_shown():
    """The console ranks hosts against each other at one index. A host present
    for a different stretch would be compared at an index that is a different
    instant for it, so it is dropped rather than shown misaligned."""
    early = [30 * i for i in range(ap.MIN_WINDOWS_PER_HOST + 4)]
    # A host active only later on: no shared stretch is long enough to forecast
    # over, so whichever timeline wins, this one cannot be on it.
    late = [30 * (i + 500) for i in range(ap.MIN_WINDOWS_PER_HOST + 4)]
    states = _states({"10.0.0.1": early, "10.0.0.2": early, "10.0.0.3": late})
    hosts, timeline = ap.select_hosts(states, max_hosts=6, max_windows=96)
    assert "10.0.0.3" not in hosts
    assert set(hosts) == {"10.0.0.1", "10.0.0.2"}
    assert list(timeline) == early[CONTEXT_LENGTH - 1:]


def test_the_timeline_is_chosen_to_show_the_most_hosts_not_the_longest_one():
    """Taking the busiest host's whole span as the timeline is what a real
    capture punishes: on 1.49M packets of CIC-IDS2017 Friday, 145 hosts were
    present and exactly one covered the busiest host's 44 windows, so the
    console got a single row and nothing to rank it against. A shorter window
    that several hosts share is worth more than a long one only the busiest
    host survives."""
    long_span = [30 * i for i in range(60)]
    shared = long_span[-(CONTEXT_LENGTH + 5):]
    states = _states({
        "10.0.0.1": long_span,   # busiest by far
        "10.0.0.2": shared,
        "10.0.0.3": shared,
        "10.0.0.4": shared,
    })
    hosts, timeline = ap.select_hosts(states, max_hosts=6, max_windows=96)
    assert set(hosts) == {"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"}
    # Four hosts over the 6 windows `shared` can display beats one host over
    # the 31 the busiest could.
    assert list(timeline) == shared[CONTEXT_LENGTH - 1:]


def test_hosts_past_the_display_cap_do_not_buy_a_shorter_timeline():
    """Only `max_hosts` rows are rendered, so a span covered by more hosts than
    that is not better — and trading away windows to reach an extra row nobody
    sees makes the console worse."""
    long_span = [30 * i for i in range(80)]
    short = long_span[-(CONTEXT_LENGTH + 2):]
    states = _states({
        **{f"10.0.0.{i}": long_span for i in range(1, 4)},      # 3 hosts, long
        **{f"10.0.1.{i}": short for i in range(1, 6)},          # 5 more, short
    })
    # Cap at 3: the long span already fills every row on offer.
    hosts, timeline = ap.select_hosts(states, max_hosts=3, max_windows=96)
    assert len(hosts) == 3
    assert list(timeline) == long_span[CONTEXT_LENGTH - 1:]


def test_a_longer_timeline_wins_when_the_host_count_ties():
    """Between two timelines the same hosts cover, the longer one shows more
    of the capture."""
    long_span = [30 * i for i in range(60)]
    states = _states({"10.0.0.1": long_span, "10.0.0.2": long_span})
    hosts, timeline = ap.select_hosts(states, max_hosts=6, max_windows=96)
    assert set(hosts) == {"10.0.0.1", "10.0.0.2"}
    assert list(timeline) == long_span[CONTEXT_LENGTH - 1:]


def test_the_timeline_is_capped_from_the_end_so_the_newest_traffic_is_shown():
    long = [30 * i for i in range(200)]
    _, timeline = ap.select_hosts(_states({"10.0.0.1": long}), max_hosts=6, max_windows=40)
    assert list(timeline) == long[-40:]


def test_a_host_needs_its_context_BEFORE_the_timeline_not_just_across_it():
    """Covering the displayed windows is not enough — each one is forecast
    from the 30 windows before it, so a host that merely starts at the
    timeline's first window can forecast none of the early ones.

    Found end-to-end rather than in review: on a real 1.49M-packet capture
    every selected host covered the timeline, then produced a different number
    of forecasts, and the run aborted with "hosts came out on different
    timelines". The timeline is therefore the run-up plus the displayed part,
    and a host has to hold all of it.
    """
    L = CONTEXT_LENGTH
    full = [30 * i for i in range(L + 20)]
    states = _states({
        "10.0.0.1": full,
        "10.0.0.2": full,
        # Starts where the others' displayable stretch starts: covers every
        # displayed window, holds no history to forecast the first ones from.
        "10.0.0.3": full[L - 1:],
    })
    hosts, timeline = ap.select_hosts(states, max_hosts=6, max_windows=96)
    assert "10.0.0.3" not in hosts
    # The displayed timeline is what remains after the context run-up.
    assert list(timeline) == full[L - 1:]


def test_every_selected_host_forecasts_every_displayed_window():
    """The invariant the above exists to protect: equal-length series, so an
    index is the same instant for every host on screen."""
    L = CONTEXT_LENGTH
    full = [30 * i for i in range(L + 20)]
    states = _states({"10.0.0.1": full, "10.0.0.2": full})
    hosts, timeline = ap.select_hosts(states, max_hosts=6, max_windows=96)

    original = ap.Forecast.model_validate
    ap.Forecast.model_validate = staticmethod(lambda _f: None)
    try:
        lengths = {
            len(ap.forecast_host(_StubPredictor(), _states({h: full}), timeline))
            for h in hosts
        }
    finally:
        ap.Forecast.model_validate = original
    assert lengths == {len(timeline)}


class _StubPredictor:
    def forecast(self, states, host_id, origin_ts):
        return {"host_id": host_id, "origin_ts": origin_ts.isoformat(), "horizons": [],
                "model_version": "t", "lead_time_s": None, "observed_risk": 0.0,
                "observed_stage": {}, "top_signals": [], "driving_window": 0}


def test_a_capture_of_only_lone_packets_is_refused():
    """Every conversation a single packet: flow assembly correctly publishes
    nothing, and the refusal has to say so rather than produce an empty page."""
    lone = _packets([
        (float(i), f"10.0.0.{i}", 1, "10.0.0.250", 80, 6, 0, 54, "0x0002")
        for i in range(1, 40)
    ])
    with pytest.raises(ap.AnalysisError, match="single packet"):
        ap.states_from_packets(lone)


def test_forecasts_are_seeded_so_the_same_capture_reads_the_same_twice():
    """Risk is a Monte Carlo estimate over sampled rollouts. Without a per
    window seed, re-analysing an unchanged capture moves every number on the
    page, which reads as the traffic having changed."""
    calls: list[np.ndarray] = []

    class Recorder:
        def forecast(self, states, host_id, origin_ts):
            calls.append(float(np.random.rand()))
            return {
                "host_id": host_id, "origin_ts": origin_ts.isoformat(),
                "horizons": [], "model_version": "test", "lead_time_s": None,
                "observed_risk": 0.0, "observed_stage": {}, "top_signals": [],
                "driving_window": 0,
            }

    stamps = [30 * i for i in range(CONTEXT_LENGTH + 3)]
    g = _states({"10.0.0.1": stamps})
    timeline = np.array(stamps[CONTEXT_LENGTH - 1:])

    # Forecast.model_validate would reject the recorder's stub dict; this test
    # is about the seeding, so the validator is not what is under test here.
    original = ap.Forecast.model_validate
    ap.Forecast.model_validate = staticmethod(lambda _f: None)
    try:
        ap.forecast_host(Recorder(), g, timeline)
        first = list(calls)
        calls.clear()
        ap.forecast_host(Recorder(), g, timeline)
    finally:
        ap.Forecast.model_validate = original

    assert first == calls != []


def test_every_forecast_gets_a_full_context_and_none_is_silently_dropped():
    """Each displayed window is forecast from exactly CONTEXT_LENGTH windows.

    A window with no history behind it is an error, not something to skip:
    skipping shortens this host's series so it stops lining up with the other
    hosts', and the mismatch surfaces far from its cause."""
    seen: list[int] = []

    class Counter(_StubPredictor):
        def forecast(self, states, host_id, origin_ts):
            seen.append(len(states))
            return super().forecast(states, host_id, origin_ts)

    stamps = [30 * i for i in range(CONTEXT_LENGTH + 4)]
    states = _states({"10.0.0.1": stamps})
    original = ap.Forecast.model_validate
    ap.Forecast.model_validate = staticmethod(lambda _f: None)
    try:
        out = ap.forecast_host(Counter(), states, np.array(stamps[CONTEXT_LENGTH - 1:]))
        with pytest.raises(AssertionError, match="no context before"):
            # The run-up itself is not displayable.
            ap.forecast_host(Counter(), states, np.array(stamps))
    finally:
        ap.Forecast.model_validate = original

    assert len(out) == 5                      # windows L-1 .. L+3
    assert set(seen) == {CONTEXT_LENGTH}      # every one gets a full context


def test_the_fidelity_notes_travel_with_every_analysis():
    """A page built from reconstructed flows must not look like the labelled
    replay. The notes are data on the payload, not copy in the template, so
    the console cannot render an upload without them."""
    assert ap.FIDELITY_NOTES
    joined = " ".join(ap.FIDELITY_NOTES).lower()
    assert "urg_ratio" in joined
    assert "no attack labels" in joined


def test_the_payload_carries_everything_the_console_reads(monkeypatch, tmp_path):
    """The analyser's output is rendered by the same console components as the
    committed replay, so a missing key is a blank panel rather than an error.

    `web/src/lib/demo-replay.ts` is the contract; this pins the upload-specific
    half of it, which nothing else covers — the per-forecast fields are already
    validated against the API's own `Forecast` model inside `forecast_host`.
    """
    L = CONTEXT_LENGTH
    stamps = [30 * i for i in range(L + 8)]
    states = _states({"10.0.0.1": stamps, "10.0.0.2": stamps})

    monkeypatch.setattr(ap, "get_config", lambda: {
        "predictor": {"impl": "nidra"}, "risk_threshold": 0.75, "lead_time_m": 2,
    })
    monkeypatch.setattr(ap, "packets_from_pcap", lambda path, workdir: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(ap, "states_from_packets", lambda packets: states)
    monkeypatch.setattr(ap, "load_predictor", lambda: _StubPredictor())
    monkeypatch.setattr(ap.Forecast, "model_validate", staticmethod(lambda _f: None))

    payload = ap.analyze(tmp_path / "capture.pcap", max_hosts=6, max_windows=96)

    # DemoWorkspace ranks hosts against each other at one index, so every host
    # it shows must carry the same number of windows.
    per_host = {h: sum(1 for f in payload["forecasts"] if f["host_id"] == h)
                for h in payload["hosts"]}
    assert len(set(per_host.values())) == 1, per_host
    assert set(per_host) == {"10.0.0.1", "10.0.0.2"}

    for key in ("window_delta", "context_L", "horizon_K", "n_features",
                "risk_threshold", "lead_time_m"):
        assert key in payload["model"]["config"], key

    source = payload["source"]
    assert source["kind"] == "upload"
    assert source["fidelity"] == ap.FIDELITY_NOTES
    for key in ("filename", "packets", "hosts_in_capture", "hosts_shown",
                "window_count", "first_window_utc", "last_window_utc"):
        assert key in source["capture"], key
    assert source["capture"]["window_count"] == next(iter(per_host.values()))

    # An upload has no ground truth. The replay's episode block reports
    # precision and recall against labels; carrying it here would let the page
    # claim a score for a capture nothing was ever scored against.
    assert "episode" not in source
    assert "measured_on_this_episode" not in json.dumps(payload)


def test_the_stub_predictor_is_refused_outright(monkeypatch, tmp_path):
    """Presenting the placeholder's output as an analysis of someone's capture
    is the one failure mode worse than refusing."""
    monkeypatch.setattr(ap, "get_config", lambda: {"predictor": {"impl": "stub"}})
    with pytest.raises(ap.AnalysisError, match="real model"):
        ap.analyze(tmp_path / "capture.pcap", max_hosts=6, max_windows=96)
