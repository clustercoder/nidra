"""A clone of this repo carries the committed windowed parquet under
artifacts/processed/ but NOT the ~50GB raw CIC-IDS2017 release. These tests
pin the contract that makes that clone runnable: build_all_splits must serve
a day from its committed cache without ever touching a raw CSV, and the
cache key must be derived from the config-DECLARED packet file rather than
from what happens to be present on the local disk — otherwise a machine
without the packet parquet computes a `flowonly` key, misses every committed
file, and silently falls back to skipping the day.

The stale-cache protection this keys on is directional and must survive: a
flow-only table must never be served under a packet-enriched key.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nidra.train import pipeline as pipeline_mod
from nidra.data.windowize import CIC2017_TIMEBASE_TAG
from nidra.train.pipeline import FLOW_ONLY_TAG, build_all_splits, day_cache_path, declared_packets_tag


def _cfg(tmp_path, *, packets_dir=None):
    return {
        "dataset": {
            "cic2017_flow_dir": str(tmp_path / "raw"),
            "packets_dir": str(packets_dir or (tmp_path / "packets")),
            "mvp_row_cap_per_day": None,
            "days": {
                "monday": {"file": "Monday.csv", "role": "train", "packets": "Monday_packets.parquet"},
                "friday_ddos": {"file": "Friday-DDos.csv", "role": "test", "packets": "Friday_packets.parquet"},
            },
        },
        "windowing": {"window_seconds": 30, "min_windows_per_host": 36},
        "splits": {
            "train_days": ["monday"],
            "test_days": ["friday_ddos"],
            "holdout_days": [],
            "val_fraction_of_train_time": 0.2,
        },
        "artifacts": {"processed_dir": str(tmp_path / "processed")},
    }


def _write_cache(cfg, tmp_path, day_key, tag, n_rows=2):
    """Write a stand-in cached labelled table under the given packet tag."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    path = day_cache_path(processed, day_key, cfg["windowing"], cfg["dataset"]["mvp_row_cap_per_day"], tag)
    pd.DataFrame({
        "host_id": ["h1"] * n_rows,
        "window_ts": list(range(0, 30 * n_rows, 30)),
        "stage_label": ["benign"] * n_rows,
        "risk_label": [0] * n_rows,
    }).to_parquet(path)
    return path


@pytest.fixture
def captured_day_tables(monkeypatch):
    """build_splits is not what these tests are about — capture the day
    tables build_all_splits assembled and return a dummy SplitResult."""
    seen = {}

    def fake_build_splits(day_tables, **kwargs):
        seen.update(day_tables)
        return "split-result-sentinel"

    monkeypatch.setattr(pipeline_mod, "build_splits", fake_build_splits)
    monkeypatch.setattr(pipeline_mod, "assert_no_temporal_overlap", lambda splits: None)
    monkeypatch.setattr(pipeline_mod, "assert_no_episode_leakage", lambda splits: None)
    return seen


@pytest.fixture
def no_raw_loads(monkeypatch):
    """Any attempt to read a raw CSV is a hard failure — these tests assert
    the cache path is taken, and a silent recompute would otherwise pass."""
    def explode(*args, **kwargs):
        raise AssertionError("raw CSV load attempted, but a committed cache should have served this day")

    monkeypatch.setattr(pipeline_mod, "load_cicflowmeter_csv", explode)
    return explode


def test_declared_tag_ignores_local_disk():
    assert declared_packets_tag({"file": "Monday.csv", "packets": "Monday_packets.parquet"}) == "Monday_packets.parquet"
    assert declared_packets_tag({"file": "Monday.csv"}) == FLOW_ONLY_TAG


def test_cache_key_matches_committed_naming_convention(tmp_path):
    """The committed files are named e.g.
    monday__w30__m36__Monday-WorkingHours_packets.parquet__capNone__utc12h.parquet —
    if this format drifts, every committed cache silently stops matching."""
    path = day_cache_path(tmp_path, "monday", {"window_seconds": 30, "min_windows_per_host": 36},
                          None, "Monday-WorkingHours_packets.parquet")
    assert path.name == (
        "monday__w30__m36__Monday-WorkingHours_packets.parquet__capNone__utc12h.parquet"
    )


def test_cache_key_carries_the_timebase_tag():
    """The flow timebase is not derivable from anything else in the key, and
    getting it wrong changes every packet-derived feature in the table (the
    CSV clock defects — see windowize.parse_cic_timestamp — left 11 of 45
    features ~always zero). A cache written under the old timebase must not
    be served to a run using the new one, so the tag is part of the name."""
    key = {"window_seconds": 30, "min_windows_per_host": 36}
    path = day_cache_path("/p", "monday", key, None, "M.parquet")
    assert f"__{CIC2017_TIMEBASE_TAG}." in path.name
    # And the pre-fix naming must no longer be produced by anything.
    assert path.name != "monday__w30__m36__M.parquet__capNone.parquet"


def test_committed_cache_serves_day_with_no_raw_csv_and_no_packet_parquet(
    tmp_path, captured_day_tables, no_raw_loads
):
    """The judge's clone: artifacts/processed/ present, raw release absent."""
    cfg = _cfg(tmp_path)
    _write_cache(cfg, tmp_path, "monday", "Monday_packets.parquet")
    _write_cache(cfg, tmp_path, "friday_ddos", "Friday_packets.parquet")

    assert build_all_splits(cfg) == "split-result-sentinel"
    assert set(captured_day_tables) == {"monday", "friday_ddos"}
    assert len(captured_day_tables["monday"]) == 2


def test_day_with_neither_cache_nor_csv_is_skipped(tmp_path, captured_day_tables, no_raw_loads):
    cfg = _cfg(tmp_path)
    _write_cache(cfg, tmp_path, "monday", "Monday_packets.parquet")
    # friday_ddos gets no cache and no CSV -> skipped, not a crash
    build_all_splits(cfg)
    assert set(captured_day_tables) == {"monday"}


def test_flow_only_cache_is_not_served_under_a_declared_packet_key(
    tmp_path, captured_day_tables, no_raw_loads
):
    """Directional staleness guard: a day declaring packets must NOT be
    satisfied by a flow-only cache, even though no raw CSV exists to
    recompute from. Skipping is correct; serving degraded features under a
    full-feature key is not."""
    cfg = _cfg(tmp_path)
    _write_cache(cfg, tmp_path, "monday", FLOW_ONLY_TAG)
    build_all_splits(cfg)
    assert set(captured_day_tables) == set(), "flow-only cache must not satisfy a packet-declaring day"
