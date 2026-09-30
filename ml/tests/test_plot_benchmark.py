"""The benchmark figures are rendered from the same record the tables read;
each writer must produce its PNG from a record of that shape."""

from __future__ import annotations

from nidra.scripts import plot_benchmark as pb
from tests.test_report_tables import _record


def _figure_record() -> dict:
    m = _record()
    m["state_forecast"]["skill_vs_persistence_by_k"] = {"world_model_deterministic": [0.5, 0.4, 0.3], "ridge_two_lag": [0.6, 0.5, 0.4]}
    rel = [{"bin_center": 0.1, "observed_frequency_natural": 0.05}, {"bin_center": 0.9, "observed_frequency_natural": None}]
    m["calibration_published_label"] = {"raw": {"reliability": rel, "brier_natural": 0.001},
                                        "calibrated": {"reliability": rel, "brier_natural": 0.0009}}
    return m


def test_every_figure_is_written(tmp_path):
    m = _figure_record()
    for fn, name in ((pb.systems_ap, "systems_ap"), (pb.horizon, "horizon"), (pb.state_skill, "state_skill"), (pb.reliability, "reliability")):
        fn(m, "test", tmp_path)
        out = tmp_path / f"{name}_test.png"
        assert out.exists() and out.stat().st_size > 0, name
