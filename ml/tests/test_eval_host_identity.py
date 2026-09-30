"""The benchmark's host/timing decomposition.

Both corpora put every attack group's positives on a single host, so an
aggregate AP cannot separate "recognised this family's behaviour" from
"recognised this machine". The decomposition is computed inside the benchmark
because the per-row scores, labels and host ids are all in hand there and are
not written to the metrics JSON, so it cannot be recovered afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from nidra.eval.benchmark import _host_identity


class _Ev:
    def __init__(self, hosts):
        self.arrays = type("A", (), {"host_id": np.array(hosts, dtype=object)})()
        self._n = len(hosts)

    def __len__(self):
        return self._n


def test_decomposes_every_system():
    ev = _Ev(["a"] * 6 + ["b"] * 6)
    y = np.array([1, 1, 0, 0, 0, 0] + [0] * 6)
    scores = {"world_model": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4] + [0.01] * 6),
              "constant": np.full(12, 0.5)}
    out = _host_identity(ev, scores, y)
    assert set(out) == {"world_model", "constant"}
    assert out["world_model"]["n_positive_hosts"] == 1
    # 0.8: under the host mean the two positives tie with host a's four
    # negatives and beat host b's six — what "the ranking is the host" looks
    # like when the infected host is mostly benign.
    assert out["world_model"]["host_mean_roc"] == pytest.approx(0.8)
    assert out["constant"]["roc"] == pytest.approx(0.5)


def test_within_host_columns_ignore_the_uninfected_hosts():
    """The within-host question is asked on the hosts that carry positives;
    including the rest would just re-measure host identity."""
    ev = _Ev(["a"] * 4 + ["b"] * 100)
    y = np.array([1, 1, 0, 0] + [0] * 100)
    scores = {"m": np.array([0.9, 0.8, 0.1, 0.2] + [0.5] * 100)}
    out = _host_identity(ev, scores, y)["m"]
    assert out["within_host_rows"] == 4
    assert out["within_host_prevalence"] == pytest.approx(0.5)
    assert out["within_host_roc"] == pytest.approx(1.0)      # perfect ordering inside host a


def test_a_system_it_cannot_decompose_is_dropped_not_raised():
    """One class means nothing to decompose; that must not fail the whole
    evaluation run."""
    ev = _Ev(["a", "a", "b", "b"])
    assert _host_identity(ev, {"world_model": np.arange(4.0)}, np.zeros(4, dtype=int)) == {}


def test_a_host_identity_system_is_flagged():
    """Positives lifted as a block but unordered inside the host."""
    ev = _Ev(["infected"] * 4 + ["other"] * 20)
    y = np.array([1, 1, 0, 0] + [0] * 20)
    scores = {"m": np.array([0.3, 0.2, 0.9, 0.8] + [0.01] * 20)}
    out = _host_identity(ev, scores, y)["m"]
    assert out["is_host_identity"] is True
    assert out["within_host_lift"] < 1.1


def test_every_reported_field_is_a_plain_float_for_json():
    ev = _Ev(["a"] * 4 + ["b"] * 4)
    out = _host_identity(ev, {"m": np.linspace(0, 1, 8)}, np.array([1, 1, 0, 0, 0, 0, 0, 0]))["m"]
    import json
    json.dumps(out)                                  # raises on numpy scalars
    assert isinstance(out["within_host_roc"], float)
    assert isinstance(out["n_positive_hosts"], int)


def test_a_broken_system_does_not_fail_the_evaluation(caplog):
    """This block runs unattended across the cross-dataset matrix. A missing
    column is recoverable; a failed benchmark run is hours."""
    ev = _Ev(["a", "a", "b", "b"])
    y = np.array([1, 0, 0, 0])
    out = _host_identity(ev, {"ok": np.array([0.9, 0.1, 0.2, 0.3]),
                              "wrong_length": np.array([0.1, 0.2])}, y)
    assert set(out) == {"ok"}
