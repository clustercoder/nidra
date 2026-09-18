"""Latency benchmark test. Uses the tiny synthetic model from
tests/conftest.py — this measures that the benchmark harness itself works
and reports real numbers for a tiny model; it does NOT stand in for the
real latency target (K=6, 5 members, 200 samples/member on the full-size
architecture), which must be measured against real trained artifacts via
`python -m nidra.serve.benchmark`.
"""

from __future__ import annotations

from nidra.serve.benchmark import benchmark


def test_benchmark_runs_and_reports_percentiles(trained_predictor):
    predictor, _ = trained_predictor
    result = benchmark(predictor, n_calls=5)
    assert result["n_calls"] == 5
    assert result["mean_ms"] > 0
    assert result["median_ms"] > 0
    assert result["p95_ms"] >= result["median_ms"]
    assert result["max_ms"] >= result["p95_ms"]
