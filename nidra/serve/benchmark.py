"""Latency benchmark for NidraPredictor.forecast(). Target: under 300 ms
for K=6, 5 ensemble members, 200 samples/member, on CPU
(IMPLEMENTATION-ML.md §7 / §25). If over target, cut stochastic samples
toward 100 before touching the ensemble size — see CLAUDE.md §25.

Usage:
    python -m nidra.serve.benchmark --weights-dir artifacts/weights \\
        --scaler-path artifacts/scaler/robust_scaler.joblib --config config/default.yaml
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER
from nidra.serve.predictor import NidraPredictor


def benchmark(predictor: NidraPredictor, n_calls: int = 20) -> dict:
    rng = np.random.default_rng(0)
    states = rng.standard_normal((CONTEXT_LENGTH, len(FEATURE_ORDER))).astype("float32")

    # warm-up (first call pays for lazy CUDA/thread-pool init if any)
    predictor.forecast(states, host_id="bench", origin_ts=datetime.now(timezone.utc))

    latencies_ms = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        predictor.forecast(states, host_id="bench", origin_ts=datetime.now(timezone.utc))
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "n_calls": n_calls,
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "max_ms": float(np.max(latencies_ms)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark NidraPredictor.forecast() latency.")
    parser.add_argument("--weights-dir", type=str, required=True)
    parser.add_argument("--scaler-path", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-calls", type=int, default=20)
    args = parser.parse_args()

    predictor = NidraPredictor(weights_dir=args.weights_dir, scaler_path=args.scaler_path, config_path=args.config)
    result = benchmark(predictor, n_calls=args.n_calls)
    target = predictor.cfg["serving"]["latency_target_ms"]
    print(result)
    print(f"target: {target} ms — {'PASS' if result['p95_ms'] <= target else 'OVER TARGET'}")


if __name__ == "__main__":
    main()
