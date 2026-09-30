# Published evaluation records

Written by `python -m nidra.eval.benchmark` (Run 8, 2026-09-20) against the artifacts in
`../weights` and `../scaler`:

| path | what |
|---|---|
| `val/benchmark.json` | validation run that **selected** the operating point (`../weights/operating_point.json`) |
| `test/benchmark.json` | Friday, scored with the frozen operating point |
| `holdout/benchmark.json` | Thursday (never trained on), scored with the frozen operating point |
| `test_K10/benchmark.json` | horizon-extension check (K=10, label threshold 10 windows) on Friday |
| `*/benchmark_scores.npz` | every system's per-row score, labels, weights, strata and episode keys — enough to recompute any number in the JSON |

Each JSON starts with a provenance block (git commit, config hash, dataset digests,
geometry, seeds, checkpoint sha256, operating point, caps). Tables and figures generated
from these files: `../../reports/run8/`. The legacy Run 1–7 files (`baselines.json`,
`ablations.json`, `calibration.json`, `lead_time.json`, written by `run_eval.py` at
Δ=30 s on a balanced subsample) are preserved under the git tag `baseline-delta30-run7`
and are not comparable with these.
