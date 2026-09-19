# experiments/

Historical records that must not be overwritten. Each entry is frozen at
the commit that produced it; later work supersedes, never edits, these files.

| Entry | What it is | Commit / tag |
|---|---|---|
| `BASELINE_MANIFEST_delta30_run7.json` | Provenance of the published Δ=30 s Run 7 artifacts: config hash, checkpoint and scaler digests, processed-table digests, pooling, calibration, the published test/holdout metrics | tag `baseline-delta30-run7` (36801e5) |
| `2026-09-19_audit_delta60_seed0/` | The single-seed Δ=60 s retrain from the first-principles audit (`ml/reports/NIDRA_REEVALUATION_2026-09-19.md` §18). Unmodified training code, `window_seconds: 60`, L=30, K=6, identity scaler, seed 0. `retrain60_results.json` carries dynamics/heads metadata and the attribution table; `model_seed_0_delta60.pt` is the checkpoint; `config_w60.yaml` is the exact config (artifact paths pointed at a scratch directory at the time) | 36801e5 + config diff |

Records written after the Δ=60 rebuild land in `runs/<date>_<label>/` with a
`record.json` produced by `nidra.utils.provenance.experiment_record`, which
carries: git commit, config hash, dataset version (day-file size/mtime and
processed-table digest), window size, history length, forecast horizon,
seed(s), checkpoint digests, evaluation split, threshold, pooling statistic,
calibration, and the metrics.

Re-create a baseline manifest (refuses to overwrite without `--force`):

```bash
python -m nidra.scripts.write_baseline_manifest --config config/default.yaml --out experiments/BASELINE_MANIFEST_<label>.json --label <label>
```
