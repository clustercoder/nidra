# experiments/

Historical records that must not be overwritten. Each entry is frozen at
the commit that produced it; later work supersedes, never edits, these files.

| Entry | What it is | Commit / tag |
|---|---|---|
| `BASELINE_MANIFEST_delta30_run7.json` | Provenance of the published Δ=30 s Run 7 artifacts: config hash, checkpoint and scaler digests, processed-table digests, pooling, calibration, the published test/holdout metrics | tag `baseline-delta30-run7` (36801e5) |
| `2026-09-19_audit_delta60_seed0/` | The single-seed Δ=60 s retrain from the first-principles audit (`ml/reports/NIDRA_REEVALUATION_2026-09-19.md` §18). Unmodified training code, `window_seconds: 60`, L=30, K=6, identity scaler, seed 0. `retrain60_results.json` carries dynamics/heads metadata and the attribution table; `model_seed_0_delta60.pt` is the checkpoint; `config_w60.yaml` is the exact config (artifact paths pointed at a scratch directory at the time) | 36801e5 + config diff |

Records written after the Δ=60 rebuild land in `runs/<label>/` (written by
`nidra.scripts.run_experiment`): `config.yaml` is the exact configuration the run
loaded (its hash is `_config_hash`), `record.json` is produced by
`nidra.utils.provenance.experiment_record` and carries git commit, config hash,
dataset version (day-file size/mtime and processed-table digest), geometry,
seed(s), checkpoint digests, per-seed training histories and the stage metrics;
`artifacts/` holds that run's weights, scaler, metadata and `metrics/<split>/
benchmark.json`. Weights and per-row score dumps under `runs/` are gitignored
(reproducible from the record); everything else is committed. Only
`--label production` writes to `ml/artifacts/` itself.

| Run label(s) | What it is | Outcome |
|---|---|---|
| `geomA_L15K3`, `geomB_L30K6` | geometry comparison at Δ=60, seed 0 | D112: L=30/K=6 |
| `headsA_*`, `headsB_imb_noise` | head-recipe comparison on one checkpoint (`--init-from`) | D108: imbalanced BCE + input noise, natural-AP selection |
| `var_*` (+ `_h2` uniform head pass) | Stage-5 dynamics-loss variants at L=15/K=3, 20 epochs | D113: β-NLL 0.5 kept |
| `production` | the shipped Δ=60 five-seed ensemble (record only; artifacts in `ml/artifacts/`) | Run 8 in `REAL_DATA_RESULTS.md` |
| `lodo_without_wednesday`, `lodo_without_tuesday` | leave-one-day-out retrains, seed 0 | Run 8 §8.6 |

`baseline_delta30_run7/` is a disk copy of the Δ=30 weights and scaler (gitignored; the
same bytes are in git under the tag).

Re-create a baseline manifest (refuses to overwrite without `--force`):

```bash
python -m nidra.scripts.write_baseline_manifest --config config/default.yaml --out experiments/BASELINE_MANIFEST_<label>.json --label <label>
```
