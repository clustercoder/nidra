# Training

Two stages, strictly ordered — never merged, never re-opened once frozen.
See `nidra/train/losses.py`, `train_dynamics.py`, `train_heads.py`.

> Retraining is the **only** workflow that needs the raw CIC-IDS2017
> release and tshark. The trained model ships in the repo — to forecast,
> evaluate or benchmark on a fresh clone, see "Without the raw dataset" in
> `README.md` instead.

## Prerequisites

```bash
cd ml
pip install -e .
tshark -v   # must succeed — packet extraction depends on it
```

Real CIC-IDS2017 data under `cicids2017/` (gitignored): the `TrafficLabelling`
CSV release (not `MachineLearningCVE`, which strips Source/Destination
IP/Timestamp and cannot be windowed by host/time), and the 5 raw PCAPs.
Extract packet-level features once per PCAP (cached to parquet, never
re-parsed):

```bash
python -m nidra.data.pcap_extract cicids2017/pcap/Monday-WorkingHours.pcap cicids2017/pcap/parquet/Monday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Tuesday-WorkingHours.pcap cicids2017/pcap/parquet/Tuesday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Wednesday-workingHours.pcap cicids2017/pcap/parquet/Wednesday-workingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Thursday-WorkingHours.pcap cicids2017/pcap/parquet/Thursday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Friday-WorkingHours.pcap cicids2017/pcap/parquet/Friday-WorkingHours_packets.parquet
```

Each takes roughly 20-40s per GB (10-14GB per capture) — tshark subprocess,
streamed to parquet in 200k-row chunks, never loading a full capture into
memory. Update `config/*.yaml`'s `dataset.days.<day>.packets` if you use
different filenames.

## Day-level caching

The first train/eval invocation against a given `(windowing config, dataset
paths, packet availability)` tuple pays the full CSV-load → join → windowize
→ label cost once per day-file (~24 minutes across all 8 real day-files on
this project's reference hardware) and writes each day's labelled table to
`<artifacts.processed_dir>/<day_key>__w<Δ>__m<min_windows>__<packets_tag>__cap<row_cap>.parquet`.
Every subsequent invocation (`train_dynamics`, `train_heads`, each
`run_eval.py --split ...`) reads that cache instead of recomputing — this is
what makes training a 5-seed ensemble practical. `config/mvp_2017.yaml` and
`config/default.yaml` intentionally share one `processed_dir`
(`artifacts/processed`) since the cached tables don't depend on
training-scale knobs, only on windowing/dataset config. Delete the relevant
file(s) under `artifacts/processed/` to force a recompute (e.g. after
extracting a new PCAP for a previously flow-only day — the cache key
includes the packet parquet filename, so this happens automatically the
next time you point config at a new file).

## Profiles

| | `config/mvp_2017.yaml` | `config/default.yaml` (shipped) |
|---|---|---|
| Geometry | Δ=60 s, L=30, K=6 | Δ=60 s, L=30, K=6 |
| Dynamics samples | capped via CLI | 500,000 train / 50,000 val, stratified so every positive origin is kept |
| Head samples | capped via CLI | **every** observed state of the split (2.27M train, 894k val) |
| Epochs | 20 / 20 | 60 (dynamics; the shipped run used `--epochs 24`) / 30 (heads, onset) |
| Seeds | `[0,1,2,3,4]` | `[0,1,2,3,4]` |
| Rollout samples/member | 100 | 200 (benchmarks use 60) |

Neither config truncates the raw CSVs (`row_cap` is unset) — every row of every real
day-file is read. Validation is the trailing 30% of each training day
(`splits.val_block_per_day`), nudged so no episode straddles the cut and no episode starts
within `pre_onset_margin_minutes` (30) after it; labels are recomputed inside each split.

## Preprocessing (`nidra/data/normalize.py`)

`FeatureScaler` applies a per-feature transform chosen from the feature's kind — log1p for
counts and byte/duration totals, asinh for signed dynamics, z-score for ratios, unit for
already-bounded quantities — fit on **active training rows only** (silent zero rows would
otherwise dominate every centre), then clipped. Constant and duplicate training columns are
dropped (`model_mask`); `FEATURE_ORDER` stays 45 wide. `fit` is only ever called from the
training pipeline on train rows and the result is serialised to
`artifacts/scaler/feature_scaler.json` with an audit (`preprocessing_audit.md`) that lists
each feature's transform, centre, scale and the fraction of rows clipped. Nothing at
serving time refits anything.

## Stage 1 — dynamics (`train_dynamics.py`)

Encoder + transition only. Multi-step unrolled **β-NLL** (`losses.dynamics_loss`,
`train_dynamics.beta_nll: 0.5` — each element's Gaussian NLL weighted by
`stop_grad(σ²)^β`, so high-variance transitions are not written off by the variance head
the way they were under the plain NLL; the alternatives measured are in `DECISIONS.md`
D113), horizon-discounted
`0.85**k`, scheduled sampling (teacher-forcing probability 1.0 → 0.3 over the first 60% of
epochs). AdamW, cosine schedule, grad clip 1.0. **Checkpoint selection on the free-running
validation NLL** (`selection_metric: val_free_running_nll`, D107): the rollout the model
performs at serving time, with its own predictions fed back, not the teacher-forced loss.
Free-running MSE, skill vs persistence and 90%-band coverage per k are written to every
checkpoint's metadata.

```bash
python -m nidra.train.train_dynamics --config config/default.yaml --seed 0 --epochs 24
```

The scaler is fit exactly once (on the training split) and reused across every seed.

## Stage 2 — heads (`train_heads.py`, `head_data.py`)

Loads the seed's stage-1 weights, freezes encoder + transition
(`model.freeze_dynamics()`), initialises fresh risk/stage heads and trains them on
**observed states only** — never a rollout prediction; there is no code path that could.
`head_data.build_head_arrays` turns every row of the split into one sample, so the head
sees all 179 training positives among 2.27M states rather than a subsample.

Recipe (`train_heads` block; chosen on validation, D108/D114): imbalanced BCE with
`pos_weight = n_neg / n_pos`, Gaussian input noise σ=0.3 in scaled units, separate AdamW
optimisers, and **independent selection** — the risk head on exact natural-prevalence
validation AP (`val_auc_pr_natural`), the stage head on macro-F1. A `balanced` sampler
(`pos_repeat`, `neg_ratio`, `hard_negative_fraction` of active-benign negatives) is
available and was measured worse. Heads are frozen when done (`model.freeze_heads()`).

```bash
python -m nidra.train.train_heads --config config/default.yaml --seed 0
```

## Stage 3 — onset head (`train_onset.py`, optional, D109)

An explicit supervised baseline for the forecasting question: `OnsetHead` on `S_t`
predicts P(an episode begins on this host within h minutes) for h ∈ {1, 3, 5, 10, 15,
30}, defined only for origins outside any episode (`nidra/data/onset.py`). Trained on
balanced epochs, selected on validation natural AP at `onset.selection_horizon_min`.
Saved beside the world-model weights as `onset_head_seed_<s>.pt` and scored in the
benchmark's Task B tables as `onset_head_direct`. The production run had 3 / 9 / 15 / 29 /
39 / 69 training positives at the six horizons — the dataset's precursor ceiling, and why
this head is near the prevalence floor below 30 minutes.

## GRU classifier baseline (`nidra/eval/gru_classifier.py`)

A sequence classifier on the same L-window input (GRU → sigmoid on the published label),
trained by the `gru_baseline` stage and scored as `gru_classifier` in the benchmark. It
exists to answer "does the recurrent input alone explain the world model's score" — it is
not part of the served system.

## The experiment runner (`nidra/scripts/run_experiment.py`)

Every recorded run goes through one entry point so that provenance is uniform:

```bash
python -m nidra.scripts.run_experiment --label production --seeds 0,1,2,3,4 \
    --stages dynamics,heads,onset,gru_baseline --epochs 24 --threads 6
# variants: --set key=value overrides, --init-from LABEL to reuse another run's stage-1 weights
python -m nidra.scripts.run_experiment --label var_betanll --seeds 0 --stages dynamics,heads \
    --epochs 20 --set train_dynamics.beta_nll=0.5
```

Each run writes `experiments/runs/<label>/{config.yaml,record.json,artifacts/}`; the record
carries git commit, config hash, dataset digests, geometry, per-seed histories and
checkpoint hashes. Only `--label production` writes into `artifacts/` itself (its record
still goes to `experiments/runs/production/`). Weights under `experiments/runs/*/artifacts`
are gitignored; the shipped ones under `artifacts/` are committed.

## Rollout sample count — a measured, hardware-specific trade-off

`config/mvp_2017.yaml`'s `rollout.n_samples_per_member` was set to 200
(matching `config/default.yaml`, for ~1,000 total trajectories across the
5-member ensemble) and benchmarked:

```bash
python -m nidra.serve.benchmark --weights-dir artifacts_mvp_2017/weights \
    --scaler-path artifacts_mvp_2017/scaler/robust_scaler.joblib --config config/mvp_2017.yaml
```

Result: p95=616ms, over the 300ms target. Per the documented policy (cut
samples toward 100 before cutting ensemble members —
`nidra/serve/benchmark.py`'s own docstring), it is now set to 100/member
(500 total trajectories): p95=108ms. This is a config value, not a code
change — swap it back to 200 on faster hardware.

## Reproducing this project's current results

```bash
cd ml
pytest tests/ -q

# the shipped artifacts (Run 8): 5 seeds, β-NLL dynamics, heads + onset head on every split row
python -m nidra.scripts.run_experiment --label production --seeds 0,1,2,3,4 \
    --stages dynamics,heads,onset,gru_baseline --epochs 24     # writes artifacts/ directly
```

Then `EVALUATION.md` for the benchmark (validation selects the operating point; test and
holdout only load it) and `REAL_DATA_RESULTS.md` §Run 8 for what this produced.
