# Training

Two stages, strictly ordered — never merged, never re-opened once frozen.
See `nidra/train/losses.py`, `train_dynamics.py`, `train_heads.py`.

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

| | `config/mvp_2017.yaml` | `config/default.yaml` |
|---|---|---|
| Samples | capped via CLI (`--max-train-samples`/`--max-val-samples`) | uncapped (~6.9M train candidates) |
| Epochs | 20 (dynamics), 20 (heads) — override with `--epochs` | 60 (dynamics), 30 (heads) |
| Seeds | `[0,1,2,3,4]` (config default; override with `--seed N` for one) | `[0,1,2,3,4]` |
| Rollout samples/member | 100 (measured trade-off — see below) | 200 |

Neither config truncates the raw CSVs (`row_cap` is unset in both) — every
row of every real day-file is read; only the tensor-construction step is
capped for the MVP profile, stratified by `risk_label` so every positive
sample is kept (`nidra.data.dataset.subsample_stratified_by_risk`).

## Stage 1 — dynamics (`train_dynamics.py`)

Encoder + transition only, multi-step unrolled Gaussian NLL
(`losses.dynamics_loss`), horizon-discounted `0.85**k`, scheduled sampling
(teacher-forcing probability 1.0 → 0.3 over the first 60% of epochs). AdamW,
cosine schedule, grad clip 1.0, early stopping on validation multi-step NLL.

```bash
# One seed:
python -m nidra.train.train_dynamics --config config/mvp_2017.yaml \
    --max-train-samples 40000 --max-val-samples 8000 --epochs 30 --seed 0
# Full ensemble (loops cfg["ensemble"]["seeds"] when --seed is omitted):
python -m nidra.train.train_dynamics --config config/mvp_2017.yaml \
    --max-train-samples 40000 --max-val-samples 8000 --epochs 30
```

The scaler is fit exactly once (on the training split only) and reused
across every seed — `prepare_training_data` loads an existing
`artifacts*/scaler/robust_scaler.joblib` if present rather than refitting.

## Stage 2 — heads (`train_heads.py`)

Loads the seed's stage-1 weights, freezes encoder + transition
(`model.freeze_dynamics()`), trains risk/stage heads on **observed states
only** (`S_t`, the last window of each sample's input — never a rollout
prediction; there is no code path that could). `pos_weight=n_neg/n_pos` for
the risk head; inverse-frequency class weights for the stage head. Freezes
the heads too when done (`model.freeze_heads()`) — nothing trains after
that point in this or any later invocation.

```bash
python -m nidra.train.train_heads --config config/mvp_2017.yaml \
    --max-train-samples 40000 --max-val-samples 8000 --seed 0
```

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
pytest tests/ -q   # 166 tests, synthetic fixtures, seconds

# Full-scale production config (what the current best results use):
python -m nidra.train.train_dynamics --config config/default.yaml --max-train-samples 500000 --max-val-samples 50000
python -m nidra.train.train_heads    --config config/default.yaml --max-train-samples 500000 --max-val-samples 50000

# Faster MVP-scale iteration:
for seed in 0 1 2 3 4; do
  python -m nidra.train.train_dynamics --config config/mvp_2017.yaml --max-train-samples 40000 --max-val-samples 8000 --epochs 30 --seed $seed
  python -m nidra.train.train_heads    --config config/mvp_2017.yaml --max-train-samples 40000 --max-val-samples 8000 --seed $seed
done
```

See `EVALUATION.md` for the evaluation commands and `REAL_DATA_RESULTS.md`
for what this produced — Run 4 and the Run 3 pooled-ensemble addendum are
the current, best-supported numbers.
