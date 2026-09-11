# Running the Full-Scale Production Training — A Beginner's Guide

This guide walks you through running NIDRA's **full-scale** training
(`config/default.yaml`: 5 seeds, 60 epochs for the dynamics stage, 30
epochs for the heads stage, on the *entire* real dataset with no sample
cap) instead of the smaller "MVP" configuration
(`config/mvp_2017.yaml`) that's been used for development so far.

**Written for someone with no machine learning background.** Every step
tells you what it does and why, not just what to type. If you just want
the commands with no explanation, skip to the "Quick reference" box at the
end.

---

## 0. Should you actually run this?

Before you spend hours of compute time, understand the trade-off:

- **The MVP config** (already run, results in `REAL_DATA_RESULTS.md`)
  trains on a *sample* of the data (40,000 training examples, capped) for
  30 epochs. It finishes in well under an hour on a normal laptop.
- **The full config** trains on *every single one* of roughly 6.9 million
  candidate training examples, for 60 epochs, five separate times (once
  per random seed). This is a fundamentally bigger job — expect it to take
  **hours, not minutes**, on a CPU-only laptop. There is no way to know the
  exact time in advance without running a short timing test first (Step 4
  below tells you how).

Why bother at all, if the MVP run already produced results? Because the
MVP run's whole point was to prove the pipeline works correctly and get a
first honest read on the model's behavior *quickly*. The full run is what
you'd actually want to report as your "real" final numbers, because more
training data and more epochs generally make a neural network's learned
patterns more reliable and less noisy — assuming nothing is broken, more
data almost always helps.

**You do not need to change any code to do this.** Every command below
just points at a different configuration file (`config/default.yaml`
instead of `config/mvp_2017.yaml`) — the data pipeline, model, and
evaluation code are identical.

---

## 1. What you need before starting

### 1.1 A working NIDRA setup

If you've already run the MVP training/evaluation successfully, you have
everything you need already — skip to Step 2. If not:

```bash
cd ml
pip install -e .
tshark -v   # must print a version number, not an error
```

If `tshark -v` fails, install Wireshark (which bundles `tshark`) for your
OS first — the ML pipeline cannot extract packet-level features without
it.

### 1.2 The dataset

You need the complete CIC-IDS2017 dataset already sitting on disk:
- The 8 CSV files (the "TrafficLabelling" release specifically — not
  "MachineLearningCVE," which is missing columns this project needs).
- All 5 raw PCAP files, **already extracted to parquet** via
  `nidra.data.pcap_extract` (see `TRAINING.md`'s "Prerequisites" section
  for the exact extraction commands, if you haven't done this yet).

If you already ran the MVP config successfully, this is all in place — the
full config uses the *exact same* `dataset:` section, so nothing about the
raw data needs to change.

### 1.3 Disk space and memory — read this one carefully, it changes the command you'll actually run

- The cached, pre-processed data tables (`artifacts/processed/`) are
  already built if you've run anything before — expect a few hundred MB
  there.
- Each trained model checkpoint is small (a few MB) — 5 seeds is nothing to
  worry about, disk-wise.
- **Memory — corrected from an earlier version of this guide, which badly
  underestimated this.** The full training split has ~6.9 million candidate
  samples. Each one is a `[30 context windows, 45 features]` block of
  32-bit numbers — about 5.4 KB — plus a smaller future-window block.
  Multiply that out: **holding all 6.9M of them in memory at once takes on
  the order of 35+ GB**, before your operating system, Python, or PyTorch
  have used a single byte for anything else. A 16GB (or even 32GB) laptop
  will have its training process killed by the operating system if you try
  to build that many samples at once — this is exactly what "the process
  got killed with no error message" looks like, and it's not a crash in the
  usual sense, it's the OS protecting itself from running out of memory.
  (An earlier version of this document said "several GB" for this step —
  that was measured wrong; the real number is an order of magnitude
  higher, and the fix below exists because of that.)
- **The fix**: `--max-train-samples` / `--max-val-samples` cap how many of
  those 6.9M candidates are actually turned into training data, and the
  code was changed (this session) to apply that cap *before* building the
  expensive `[30,45]` blocks, not after — so a capped run's peak memory is
  proportional to the cap you choose, not to the full 6.9M. **A cap does
  not mean "all your real data is thrown away"**: the selection keeps every
  single positive (attack-adjacent) sample and only subsamples the
  overwhelmingly larger benign majority — see `nidra/data/dataset.py`'s
  `build_windowed_arrays` docstring.
- **Recommended, measured-safe command for a 16GB machine**:
  `--max-train-samples 500000 --max-val-samples 50000`. This was actually
  run (not just estimated) against the real dataset on a 16GB Apple M1: it
  used about **8.7GB of RAM** and took about **166 seconds per Stage-1
  epoch** on that machine. That's roughly 12x more training data than the
  MVP config's default cap, while staying safely inside a 16GB budget. If
  your machine has more RAM, you can raise these numbers proportionally
  (memory scales roughly linearly with the cap); if you have 64GB+ of RAM
  available, you can omit both flags entirely for the literal, fully
  uncapped run the config was originally designed for.

---

## 2. Understanding what's about to happen (read this before running anything)

Training happens in two separate stages, and you run a *separate command*
for each — this isn't a bug, it's deliberate (see `PROJECT_DEEP_DIVE.md`
Part 2.6 for the full "why" if you're curious; the short version is that
training the two stages together would let the system cheat in a way that
makes the results untrustworthy).

**Stage 1 ("dynamics")**: the model learns to predict "what does this
network host's behavior look like a few seconds/minutes from now,"
without knowing anything about which windows are attacks. This is the
expensive stage — it's what "60 epochs" refers to.

**Stage 2 ("heads")**: with Stage 1's learning locked in place (frozen —
it cannot change anymore), a much smaller and faster stage trains the
part of the model that turns "predicted network behavior" into "risk
score" and "attack stage." This is what "30 epochs" refers to, and it's
much cheaper because it's a tiny classifier riding on top of already-
learned representations.

You do this **five times total** (once per "seed" — a seed is just a
different random starting point, used to check that the results aren't a
fluke of one particular random initialization). Seeds are numbered
`0, 1, 2, 3, 4`.

---

## 3. Step-by-step commands

Run everything from inside the `ml/` directory.

### Step 1 — Confirm your test suite still passes

This costs seconds and confirms nothing is broken before you commit hours
of compute:

```bash
cd ml
pytest tests/ -q
```

You should see something like `147 passed`. If anything fails, stop and
fix it (or ask for help) before proceeding — don't spend hours training
against a broken pipeline.

### Step 2 — Look at the full-scale config

Open `config/default.yaml` and compare it to `config/mvp_2017.yaml`. The
`dataset:`, `windowing:`, `splits:`, and `model:` sections are identical —
only `train_dynamics.epochs` (60 vs 20) and `train_heads.epochs` (30 vs 20)
differ in the file itself. The sample-count cap (see §1.3 above) is passed
as a command-line flag, not written into either config file, so you choose
it per-run based on your machine's RAM.

### Step 3 — A one-epoch timing probe (do this first, always)

**Do not skip this, and do not omit `--max-train-samples`/`--max-val-samples`
unless you have confirmed (§1.3) your machine has enough RAM to hold the
full uncapped ~6.9M-sample tensor (~35+ GB).** This runs exactly one epoch
of Stage 1 training against the real, full-scale data, so you can measure
how long one epoch actually takes — and confirm your machine doesn't run
out of memory — before committing to all 60:

```bash
python -m nidra.train.train_dynamics --config config/default.yaml --epochs 1 --seed 0 \
    --max-train-samples 500000 --max-val-samples 50000
```

(Drop both flags only if you have 64GB+ of RAM and want the literal
uncapped run — see §1.3.)

**Bonus**: this probe's scaler fit is not wasted — Step 4 below reuses it
automatically (the scaler is cached to `artifacts/scaler/robust_scaler.joblib`
and loaded rather than refit on every subsequent run) **as long as you pass
the exact same `--max-train-samples`/`--max-val-samples` values in both
steps**, which is what the commands below do. **If you change the cap
between the probe and the real run** (or between the probe and a different
config), delete `artifacts/scaler/robust_scaler.joblib` and
`artifacts/scaler/scaler_metadata.json` first — the code checks only
whether those files exist, not what data they were fit on, so a mismatched
cap would silently reuse a scaler fit on differently-sized data instead of
refitting.

Watch the terminal output. You'll see log lines like:

```
... seed=0 epoch=0 train_nll=... val_nll=... tf_p=...
```

Note the **wall-clock time** this single line took to appear (use your
terminal's timestamps, or just time it with a stopwatch). This is your
"time per epoch" estimate.

**Do the math**: `(time for 1 epoch) × 60 epochs × 5 seeds` = rough total
time for all of Stage 1. Stage 2 (the heads) is much cheaper — it trains a
tiny classifier on top of an already-fixed representation, typically a
small fraction of Stage 1's time per epoch — so you can budget roughly an
extra 10-20% of the Stage 1 estimate for Stage 2 across all 5 seeds.

For reference, one real measurement at the recommended
`--max-train-samples 500000 --max-val-samples 50000` cap on a 16GB Apple
M1 (CPU only, no GPU): **~166 seconds per Stage-1 epoch**, which projects
to `166s × 60 × 5 ≈ 13.8 hours` for Stage 1 alone. Your machine will give a
different number — this is why the probe exists, not a promise.

If that total is longer than you're willing to wait (say, more than a
comfortable overnight run), you have three honest options, in order of
preference:
1. Run it anyway, overnight or over a weekend — this is normal for
   full-scale ML training.
2. Reduce the epoch count in `config/default.yaml` (e.g. 40/20 instead of
   60/30) — document that you did this; it's a legitimate, disclosed
   trade-off, not a shortcut you need to hide.
3. Stick with the MVP config's results and report them as-is — they're
   already honest, real numbers on the complete dataset, just at a smaller
   training sample size.

### Step 4 — Run the real thing

Once you're comfortable with the time estimate, run all 5 seeds. This will
take a long time — start it, then leave your computer alone (don't let it
sleep — check your OS's power settings) until it's done:

```bash
python -m nidra.train.train_dynamics --config config/default.yaml \
    --max-train-samples 500000 --max-val-samples 50000
python -m nidra.train.train_heads    --config config/default.yaml \
    --max-train-samples 500000 --max-val-samples 50000
```

Note there's no `--seed` flag here — **omitting `--seed` trains all 5
ensemble seeds in one process run**, reusing the same windowed data and
scaler across every seed (it fits/loads the scaler once, then loops over
`config/default.yaml`'s `ensemble.seeds` internally). This is not just more
convenient than looping over `--seed 0..4` yourself in a shell `for` loop —
it's meaningfully faster, since building the capped ~500K-sample tensor
from raw data is itself a non-trivial cost you'd otherwise pay 5 times
instead of once. Only pass `--seed N` explicitly if you want to (re-)run
one specific seed on its own (e.g. after Step 4's "if a seed crashes"
note below).

There's also no `--epochs` flag — omitting it means "use whatever
`config/default.yaml` says" (60 for dynamics, 30 for heads), which is what
you want for the real run (the `--epochs 1` from Step 3 was only for the
timing probe).

**What to watch for while it runs:**
- Each seed logs one line per epoch (`train_nll=... val_nll=...`). The
  "train" number should generally trend down. The "val" number matters
  more — if it stops improving for several epochs in a row, training will
  automatically stop early for that seed (this is normal, expected
  behavior — it's called "early stopping" and it's there to prevent
  wasting time once a model has stopped genuinely improving).
- If a seed crashes partway through, re-run just that seed with `--seed N
  --max-train-samples 500000 --max-val-samples 50000` (same caps as the
  original run) for both `train_dynamics` and `train_heads` — earlier
  completed seeds are unaffected, since each seed saves its own separate
  checkpoint file, and the scaler (already fit and saved to disk from the
  first run) is loaded rather than refit, so re-running one seed doesn't
  change the scaler the other seeds are already using.

### Step 5 — Fit calibration on the newly-trained ensemble

The risk head's raw output needs a calibration pass fit against this
specific set of trained weights (see `PROJECT_DEEP_DIVE.md` Part 10 for
why this step exists — in short, it corrects for the model's probability
outputs being compressed below the 0.75 decision threshold used
throughout evaluation):

```bash
python -m nidra.scripts.fit_calibration --config config/default.yaml
```

This is much faster than training — it only needs the validation split,
not the full training set, and doesn't touch the model's weights at all.

### Step 6 — Evaluate

Run the same evaluation harness used for the MVP config, just pointed at
the new config:

```bash
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000
```

This writes JSON result files under `artifacts/metrics/test/` and
`artifacts/metrics/holdout/` (note: `config/default.yaml` uses a plain
`artifacts/` directory, not `artifacts_mvp_2017/` — the two configs never
overwrite each other's results).

### Step 7 — Benchmark serving latency

Confirm the trained ensemble still serves fast enough:

```bash
python -m nidra.serve.benchmark --weights-dir artifacts/weights --scaler-path artifacts/scaler/robust_scaler.joblib --config config/default.yaml
```

Look for `p95_ms` in the output and compare it to the 300ms target printed
alongside it. If it's over target, see `TRAINING.md`'s note on reducing
`rollout.n_samples_per_member` in the config.

### Step 8 — Generate the report artifacts (plots, metadata)

```bash
python -m nidra.scripts.generate_report --config config/default.yaml \
    --test-metrics-dir artifacts/metrics/test \
    --holdout-metrics-dir artifacts/metrics/holdout \
    --reports-dir reports \
    --metadata-dir artifacts/metadata
```

This reads the JSON files Step 6 produced and renders plots + metadata
JSON — it never computes a new number itself, so if a metric looks wrong
here, the bug (if any) is in Step 6's output, not here.

---

## 4. How to read the results once you have them

Open the newly-written `artifacts/metrics/test/baselines.json` and
`artifacts/metrics/holdout/baselines.json`. Compare the `world_model` row's
`auc_pr` value against the `persistence` row's `auc_pr` value — **the
world model should score higher**. This is the project's central claim.
If it doesn't, that's a real, honestly-reportable finding, not something
to hide (see `PROJECT_DEEP_DIVE.md` Part 6.4 on why negative ablation
results are treated as legitimate outcomes here, not failures to cover up).

Compare these full-scale numbers against the MVP-scale numbers already
recorded in `REAL_DATA_RESULTS.md`. If they're in the same ballpark, that's
reassuring evidence the MVP-scale run was already a reliable preview. If
they're meaningfully different (better or worse), that's worth writing
down and thinking about — more data changing the picture is itself a
finding.

**Update `REAL_DATA_RESULTS.md`** with a new "Run 3" section (following the
same pattern as the existing "Run 2 (current)" section) once you have
these numbers — don't overwrite Run 2; keep it for provenance, exactly as
Run 1 was kept when Run 2 superseded it.

---

## Quick reference (no explanations)

```bash
cd ml
pytest tests/ -q

# Timing probe — ALWAYS do this first. --max-train-samples/--max-val-samples
# keep peak RAM around ~8.7GB (measured); the full uncapped split needs
# ~35+ GB and will get your process killed on anything under that. Drop
# both flags only if your machine actually has that much RAM (see §1.3).
python -m nidra.train.train_dynamics --config config/default.yaml --epochs 1 --seed 0 \
    --max-train-samples 500000 --max-val-samples 50000

# Full training — will take hours. No --seed: trains all 5 ensemble seeds
# in one process, reusing one windowed-data build and one fitted scaler.
python -m nidra.train.train_dynamics --config config/default.yaml \
    --max-train-samples 500000 --max-val-samples 50000
python -m nidra.train.train_heads    --config config/default.yaml \
    --max-train-samples 500000 --max-val-samples 50000

# Calibration + evaluation
python -m nidra.scripts.fit_calibration --config config/default.yaml
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000

# Latency benchmark + report generation
python -m nidra.serve.benchmark --weights-dir artifacts/weights --scaler-path artifacts/scaler/robust_scaler.joblib --config config/default.yaml
python -m nidra.scripts.generate_report --config config/default.yaml \
    --test-metrics-dir artifacts/metrics/test --holdout-metrics-dir artifacts/metrics/holdout \
    --reports-dir reports --metadata-dir artifacts/metadata
```
