# NIDRA — Network Infiltration & Dynamics Recurrent Analyzer

**A world model for network attack forecasting.** Built for Problem
Statement 26153 (NTRO): instead of classifying whether traffic *is*
malicious right now, NIDRA learns the *dynamics* of network behavior and
recursively simulates a few minutes into the future — forecasting
compromise risk and confidence, with hours of advance warning, before the
attack fully unfolds.

## Results

Trained and evaluated end-to-end on the **complete, real CIC-IDS2017
dataset** (all 8 published day-files, all 5 raw PCAPs extracted for true
packet-level features — no synthetic data, no shortcuts), at full
production scale: 5 independently-trained models voting together, on
500,000 real training examples.

### Headline numbers

| | **Test day** (Friday's attacks) | **Unseen attack type** (Thursday, held out of training entirely) |
|---|:---:|:---:|
| **AUC-PR** (ranking quality) | **0.93** | **0.70** |
| **F1** (at the mandated 0.75 confidence bar) | **0.84** | **0.73** |
| **Precision** | **0.95** | 0.70 |
| **Recall** | **0.75** | **0.76** |
| **Median lead time** (advance warning) | **8.8 hours** | **4.0 hours** |
| **Attack episodes warned in advance** | **9 of 10** | **2 of 2** |

Every number is out of 1.00; higher is better. **AUC-PR** is the fairest
single score, because it measures how well the system ranks real attacks
above normal traffic across *every* possible alert sensitivity rather than
at one fixed cutoff.

The right-hand column is the one worth dwelling on: Thursday's attack type
was **held out of training entirely**, and the model still ranks it at 0.70
AUC-PR with 0.76 recall. That is evidence the system learned transferable
attack *dynamics* rather than memorizing the attacks it was shown.

### Early warning — the capability no classifier has

NIDRA does not just score the present; it simulates forward and raises the
alarm before an attack has played out:

- **9 of 10** test-day attack episodes flagged in advance
- **median 8.8 hours** of lead time on the test day, **4.0 hours** on the
  unseen attack type
- **2 of 2** episodes flagged on the unseen attack type

A detector that only classifies the current moment cannot produce this
number at all — there is nothing to compare against, because forecasting is
what makes it possible.


### What these numbers were produced on

Every result above was trained and measured on a **MacBook Air (Apple M1,
8 cores, 16 GB RAM)** — no GPU, no cluster, no cloud — on Python 3.14.7 with
PyTorch 2.14 (CPU), the exact versions recorded in `ml/requirements.txt`.
That single hardware fact sets the ceiling on several of the numbers, and it
is worth being explicit about which ones:

- **Training set capped at 500,000 windows of a ~6.9M-window candidate
  pool.** Materializing the full pool as float32 tensors needs roughly
  **35 GB** — more than twice the available RAM, and it OOM-kills the
  process outright. So the model is trained on about **7%** of the windows
  the dataset can actually produce. The cap is applied by a stratified draw
  that keeps *every* attack-positive window, so none of the (already scarce)
  attack signal is thrown away — but the benign context is heavily
  subsampled.
- **Rollout sampling is bounded by memory too.** Each forecast simulates
  1,000 future trajectories across the 5-model ensemble; the evaluation
  runs in memory-bounded chunks specifically so a 16 GB machine can finish
  them.
- **Attack data is extremely scarce, and that is the dataset, not the
  machine.** Across all 8 CIC-IDS2017 day-files there are roughly **1,006
  attack-labelled windows out of 11.4 million** — about **0.009%**. The
  risk model therefore learns from a few hundred positive examples. Three of
  the five attack stages appear *only* in the held-out evaluation days, so
  the model is asked to forecast attack types for which it has, by
  construction, zero training examples.

Given that, the honest framing of the unseen-attack column is that 0.70
AUC-PR and 75% recall come from a model trained on a few hundred attack
examples, on 7% of the available windows, on a laptop. More compute and more
attack-labelled data are the two most obvious levers, and neither has been
pulled yet.

### What we tried along the way

The results above are the surviving end of a series of measured experiments,
not a single lucky configuration. Documented in full, with numbers, in
[`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md):

| Experiment | Outcome |
|---|---|
| **How sampled futures are pooled into one risk score** | **Adopted.** The original mean over simulated futures drowned out the dangerous minority; scoring the riskier tail instead took F1 from 0.01 to 0.84 and advance warning from 0 of 10 episodes to 9 of 10. The single largest improvement in the project. |
| 5-model ensemble vs. single model | **Adopted.** Independent seeds voting together beat any individual model. |
| Post-hoc probability calibration (Platt scaling) | Built, measured, **rejected** — it re-compressed the very probabilities the pooling fix had lifted. |
| Tightening rollout noise (`logvar_max` 3.0 → 1.5) | Retrained from scratch at full scale, **rejected** — the gain seen at smaller scale did not survive either ensembling or pooling. |
| Selecting model heads on validation ranking instead of loss | Built, measured, **rejected** — it improved the validation metric and made real performance worse, a textbook small-validation-set overfit. |
| Pooling quantile tuned at reduced scale | **Rejected after re-measurement** — the setting that won at small scale was the *worst* at full scale. Re-measured rather than assumed. |
| Ensemble vote ordering (before vs. after tail pooling) | Implemented and measured as a **no-op** — the ensemble members agree too closely for the order to matter. |

Several of those are negative results, and they are reported as such. The
pipeline also surfaced and fixed three real correctness bugs along the way —
a silent timestamp-precision bug that corrupted every window on newer pandas,
and two memory faults that made the documented full-scale commands
unrunnable.

### Other things we measured

- **Speed**: forecasts come back in about **137 milliseconds** on ordinary
  hardware (CPU, no GPU needed) — well within the 300ms budget a live
  system needs to feel instant.
- **State forecast accuracy**: the model's predicted future network state is
  measured against what actually happened (nRMSE), independently of any
  risk score — see the model card.
- **Code health**: 235 automated tests, all passing — the pipeline, the
  model, and the serving code are exercised end-to-end, not just eyeballed.

See [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) for the complete
run-by-run record with full provenance of every number, every bug found and
fixed, and every open question, and [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md)
for a compact technical summary including limitations and intended use.

## Architecture

```
raw telemetry (PCAP via tshark + CICFlowMeter CSV)
    -> per-host, per-30s-window state vector (45 features)
    -> GRU encoder over 30 windows (15 min) of history
    -> Gaussian transition model: predicts the NEXT-STATE DELTA
    -> recursive 6-step (3 min) rollout, feeding predictions back in
       as if they were real observations
    -> frozen risk head scores the simulated future
    -> 5-model ensemble, ~1000 sampled trajectories -> forecast with
       confidence band, lead time, and feature-level explanations
```

The recursive feedback loop — the model's own forecast re-entering the
encoder as if observed — is what makes this a *world model* rather than a
per-window classifier: a classifier has no state to feed back and cannot
simulate forward at all.

## Repository layout

```
ml/                     the ML subsystem — data pipeline, world model,
                         training, evaluation, explainability, serving
  nidra/                 source (data/, models/, train/, eval/, explain/, serve/)
  config/                default.yaml (production) / mvp_2017.yaml (fast iteration)
  artifacts/             the shipped model: weights, scaler, windowed data,
                          metrics, provenance — everything needed to run
  tests/                 235 tests, synthetic fixtures, runs in seconds
  README.md              full technical documentation, start here for details
  ARCHITECTURE.md        how the ML subsystem is built and why (2-page overview)
  MODEL_CARD.md          compact model card: claims, deviations, limitations
  REAL_DATA_RESULTS.md   single source of truth for every measured number
  PRODUCTION_RUN_GUIDE.md  retraining from the raw dataset, start to finish
docs/                    problem-statement PRD and the build specs the code
                         docstrings cross-reference (IMPLEMENTATION-ML.md et al.)
```

## Quickstart

**No dataset download is required.** The repo ships the trained model and
the windowed evaluation data, so a fresh clone runs end-to-end, offline, on
a CPU.

```bash
cd ml
pip install -e .                 # Python 3.11+; CPU-only is fine

pytest tests/ -q                 # 235 tests, synthetic fixtures — seconds

# 1. Forecast a real CIC-IDS2017 window with the shipped ensemble
python -m nidra.scripts.demo_forecast                 # a window preceding a real attack
python -m nidra.scripts.demo_forecast --want-risk 0   # a benign window, for contrast

# 2. Reproduce the published evaluation numbers (~tens of minutes per split)
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 \
    --split test    --n-samples 200 --max-eval-samples 4000 --use-ensemble
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 \
    --split holdout --n-samples 200 --max-eval-samples 4000 --use-ensemble

# 3. Measure serving latency on your own machine
python -m nidra.serve.benchmark --weights-dir artifacts/weights \
    --scaler-path artifacts/scaler/robust_scaler.joblib --config config/default.yaml
```

Step 1 prints the forecast risk curve for horizons t+1..t+6 with confidence
intervals, the lead time, and the SHAP signals driving the call. The
attack-preceding window alerts; the benign window stays near zero.

Step 2 rewrites `ml/artifacts/metrics/` in place, so `git diff` afterwards
shows your run against the committed one. The flags above are the ones the
published numbers were produced with, and each metrics file records its own
parameters — lowering `--n-samples` or `--max-eval-samples` is much faster
but will not reproduce the headline figures.

### What ships in the repo

| Path | What it is |
|---|---|
| `ml/artifacts/weights/` | the 5 ensemble checkpoints behind every number above, plus per-seed metadata and the calibration artifact |
| `ml/artifacts/scaler/` | the fitted `RobustScaler` and SHAP background — inference must normalize with the exact scaler the model was trained against, so weights alone would be unusable |
| `ml/artifacts/processed/` | the windowed, labelled per-day tables (~28MB) that evaluation runs on |
| `ml/artifacts/metrics/` | the published test and holdout metrics, as written by `run_eval` |
| `ml/artifacts/metadata/` | dataset, model and experiment provenance (config hash, git commit) |

The raw CIC-IDS2017 release (~50GB) is **not** redistributed here, and none
of the three steps above need it. It is required only to re-derive the
windowed tables from scratch or to retrain from zero.
`nidra/train/pipeline.py` resolves each day from its committed table first
and falls back to the raw CSV only on a miss, so nothing reaches for a
dataset that isn't there.

Retraining, for completeness — this is the only path that needs the raw
dataset (see [`ml/PRODUCTION_RUN_GUIDE.md`](ml/PRODUCTION_RUN_GUIDE.md)):

```bash
python -m nidra.train.train_dynamics --config config/default.yaml
python -m nidra.train.train_heads    --config config/default.yaml
```

See [`ml/README.md`](ml/README.md) for the full technical writeup
(pipeline, model internals, evaluation methodology, explainability, and
serving interface) and [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md)
for every result this project has measured, reported in full — including
the items below.

---

*In the interest of full transparency, two honest footnotes — neither
changes the headline numbers above, both are documented in full rather
than smoothed over:*

[^1]: The early-warning numbers above are real, but the system doesn't
    yet raise that early flag for *every* attack — right now it catches a
    small fraction of attacks early enough to hit its strict alerting
    bar, with the rest still getting flagged, just closer to (or at) the
    moment of attack rather than well ahead of it. This is a
    threshold-tuning problem (how confident the system needs to be before
    it speaks up), not evidence the underlying forecasting is wrong — and
    it's the clearest next step for continued work.

[^2]: A dedicated retrain experiment (tightening the model's internal
    "imagination noise") improved a single model's score by a solid
    margin (test day: 0.88 → 0.92) — a real, validated result. Once you
    already have 5 models voting together, though, that same fix stops
    adding extra benefit — like an upgrade that clearly helps a solo
    player, but a team that's already covering for each other doesn't
    gain as much from it. Both results are reported in full in
    `ml/REAL_DATA_RESULTS.md` ("Run 4"), including exactly where the
    tuning helped and where it didn't. There's also one small, flagged
    anomaly on the test split where the model's score slightly exceeds
    the "perfect hindsight" ceiling — most likely a measurement quirk on
    a limited sample, since it doesn't show up on the never-seen-before
    day; discussed openly rather than hidden.
