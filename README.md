# NIDRA — Network Infiltration & Dynamics Recurrent Analyzer

**A world model for network attack forecasting.** Built for Problem
Statement 26153 (NTRO): instead of classifying whether traffic *is*
malicious right now, NIDRA learns the *dynamics* of per-host network
behaviour and recursively simulates the next six minutes — producing a
per-horizon risk curve with an uncertainty band, a projected attack-stage
sequence mapped to ATT&CK tactics, and the attributions behind the forecast.

## Results

Trained and evaluated end-to-end on the **complete, real CIC-IDS2017
dataset** (all 8 published day-files, all 5 raw PCAPs extracted with tshark
for true packet-level features), five independently seeded models voting.
Every number below is measured **at the natural prevalence of attacks**
(0.03–0.4 % of host-minutes) with every tunable — how sampled futures are
pooled, the calibration, the alert threshold — chosen on validation and
frozen before the test or holdout day was scored. That protocol is new as of
2026-09-20 (Run 8 in [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md));
the earlier headline "AUC-PR 0.96 / F1 0.91" was measured on a balanced
subsample with a pooling statistic chosen on the test day and is withdrawn
as a description of the system. It is kept there for provenance.

### Headline numbers

| | **Test day** (Friday: Botnet, PortScan, DDoS) | **Unseen attack type** (Thursday: Web attacks, Infiltration — held out of training entirely) |
|---|:---:|:---:|
| **AP** (average precision at natural prevalence) | **0.058** [0.018, 0.199] | **0.439** [0.000, 0.768] |
| the same risk head on the *current* state only ("assume nothing changes") | 0.065 | 0.295 |
| strongest non-forecasting baseline | 0.164 (GRU sequence classifier) | 0.443 (logistic regression on 30 min of history) |
| **Precision / recall / F1** at the alert threshold | 0.89 / 0.03 / 0.06 | 0.85 / 0.32 / 0.46 |
| false alarms per hour across ~2,000–2,600 monitored hosts | 0.48 | 0.86 |
| attack episodes alerted (before onset / inside the episode) | 0 / 3 of 15 | 0 / 3 of 5 |
| **next-state forecast: MSE reduction vs "nothing changes"** (linear reference) | **0.587** (0.565) | **0.616** (0.595) |

Brackets are 95 % intervals from resampling whole attack episodes. Higher is
better everywhere; AP is the fairest single score because it does not depend
on one threshold. The threshold is the validation-chosen 0.718; the mandated
0.75 gives the same recall.

### What these numbers mean

- **The model learns how a host's traffic evolves.** On every split its
  six-minute state forecast has 59–67 % lower squared error than assuming
  nothing changes, and 5–6 % lower than a linear model fit to the same task.
  This is the part of the "world model" claim the data supports, and it is
  measured directly rather than inferred from a risk score.
- **On an attack type it never saw (Thursday), forecasting from those
  dynamics helps.** AP 0.439 against 0.295 for the identical risk head
  applied to the current state only — +0.14, interval [−0.00, +0.28] over
  five episodes[^1] — matching the best history-reading classifier. Three of
  five episodes are alerted, 1 to 29 minutes after they start.[^2]
- **On Friday it fails, and so does every baseline.** 833 of Friday's 946
  attack minutes are Botnet command-and-control traffic on five workstations
  — an attack stage with zero training examples, and traffic that looks like
  a quiet workstation to a risk head reading one minute of state. That head
  ranks those minutes *below* silence. A sequence classifier over the full
  30-minute history does rank them (ROC-AUC 0.976) but is still only 0.164 AP
  at natural prevalence; the world model is at 0.058, the same as its own
  head on the current state. The diagnosis, and the architectural change it
  points to, are in Run 8 §8.7.
- **No advance warning is demonstrated.** No episode on any split crosses
  the threshold before its first attack-labelled minute. CIC-IDS2017's
  attacks are launched from an attacker VM with no same-host run-up — there
  are 3 / 9 / 15 training examples of "an attack begins within 1 / 3 / 5
  minutes" among 2.27 million host-minutes — and every system, including one
  trained directly on that question, sits at the prevalence floor there.
  Earlier versions of this README claimed hours, then "8 of 10 episodes";
  both are withdrawn.

### What was done to get here (all recorded, all reproducible)

- Δ=60 s windows (CIC-IDS2017 flow timestamps have minute resolution, so
  Δ=30 produced an artificial all-zero state every other window); flow/packet
  fusion fixed; per-feature preprocessing with an audit; validation cut that
  keeps every attack's run-up on its own side.
- A natural-prevalence benchmark with episode-level bootstrap intervals,
  Task A (published label) / A′ (detection) / B (onset within h minutes) /
  C (per horizon), attribution pairs (persistence, persistence + learned
  noise, isotropic noise, deterministic rollout, ridge two-lag dynamics,
  oracle on the true future, LR ×2, GBDT, GRU classifier), state-forecast
  skill, per-episode lead time — all from one record per split with git
  commit, config hash, dataset digests and checkpoint hashes.
- Dynamics: β-NLL loss (chosen among seven variants on validation), free-running
  checkpoint selection, L=30/K=6 geometry (chosen against L=15/K=3 on
  validation). Heads: trained and selected on every observed state of the
  split at natural prevalence. Operating point (pooling statistic,
  per-horizon calibration, threshold) selected on validation only and read by
  the serving predictor.
- New surfaces: multi-horizon risk curve with a trajectory band, projected
  stage sequence → ATT&CK tactics and techniques (curated, labelled
  model-internal), signed per-forecast attributions with a deletion
  faithfulness check, an offline PCAP / CICFlowMeter-CSV forecasting CLI with
  a static HTML report, and an explicit onset-target head as a supervised
  baseline for the forecasting question.

Everything measured on a MacBook Air (Apple M1, 8 cores, 16 GB, CPU only):
one five-seed run is ~4.8 h; forecasts are served from CPU. Full run history,
every bug found, every rejected variant: [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md);
compact technical summary with limitations: [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md);
requirement-by-requirement compliance with the problem statement:
[`ml/reports/NTRO_COMPLIANCE_2026-09-20.md`](ml/reports/NTRO_COMPLIANCE_2026-09-20.md).

## Architecture

```
raw telemetry (PCAP via tshark + CICFlowMeter CSV)
    -> per-host, per-60s-window state vector (45 features)
    -> GRU encoder over 30 windows (30 min at Δ=60 s) of history
    -> Gaussian transition model: predicts the NEXT-STATE DELTA
    -> recursive 6-step (6 min) rollout, feeding predictions back in
       as if they were real observations
    -> frozen risk and stage heads score every simulated future state
    -> 5-model ensemble, sampled trajectories pooled with the statistic,
       calibration and threshold frozen on validation -> per-horizon risk
       curve with a band, projected stage sequence -> ATT&CK tactics,
       lead time, and signed feature-level attributions
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

### The dashboard, and your own captures

```bash
cd web && npm install && npm run dev      # http://localhost:3000/demo
```

The console replays real CIC-IDS2017 Thursday-afternoon traffic (the held-out
Infiltration episode on an internal workstation) scored by the
trained ensemble, at 30x by default (one 60-second window every two seconds). It
reads a committed fixture, so it renders with every container stopped.

Its **Analyse a capture** panel takes a pcap the model has never seen and
scores it through the same predictor, locally — the file is not uploaded
anywhere and is deleted when the analysis returns. The capture needs at
least 31 minutes of traffic from one host, because the model reads 30
minutes of history before it will forecast. It needs `tshark` on PATH and
the Python environment from the Quickstart above.

Two things are different about an uploaded capture, and the page says both
rather than leaving you to infer them. It has no ground truth, so nothing on
it is scored against labels. And with no CICFlowMeter CSV the fifteen flow
features are reconstructed from the packets: volume and timing track the
reference closely (Spearman 0.85-0.92 on the one capture where both views
exist), the TCP flag ratios do not, and `urg_ratio` cannot be reconstructed
at all. `scripts/validate_flow_assembly.py` is the measurement, and
`scripts/analyze_pcap.py` is the same analysis on the command line.

Step 2 rewrites `ml/artifacts/metrics/` in place, so `git diff` afterwards
shows your run against the committed one. The flags above are the ones the
published numbers were produced with, and each metrics file records its own
parameters — lowering `--n-samples` or `--max-eval-samples` is much faster
but will not reproduce the headline figures.

### What ships in the repo

| Path | What it is |
|---|---|
| `ml/artifacts/weights/` | the 5 ensemble checkpoints behind every number above, the 5 onset heads, the GRU-classifier baseline, per-seed metadata, and `operating_point.json` (pooling, calibration, threshold — all selected on validation) |
| `ml/artifacts/scaler/` | the fitted per-feature scaler (`feature_scaler.json`), its preprocessing audit, and the SHAP background — inference must normalise with the exact scaler the model was trained against, so weights alone would be unusable |
| `ml/artifacts/processed/` | the windowed, labelled per-day tables (Δ=60 s, ~32MB) that evaluation runs on |
| `ml/artifacts/metrics/` | the published validation / test / holdout benchmark records (`benchmark.json`), plus the K=10 horizon check |
| `ml/artifacts/metadata/` | dataset audit and provenance |
| `ml/experiments/` | the config + provenance record of every recorded run (geometry, head-recipe, loss-variant screens, production, leave-one-day-out) and the Δ=30 baseline manifest |

The raw CIC-IDS2017 release (~50GB) is **not** redistributed here, and none
of the three steps above need it. It is required only to re-derive the
windowed tables from scratch or to retrain from zero.
`nidra/train/pipeline.py` resolves each day from its committed table first
and falls back to the raw CSV only on a miss, so nothing reaches for a
dataset that isn't there.

Retraining, for completeness — this is the only path that needs the raw
dataset (see [`ml/PRODUCTION_RUN_GUIDE.md`](ml/PRODUCTION_RUN_GUIDE.md)):

```bash
cd ml && python -m nidra.scripts.run_experiment --label production --seeds 0,1,2,3,4 \
    --stages dynamics,heads,onset,gru_baseline --epochs 24
```

See [`ml/README.md`](ml/README.md) for the full technical writeup
(pipeline, model internals, evaluation methodology, explainability, and
serving interface) and [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md)
for every result this project has measured, reported in full — including
the items below.

---

*Two footnotes, both documented in full in `ml/REAL_DATA_RESULTS.md` rather
than smoothed over:*

[^1]: The world model's margin over "assume nothing changes" at the risk
    level is not resolved from zero on any single split (−0.02 validation,
    −0.01 test, +0.14 holdout with a lower bound of −0.00). Its margin at the
    *state* level — how well it predicts the next six minutes of traffic —
    is, on every split. The honest statement is that the learned dynamics
    are real and that turning them into better risk ranking depends on a
    risk head that can recognise unseen attack families, which the current
    per-state head cannot (Run 8 §8.7).

[^2]: The oracle row (the same head applied to the *true* future) scores
    below the forecast on Thursday. That is not a measurement quirk this
    time: the oracle is only as good as the head on benign futures too, and
    it fires on 34 benign minutes whose true next states look like attacks
    to the head, where the rollout's smoothed futures do not. The oracle
    bounds what the head can recognise, not how good a forecast can be.
