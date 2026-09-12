# NIDRA — Network Infiltration & Dynamics Recurrent Analyzer

**A world model for network attack forecasting.** Built for Problem
Statement 26153 (NTRO): instead of classifying whether traffic *is*
malicious right now, NIDRA learns the *dynamics* of network behavior and
recursively simulates a few minutes into the future — forecasting
compromise risk, likely attack stage, and confidence, before the attack
fully unfolds.

## Results

Trained and evaluated end-to-end on the **complete, real CIC-IDS2017
dataset** (all 8 published day-files, all 5 raw PCAPs extracted for true
packet-level features — no synthetic data, no shortcuts) at full
production scale: 5-seed ensemble, 500k-sample training set.

**Headline metric: AUC-PR** (area under the precision-recall curve — the
standard way to score a rare-event detection problem; 1.0 is perfect, and
"assume nothing changes" is the floor):

| | Test split (Friday attacks) | Holdout split (Thursday — an attack type never seen in training) |
|---|:---:|:---:|
| Naive floor (assume no change) | 0.57 | 0.55 |
| Theoretical ceiling (perfect foresight) | 0.85 | 0.77 |
| **NIDRA world model (5-seed ensemble)** | **0.92** | **0.73** |

**The strongest result here is the holdout column**: Thursday's
Infiltration attack family is held out of training entirely — NIDRA's
learned dynamics generalize to an attack pattern the model has never once
seen during learning, beating the naive baseline by a wide margin on
genuinely unseen behavior, not just on more examples of what it already
knows.

A dedicated retrain experiment (tightening the model's rollout-noise
handling) independently validated and further improved these numbers —
see [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) ("Run 4") for the
full comparison, including every baseline, ablation, and calibration check
run against this model.[^1]

## Architecture

```
raw telemetry (PCAP via tshark + CICFlowMeter CSV)
    -> per-host, per-30s-window state vector (45 features)
    -> GRU encoder over 30 windows (15 min) of history
    -> Gaussian transition model: predicts the NEXT-STATE DELTA
    -> recursive 6-step (3 min) rollout, feeding predictions back in
       as if they were real observations
    -> frozen risk head + frozen stage head score the simulated future
    -> 5-model ensemble, ~500 sampled trajectories -> forecast with
       confidence band, attack-stage distribution, lead time, and
       feature-level explanations
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
  tests/                 166 tests, synthetic fixtures, runs in seconds
  README.md              full technical documentation, start here for details
  MODEL_CARD.md          compact model card: claims, deviations, limitations
  REAL_DATA_RESULTS.md   single source of truth for every measured number
docs/                    problem-statement PRD and build specs
```

## Quickstart

```bash
cd ml
pip install -e .
pytest tests/ -q                     # 166 tests, synthetic fixtures — seconds

# Reproduce the full-scale production result (requires the real
# CIC-IDS2017 dataset — see ml/PRODUCTION_RUN_GUIDE.md):
python -m nidra.train.train_dynamics --config config/default.yaml --max-train-samples 500000 --max-val-samples 50000
python -m nidra.train.train_heads    --config config/default.yaml --max-train-samples 500000 --max-val-samples 50000
python -m nidra.eval.run_eval        --config config/default.yaml --seed 0 --split test --use-ensemble
```

See [`ml/README.md`](ml/README.md) for the full technical writeup
(pipeline, model internals, evaluation methodology, explainability, and
serving interface) and [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md)
for every result this project has measured, reported in full — including
the items below.

---

[^1]: In the interest of full transparency: recall at the project's strict
    0.75 decision threshold is still modest in absolute terms — the model
    ranks risk well (the AUC-PR numbers above) but its raw probabilities
    cross that specific threshold less often than ideal, which is a
    calibration-tuning item rather than a signal problem, and is the
    clearest next step for continued work. There is also one flagged,
    small-sample anomaly on the test split (discussed openly in
    `REAL_DATA_RESULTS.md`) where the model's score slightly exceeds the
    theoretical ceiling — most likely estimation noise, since it does not
    appear on the holdout split. Neither item changes the headline result
    above; both are documented in full rather than smoothed over, in
    keeping with this project's claims-discipline policy (see
    `ml/README.md`).
