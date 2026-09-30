# NIDRA — Architecture

**A world model for network attack forecasting** · NTRO Problem Statement 26153 · fully
offline, CPU-only, open source. Detail: [`ARCHITECTURE_DETAIL.md`](ARCHITECTURE_DETAIL.md).

## What the evidence supports

Measured on the complete CIC-IDS2017 dataset at the natural prevalence of attacks
(0.03–0.4 % of host-minutes), with every tunable frozen on validation before test or
holdout was scored. Test = Friday (Botnet, PortScan, DDoS); holdout = Thursday (Web
attacks, Infiltration), attack types never seen in training.

**1. It learns how network state evolves — P(S_t+1 | S_t).** Six-minute state forecasts
have lower error than both reference models on every split, at every horizon from one
to six minutes, and on active hosts alone:

| MSE reduction vs "nothing changes" | val | test | holdout |
|---|---:|---:|---:|
| NIDRA world model | **0.672** | **0.587** | **0.616** |
| two-lag linear model, same task | 0.653 | 0.565 | 0.595 |
| NIDRA's margin over the linear model | 5.6 % | 5.2 % | 5.2 % |

The margin over the linear model is modest and it is consistent. The 90 % forecast
band covers the true next state 98 % of the time (validation).

**2. It raises better alerts than the PS's logistic-regression baseline.** Each system
is scored at its own validation-chosen threshold (`reports/PS_BASELINE_BENCHMARK.md`):

| | NIDRA test | LR test | NIDRA holdout | LR holdout |
|---|---:|---:|---:|---:|
| AP (threshold-free) | **0.058** | 0.033 | **0.438** | 0.241 |
| precision | **0.89** | 0.48 | **0.85** | 0.46 |
| recall | 0.03 | 0.04 | **0.32** | 0.28 |
| F1 | 0.06 | **0.07** | **0.46** | 0.35 |
| false alarms per hour | **0.48** | 4.80 | **0.86** | 4.89 |

That is 10× and 5.7× fewer false alarms at higher precision, and better F1 on the
unseen-attack day. The AP margin's paired interval excludes zero on test
(+0.025 [+0.005, +0.058]) and touches it on holdout (+0.197 [−0.000, +0.366]).

**3. Every number is reproducible and every prediction is explained.** Each run records
its git commit, config hash, dataset digests and seeds; 28 recorded cells re-ran
bit-identically. Each forecast carries SHAP attributions over the 45 named features,
temporal saliency, and integrated-gradient attributions with a faithfulness check.

## What it does not do

- **No advance warning is demonstrated.** No attack episode crosses the threshold before
  its first attack minute (0 of 15 test, 0 of 5 holdout). CIC-IDS2017 attacks come from an
  external VM with almost no same-host run-up to learn from.
- **The forward simulation does not yet add attack-forecasting skill.** The same risk
  head on the *current* state scores about as well (test AP 0.065 vs 0.058). Advantage 2
  comes from the state representation and head, not demonstrably from the rollout.
- **History-reading classifiers match it.** LR on 30 minutes of history is statistically indistinguishable on AP; a
  GRU classifier ranks Friday's attacks better (AP 0.164).
- **Projected stages are unreliable where training had none.** Recon, C2 and lateral
  movement never occur in the training days; stage accuracy on test/holdout is 0.00.
- **It is correlational, not causal.** The model learns dynamics from observational
  data. A simulated trajectory is the model's projection, not a claim about why the
  network behaves as it does; what-if outputs are labelled "model-internal what-if".

## Pipeline

```
PCAP ─tshark─► packet features ─┐
CICFlowMeter CSV ─► flow features ┼─► per-host, per-60 s state S_t (45 features)
(PCAP only: flows reassembled) ───┘        │  scaler fit on training days only
                                           ▼
 30 windows of history ─► GRU encoder (2×128) ─► h_t ─► Gaussian transition ─► ΔS
         ▲                                                                     │
         └─────────── predicted S_t+1 fed back as if observed (×6) ◄───────────┘
                                           ▼
     frozen RiskHead + StageHead score every simulated state ─► 5-seed ensemble,
     1,000 trajectories ─► per-horizon risk curve with band, projected ATT&CK
     stages, SHAP / saliency / attributions
```

**State (45 features, one canonical order in `nidra/data/schema.py`).** 15 flow (TCP flag
ratios, bytes, packets, duration, IAT mean/variance/max, bidirectional ratios), 11 packet
(TTL mean/variance, TCP window, fragment flags, payload distribution, retransmissions), 8
graph scalars (degree, port entropy, new peers, reciprocity) and 11 dynamics/activity
features. Silent windows are real zero rows, never dropped.

**Transition.** An MLP on h_t predicts the mean and log-variance of the next-state
*delta* (log-variance clamped to [−6, 3], state to ±10). Predicting a delta makes
"nothing changes" the μ = 0 case, so persistence is an exact ablation.

**Training, strictly ordered.** Stage 1 trains encoder and transition on a multi-step
unrolled β-NLL loss with scheduled sampling, selected on free-running validation
error. Stage 2 freezes them and trains the heads on *observed* states only. The heads
are never fine-tuned on simulated states: every unit of forward-looking skill must
come from the transition model, which keeps the forecasting claim falsifiable.

**Operating point.** Trajectory pooling, per-horizon Platt calibration and the alert
threshold (0.718) are chosen on validation and read unchanged by the benchmark and by
serving. The mandated 0.75 is reported alongside.

## Inference and interface

`NidraPredictor.forecast()`, the only ML surface the backend imports, is stateless per
host and returns the t+1…t+6 risk curve with a band, the projected stage sequence mapped
to MITRE ATT&CK tactics (a curated presentation mapping), predicted features in raw units
and explanations — about 128 ms on an idle M1 CPU against a 300 ms target. Offline
surfaces: `python -m nidra.cli.forecast --pcap | --csv` (forecasts, alerts, HTML report)
and the web console's replay and capture upload. None needs a network connection.

## Evaluation protocol

Splits are temporal by day, never random. Thursday is held out of training entirely.
Metrics are natural-prevalence AP, precision, recall, F1, FPR and false alarms/hour,
with 95 % intervals from resampling whole attack episodes. Baselines: persistence,
persistence plus learned noise, ridge two-lag dynamics, LR on the current state and on
30 min of history, GBDT, a GRU sequence classifier, and an oracle head on the true future.

**Run 9** added CTU-13. A history-aware head won its training objective but failed a
pre-registered benchmark test, so the per-state head was kept (D146). Numbers:
`REAL_DATA_RESULTS.md`, `reports/RUN9_FINAL_REPORT_2026-09-23.md`,
`reports/PS_BASELINE_BENCHMARK.md`.
