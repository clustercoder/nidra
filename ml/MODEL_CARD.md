# Model Card — NIDRA world model (Run 8, Δ=60 s, 2026-09-20)

This card describes the artifacts under `ml/artifacts/` produced on 2026-09-20 by
`experiments/runs/production/record.json` (git 8beccae, config hash 91e7e0ab7162ec53,
dataset digests and per-seed checkpoint hashes inside). The measured numbers are in
`REAL_DATA_RESULTS.md` §Run 8 and in `artifacts/metrics/<split>/benchmark.json`; this
card summarises them and states what they do and do not support. The previous card
(Runs 1–7, Δ=30 s) is superseded — Run 8's first paragraph says why its numbers are not
comparable, and the Δ=30 artifacts are frozen under the git tag `baseline-delta30-run7`.

## What it is

A per-host network **world model**: 45 features per host per minute (15 flow, 11 packet,
8 graph scalars, 11 dynamics) → GRU encoder over the last 30 minutes → diagonal-Gaussian
transition over the next-state delta → recursive 6-step rollout (own predictions fed
back, no data after "now") → frozen risk and stage heads scored on every sampled future
state → pooled and calibrated risk curve with a trajectory band. Five independently
seeded models vote. It is **not** an intrusion classifier: the heads never see a
predicted state during training, so any forward-looking skill has to come from the
transition model, and the benchmark measures whether it does.

| Component | Value |
|---|---|
| Window Δ / context L / horizon K | 60 s / 30 windows (30 min) / 6 windows (6 min) |
| Features | 45 (`nidra/data/schema.FEATURE_ORDER`, asserted at every boundary); none dropped by the production scaler |
| Preprocessing | per-feature log1p / asinh / z-score / unit, fit on active TRAIN rows only; audit next to the scaler (`artifacts/scaler/preprocessing_audit.md`) |
| Encoder / transition | 2-layer GRU (128) / MLP(256) → μ, log σ² clamped [−6, 3]; states clamped ±10; no 32-dim bottleneck (the 128-dim `h_t` is the latent — a documented deviation from the original design target, unchanged since the first card) |
| Dynamics loss | β-NLL, β = 0.5, multi-step unrolled with scheduled sampling (D113); checkpoint selected on free-running validation NLL (D107) |
| Heads | risk 45→64→1, stage 45→64→6, onset 45→64→6 horizons; trained on observed states only (every row of the split, D114), then frozen |
| Ensemble / trajectories | 5 seeds × 200 samples in serving (benchmarks: 5 × 60) |
| Operating point | selected on validation, `artifacts/weights/operating_point.json`: **median** over sampled trajectories, max over horizons, per-horizon Platt, threshold **0.718**; the mandated 0.75 is reported alongside (within 0.01 F1 of it on every split) |

## Intended use and out-of-scope use

Research prototype for NTRO PS 26153: short-horizon (≤ 6 min) per-host forecasting of
attack likelihood and projected attack stage from flow + packet telemetry, offline or
replayed. Not validated for production security operations; not a detector; not
attribution; not a causal model of attacker behaviour (every projected sequence is
"model-internal"). It should not be relied on for attack families it has not seen —
see the Friday result below.

## Data

CIC-IDS2017, all 8 day-files, packets extracted with tshark for every day. Train:
Monday (benign) + first 70 % of Tuesday and Wednesday (FTP-Patator, DoS ×4; 179
positives among 2.27 M host-minutes). Validation: trailing 30 % of Tuesday and Wednesday
(SSH-Patator, Heartbleed; 2 episodes). Test: Friday (Bot, PortScan, DDoS; 15 episodes).
Holdout, never trained on: Thursday (Web ×3, Infiltration; 5 episodes). Every evaluation
family is unseen in training. Full composition in Run 8 §8.1.

## Evaluation protocol

`nidra.eval.benchmark` (`EVALUATION.md`): every positive and every pre-onset window
(30 min before each onset) of the split, plus 15,000 active-benign and 5,000
silent-benign windows weighted to the split's natural prevalence (test 0.0042, holdout
0.00034). Primary metric: natural-prevalence AP on the published label (attack within K)
with an episode-bootstrap interval; also ROC-AUC, P/R/F1 at the frozen threshold and at
0.75, false alarms per hour, detection (Task A′), onset within h min (Task B),
per-horizon (Task C), state-forecast skill, per-episode lead time. Baselines scored on
the identical rows: persistence (risk head on S_t), persistence + learned noise,
isotropic noise, deterministic rollout, ridge two-lag dynamics, oracle on the true future,
LR on S_t, LR on the 30-window history, GBDT on S_t, a GRU sequence classifier, the onset
head. Everything tunable was frozen on validation before test or holdout was scored.

## Measured performance (5-seed ensemble, natural prevalence)

| | validation | **test (Friday)** | **holdout (Thursday, unseen families)** |
|---|---|---|---|
| positives / episodes / prevalence | 89 / 2 / 0.00012 | 946 / 15 / 0.0042 | 122 / 5 / 0.00034 |
| **NIDRA AP** (episode-bootstrap 95 % CI) | 0.726 [0.01, 0.97]* | **0.058** [0.018, 0.199] | **0.439** [0.000, 0.768] |
| persistence — same head on the current state | 0.749 | 0.065 | 0.295 |
| best other baseline | 0.756 (persistence + learned noise) | 0.164 (GRU sequence classifier) | 0.443 (LR on 30-min history) |
| oracle — same head on the true future | 0.770 | 0.117 | 0.400 |
| ΔAP NIDRA − persistence (paired bootstrap) | −0.024 [−0.045, +0.052] | −0.007 [−0.026, +0.011] | +0.143 [−0.000, +0.283] |
| ROC-AUC | 0.995 | 0.40 | 0.65 |
| P / R / F1 at threshold 0.718 | 1.00 / 0.65 / 0.79 | 0.89 / 0.03 / 0.06 | 0.85 / 0.32 / 0.46 |
| P / R / F1 at the mandated 0.75 | 1.00 / 0.65 / 0.79 | 0.94 / 0.03 / 0.06 | 0.89 / 0.31 / 0.46 |
| false alarms per hour (all hosts) / share of active-benign minutes | — | 0.48 / 0.013 % | 0.86 / 0.020 % |
| episodes alerted before onset / inside episode | 0 / 1 of 2 | 0 / 3 of 15 | 0 / 3 of 5 (latency 1, 8, 29 min) |
| Task B — onset within 5 / 15 min (AP; positives) | 0.002 / 0.005 (6 / 16) | 0.010 / 0.013 (60 / 161) | 0.001 / 0.002 (20 / 52) |
| per-horizon AP, k = 1 → 6 | 0.83 → 0.82 | 0.063 → 0.024 | 0.58 → 0.29 |
| state-forecast skill vs persistence (ridge two-lag) | 0.672 (0.653) | 0.587 (0.565) | 0.616 (0.595) |
| 90 % band coverage (validation, free-running) | 0.98 | — | — |

`*` two positive episodes: the interval is not informative. Tables for every system,
horizon and episode: `reports/run8/benchmark_tables.md`; figures: `reports/run8/`.

## What the numbers support

- **Learned dynamics, measurably.** The deterministic rollout reduces next-state MSE
  by 59–67 % against persistence on every split and by 5–6 % against a ridge two-lag
  linear model fit to the same task; the free-running validation NLL improves through
  training and selects epochs 12–19. This is the claim the benchmark confirms.
- **On Thursday's unseen families, forecasting from those dynamics adds risk-level
  information**: +0.14 AP over the same head applied to the current state (interval
  [−0.00, +0.28] over five episodes), on par with the strongest history-reading
  classifier, with 0.86 false alarms per hour across 2,557 hosts and 3 of 5 episodes
  alerted 1–29 minutes after onset.
- **On Friday it does not, and the reason is located**: the risk head, trained on 179
  positives from two families, ranks Friday's attack minutes (833 of 946 are Botnet C2 on
  five workstations) below silence (ROC-AUC 0.37); the oracle on the true future is only
  0.117. A GRU sequence classifier on the same rows reaches ROC-AUC 0.976 (AP 0.164), so
  the 30-minute history carries family-general signal that a per-state head cannot read.
- **Calibration** is per-horizon Platt on validation; the calibrated score's Brier is
  within 1 % of the raw one on test and holdout (8 % lower on validation, where it was
  fit), and 0.75 and 0.718 give the same recall on
  test (0.03) and nearly the same on holdout (0.31 vs 0.32) — the mandated threshold is
  not what limits recall.

## Limitations — read before citing

1. **No advance warning is demonstrated.** No episode on any split crosses the threshold
   before its first attack-labelled minute. Onset-within-h positives in the whole training
   split are 3 / 9 / 15 / 29 / 39 / 69 at 1 / 3 / 5 / 10 / 15 / 30 min; every system,
   including an explicitly supervised onset head and the oracle, is near the prevalence
   floor on Task B. The dataset's attacks have no same-host run-up.
2. **The per-state risk head does not transfer to unseen attack families** (Friday). The
   world model's forecast cannot exceed what the head can recognise. The architectural
   fix (a head on the encoder state at each rollout step, trained on observed pairs and
   frozen) is identified, not implemented — it needs a family-transfer selection split.
3. **The world model's risk-level margin over persistence is not resolved from zero on
   any split** (validation −0.02, test −0.01, holdout +0.14 with a lower bound of −0.00).
   The state-level margin is.
4. **Stage forecasting is 0.00 top-1 on Friday and Thursday futures**: `recon`, `c2`,
   `lateral` never occur in training. The ATT&CK mapping labels every sequence
   "projected stage sequence (model-internal)"; on those days the stage output is not a
   finding.
5. **Sampled noise costs AP on test** (deterministic − stochastic +0.023 [+0.003, +0.048])
   and gains on holdout (−0.058 [−0.169, +0.014]). The stochastic path is served because
   the band and P(attack within horizon) need it.
6. **Validation has two episodes**, both similar to training families; it can select
   pooling and threshold but cannot measure family transfer. Leave-one-day-out retrains
   (Run 8 §8.6) are the only within-training-days check with unseen families.
7. **Hardware caps**: dynamics trained on a 500k/50k stratified subsample of the 2.27 M /
   894 k windows (every positive kept); benchmarks at 60 trajectories per member (serving:
   200). One M1 laptop, ~4.8 h per five-seed run.
8. **One dataset.** CIC-IDS2017 only; CSE-CIC-IDS2018 excluded (size, no valid temporal
   mapping in the time available), CTU-13 identified as the next candidate
   (`reports/DATASET_ASSESSMENT_2026-09-20.md`).

## Claims discipline

This system performs **learned dynamics** and **temporal forecasting**, not causal
inference. `nidra/explain/counterfactual.py` outputs are labelled `"model-internal
what-if"` everywhere. The stage → ATT&CK tactic and label → technique tables
(`nidra/data/attack_mapping.py`, `docs/ATTACK_MAPPING.md`) are a curated presentation
mapping validated against the published attack descriptions, not technique-level ground
truth observed in the traffic. See `README.md`'s "Claims discipline" section.

## Reproduction

```bash
cd ml
python -m nidra.scripts.run_experiment --label production --seeds 0,1,2,3,4 \
    --stages dynamics,heads,onset,gru_baseline --epochs 24
python -m nidra.eval.benchmark --split val --select-operating-point --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split test    --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split holdout --n-samples 60 --n-resamples 300
python -m nidra.scripts.report_tables --run . --splits val,test,holdout --out reports/run8/benchmark_tables.md
python -m nidra.scripts.plot_benchmark --run . --splits val,test,holdout --out reports/run8
```
