# CTU-13 integration and model improvement — running experiment log

**Branch** `research/run9-ctu13`. **Baseline** Run 8, frozen at tag `baseline-run8`
(`ad1430c`); no Run 8 artifact, metric or report is modified by anything here.

This file is written as the experiments run, not afterwards. Results appear in the order
they were measured, including the ones that did not help. The final synthesis lives in
`reports/RUN9_FINAL_REPORT_<date>.md`; this is the log it is built from.

---

## 0. What Run 8 left open

Run 8's own diagnosis (`REAL_DATA_RESULTS.md` §8.7, `DECISIONS.md` D115) was specific:
the transition model learns useful dynamics — state-forecast skill over persistence 0.587
on test and 0.616 on holdout — while the **per-state risk head** is the ceiling. On
Friday it ranked Bot-C2 windows below silence (ROC-AUC 0.37) where a GRU sequence
classifier on the identical rows reached 0.976. Two questions follow, and neither can be
answered inside CIC-IDS2017:

1. Is the head's failure on botnet C2 a property of the model, or of one dataset whose
   botnet traffic is five workstations on one afternoon?
2. Does the project have anything to say about **advance warning**? CIC-IDS2017 supplies
   3 / 9 / 15 independent pre-onset episodes at 1 / 5 / 30 minutes. That is not enough to
   measure a lead time, however the metric is computed.

CTU-13 exists in this phase to answer both, and to make "generalises to unseen attack
families" a cross-dataset claim rather than a within-dataset one.

---

## 1. Dataset (summary; full audit in `reports/CTU13_DATASET_ASSESSMENT_2026-09-23.md`)

13 Argus `.binetflow` captures, CVUT, 10–19 August 2011, 19,976,700 flows, 2.73 GB, seven
botnet families. Eleven captures window at Δ=60 s; scenarios 7 and 11 (0.35 h and 0.27 h)
are too short for L=30 + K=6 and are excluded.

Two findings shaped everything downstream.

**The public PCAPs are botnet-only.** The full captures were withheld for privacy, so any
packet feature derived from them would be a perfect label — only infected hosts would
have one. CTU-13 is therefore treated as flow-only, and the eleven packet aggregates plus
`iat_max` and `d_retrans_rate` are *unavailable*, not missing. They are declared drops in
a named feature regime (`cross_core`, 32 of 45; `cross_strict`, 24) rather than zero-filled
(D117).

**CTU-13 has pre-onset structure that CIC-IDS2017 does not**: 160 independent pre-onset
episodes across the 1–60 minute horizons, against CIC's 3 / 9 / 15. The qualification
matters and is stated wherever the number is used — 10 hosts in 4 captures produce most
of them, and 5 of the 11 usable captures contribute none.

---

## 2. Experiment 1 — does the transition model learn CTU-13's dynamics?

`experiments/runs/ctu_dyn`, seed 0, 20 epochs (early stopped at 17, best epoch 9),
`config/ctu13.yaml`, regime `cross_core`, 30 of 45 features kept after two constants
(`urg_ratio`, `reciprocity`) were dropped on the training rows. Train captures 1, 2, 3;
validation captures 4 and 6 (held out, not carved — D118). 500,000 training sequences,
1,384 positive; 50,000 validation, 283 positive. 45 min wall clock on the M1.

Free-running rollout on validation, against persistence:

| k (minutes ahead) | 1 | 2 | 3 | 4 | 5 | 6 | mean |
|---|---|---|---|---|---|---|---|
| MSE skill vs persistence | 0.609 | 0.564 | 0.490 | 0.486 | 0.457 | 0.471 | **0.515** |
| 90% interval coverage | 0.956 | 0.952 | 0.948 | 0.949 | 0.949 | 0.945 | 0.950 |

**Read:** the transition model reduces six-step state-forecast error by roughly half
against persistence on a dataset it has never seen before this phase, on 32 features
rather than 45, and its predictive intervals are calibrated to within half a point of
nominal at every horizon. The skill decays smoothly with k (0.61 → 0.47), which is what a
model that has learned dynamics looks like; a flat curve would be the leakage signature
`CLAUDE.md` warns about. For scale, Run 8's CIC figures are 0.587 (test) and 0.616
(holdout) on the full 45 features — the same order, and the gap is the expected cost of a
13-feature mask plus a harder, noisier capture set.

This is a state-forecasting result. It says nothing yet about attack-risk forecasting,
which is what §3 measures and where Run 8 failed.

---

## 3. Experiment 2 — the history-aware risk head (in progress)

Six risk-head variants on **one** frozen dynamics run (`ctu_dyn`), so the only thing
that differs between them is the head's input:

`state` (the Run 8 architecture exactly) · `hidden` · `state+hidden` · `state+logvar` ·
`state+hidden+logvar` · `state+hidden+delta+logvar`

Selection is on the CTU **validation** captures and nowhere else. Test and holdout are
not read until one variant is chosen and frozen.

_Results appended as each variant's benchmark completes._

---

## 4. Corrections made to the evaluation harness itself

**Observation time** (D122). `false_alarms_per_hour` divided by
`max(window_ts) − min(window_ts)`, which is the observation time only of a split that is
one continuous capture. CTU-13's splits are captures made on different days: the naive
range overstates the test split by 1.66× and the **holdout by 5.02×** (91.03 h of range
over 18.13 h of capture), which would have divided the CTU false-alarm rate by five.
Observation time is now distinct window timestamps × Δ. On the real CIC tables the change
is +0.25% (test 8.05 → 8.07 h), so Run 8's rates stand and the two phases remain
comparable.

**Attribution through the head that produced the score.**
`explain/forecast_attribution.py` scored through `score_states`, which cannot pass a
history-aware head its hidden state or log-variance. It now goes through
`score_trajectory`.

**Serving built its own architecture.** `NidraPredictor._build_model` constructed a
`WorldModel` by hand, so a config declaring a history-aware head would have produced a
per-state head at serving time and a trajectory head everywhere else. It now uses the
shared constructor.

None of these changes alter a Run 8 number by more than 0.25%, and none of them touch a
Run 8 artifact.

### 3.1 The Run 8 head on CTU-13 (`ctu_heads__state`, validation)

Benchmarked first, on its own, because everything else is measured against it. Seed 0,
60 trajectories, natural prevalence 0.00326, 20,338 scored rows, operating point selected
here (`mean|max`, threshold 0.182).

| system | AP | 95% CI | ROC-AUC | P / R / F1 @thr | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|
| **gbdt_current_state** | **0.463** | [0.002, 0.745] | 0.766 | 0.11 / 0.53 / 0.18 | 0.631 | 0.001 / 0.002 |
| world model (stochastic) | 0.412 | [0.001, 0.737] | 0.760 | 0.41 / 0.49 / 0.45 | 0.565 | 0.000 / 0.001 |
| world model (calibrated) | 0.408 | — | 0.763 | 0.66 / 0.45 / 0.54 | 0.559 | 0.000 / 0.001 |
| persistence + learned noise | 0.402 | [0.001, 0.730] | 0.733 | 0.33 / 0.50 / 0.40 | 0.553 | 0.000 / 0.001 |
| **oracle: head on the true future** | **0.381** | — | 0.788 | 0.17 / 0.55 / 0.26 | 0.482 | 0.022 / 0.010 |
| world model (deterministic) | 0.374 | [0.002, 0.705] | 0.770 | 0.36 / 0.48 / 0.41 | 0.507 | 0.001 / 0.002 |
| ridge two-lag + head | 0.368 | [0.001, 0.729] | 0.744 | 0.07 / 0.51 / 0.12 | 0.505 | 0.001 / 0.001 |
| persistence (rollout, transition disabled) | 0.350 | [0.001, 0.690] | 0.726 | 0.37 / 0.49 / 0.42 | 0.479 | 0.001 / 0.002 |
| persistence (head on S_t) | 0.350 | [0.001, 0.690] | 0.726 | 0.37 / 0.49 / 0.42 | 0.479 | 0.001 / 0.002 |
| logistic regression on S_t | 0.346 | [0.001, 0.720] | 0.723 | 0.04 / 0.53 / 0.08 | 0.475 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.277 | [0.001, 0.699] | 0.751 | 0.21 / 0.53 / 0.30 | 0.355 | 0.002 / 0.003 |

Paired episode-bootstrap margins: world model − persistence **+0.063 [−0.000, +0.119]**;
− ridge +0.044 [−0.036, +0.112]; − deterministic +0.039 [−0.003, +0.083];
− **gbdt −0.051 [−0.158, +0.014]**. Only eight episodes carry the validation split, so
every interval here is wide; none excludes zero.

Four things are worth stating plainly.

**The transition model helps, the head limits it.** `persistence_rollout` — the identical
forward simulation with only the predicted change removed — scores 0.350, exactly the same
as the per-state persistence baseline, as it must for a head that reads nothing but the
state. The full rollout scores 0.412 on top of that, and the deterministic-state skill
against persistence is 0.548. The dynamics are doing work.

**The oracle is BELOW the forecast (0.381 < 0.412).** This is the Run 8 diagnosis
reproduced on a second dataset and a second attack population. The oracle applies the same
head to the *true* future state, so it is the ceiling this head can reach with perfect
state prediction — and it is lower than what the model reaches by averaging 60 sampled
trajectories. Better state forecasting cannot help this head; the head is not extracting
risk from a state vector. There is no version of this result in which the limitation is
the transition model.

**A gradient-boosted tree on the current observed state beats the whole system** (0.463 vs
0.412, margin −0.051 [−0.158, +0.014]). Reported because it is true. It is also a
detection result, not a forecasting one — GBDT's onset AP is 0.002 against the oracle's
0.022 — but "our forecast ranks attack windows worse than a tree on the current state"
is the baseline comparison §23 exists to force, and it has to appear in the headline
table rather than a footnote.

**No advance warning.** Of the 8 validation episodes, 0 were warned before onset; 2 were
detected inside the episode, median latency 3.5 minutes, no lead time at all. Onset AP at
every horizon is within noise of zero while the oracle reaches 0.022 — so even a perfect
state forecast would give almost nothing here, which again points at the head.

Per attack group (one-vs-rest, `<capture>:<stage>`):

| group | positives | episodes | AP | ROC-AUC | recall@thr | oracle AP | persistence AP | state skill |
|---|---|---|---|---|---|---|---|---|
| ctu_6:exfil (Menti) | 116 | 1 | 0.768 | 0.990 | 0.879 | 0.651 | 0.640 | 0.483 |
| ctu_4:exfil (Rbot) | 61 | 5 | 0.207 | 0.720 | 0.410 | 0.149 | 0.139 | 0.382 |
| ctu_4:c2 (Rbot) | 56 | 6 | 0.001 | 0.528 | 0.000 | 0.005 | 0.001 | 0.150 |
| ctu_4:recon (Rbot) | 44 | 6 | 0.001 | 0.558 | 0.000 | 0.001 | 0.001 | 0.435 |
| ctu_6:c2 (Menti) | 6 | 1 | 0.000 | 0.478 | 0.000 | 0.052 | 0.000 | 0.096 |

The whole validation result is one Menti exfiltration episode. Every C2 and recon group is
at or below chance — **the same failure mode as Run 8's Bot-C2 on Friday, on a different
dataset, a different family and a different year.** That is now a property of the head,
not of CIC-IDS2017.


### 3.2 Head ablation, stage A: validation AP on observed states

All six variants sit on the identical frozen `ctu_dyn` dynamics, the identical
scaler, the identical rows. The only thing that differs is what the head reads.
This is the head-training selection metric — natural-prevalence AP on the CTU
validation captures' observed states — which is cheap enough to screen all six and
is *not* the forecast number; that is §3.3.

| head input | validation AP | Δ vs Run 8 head | input dim | best epoch |
|---|---|---|---|---|
| `hidden` | **0.496** | **+0.143** | 128 | 13 |
| `state+hidden` | 0.489 | +0.136 | 173 | 7 |
| `state+hidden+delta+logvar` | 0.470 | +0.117 | 263 | 8 |
| `state+hidden+logvar` | 0.454 | +0.101 | 218 | 7 |
| `state+logvar` | 0.373 | +0.020 | 90 | 1 |
| `state` (Run 8 architecture) | 0.353 | — | 45 | 28 |

Three answers fall out, and two of them are answers to questions the roadmap
asked directly.

**History is the missing input, and it is worth 40% relative.** Run 8's diagnosis
said the per-state head could not see what a GRU sequence classifier could. Giving
the head the encoder's hidden state — the same recurrent summary, under the same
frozen-head discipline, trained on observed states only — recovers most of that
gap. This is the phase's central hypothesis and on this screening metric it holds.

**Uncertainty helps, but barely** (§6). `state+logvar` is +0.020 over `state`, and
adding `logvar` on top of `hidden` makes things *worse* (0.454 vs 0.489). The
transition's predicted variance carries a little signal about risk and mostly
carries parameters. The honest answer to "does uncertainty help" is: measurably,
by about a seventh of what history is worth, and not additively.

**More components is not better.** `hidden` alone beats every richer variant. With
1,384 positive training sequences, a 263-dimensional input is being fit on very
little; the best-epoch column shows it — the state-only head needed 28 epochs, the
history-aware ones converge by 7–13. Nothing here supports adding `delta`.

These are screening results on one seed (§19 Stage A). No architecture is adopted
on them: the forecast benchmark is the selection metric, and the winner then gets
multi-seed confirmation.

