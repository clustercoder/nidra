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

### Where to find the answers

This is an append-only log, so its conclusions are spread across it and some of
them retract earlier entries. Three companion documents collect them:

- **`RUN9_QUESTIONS.md`** — the roadmap's fourteen questions, each with its
  status (answered / partial / pending) and the section it rests on.
- **`RUN9_NEGATIVE_RESULTS.md`** — the nine things that did not work and what
  each one changed about the next experiment.
- **`tables/experiment_matrix.md`** — every run on disk, generated from its own
  provenance record.

The findings that later sections depend on, in the order they bind:

| | finding | where |
|---|---|---|
| 1 | Train and validation have **one infected host each**, the same address the model trained on. No validation number in this phase is cross-host; both test splits carry nine hosts never infected in training. | §3.14, §3.17 |
| 2 | The history-aware head's advantage is **timing signal on CIC and largely host identity on CTU** — aggregates agree across the datasets, the decomposition does not. | §3.11 |
| 3 | The risk head ranks two attack stages **below chance** where the frozen stage head ranks them at 0.85 and 0.64. The signal is in the model; no scalar fusion recovers it. | §3.9 |
| 4 | A fifth of CTU's forecast positives are the **silence floor** — bit-identical to 47,650 negatives. CTU cannot support an advance-warning claim; CIC weakly can. | §3.10, §3.15 |
| 5 | Two conclusions **withdrawn**: "Rbot C2 is not separable" (§3.8) and "cross-host by construction" (§3.17). Both are struck through rather than deleted. | §3.8, §3.17 |
| 6 | The history-aware head **could not be served at all** until D141, and once served, SHAP over the 45 named features attributes a risk of 1.00 to nothing larger than 0.0003. The AP gain costs the project's explainability. | §3.20 |
| 7 | The history-aware head is also the **worse-calibrated** one, and the only within-dataset arm whose calibrated score loses to predicting the prevalence and never moving. Second cost on the same candidate. | §3.23 |
| 7b | **Stage B, three seeds on the forecast benchmark:** the history-aware head halves false alarms and wins F1 — and its within-host ROC *falls* (0.816 → 0.697) while its margin over persistence is −0.0100 [−0.0342, +0.0034]. It wins the alert, not the forecast. A GRU classifier on identical rows beats both. | §3.29, §3.32 |
| 8 | **The oracle is not an upper bound on the deployed system** — beaten in 11 of 28 cells even by the uncalibrated sampled arm. ~~Every "% of oracle" statement has to be against the deterministic arm.~~ *Withdrawn in §3.40: the deterministic arm is below chance in 12 of 24 cells and cannot serve as a reference.* Two cells break the bound and remain unexplained. | §3.24, §3.26, §3.27, §3.40 |
| 9 | **The rollout manufactured state in features that carry no gradient.** The transition loss masks dropped features, so the network is untrained there; the rollout fed that output back, compounding to rms 2.03 by k=6 in slots that are exactly zero in the input and in the truth — more magnitude than the real features carry. Every CTU and cross-dataset number in this report was produced under it. Run 8 is unaffected. | §3.37 |
| 10 | **All 28 recorded cells reproduce bit-identically**, and the re-run took confidence-interval coverage from 6 of 28 to 28 of 28. | §3.36 |
| 11 | On CIC the **decomposable `state+logvar` head reaches 99.1%** of the history-aware head, against 14% on CTU — so finding 6's explainability cost is dataset-dependent, not intrinsic. One seed, and the arm most exposed to finding 9. | §3.38 |
| 12 | **On an unseen attack family (Neris withheld) the world model loses to persistence** — 0.717 against 0.772 — while its within-host ROC is 0.949 across ten infected hosts. | §3.39 |

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


---

## 5. What "the whole dataset" means here (§28)

Generated by `nidra.scripts.data_accounting` from the same cached tables the training
run reads (`reports/tables/ctu13_data_accounting.md`), so it cannot drift from what was
actually trained.

| split | captures | raw flows | canonical states | active | eligible origins | positives | sampled / epoch | coverage / epoch | mean touches | state exposures |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 3 | 9,343,396 | 1,419,278 | 244,235 | 1,401,778 | 1,384 | 500,000 | 35.7% | 7.1 | 300,000,000 |
| val | 2 | 1,679,995 | 98,876 | 35,014 | 86,941 | 283 | 50,000 | 57.5% | 11.5 | 30,000,000 |
| test | 3 | 6,351,529 | 508,517 | 123,963 | 492,417 | 3,688 | — | — | — | — |
| holdout | 3 | 2,380,452 | 267,388 | 56,866 | 253,983 | 1,040 | — | — | — | — |

11 usable captures, 19,755,372 raw flows, 2,294,059 canonical per-host states of which
460,078 are active, 2,235,119 eligible sequence origins of which 6,395 are positive.

The sampler draws 500,000 of the 1,401,778 eligible training origins per epoch — 35.7% —
and over 20 epochs an origin is drawn 7.1 times on average, so effectively all of the
training population is reached, repeatedly. 330 million state vectors pass through the
encoder over a run. Nothing here is a subsample chosen for convenience: the cap exists
because the uncapped `[N, 30, 45]` float32 tensor for 1.4M origins is 7.6 GB and the
machine has 16.

The number that governs everything downstream is the last one on the training row:
**1,384 positive origins.** Every head in §3 is fit on that, which is why a
263-dimensional head does worse than a 128-dimensional one, and why every confidence
interval on the validation split is wide enough to contain zero.

### 3.3 Head ablation, stage B: the forecast benchmark on CTU validation

The screening metric in §3.2 scores a head on observed states. This is the
system: stochastic rollout, 60 trajectories, pooling and threshold selected on this
split from nine candidates, natural prevalence 0.00326, 20,338 rows, 8 episodes.

`state` and `hidden` first, because the difference between them is not a change in
one number — it is a change in the shape of the result.

| | `state` (Run 8) | `hidden` |
|---|---|---|
| world model, pooled AP | 0.412 [0.001, 0.737] | 0.459 [0.001, 0.812] |
| world model, calibrated | 0.408 | **0.463** |
| persistence (head on S_t) | 0.350 | **0.489** |
| persistence rollout (transition disabled) | 0.350 | 0.459 |
| oracle (head on the true future) | 0.381 | 0.508 |
| gradient-boosted trees on S_t | 0.463 | 0.463 |
| P / R / F1 at the operating point (calibrated) | 0.66 / 0.45 / 0.53 | **0.77 / 0.48 / 0.59** |
| false alarms / h (calibrated) | 10.22 | **6.39** |
| ΔAP world model − persistence | +0.063 [−0.000, +0.119] | **−0.031 [−0.080, +0.007]** |

Two of those lines point in opposite directions and both are real.

**The oracle gap inverts, which is the result this phase was run to get.** With the
Run 8 head the oracle scores 0.381 against the forecast's 0.412 — *below* it. A head
that does worse when handed the true future is a head that is not reading the state;
no improvement to the transition model could have helped it. With the history-aware
head the oracle is 0.508 against 0.459, a gap of +0.049 in the right direction. For
the first time in this project, better state forecasting would now translate into
better risk forecasting.

**But pooled AP no longer beats the head applied to the present.** `persistence` —
the same history-aware head on the observed origin — scores 0.489 against the
rollout's 0.459, and the rollout with the transition disabled scores 0.459 too. The
cost is not what the transition predicts; it appears as soon as the encoder ingests
*any* synthetic window. The head was fit on hidden states the encoder reached over
real observations and is asked at inference about hidden states it reached over six
of its own. That is textbook exposure mismatch, and it is the motivation for the
`context_noise` sweep in §3.5.

**The per-horizon table is where the improvement actually lives.** Task C asks the
unpooled question — is the window at t+k an attack window? — at each horizon
separately, so it is not a max over six correlated scores:

| k (min ahead) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `state` head, AP | 0.560 | 0.519 | 0.442 | 0.363 | 0.297 | **0.209** |
| `state` head, oracle AP | 0.491 | 0.483 | 0.488 | 0.526 | 0.501 | 0.476 |
| `hidden` head, AP | 0.602 | 0.617 | 0.586 | 0.547 | 0.539 | **0.512** |
| `hidden` head, oracle AP | 0.654 | 0.637 | 0.647 | 0.650 | 0.671 | 0.686 |
| state-forecast skill vs persistence | 0.635 | 0.598 | 0.524 | 0.515 | 0.498 | 0.506 |

The Run 8 head collapses across the horizon — 0.560 to 0.209, −63% — while its oracle
stays flat near 0.49. Flat oracle, collapsing forecast: all of the loss is compounding
state error, and the ceiling is low everywhere. The history-aware head degrades from
0.602 to 0.512, −15%, and **beats the Run 8 head by 2.4× at k=6**. Its oracle sits
above it at every horizon and the gap widens with k (0.05 at k=1 to 0.17 at k=6),
which is the signature of rollout error compounding against a head that can use the
difference.

So the pooled-AP line and the per-horizon line are not in conflict. Pooled AP takes a
max over six horizons and the present-state baseline is a single clean score, which
flatters the baseline; the forecasting question the project exists to answer is asked
per horizon, and there the history-aware head is decisively better and the transition
model is doing work at every k.

**Per attack group**, `hidden` against `state`: ctu_6:exfil 0.857 (from 0.768, ROC
1.000), ctu_4:exfil 0.216 (from 0.207), ctu_6:c2 0.039 at ROC 0.999 on 6 positives
(from 0.000 at ROC 0.478), ctu_4:c2 0.001 (unchanged), ctu_4:recon 0.001 (unchanged).
Menti's C2 goes from below chance to essentially perfect ranking on a handful of
windows; Rbot's C2 and recon stay at chance under both heads. The failure is narrower
than it was but it has not gone away.

**Advance warning is unchanged: 0 of 8 episodes warned before onset** under either
head, median detection latency 3.5 min (`state`) and 4.5 min (`hidden`). Whatever the
history-aware head fixed, it did not produce lead time.

### 3.4 All six variants, and the selection

Generated by `nidra.scripts.compare_runs`
(`reports/tables/ctu_head_ablation_val.md`). Ordered by pooled AP of the
stochastic rollout; every row is seed 0 on `ctu_dyn`, benchmarked on the CTU
validation captures with its own validation-selected operating point, so the
P/R/F1 and FA/h columns are each at a different threshold and rank nothing.

| variant | AP | ΔAP vs persistence | ΔAP vs persistence_rollout | calibrated AP | ROC-AUC | oracle AP | gap | AP at k=6 |
|---|---|---|---|---|---|---|---|---|
| **state+hidden** | **0.473** | −0.008 [−0.021, +0.004] | **+0.033 [+0.0002, +0.055]** | **0.478** | 0.814 | 0.506 | +0.034 | **0.574** |
| hidden | 0.459 | −0.031 [−0.080, +0.007] | −0.000 [−0.038, +0.054] | 0.463 | 0.757 | 0.508 | +0.049 | 0.512 |
| state+hidden+logvar | 0.436 | −0.025 [−0.054, +0.002] | — | 0.452 | 0.780 | 0.485 | +0.049 | — |
| state+hidden+delta+logvar | 0.430 | −0.036 [−0.073, +0.000] | — | 0.435 | 0.788 | 0.482 | +0.052 | — |
| state (Run 8) | 0.412 | +0.063 [−0.000, +0.119] | +0.063 [−0.000, +0.119] | 0.408 | 0.760 | 0.381 | **−0.031** | 0.209 |
| state+logvar | 0.396 | +0.019 [−0.011, +0.032] | — | 0.393 | 0.821 | 0.416 | +0.020 | — |

**Selected: `state+hidden`.** Not because it has the top AP by 0.014 — with
283 positives and 8 episodes on this split, 0.014 is noise — but because it is
the only variant that is best or near-best on every axis that is not noise:

> **Read this section with §3.11, §3.14 and §3.17.** The selection stands: it was
> made on validation, which is the protocol, and the reasons below are the
> reasons. What later sections change is the *interpretation* of the margin.
> Every number in this table comes from a split whose 288 positives sit on a
> single host — `147.32.84.165`, the same address the model trained on — and
> §3.11 shows that on CTU most of `state+hidden`'s aggregate advantage is
> better recognition of that host rather than of the moment. The
> `persistence_rollout` margin is about the transition model and is not
> affected by that; the head-to-head AP ordering is.


* the **only** variant whose margin over `persistence_rollout` excludes zero
  (+0.033 [+0.0002, +0.055]). That is the strict ablation: the same head, the
  same encoder advance, only the predicted change removed. It is the first
  interval in this project that isolates the transition model's own
  contribution to risk and does not contain zero. It just barely excludes it,
  on one seed, and is reported as a screening result until the §19 Stage B
  multi-seed run confirms it (queued; the reference here originally pointed at
  §3.6, which the numbering later gave to the onset 2×2);
* the flattest horizon curve (0.631 → 0.574, −9%, against the Run 8 head's
  0.560 → 0.209, −63%) and **2.7× the Run 8 head at k=6**;
* an oracle above it at every horizon, so better state forecasting would now
  translate into better risk forecasting;
* the best served operating point of the six: calibrated P/R/F1
  0.79 / 0.48 / 0.60 at **5.7 false alarms per hour**, against the Run 8
  head's 0.66 / 0.45 / 0.53 at 10.2;
* and it keeps a state component, so feature-level attribution over the input
  remains meaningful rather than being a claim about a 128-dimensional vector.

**Uncertainty does not help the forecast** (§6, answered). `logvar` raised the
observed-state screening metric (`state+logvar` 0.373 vs `state` 0.353) and
*lowered* the forecast benchmark (0.396 vs 0.412); added to `state+hidden` it
costs 0.037. The two metrics disagree and the forecast one is the one that
matters. The plausible reading is that predicted variance is informative about
the *state* and the head spends capacity on it that it needs elsewhere — 90
extra input dimensions against 1,384 positive training origins. Recorded as a
negative result; `logvar` is not adopted.

Adding `delta` costs another 0.006 on top of that. Nothing beyond
`state+hidden` earns its parameters.

### 3.5 Does the history-aware head actually use history? (§27)

The ablation says the hidden state is worth +0.061 AP. That is a claim about a
number. Integrated gradients through the rollout back to the input history
answer the mechanism question: 24 highest-risk validation origins, the same
origins for every head, `reports/tables/ctu_explanation_comparison.md`.

| head | share of \|attribution\| in the last 5 of 30 windows | attribution-weighted mean age | drop from deleting the top-8 cells | drop from 8 random cells | beats random |
|---|---|---|---|---|---|
| `state` | 0.631 ± 0.341 | 6.1 ± 5.9 min | 0.579 ± 0.490 | 0.006 ± 0.020 | 0.75 |
| `hidden` | 0.342 ± 0.252 | 11.2 ± 4.9 min | 0.100 ± 0.270 | 0.005 ± 0.023 | 0.45 |
| `state+hidden` | 0.368 ± 0.244 | 11.4 ± 4.5 min | 0.055 ± 0.199 | 0.000 ± 0.000 | 0.67 |

Uniform attribution over a 30-window history would be 0.167 recent and 14.5
minutes of mean age.

**Yes — and it is not a small effect.** The Run 8 head puts 63% of its
attribution in the last five minutes of a thirty-minute history and has a mean
attribution age of 6 minutes. The history-aware heads put 34–37% there and
reach back 11 minutes. They are not a per-state head with extra parameters;
they read roughly twice as far into the past.

**The cost is that they are harder to explain with a short list.** Deleting the
eight highest-attributed cells of 1,350 collapses the Run 8 head's forecast by
0.579 — its score depends on a handful of cells — and moves the history-aware
heads by 0.100 and 0.055. That is diffuse dependence rather than unfaithful
attribution, but it has a practical consequence: a top-8 explanation of a
history-aware forecast is a much weaker statement, and for `hidden` alone the
top-8 beats a random 8 only 45% of the time, which is no better than chance.
`state+hidden` retains 67%, another reason to prefer it over `hidden`.

This is reported as a property of the selected model, not as a selection
signal — no head was chosen on it.

### 2.1 What the cross-dataset feature mask costs, measured on CIC-IDS2017

`cic_core_dyn` is the control arm: the Run 8 protocol with the 13 features CTU-13
cannot produce declared as drops. Without it, any CIC→CTU number confounds the
domain change with the feature change.

| run | features | free-running skill vs persistence, mean over k | k=1 → k=6 | 90% coverage |
|---|---|---|---|---|
| Run 8, CIC, all 45 (test / holdout) | 45 | 0.587 / 0.616 | — | — |
| `cic_core_dyn`, CIC, masked (val) | 32 | 0.495 | 0.618 → 0.417 | 0.984 |
| `ctu_dyn`, CTU, masked (val) | 30 | **0.515** | 0.609 → 0.471 | 0.950 |

Two things worth stating. The mask costs CIC roughly 0.09–0.12 of state-forecast
skill, so a CIC→CTU transfer number has to be read against 0.495, not against Run
8's 0.587. And on the same feature set **CTU-13's dynamics are not harder than
CIC-IDS2017's** — 0.515 against 0.495, with better-calibrated predictive intervals
(0.950 against 0.984 for a nominal 0.90; CIC's are too wide at this feature set).
Whatever makes CTU hard for this project, it is not the state-forecasting problem.

### 3.6 Advance warning: the onset head 2×2 (§10, §11)

Four onset heads on the same frozen `ctu_dyn` dynamics: what the head reads
(`state` vs `state+hidden`) crossed with how its horizons are parameterised
(independent per-horizon BCE, the Run 8 form, vs the discrete-time hazard).
The architecture is identical in every cell — only the input width and the loss
differ — which is what makes this an ablation rather than four models.

The population is the binding constraint and has to be stated first. Of
1,417,909 eligible training origins (outside every episode, with L windows of
history) the positives are **10 / 30 / 50 / 100 / 143 / 263** at 1 / 3 / 5 / 10 /
15 / 30 minutes. Validation has 98,627 eligible origins and **8 / 24 / 40 / 75 /
90 / 105**. A base rate of 0.008% at one minute is what any of these numbers has
to be read against.

Validation AP at natural prevalence, with the lift over the base rate in
brackets:

| head / parameterisation | 1 min | 3 min | 5 min | 10 min | 15 min | 30 min |
|---|---|---|---|---|---|---|
| `state`, independent (Run 8) | 0.0002 (2.5×) | 0.0005 (2.1×) | 0.0008 (2.0×) | 0.0015 (2.0×) | 0.0016 (1.8×) | 0.0019 (1.8×) |
| `state`, hazard | 0.0002 (2.5×) | 0.0005 (2.1×) | 0.0008 (2.0×) | 0.0014 (1.8×) | 0.0016 (1.8×) | 0.0019 (1.8×) |
| `state+hidden`, independent | 0.0002 (2.5×) | 0.0010 (4.1×) | 0.0014 (3.5×) | 0.0020 (2.6×) | 0.0022 (2.4×) | 0.0024 (2.3×) |
| `state+hidden`, hazard | 0.0002 (2.5×) | 0.0007 (2.9×) | 0.0009 (2.2×) | 0.0014 (1.8×) | 0.0015 (1.6×) | 0.0017 (1.6×) |

| base rate | 0.00008 | 0.00024 | 0.00041 | 0.00076 | 0.00091 | 0.00106 |

**There is pre-onset signal on CTU-13, and it is small.** The Run 8 onset head
ranks pre-onset origins 1.8–2.5× better than chance; the history-aware one
reaches 3.4–4.1× at the 3 and 5 minute horizons, roughly doubling the lift where
lead time would actually be useful. That is a real effect and it is the second
place in this phase where the encoder's hidden state is what supplies it.

**It is also nowhere near a usable warning.** Four times a 0.04% base rate is
0.14%. At the risk head's operating point, 0 of the 8 validation episodes were
warned before onset under any configuration; detection happens 3.5–4.5 minutes
*into* the episode. The honest statement stays the one Run 8 made: NIDRA has not
demonstrated advance warning, on either dataset. What CTU-13 adds is that the
ceiling is a data property — 10 positive training origins at one minute — and not
obviously a model property.

**The hazard parameterisation is not adopted** (§11, answered with the ablation it
asks for). It is identical to independent BCE for the state head and slightly
*worse* for the history-aware one (0.0007 vs 0.0010 at 3 min). Its argument was
coherence — independent per-horizon BCE can and did report P(within 1 min) above
P(within 30 min) — and it delivers that by construction. But it fits each bucket
only on the rows still at risk in it, and the buckets hold 10 / 20 / 20 / 50 / 43
/ 120 events; splitting 263 events six ways costs more variance than the
monotonicity is worth here. Recorded as a negative result. It stays in the code
behind `onset.parameterisation` because the argument would come back the moment a
dataset with more onsets appears.


### 3.7 Context noise: the fix works, and it erases the thing it was meant to protect

§3.3 measured the exposure mismatch: the head is fit on hidden states the encoder
reached over real observations and asked at inference about hidden states it
reached over six of its own. `train_heads.context_noise` applies the regularizer
the state component already carries to the context components, scaled per
component by its own batch standard deviation. Same head (`state+hidden`), same
frozen `ctu_dyn` dynamics, same rows.

| σ | world model AP | calibrated | persistence | persistence_rollout | oracle | **ΔAP world model − persistence_rollout** | AP at k=6 |
|---|---|---|---|---|---|---|---|
| **0.0** (adopted) | 0.473 | 0.478 | 0.481 | 0.440 | 0.506 | **+0.033 [+0.0002, +0.055]** | 0.574 |
| 0.1 | 0.480 | 0.484 | 0.500 | 0.490 | 0.536 | −0.010 [−0.075, +0.020] | 0.585 |
| 0.3 | 0.455 | 0.462 | 0.484 | 0.474 | 0.509 | −0.018 [−0.084, +0.030] | 0.507 |

**Not adopted.** At σ=0.1 it does what it was designed to do: the rollout's AP
rises 0.473 → 0.480, the oracle rises 0.506 → 0.536, the k=6 tail rises 0.574 →
0.585. It also lifts `persistence_rollout` from 0.440 to 0.490 — and that is the
whole story. Making the head robust to a perturbed hidden state makes it robust
to *which* windows the encoder ingested, so a rollout that freezes the state
scores as well as one that predicts it, and the transition ablation margin goes
from +0.033 [+0.0002, +0.055] to −0.010 [−0.075, +0.020].

The exposure mismatch and the transition model's contribution turn out to be the
same quantity seen from two sides. A head that cannot tell the difference between
ingesting S_t six times and ingesting six predicted states does not suffer from
the mismatch and does not benefit from the prediction either. Buying +0.007 AP —
comfortably inside the noise of 283 positives — at the cost of the only
statistically supported transition signal in the project is not a trade worth
making. σ=0.3 is worse on every column.

Recorded as a negative result. `context_noise` stays in the code, defaulting to
0.0, because the diagnosis it came from is correct and a different mechanism for
the same problem — one that closes the gap without flattening the head's
sensitivity to its input — is the obvious next thing to try.

### 3.8 Why is Rbot invisible? A supervised probe says: two different reasons (§26, §31 Q13)

Every head scores `ctu_4:c2` at AP 0.001 and ROC below chance. Before blaming the
model, `nidra.scripts.group_separability` fits a gradient-boosted probe on the
**training** captures' rows for that attack *stage* and scores the validation
group. Fitting on captures 1–3 and scoring capture 4 makes it cross-capture and
cross-family — it asks much of what the model is asked. It reads labels the model
does not, so it is an upper bound in the same sense the oracle is: a diagnostic,
never a system, and nothing in the pipeline reads it.

**Correction (§3.17).** This paragraph originally said "cross-host and
cross-capture by construction, so it cannot answer with host identity". That is
wrong. CTU-13 reuses the same infected address, `147.32.84.165`, across
scenarios: it is the infected host in captures 1, 2 and 3 (train) *and* in 4 and
6 (validation). The probe transfers across captures and malware families, not
across hosts. Its one-vs-rest construction does penalise a pure
host-recognition strategy — that host's own benign windows are negatives — but
that is a partial control, not the structural guarantee claimed here.

| group | positives | positive hosts | probe AP | probe ROC | world model ROC | verdict |
|---|---|---|---|---|---|---|
| ctu_6:exfil (Menti) | 122 | 1 | 0.814 | 0.992 | 1.000 | model at the ceiling |
| ctu_4:exfil (Rbot) | 50 | 1 | 0.091 | 0.829 | 0.704 | partial gap |
| ctu_4:recon (Rbot) | 17 | 1 | 0.033 | **0.970** | ~0.42 | **model failure** |
| ctu_4:c2 (Rbot) | 23 | 1 | 0.008 | **0.443** | ~0.47 | probe missed it — ~~not separable~~ **see §3.9** |

**Correction (§3.9).** This section originally read the C2 row as a
representation limit and concluded that no head architecture reaches it. That
conclusion is withdrawn. NIDRA's own frozen stage head — trained on the same
training captures, scored on the same 23 windows, cross-host and cross-capture by
the same construction — ranks them at ROC **0.864**. The probe was the weaker
learner, not the ceiling. The original text is kept below with the error marked,
because a retracted conclusion that quietly disappears is worse than one that is
visibly wrong.

> ~~**Rbot C2 is not there to be found.** A probe handed the C2 label from the
> training captures ranks capture 4's C2 windows *below* chance (ROC 0.443).
> Rbot's command-and-control on this host does not resemble Rbot's
> command-and-control on the training hosts in 32 flow features at Δ=60 s. No head
> architecture fixes that; it is a representation limit, and the honest options are
> a finer Δ, features the flow record does not currently carry, or accepting it.~~

**Rbot C2 is a model failure too.** Both Rbot stages are. The probe's verdict is
one-sided and was read as two-sided: a probe that finds a group proves the signal
is there, a probe that misses one proves only that it missed. `group_separability`
now says so in its docstring, in the `separable` property and in the table it
writes.

**Rbot recon is a model failure.** The same probe reaches ROC 0.970 on the same
capture, on 17 windows, transferring across hosts — the signal is there, it
generalises, and NIDRA scores it at chance. This is the most actionable finding
of the phase: a quantified gap with an upper bound attached, on a stage that is
*early* in the kill chain and therefore exactly where advance warning would come
from.

**A caveat that applies to every row and to Run 8's Friday result equally: each
group's positives sit on exactly ONE host.** The validation captures do not
contain a second infected host for any attack stage, so nothing here — the model's
numbers included — separates "learned the behaviour" from "learned the host". The
within-split probe, which is allowed to see other windows from the same host,
reaches ROC 0.913–0.996 on all four groups, including the one that does not
transfer at all. That is the measurement of how much host identity alone buys.

### 3.9 The signal the risk head misses is already inside the model — in the other frozen head (§26, §31 Q13)

§3.8 left one question open. The cross-host probe reaches ROC 0.970 on Rbot recon
where the risk head is at chance, so the signal transfers; what stops NIDRA from
using it? One candidate explanation is cheap to test, because it is about the
objective rather than the architecture: the probe is fit on **that stage**
one-vs-rest, while the risk head is fit on `risk_label` with every stage pooled
into one positive class. NIDRA already trains a second head on stage labels, and
it is frozen under the same discipline. `nidra.scripts.stage_head_diagnostic`
scores both frozen heads on the same observed validation states.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| recon | 17 | 0.00017 | 0.003 | 20× | **0.638** | 0.000 | **0.320** |
| c2 | 24 | 0.00024 | 0.001 | 6× | **0.854** | 0.000 | **0.434** |
| exfil | 172 | 0.00174 | 0.301 | 173× | 0.910 | 0.799 | 0.915 |

Observed states, so rollout error is not in the picture: both heads are scored
exactly where they were trained, and the gap is about what they were asked to
learn.

**The two heads disagree, and the stage head is right.** On c2 it ranks at 0.854
where the risk head is at 0.434; on recon, 0.638 against 0.320. The risk head is
*below* chance on both — not indifferent to those windows but actively ordering
them beneath benign traffic. Exfil, the only stage with enough positives to
dominate the pooled label, is the one stage where the two agree (0.910 / 0.915).
That is the shape the pooled-objective hypothesis predicts.

**This is what overturns §3.8's C2 conclusion.** The stage head's 0.864 on
`ctu_4:c2` alone (0.766 on ctu_6's single window; 0.854 pooled) is measured under
the same cross-capture, cross-family construction as the probe's 0.443 — and,
per §3.17, on the same infected address in both, so neither is cross-host. A
trained NIDRA head reaches what the probe could not, so "not separable" was a
statement about the probe. Both Rbot failures are model failures.

**But the information cannot be harvested by fusing the two heads at inference.**
That was the obvious next move — both heads are frozen and already trained, so a
scalar rule adds no parameter and fits nothing on the rows it scores.
`nidra.scripts.head_fusion_screen` tried five parameter-free rules
(`reports/tables/ctu_head_fusion_val.md`):

| rule | AP | ΔAP vs published | ROC |
|---|---|---|---|
| risk head alone (published) | 0.4894 | — | 0.7441 |
| noisy-or | 0.4868 | −0.0026 | 0.7380 |
| max | 0.4797 | −0.0097 | 0.7354 |
| mean | 0.4662 | −0.0233 | 0.7380 |
| geometric mean | 0.4516 | −0.0379 | 0.7327 |
| stage 1-P(benign) alone | 0.2946 | −0.1948 | 0.7411 |

Every rule loses. The per-stage view says why: on c2 a fused score lands at
0.603–0.682, *between* the two heads rather than above either. The risk head's
confident scores on the 172 exfil windows outrank the stage head's correct
ordering of the 24 c2 ones, so a single pooled ranking cannot hold both orderings
at once. Collapsing the stage head to one non-benign scalar also costs it: its own
c2 column reaches 0.854 where `1 − P(benign)` reaches 0.828.

**What this directs.** The gap is in the risk head's *objective*, and it is not
reachable by post-hoc arithmetic on two heads' outputs — which is a negative
result worth having, because fusion is the cheap thing one would try first. A
risk head trained with a stage-aware term, so that a rare stage is not required
to outrank a common one on a single pooled scale, is the experiment the evidence
points at (§12 multi-task). It is motivated here rather than speculated at. Not
started: §7's dynamics runs own the machine until the cross-dataset matrix is in.

Weight this against §3.8's host caveat, which has not gone away: every attack
group on these validation captures has its positives on exactly one host. The
stage head's advantage over the risk head is measured on the same rows for both,
so the *comparison* is sound, but neither number separates behaviour from host
identity. Confirming any of this needs a capture with two infected hosts in the
same stage.

### 3.10 A fifth of CTU's forecast positives contain no information at all, and the history-aware head answers them with the host's name (§10, §31 Q6, Q8)

§3.9 left the risk head below chance on two stages and pointed at the objective.
Asking *what the head keys on* turned up something about the data instead.

`risk_label[t]` is 1 when an attack starts within the horizon, so a positive row
need not contain attack traffic — it is the window *before*. On CTU, **every one
of those pre-onset windows is silent**: 71 of 71 on train, 84 of 84 on val have
`is_active == 0`. Most are silent in the strongest sense — all 45 features within
1e-5 of the scaler's silent state, bit-identical to the vector every other silent
window has. `nidra.scripts.silent_positive_audit`:

| split | rows | positives | pre-onset | of those, silent | at the exact floor | floor stratum | state-only AP ceiling | share of all positives |
|---|---|---|---|---|---|---|---|---|
| CTU train | 1,419,278 | 1,418 | 71 | 71 | 63 | 1,026,446 | 0.000061 | 4.4% |
| CTU val | 98,876 | 288 | 84 | 84 | 58 | 47,708 | 0.001216 | **20.1%** |
| CIC train | 2,274,548 | 179 | 45 | 24 | 20 | 2,075,362 | 0.000010 | 11.2% |
| CIC val | 893,701 | 94 | 22 | 9 | 8 | 808,496 | 0.000010 | 8.5% |

**One fifth of CTU validation's positives are input vectors identical to 47,650
negatives.** No function of the state can order them above those negatives, so
inside that stratum the best AP any state-only head can reach is the stratum's own
prevalence, 0.001216. This is a ceiling, not a complaint, and it doubles as a leak
check: a state-only head that beats it is reading something the state does not
contain. The published state-only head emits **exactly one distinct score** across
all 47,708 rows and lands on 0.001216 with ROC 0.5000 — the bound is tight and the
check passes.

The datasets differ in a way worth naming. CIC's pre-onset windows are silent only
about half the time (24 of 45 on train, 9 of 22 on val); CTU's are silent always.
CTU's advance-warning signal, such as it is, is a *silence* phenomenon — the host
goes quiet, then acts. Anything NIDRA predicts there it predicts from history, by
construction.

**So does history deliver?** The `state+hidden` head produces 11,599 distinct
scores on those identical inputs and reaches AP 0.0707 — **58× the state-only
ceiling**. Taken alone that is the strongest number in the phase. It does not
survive the next question.

| head | AP | lift over ceiling | ROC | host-mean AP | host-mean ROC | within-host prev. | within-host AP | within-host lift | within-host ROC |
|---|---|---|---|---|---|---|---|---|---|
| state-only | 0.001216 | 1.00× | 0.5000 | 0.0012 | 0.5000 | 0.4567 | 0.4567 | 1.00× | 0.5000 |
| state+hidden | 0.070700 | **58.15×** | 0.4242 | 0.4567 | **0.9993** | 0.4567 | 0.4460 | **0.98×** | 0.3836 |

(The state-only row is a constant across all 47,708 rows, so every column is what
a constant gives: AP at the prevalence, ROC 0.5, and a within-host "AP" that is
just the within-host prevalence. It is the ceiling, not a competitor.)

All 58 floor positives sit on one host, `147.32.84.165`, which contributes 127 of
the stratum's 47,708 rows. Replacing every score with its **host's mean** — throwing
away everything the head said about *which window* — reproduces the ranking at ROC
0.9993. And restricting to that host, where identity is constant and only the
timing question remains, the head scores AP 0.4460 against a prevalence of 0.4567:
**lift 0.98×, worse than a constant**, ROC 0.3836.

The head has learned to recognise the infected host. It has not learned when that
host is about to act. The 58× is the same host caveat as §3.8 and §3.9, measured
directly for once rather than inferred, and the ROC of 0.4242 was the tell — a
ranking cannot be 58× lift and below chance at the same time unless the gain is one
block of rows lifted wholesale.

**What this does and does not retract.** §3.4's composite result stands: the
history-aware head's AP improvement and the `persistence_rollout` margin were
measured on the full split, most of whose positives are attack windows carrying
real traffic, and nothing here touches them. What it retracts is any reading of
that improvement as *advance warning from silence*. On the rows where advance
warning is the only thing being asked, the head contributes nothing beyond knowing
which host is infected — and a deployment knows that already or does not, in which
case the 58× is unavailable.

**Consequences for the phase.** Three, in order of how much they change what
follows.

1. **Task B's near-floor numbers in Run 8 now have a mechanism, not just a
   prevalence excuse.** Run 8 reported onset forecasting near the prevalence floor
   and attributed it to CIC having almost no same-host precursors. CTU has
   precursors — 84 of them on validation — and they are *empty*. Two datasets, two
   different reasons, the same conclusion: this corpus family cannot support a
   strong advance-warning claim, and the honest number is the floor.
2. **`n_positive_hosts` must be reported beside every per-group result**, and a
   result on a single-positive-host group needs the within-host decomposition
   before it means anything. `floor_stratum_probe` does this generically and flags
   `is_host_identity`; it should run over the cross-dataset matrix when that lands.
3. **The single-host confound is now the phase's binding limitation**, ahead of the
   architecture questions. Neither CIC-IDS2017 nor CTU-13 has two infected hosts in
   the same attack stage on the same split. No head, objective or curriculum fixes
   that — it is a property of the corpora, and the §4/§12 experiments will all
   inherit it.

Not adopted, not tuned, nothing removed. The audit is committed and runs on both
datasets so the next model is measured against the same ceiling.

### 3.11 The history-aware head's advantage is real timing signal on CIC and mostly host identity on CTU (§2, §4, §16, §31 Q4, Q12)

§3.10 built the host/timing decomposition for the silence floor. Nothing in it is
specific to that stratum, so `--probe-stratum all` asks the same question of a
whole split — which is what §3.4's headline deserves, since every attack group in
both corpora has its positives on one host.

**CTU-13 validation** (98,876 rows, 288 positives, 2 infected hosts):

| head | AP | ROC | host-mean AP | host-mean ROC | within-host prev. | within-host AP | within-host lift | within-host ROC |
|---|---|---|---|---|---|---|---|---|
| state-only | 0.3531 | 0.7281 | 0.7579 | 0.9995 | 0.7579 | 0.8763 | 1.16× | 0.6559 |
| state+hidden | 0.4894 | 0.7441 | 0.7579 | 0.9995 | 0.7579 | 0.8951 | 1.18× | 0.6621 |

**CIC-IDS2017 validation** (893,701 rows, 94 positives), same construction, same
32-feature `cross_core` regime, transition model held fixed within each dataset:

| head | AP | ROC | host-mean AP | host-mean ROC | within-host prev. | within-host AP | within-host lift | within-host ROC |
|---|---|---|---|---|---|---|---|---|
| state-only | 0.6781 | 0.8507 | 0.4196 | 0.9999 | 0.4196 | 0.8552 | 2.04× | 0.8256 |
| state+hidden | 0.7825 | 0.9797 | 0.4196 | 0.9999 | 0.4196 | 0.9224 | **2.20×** | **0.9245** |

The head ablation replicates on CIC: `state` → `state+hidden` moves val AP from
0.678 to 0.783, the same direction and a similar size as CTU's 0.353 → 0.489. Read
only as aggregates, the two datasets agree. The decomposition says they do not.

**On CTU both heads score *below* a host-level constant.** Collapsing every score
to its host's mean — discarding everything the head said about which window — gives
AP 0.7579, against the heads' 0.3531 and 0.4894. Flagging every window of the
infected host would outscore the model. Within those hosts the heads reach ROC
0.6559 and 0.6621: a little above chance, not much.

**On CIC both heads beat the host-level constant, and the history-aware one beats
it decisively.** Host identity alone gives AP 0.4196; the heads give 0.6781 and
0.7825. Within the infected hosts the history-aware head reaches **ROC 0.9245**
against the state-only head's 0.8256, and its lift over the within-host base rate
is 2.20× against 2.04×. The improvement is *reproduced inside the host*, where host
identity is constant and only timing is left. On CIC it is timing signal.

Two things make this comparison fair and one makes it approximate. Fair: within-host
ROC is prevalence-independent, and both datasets are scored with the same code, the
same feature regime and one seed each. Approximate: the within-host *lift* is not
comparable across the two, because CTU's infected hosts are 75.8% attack windows
while CIC's are 42.0%, so there is far less room on CTU for a within-host ranking to
be right. Neither figure resembles a deployment, where an infected host is mostly
benign; both flatter the model.

**What this answers.** The roadmap asked whether Run 8's risk-head limitation is a
CIC-IDS2017 artifact. On this evidence it is not, and the direction is the reverse
of the expected one: CIC is where the history-aware head does genuine within-host
temporal work, and CTU is where its apparent advantage is substantially the ability
to pick out the compromised host. A cross-dataset scorecard built on aggregate AP
alone would have reported the opposite, because CTU's aggregate AP gap (+0.136) is
larger than CIC's (+0.104).

**What it does not answer.** Both corpora have one infected host per attack stage
per split, so "recognising the host" and "recognising this family's behaviour on
this host" are not separated anywhere. The within-host numbers are the part of the
result that survives that confound, and they are what §17's scorecard should carry
beside the aggregates. `host-mean ROC` of 0.9995 and 0.9999 says the confound is
not marginal on either dataset: host identity is almost perfectly recoverable from
the scores.

Every number here is one seed on validation, which is where §19 says selection may
happen and nothing else may. No test or holdout split has been read.

### 3.12 The head ablation replicates in all three training regimes (§2, §4, §19 Stage A)

§4 asks for the history-aware head to be tested with the transition model held
identical. Three dynamics models were trained — CTU-only, CIC-only at the
`cross_core` 32-feature mask, and CIC+CTU combined — and each one had both head
variants fitted on top of its own frozen encoder and transition, so within a row
the only thing that changes is what the head reads.

| training regime | rows (train) | `state` | `state+hidden` | Δ |
|---|---|---|---|---|
| CTU-13 only | 1,419,278 | 0.3531 | 0.4894 | +0.1363 |
| CIC-IDS2017 only | 2,274,548 | 0.6781 | 0.7825 | +0.1044 |
| CIC + CTU combined | 3,693,826 | 0.3850 | 0.5130 | +0.1280 |

Validation AP at natural prevalence, one seed, head-training metric on observed
states — not the benchmark's forecast AP, and not comparable to Run 8's numbers.
The direction holds in all three regimes and the size is similar in all three.

Two things this table does **not** say. The three rows are not comparable *to each
other*: each regime has its own validation set with its own prevalence and its own
difficulty, so the combined regime's 0.5130 being below CIC-only's 0.7825 reflects
the mixture, not a loss from combining. And §3.11 has already shown that on CTU
most of the +0.1363 is the head getting better at picking out the infected host
rather than the moment, while on CIC the gain survives the within-host test. An
aggregate replication is evidence the effect is not a fluke of one split; it is not
evidence the effect is the one we want.

This is §19 Stage A screening. One seed per cell, selection on validation only.
Stage B — three seeds for the two surviving variants — is queued behind the
cross-dataset matrix.

### 3.13 Pre-registration: the stage-balanced risk objective (§12)

Written **before** the run, because the expected outcome includes a metric going
down and a criterion decided afterwards would be a criterion fitted to the result.

**The hypothesis.** §3.9 found the risk head ranks recon and c2 *below chance*
(ROC 0.320 and 0.434) while the stage head, trained on the same observed states
with class weights, ranks them at 0.638 and 0.854 — and §3.9 also showed no scalar
fusion of the two recovers it, because one pooled ranking cannot hold both
orderings. The proposed mechanism is the objective: `risk_label` pools every stage
into one positive class, so the stage with most positives owns the gradient. On CTU
that is exfil at 172 of 213 attack windows.

**The intervention.** `train_heads.stage_balanced_positives` (default `false`, so
every earlier run reproduces bit for bit) weights each positive by the inverse
frequency of its own stage, then rescales so the positive class's **total** weight
is unchanged. The positive/negative balance, `pos_weight`, the sampler, the frozen
encoder and transition, the seed and the data are all held fixed. The only thing
that moves is the mix *inside* the positive class. Pre-onset positives carry the
benign stage and form their own group, which is correct — on CTU they are 29% of
the positives and a distinct kind of row (§3.10).

**What would count as working**, in order:

1. **Primary:** the risk head's per-stage ROC on `recon` and `c2` rises above 0.5
   on observed validation states, from 0.320 and 0.434. Below-chance ranking is the
   defect being targeted; anything that leaves it below chance has not addressed it.
2. **Secondary:** within-host ROC (§3.11) does not fall. A gain that shows up only
   in the aggregate is the host-identity failure again.
3. **Guardrail:** exfil's ROC does not fall below 0.85 (from 0.915). Rebalancing
   that fixes the rare stages by breaking the common one is not an improvement.

**What would NOT count.** A rise in aggregate validation AP alone. Under this
intervention aggregate AP is *expected to fall*, because AP on CTU is dominated by
exfil and the objective deliberately stops exfil from owning the gradient. §32 says
a model with slightly lower overall AP but better generalisation may be
scientifically preferable; this is the case that tests whether we mean it. A fall
in aggregate AP with criteria 1–3 met is a **success** and will be reported as one,
with the AP cost stated.

**What would kill it.** Criterion 1 failing. If forcing the rare stages to own an
equal share of the gradient still leaves them below chance, the pooled objective is
not the explanation and the §12 direction is wrong — which is a useful thing to
learn, and will be reported whichever way it lands.

One seed, CTU-only, validation only, both head variants, same `ctu_dyn` checkpoint
as §3.4. Screening. Queued behind the cross-dataset matrix; the machine is full.

### 3.14 Correction to §3.11's scope: train and validation are single-host, test and holdout are not

§3.11 drew its conclusion on validation. Counting infected hosts per split shows
what that does and does not license:

| split | rows | hosts | positives | infected hosts |
|---|---|---|---|---|
| CIC train | 2,274,548 | 6,413 | 179 | **1** (`172.16.0.1`) |
| CIC val | 893,701 | 5,224 | 94 | **1** (`172.16.0.1`) |
| CIC test | 299,392 | 2,072 | 982 | **10** |
| CIC holdout | 450,482 | 2,557 | 151 | 2 |
| CTU train | 1,419,278 | 500 | 1,418 | **1** (`147.32.84.165`) |
| CTU val | 98,876 | 341 | 288 | **1** (`147.32.84.165`) |
| CTU test | 508,517 | 460 | 3,814 | **10** |
| CTU holdout | 267,388 | 383 | 1,134 | 3 |

**Training and validation see exactly one infected host, on both datasets.** Not
one per attack group — one per split, full stop. Every positive NIDRA has ever
been trained on, and every positive any validation number in this report is
computed from, comes from a single machine in each corpus.

**Test and holdout do not.** Ten infected hosts on each dataset's test split, most
of them never infected during training. So the evaluation design *does* test
cross-host generalisation — the limitation is on what validation can reveal, not
on what the protocol measures.

This bounds §3.11 rather than overturning it. What §3.11 established stands: on a
single-host validation split, CTU's aggregate head-ablation gain is largely the
ability to pick out that one host while CIC's survives the within-host test. What it
cannot establish is how either behaves across ten infected hosts, because validation
contains no such thing. The cross-dataset matrix now running produces exactly those
test and holdout numbers, and they are where the question is actually settled.

One consequence for the method. With several infected hosts the pooled within-host
columns regain a between-host component — host identity one level down — so the
probe now also reports **per-host ROC (macro)**, computed inside each infected host
and averaged. On a single-host split the two are identical, which is why it did not
matter until now; on the test splits it will. A test fixture pins the case where
pooling reads 0.81 and the macro average reads 0.67 on hosts that are individually
identical in quality.

A note on what the infected host *is*, since it differs between corpora and affects
what the CIC numbers mean. `host_id` is the flow's source address, so a positive
attaches to the host emitting the attack traffic. On CTU that is `147.32.84.165`, an
internal infected workstation — the compromised-host framing the project is built
around. On CIC train and validation it is `172.16.0.1`, the dedicated attack machine
outside the victim network. Forecasting that the attack box is about to attack is a
different and easier problem than forecasting that an internal host has been
compromised, so CIC's validation figures should be read with that in mind. CIC's
*test* split is the one that includes internal `192.168.10.x` hosts.

### 3.15 The Δ=15 direction is dead, and the advance-warning question belongs on CIC (§10, §31 Q1, Q10, Q14)

§3.10 established that CTU's pre-onset windows are empty. The obvious follow-up
was whether a finer window would recover a precursor that Δ=60 rounded away — and
that is worth measuring before paying for it, since changing `window_delta`
invalidates every trained artifact and every recorded metric in the project.

**How long is the infected host silent before an attack starts?**

| | onsets | silent windows before onset (median / mean / max) | seconds since last active (median / min / p90) |
|---|---|---|---|
| CTU train | 16 | 29 / 193.6 / 1254 | 2820 / 120 / 27060 |
| CTU val | 23 | 3 / 7.3 / 56 | 270 / 120 / 780 |
| CIC train | 13 | 0 / 19.2 / 188 | 90 / 60 / 2598 |
| CIC val | 12 | **0** / 2.6 / 30 | **60** / 60 / 60 |

**A finer Δ cannot help CTU.** The host has emitted nothing for a median of 4.5
minutes (validation) or 47 minutes (train) before an attack begins. Splitting a
silent minute into four silent quarter-minutes produces four silent windows.
There is no precursor being rounded away; there is no precursor. The Δ=15
direction is closed by measurement, at the cost of one query.

**CIC is the opposite.** In 83% of validation onsets the attacker host is active
in the window immediately before the attack — median gap 60 s, p90 60 s, meaning
essentially every onset follows an active window. CIC's precursor windows are not
empty, so the reason Run 8 reported Task B near the prevalence floor there cannot
be the one that applies to CTU.

**So is there anything in them?** A gradient-boosted probe fit on the infected
host's *training* rows and scored on its *validation* rows — same host on purpose,
so host identity is not available and only the timing question remains, and
forward in time, so nothing leaks:

| | pre-onset rows (train → val) | base rate | probe AP | lift | probe ROC | `is_active` alone |
|---|---|---|---|---|---|---|
| CIC | 45 → 22 of 150 | 0.1467 | 0.3411 | **2.33×** | **0.7500** | 0.6783 |
| CTU | 71 → 84 of 167 | 0.5030 | 0.5030 | **1.00×** | **0.5000** | degenerate |

**CTU: exactly chance, to four decimal places.** AP equal to the base rate and ROC
of exactly 0.5000 is what a constant predictor gives, which is all a probe can
produce when every row it is scoring is the same vector. Q1 is answered: CTU-13
does not contain enough temporal precursor information for genuine advance
warning, and this is a property of the capture, not of NIDRA.

**CIC: a real but modest signal, most of which is not interesting.** The probe
reaches ROC 0.7500 where the model is at the floor — so there *is* something to
find. But `is_active` alone reaches 0.6783 of it. Most of what distinguishes a CIC
pre-onset window from the attack host's other benign windows is that the host is
doing anything at all; the genuine precursor content beyond mere activity is the
gap between 0.75 and 0.68.

**What this changes.**

- **Q1 and Q10 split by dataset.** CTU: no, structurally. CIC: a weak yes, with a
  measured ceiling of ROC 0.75 against a model at the floor. Reporting a single
  answer for both would have been wrong in one direction or the other.
- **Q14 direction 3 is withdrawn** and replaced: the advance-warning work belongs
  on CIC, where the precursor windows are non-empty, and its honest ceiling is
  modest. A phase that chases advance warning on CTU is chasing an artifact.
- **The probe's caveat applies here too** (§3.8): a probe that finds something
  proves it is there; this one found ROC 0.75 on CIC, so that much is real. Its
  0.5000 on CTU would normally prove nothing — except that here the rows being
  scored are literally identical, so the ceiling argument of §3.10 carries it
  rather than the probe.
- Same-host by construction, so neither number says anything about transferring to
  a host that was never infected during training.

### 3.16 The two "validations" are not the same kind of thing (§3, §15, §22)

Building the experiment matrix (`reports/tables/experiment_matrix.md`, generated
from the provenance records) turned up a difference that §3.11 put side by side
without naming.

| run | validation is… |
|---|---|
| `ctu_*` | `val_days = [ctu_4, ctu_6]` — two **held-out captures**, different scenarios, different families (Rbot, Menti), never trained on |
| `cic_core_*` | no `val_days`; validation is the last 30% of the **training days' own time** (`val_fraction_of_train_time: 0.3` over Monday–Wednesday) |
| `comb_*` | both: `val_days = [ctu_4, ctu_6]` **and** a temporal carve from Monday–Wednesday |

CTU validation is out-of-capture. CIC validation is in-distribution — the same
days, later in the clock. The second is an easier target, and it is part of why
CIC's numbers in §3.11 and §3.12 sit above CTU's. Those comparisons were always
*within* a dataset, which is the comparison that matters there, but the
cross-dataset reading of §3.11 must carry this: CIC's within-host ROC of 0.9245 is
measured on a temporal tail of days the model trained on, and CTU's 0.6621 is
measured on captures it has never seen. Some of the gap is the split design, not
the model.

Both are legitimate and neither leaks — `assert_no_temporal_overlap` and
`assert_no_episode_leakage` run on every split construction, and the CIC carve
respects a 30-minute pre-onset margin. What they are not is the same experiment.

**A provenance defect found and fixed on the way.** `dataset.days[].role` is a
hand-written annotation in the config and had drifted from `cfg["splits"]`, which
is what actually assigns days: `ctu_4` and `ctu_6` are annotated `role: test` and
are the validation captures. Every CTU record written before today therefore
misstates the split those two captures landed in. Nothing downstream read the
annotation — `build_all_splits` uses `cfg["splits"]` — so no result is affected,
and the numbers in this report are validation numbers as labelled. But a
provenance record that misstates a split assignment is worse than no record, so
`provenance.py` now derives the role from `cfg["splits"]` and keeps the
annotation beside it under `role_annotated_in_config` only where the two disagree.
Records written before this change carry the old annotation; the matrix is
generated from the corrected derivation.

**Recorded compute so far:** 15.2 hours of wall clock across 36 runs with
provenance records, single machine (Apple M1, 8 cores, 16 GB). Runs overlapped,
so elapsed time is less than the sum.

### 3.17 Correction: nothing in this phase is cross-host before the test split

Checking which hosts carry positives, capture by capture, turned up an error
repeated in §3.8, §3.9 and (by implication) §3.11.

| split | infected (host, capture) |
|---|---|
| CTU train | `147.32.84.165` in ctu_1, ctu_2, ctu_3 |
| CTU val | **`147.32.84.165`** in ctu_4, ctu_6 |
| CTU test | `147.32.84.165` in ctu_8/9/10, plus `…191`, `…192` and 7 more — **9 hosts never infected in training** |
| CIC train | `172.16.0.1` in tuesday, wednesday |
| CIC val | **`172.16.0.1`** in tuesday, wednesday (temporal carve) |
| CIC holdout | `172.16.0.1`, plus `192.168.10.8` |
| CIC test | `172.16.0.1`, plus `192.168.10.12/14/15/17` and 5 more — **9 hosts never infected in training** |

**CTU-13 reuses the same infected address across scenarios.** `147.32.84.165` is
the bot in captures 1 through 4 and 6 — different malware families (Neris, Rbot,
Menti), the same machine. So the transfer probe of §3.8, described there as
"cross-host and cross-capture by construction, so it cannot answer with host
identity", is **cross-capture and cross-family but not cross-host**. The same
correction applies to §3.9's stage-head comparison, which uses the same
construction.

The one-vs-rest label does penalise a pure host-recognition strategy — that
host's own benign windows are negatives in every one of those probes, so a model
that simply recognised the machine would rank them high and lose. That is a
partial control. It is not the structural guarantee the text claimed, and the
difference matters because host identity is precisely the confound this phase has
been chasing.

**What changes and what does not.**

- §3.8 and §3.9's *measurements* stand — the probe's 0.443, the stage head's
  0.864, the risk head's below-chance 0.434 and 0.320. What changes is what they
  exclude: they exclude the model having merely memorised a *capture*, not the
  model having memorised a *machine*.
- §3.11's CTU finding gets **stronger**. Host-mean ROC of 0.9995 on validation is
  not a within-split curiosity: the model was trained on `147.32.84.165`'s traffic
  in captures 1–3 and is being scored on the same address in captures 4 and 6. It
  had the opportunity to learn that specific machine, and the decomposition says
  it took it.
- §3.9's central claim survives intact, because it is a comparison between two
  heads on identical rows. Whatever host information is available is available to
  both, so the stage head beating the risk head by 0.42 ROC on c2 is not explained
  by it.
- **Nothing in this phase is a cross-host result yet.** Both test splits contain
  nine hosts never infected during training, and both holdouts contain one. Those
  are the only genuinely cross-host evaluations available, and they are what the
  running matrix produces. Every validation number in this report — every one —
  is same-host.

`group_separability`'s docstring now states this, and a test pins the wording so
the claim cannot quietly reappear.

### 3.18 A scorecard column that reported the wrong quantity, and the number it was hiding

Smoke-testing `cross_dataset_scorecard` against the first completed arm of the
matrix — rather than waiting to run it once at the end — found it reading
`active_benign_false_alarm_rate` and printing it under a heading that said
**FA/h**. Those are different quantities: one is a fraction of active-benign
*rows*, the other is alarms per *hour*. The fraction rounds to `0.00` at two
decimals, so every regime in the table appeared to produce no false alarms at
all. §33 asks specifically for false alarms per hour, and the scorecard was
answering with something else under that name.

`report_tables.py` and `compare_runs.py` both read the correct field; only the
scorecard was wrong. It now reports all three — alarms/hour, false alarms/hour,
and the active-benign rate as its own labelled column — and a test pins the
distinction.

What the correct column shows on the arms finished so far:

| training → evaluation | split | precision | recall | alerts/h | **FA/h** |
|---|---|---|---|---|---|
| CIC → CIC | val | 0.955 | 0.753 | 8.40 | **0.38** |
| CIC → CIC | test | 0.745 | 0.033 | 5.16 | **1.32** |
| CIC → CIC | holdout | 0.511 | 0.279 | 8.22 | **4.02** |
| CIC → CTU | test | 0.011 | 0.019 | 210.32 | **207.97** |
| CIC → CTU | holdout | 0.002 | 0.005 | 186.46 | **186.18** |

The transfer arms fire roughly 200 alarms an hour, essentially all of them
false. That is not a marginal degradation, it is an unusable operating point:
the threshold frozen on CIC validation does not transfer to CTU at all. The
broken column reported both of those rows as `0.00` — the single most
operationally important number in the table, inverted into its opposite.

One seed, state-only head, and the rest of the matrix is still running, so these
are not the phase's conclusions. They are reported here because the defect is,
and because a table that renders a catastrophic false-alarm rate as silence is
exactly the failure the §33 reporting rules exist to prevent.

### 3.19 Verifying that the transfer arms carry the source domain's operating point

§2 requires a transfer evaluation to be a transfer: the target domain must
contribute nothing — not weights, not the scaler, not the threshold. That is how
the queue is written, and it is worth checking rather than trusting, because the
failure mode is silent and would make every transfer number optimistic.

Reading the threshold and pooling rule actually used out of each finished
benchmark:

| arm | split | threshold used | pooling |
|---|---|---|---|
| CIC → CIC | val, test, holdout | 0.501969 | `mean｜q=-｜integrated` |
| CIC → CTU | test, holdout | **0.501969** | `mean｜q=-｜integrated` |
| CTU → CIC | test, holdout | **0.181961** | `mean｜q=-｜max` |
| CIC+CTU → CIC | val, test | 0.040888 | `p_above_half｜q=-｜max` |
| CIC+CTU → CTU | test | **0.040888** | `p_above_half｜q=-｜max` |

Every transfer arm carries its **source** run's number unchanged into the target
domain, and CTU→CIC's 0.181961 is exactly the value in
`ctu_heads__state/artifacts/weights/operating_point.json`, selected on CTU
validation before any of this ran. Nothing was selected on a target domain and
nothing was re-selected on test or holdout. The safeguard holds.

**A caveat this surfaces for reading the scorecard.** The three source runs
selected *different pooling rules* on their own validation — CIC chose
`integrated` over the horizon, CTU and the combined run chose `max`, and the
combined run chose a different member-aggregation (`p_above_half` rather than
`mean`). Each choice is legitimate and each was made on that run's own
validation, which is the rule. But it means a row-to-row comparison in the
scorecard compares *systems*, pooling rule included, not just training sets. That
is the right unit for "which configuration would you deploy" and the wrong unit
for "does adding CTU help the dynamics", which is exactly what §31 Q2 and Q3
ask. The latter needs the readout held fixed, so `benchmark.py` gained a
`--force-pooling` flag and four cells are queued at lowest priority behind
everything else: CIC-only and combined on CIC test, CTU-only and combined on
CTU test, all at `mean｜q=-｜max`. Those runs are **not** deployable
configurations — the threshold and calibration still come from each run's own
selection under a different rule — and their metrics record `pooling_forced:
true` alongside the rule they replaced, so a forced run can never be mistaken
for a selected one in the scorecard. If the machine does not get to them, the
scorecard is reported with this caveat attached rather than without it.

This is also the mechanism behind §3.18's 208 false alarms an hour: a threshold
of 0.501969 chosen against CIC's score distribution means something entirely
different against CTU's.

**The comparison pairs do evaluate identical rows.** Checked rather than assumed,
because a cross-dataset table whose cells are scored on different data is worse
than no table: `ctu2ctu` and `comb2ctu` both test on ctu_8/9/10, `cic2cic` and
`comb2cic` both on friday morning/portscan/ddos, and the two transfer arms land
on those same sets. One declaration differs — the CTU-only config lists `ctu_5`
in `holdout_days` and the cross-eval configs do not — but `ctu_5` contributes
**zero rows**: at 17.7 MB it is too short for any host to reach the 36 windows
`min_windows_per_host` requires. Both holdouts are ctu_12 + ctu_13, 267,388 rows
and 1,134 positives, and are comparable.

### 3.20 The selected head could not be served, and what fixing it revealed about explainability (§4, §27, §36 items 25/26)

`NidraPredictor` is the whole ML surface the backend imports, and it could not
run a `TrajectoryRiskHead` at all. Every one of `forecast`, `forecast_batch`,
`counterfactual` and `explain` raised: four call sites asked `score_states` for
a risk number, which a head reading the encoder hidden state cannot produce
from a bare state, and two asked `explain_current_risk` for an attribution
without the context it holds fixed. Nothing caught it because every serving
test built the Run 8 per-state head. §4 would have selected a head that
evaluates and does not serve. Fixed in D141; per-state numbers are unchanged
(`score_trajectory` reduces to `score_states(states)` and `score_observed` to
`score_states(x[:, -1, :])` for a per-state head, and the existing tests that
pin the predictor's observed risk to `score_states` still pass).

Verified against **real trained weights**, not the synthetic fixture: the
`cic_core_heads__state+hidden` checkpoint, loaded through `NidraPredictor`
exactly as the backend loads it, on a real Wednesday window.

| probe | observed risk |
|---|---|
| host 172.16.0.1, 30 pre-onset windows | 0.0000128 |
| same window, history rows reversed | 0.0000128 |
| **same origin state**, that host's busiest 29 windows spliced in behind it | **1.0000000** |

The reversal moving nothing is not a bug: that host's history is one distinct
row repeated 29 times, so reversing it is a no-op — a useful reminder that a
"different history" probe has to be checked for actually being different. The
splice is the real test, and a five-order-of-magnitude swing behind a
byte-identical origin state says the history reaches the head.

It also says something less comfortable. **The origin state is contributing
almost nothing.** On the spliced window the head scores 1.0, and KernelSHAP
over all 45 named features attributes it to nothing: the largest absolute
attribution is 0.0003, on `active_flow_count`. That is not a broken
attribution — it is a correct one. The conditional attribution holds the
hidden state fixed and perturbs the state, and the honest answer is that the
state is not what moved the score. The drive is inside the 128-dimensional
encoder hidden state, which the 45-feature SHAP surface cannot decompose.

So the history-aware head buys AP (§3.12: 0.678 → 0.783 on CIC val, replicated
in all three regimes) at the cost of the explainability the project ships. A
console that says "risk 1.00, top signal `active_flow_count` (+0.0003)" is
worse than one that says nothing. Under §32's ranking this does not disqualify
the head — explainability is not on the list — but it is a real trade-off that
belongs in the model card and in the §31 Q13 answer about what still limits
NIDRA, and it is the strongest argument yet for §5's trajectory-aware variants
over the raw hidden state: `delta` and `logvar` are decomposable in a way the
hidden state is not.

The corroboration with §3.11 runs the other way too. The floor-stratum probe
found the `state+hidden` head's 58× lift is host identity (host-mean ROC
0.9993, within-host lift 0.98×). A head whose score is set by the hidden state
rather than the origin state, and whose hidden state encodes which host this
is, is exactly the head that probe described. The two measurements were taken
independently and agree.

Six tests in `tests/test_predictor_trajectory_head.py` drive the served surface
with a `("state", "hidden")` head. Its weights are random, so no numbers are
asserted — except that two windows sharing a final state but differing earlier
must score differently, which is the check that the history arrives rather than
arriving zeroed.

`eval/baselines.py` was deliberately left per-state: it serves only
`run_eval.py`, the legacy balanced-subsample harness whose numbers are already
not comparable to this protocol. `score_states` now raises a message naming
`score_trajectory`, `score_observed` and `score_stage` instead of an arity
error from inside `nn.Module.__call__`, so the legacy path fails legibly rather
than mysteriously. **Serving latency for this head is not yet measured** — the
machine has two training jobs on it and a timing number taken under contention
would be worthless. It is outstanding for §36 item 25, and it matters: the
head's input is 173 wide against the per-state head's 45.

### 3.21 Serving latency: the history-aware head costs nothing measurable (§36 item 25)

Measured after the D141 fix, on the real `cic_core_heads__*` checkpoints through
`NidraPredictor`, at `n_samples_per_member: 200`, `K=6`, `torch_num_threads: 2`.
**The machine was running two training jobs throughout (1-minute load average
6.1).** Absolute numbers are therefore contended and are an upper bound; the
comparison is not, because the two arms' calls were interleaved A,B,A,B so load
drift hits both equally.

| head | median | p95 | max |
|---|---|---|---|
| `state` (45-wide) | 68.2 ms | 75.0 ms | 75.5 ms |
| `state+hidden` (173-wide) | 70.4 ms | 77.9 ms | 80.6 ms |

Ratio 1.033×, over 12 paired calls, one ensemble member. A second run
decomposing the call put the two at 71.5 ms and 71.0 ms — the trajectory head
nominally *faster*, which is the clearest statement available that the
difference is noise. **The extra 128 input dimensions cost nothing worth
reporting**, because they are one wider `Linear` on a path whose cost is
elsewhere:

| stage | `state` | `state+hidden` | scales with ensemble size? |
|---|---|---|---|
| ensemble rollout | 49.6 ms | 48.6 ms | **yes** |
| KernelSHAP current-risk attribution (nsamples=100) | 8.9 ms | 9.7 ms | no |
| temporal saliency | 9.1 ms | 9.4 ms | no |
| everything else | 3.9 ms | 3.3 ms | no |
| **total, 1 member** | **71.5 ms** | **71.0 ms** | |

Substituting five members for one — the shipped `ensemble_seeds: [0,1,2,3,4]` —
gives **270 ms** and **266 ms** against the 300 ms target. Two caveats, both
load-bearing:

1. That is an **estimate by substitution, not a measurement.** No 5-seed
   trajectory-head artifact exists yet; §19 Stage B produces three. The
   estimate assumes members cost the same and run sequentially, which is what
   `_ensemble_rollout` does today.
2. It is 270 ms of a 300 ms budget **on a machine under load 6.1**. A quiet
   machine has more headroom than that, but the margin is thin enough that the
   final ensemble must be measured rather than extrapolated before any latency
   claim is made.

If it does come in over target, CLAUDE.md's instruction applies unchanged — cut
stochastic samples toward 100 before touching the ensemble size — and it applies
equally to both heads, since the rollout is the term that scales and the head is
not why.

#### D145's mask costs nothing measurable (added after §3.37)

The fix adds an elementwise multiply on `nxt`, `mu` and `logvar` at every
rollout step. Paired A,B,A,B under the same contention as above, one member,
`n_samples=200`, 13 dropped slots of 45:

| arm | median | p95 | min |
|---|---|---|---|
| masked | 65.22 ms | 69.46 ms | 56.26 ms |
| unmasked | 64.82 ms | 70.11 ms | 57.29 ms |

Ratio of medians 1.006; ratio of **minima** 0.982, i.e. the masked arm is
nominally faster on the least-contended sample. Paired difference median
−0.08 ms, 5th–95th −6.4 to +6.9 ms. Sign test: **masked is slower in 19 of 40
pairs**, which is a coin flip.

Worth recording how this looked at fifteen pairs, because it is the same trap as
everywhere else in this log: the ratio of medians was **1.063** and would have
been reported as a 6% cost. What said otherwise before the longer run was that
the p95 ordering *reversed* between the arms and the minima agreed to 0.13 ms —
a real 6% cost does not do either. Fifteen paired samples is not enough to
resolve a difference smaller than this machine's load noise.

### 3.22 The explainability loss may not be a component choice (§5, §27, §31 Q14)

§3.20 argued that a head driven by the encoder hidden state costs the project its
explanation surface, and suggested trajectory components — `delta`, `logvar` —
as a decomposable alternative worth a lower AP under §32's ranking. **The CTU
half of that question was already answered on disk and the answer is negative.**
From §4's component table, one seed, validation AP at natural prevalence:

| head reads | val AP | over `state` | fraction of what `hidden` is worth |
|---|---|---|---|
| `hidden` alone | 0.4958 | +0.1427 | 100% |
| `state+hidden` | 0.4894 | +0.1363 | 96% |
| `state+logvar` | 0.3731 | +0.0200 | **14%** |
| `state` | 0.3531 | — | 0% |

The decomposable component carries a seventh of what the recurrent summary
carries, and `logvar` on top of `hidden` makes things *worse* (0.4538). This
was reported at §4 as "uncertainty helps, but barely"; read against §3.20 it
says something sharper. **The AP gain and the explainability loss may be the
same thing** — not a trade-off a different component set can route around.

`hidden` alone beating `state+hidden` is the same observation §3.20's real-weight
probe made from the other end: on a served forecast, splicing a different history
behind a byte-identical origin state moved the risk from 0.0000128 to 1.0000000.
Two independent measurements, months of code apart, saying the origin state is
not what the head is using.

**What is genuinely untested is CIC**, where the ablation only ever ran `state`
and `state+hidden`, and where §3.11 found the history-aware gain survives the
within-host test that CTU's does not. That is a real difference between the
corpora and it is the reason not to assume CTU's verdict transfers. Four CIC arms
— `state+logvar`, `state+delta`, `state+delta+logvar`, `hidden` alone — are
queued behind the pre-registered stage-balanced run, off the same `cic_core_dyn`
checkpoint so the transition model is byte-identical and only the head's inputs
change. A negative closes the direction properly rather than leaving it standing
as an untested suggestion.

Recorded as a correction to this log's own Q14 entry, which was written before
the CTU table was read against this question.

**Batch memory, checked because the fix looked like it should have cost some.**
D141 made `forecast_batch` pool the whole `RolloutOutput` instead of just
`.states` — four extra tensors per member held alive at once, which arithmetic
puts at 166 MB per member against 28 MB, or 1.66 GB at five members. Measured
in separate processes, same head, same batch, the only difference being which
tensors get pooled:

| batch path | peak RSS, 1 member, chunk 128 |
|---|---|
| pre-D141, `.states` only | 2795 MB |
| current, whole output | 2615 MB |

**No regression** — the current path is nominally lower, which is the honest way
to say the difference is below the noise of the thing that actually dominates.
What dominates is the rollout's own intermediate activations at `chunk=128`,
`n_samples_per_member=200`, `K=6`; the pooled tensors are a few hundred MB of a
2.6 GB peak.

That 2.6 GB is itself worth writing down, and it is **pre-existing, not
introduced here**. The backend's replay path is `forecast_batch`, the shipped
ensemble is five seeds, and only the retained outputs accumulate across members
(the rollouts run sequentially), which puts a five-member call near 4 GB on a
16 GB machine that is also running inference workers. The lever is `chunk`,
which is a caller argument defaulting to 128 and scales the peak almost
linearly. Nothing here needs changing today; it needs to be known before someone
sets `chunk` from a config file.

### 3.23 Calibration, and the constant that makes it readable (§36 item 19, §32 criterion 8)

Every benchmark has been writing a calibration block since Run 8 and nothing was
reading it. It now reaches both the per-run tables and the scorecard, with one
column that changes how the rest of them read.

At these prevalences — 0.0003 to 0.007 — the Brier score is dominated by the
negatives. A model that predicts 0.004 for every row and never moves scores
p(1−p) = 0.00415, which *looks* excellent. So the tables carry that constant
next to the model's Brier, and the comparison is not always flattering:

| training → eval | split | Brier raw | ECE raw | Brier cal. | ECE cal. | constant | beats it? |
|---|---|---|---|---|---|---|---|
| CIC → CIC, `state` | test | 0.0872 | 0.266 | 0.00402 | 0.046 | 0.00415 | yes |
| CIC → CIC, `state` | holdout | 0.0886 | 0.277 | 0.00030 | 0.050 | 0.00034 | yes |
| **CIC → CIC, `state+hidden`** | test | 0.0633 | 0.211 | **0.00565** | 0.048 | 0.00415 | **no** |
| **CIC → CIC, `state+hidden`** | holdout | 0.0653 | 0.227 | **0.00039** | 0.050 | 0.00034 | **no** |
| CIC → CTU, `state` | test | 0.1332 | 0.303 | 0.01901 | 0.056 | 0.00743 | **no** |
| CIC → CTU, `state` | holdout | 0.1153 | 0.288 | 0.01571 | 0.059 | 0.00408 | **no** |
| comb → CIC, `state` | test | 0.0038 | 0.050 | 0.00407 | 0.046 | 0.00415 | yes |
| comb → CIC, `state+hidden` | test | 0.0038 | 0.046 | 0.00404 | 0.046 | 0.00415 | yes |
| comb → CTU, `state` | test | 0.0093 | 0.058 | 0.00735 | 0.043 | 0.00743 | yes |
| CTU → CTU, `state` | test | 0.0067 | 0.047 | 0.00706 | 0.044 | 0.00743 | yes |

Three things fall out, and two of them matter for the architecture decision.

**Platt calibration is doing real work.** Raw ECE on the CIC-trained arms is 0.21
to 0.30 and calibration takes it to 0.046–0.050 everywhere. The per-horizon Platt
layer is not decoration.

**The history-aware head is the worse-calibrated one.** On the same rows, same
split, same calibration procedure, `state+hidden` scores 0.00565 against `state`'s
0.00402 on CIC test, and 0.00039 against 0.00030 on holdout — and it is the only
within-dataset arm that loses to the constant. §32 ranks calibration eighth of
ten, so this does not by itself overturn the AP gain, but it is the second cost
the head has now been shown to carry, after §3.20's explainability. Both should
be on the table when the architecture is chosen, not discovered afterwards.

**Training on both corpora produces a far better-calibrated raw score.** The
combined arms' *raw* ECE is 0.046–0.058, where CIC-only raw is 0.211–0.277 — the
combined model's uncalibrated output is already about as good as the CIC-only
model's *calibrated* one. That is a point in favour of the combined regime that
AP alone does not show, and it belongs in the Q2/Q3 answers alongside the AP
comparison rather than in place of it.

The transfer arms are the worst calibrated (CIC → CTU raw ECE 0.303, calibrated
Brier 2.6× the constant), which is the same failure §3.18 measured as 208 false
alarms an hour from the other direction: a threshold and a calibration fitted on
one corpus's score distribution do not mean the same thing on another's.

A caveat that travels with the whole table: **Brier at extreme imbalance is a
weak instrument**, and "beats the constant" is a floor, not a standard. A model
can rank well and score badly here, which is exactly why AP is the headline
metric and this is a supporting one. What it does establish is that a
`state+hidden` probability shown to an operator as a probability would be less
trustworthy than a `state` one, and that is a deployment fact rather than a
statistical artefact.

### 3.24 The oracle is not the upper bound the scorecard was treating it as (§25, §36 item 22)

§25 asks for the oracle gap to be investigated. Collected across every
cross-evaluation arm measured so far, the gap does not behave like a gap:

| training → eval | split | published AP | oracle AP | published ÷ oracle |
|---|---|---|---|---|
| CIC → CIC, `state` | test | 0.111 | 0.155 | 0.72 |
| CIC → CIC, `state` | holdout | 0.332 | 0.489 | 0.68 |
| CIC → CIC, `state+hidden` | test | 0.064 | 0.051 | **1.26** |
| CIC → CIC, `state+hidden` | holdout | 0.380 | 0.375 | **1.01** |
| CIC → CTU, `state` | holdout | 0.035 | 0.034 | **1.05** |
| comb → CIC, `state` | test | 0.164 | 0.089 | **1.85** |
| comb → CIC, `state+hidden` | test | 0.158 | 0.052 | **3.05** |
| comb → CTU, `state` | holdout | 0.307 | 0.232 | **1.32** |
| CTU → CTU, `state` | holdout | 0.188 | 0.188 | **1.00** |
| *(the remaining 5 cells)* | | | | 0.28 – 0.92 |

**The published system beats the oracle in 7 of 14 cells.** A ratio to an upper
bound cannot exceed 1, so either the model is doing something impossible or the
oracle is not bounding what the column implies. It is the second, and the reason
is a readout mismatch rather than a defect in `_oracle_risk`:

- `oracle_true_future` is the frozen head on the **true** future state, reduced
  over horizons. One trajectory, no pooling, **no Platt layer**.
- `world_model_calibrated` is the frozen head on ~200 **predicted** trajectories,
  pooled by the validation-selected rule, **then** per-horizon Platt.

Pooling and calibration are part of the system under test and the oracle is
given neither. Attributing each inversion by comparing against the pooled but
uncalibrated `world_model`: in **5 of the 7**, pooling alone already beats the
oracle; in the other 2, calibration flips it. The readout, not the state
forecast, is what the oracle is missing.

**The matched comparison is `world_model_deterministic`** — one predicted
trajectory, no pooling, the same reduction the oracle gets — and there the
oracle behaves: 0.014 against 0.051, 0.236 against 0.375, 0.005 against 0.011.
It bounds the deterministic system in **12 of the 14** cells. So the oracle is a
sound upper bound on *state-forecasting error at a fixed single-trajectory
readout*, which is what it was built to isolate, and an unsound one on the
deployed system. The scorecard now prints `oracle AP / deterministic AP` with
that stated, because the one-column version invited exactly the reading it
cannot support.

**Two cells break even the matched bound** and are not explained here:
`comb → CIC` test (deterministic 0.157 against oracle 0.089) and `comb → CTU`
holdout (0.339 against 0.232). Both are combined-trained and both use the
per-state head, where the oracle is simply `score_states` on the true future —
so the frozen head ranks the transition model's *predicted* states better than
the real ones. A rollout acting as a learned prior that pushes states toward
regions the head separates well would produce this, and so would an artefact of
the horizon `max` reduction interacting with a smoother predicted trajectory.
**Neither is tested**, and they are named as candidates, not causes. It is
recorded as an open item rather than resolved, because the honest position is
that a 2-of-14 violation of a bound we thought was structural is not yet
understood.

What this does *not* change: the oracle gap where it is a gap. CIC → CIC with
the `state` head reaches 68–72% of the oracle, and that remains the cleanest
statement of how much of the head's ceiling the forecast actually delivers.

### 3.25 Hardware and training-time accounting (§7, §19, §36 item 25)

**Hardware:** one Apple M1, 8 cores, 16 GB, CPU only. No GPU path exists in this
project and none was added. `torch.set_num_threads` is 2 or 3 per job and two to
three jobs run concurrently, which is what an 8-core machine absorbs before the
jobs start stealing from each other — measured, not assumed: a full `pytest`
that takes 104 s against one background job takes 8 m 35 s against three.

Parsed from the queue logs' own START/END markers, so these are what the machine
spent rather than what a re-timing would say:

| stage | runs | total (h) | median (min) | longest |
|---|---:|---:|---:|---|
| dynamics (stage 1) | 5 | 6.09 | 79 | `ctu_dyn_s2` (86 min) |
| benchmark | 30 | 3.90 | 9 | `bench ctu_sh_cn03` (18 min) |
| head ablation (stage 2) | 3 | 0.83 | 16 | CTU, six variants (26 min) |
| onset head | 2 | 0.11 | 3 | `ctu_onset_hidden_indep` (3 min) |
| GRU baseline | 1 | 0.06 | 4 | `ctu_gru` (4 min) |
| other | 1 | 0.11 | 7 | `ctu_sh_cn03` (7 min) |
| **total** | **42** | **11.09** | | |

~~All 42 exited zero.~~ **Withdrawn (§3.30):** the queues write
`rc=$?` after a pipeline, so they were recording `tail`'s exit status, not the
job's. Three runs in that count crashed and were logged `rc=0`. The wall-clock
figures are unaffected — a crashed run still started and ended — but the
exit-code column meant nothing and is struck rather than quietly deleted.

Several ran concurrently, so 11.09 h is **CPU occupancy, not elapsed time** —
the phase's wall clock is shorter and the two must not be confused.

The shape of this table is the argument for §19's two-stage discipline. Stage 1
is 55% of the budget in 5 runs; a head ablation is 16 minutes because it
initialises from a frozen stage-1 checkpoint and trains only the head. That is
why the architecture search runs on heads over shared dynamics and why the
multi-seed confirmation is scheduled rather than run on every variant: five more
dynamics seeds would cost more than everything else in this table combined.

A note on how this was measured, because the first version of it was wrong: the
queue markers carry labels with spaces (`END head ablation rc=0 …`), and a
`\S+` capture matched none of those, silently dropping six runs and 1.3 hours.
The dropped runs were the ones with the most descriptive names. Corrected before
publication; recorded because a log parser that under-reports without erroring
is the same failure mode as a metric that looks plausible.

### 3.26 One of §3.24's two candidates is eliminated, and the survivor is a warning sign

§3.24 left two cells breaking even the readout-matched oracle bound, and named
two candidate mechanisms without testing either. One is now eliminated with no
new compute.

**It is not the horizon reduction.** The composite takes `max` over k, so a
smoother predicted trajectory could win on the max while losing at every
individual horizon. It does not: in both anomalous cells the deterministic
system beats the oracle at **6 of 6 individual horizons**.

| arm | split | det k=1 | det k=6 | k6/k1 | oracle k=1 | oracle k=6 | det > oracle |
|---|---|---|---|---|---|---|---|
| CIC → CIC, `state` | test | 0.096 | 0.017 | 0.18 | 0.100 | 0.098 | 0/6 |
| CTU → CTU, `state` | test | 0.202 | 0.006 | 0.03 | 0.197 | 0.200 | 1/6 |
| **comb → CIC, `state`** | **test** | 0.081 | **0.105** | **1.29** | 0.053 | 0.051 | **6/6** |
| **comb → CTU, `state`** | **holdout** | 0.212 | **0.431** | **2.04** | 0.202 | 0.217 | **6/6** |
| *(the other 11 cells)* | | | | 0.01 – 0.68 | | | 0–4/6 |

Two things in that table are worth separating.

**The oracle is flat in k, everywhere.** 0.100 → 0.098, 0.197 → 0.200, 0.202 →
0.217. That is a good sign about the oracle's construction: the frozen head's
ceiling does not depend on how far ahead the state is, so a forecast approaching
it should decay toward it, never through it.

**The forecast's AP RISES with horizon in exactly those two cells** — 1.29× and
2.04× from k=1 to k=6 — against 0.01–0.68× in the other thirteen. Every normal
cell decays as forecast error accumulates. These two improve.

A forecast that gets more accurate the further ahead it looks is not forecasting.
The rollout is recursive, `S[t+k] = S[t+k-1] + mu`, so over six steps it drifts
toward whatever the transition dynamics attract to; if that attractor separates
infected hosts from benign ones better than the real states do, AP rises with k
and passes the oracle. That is the surviving candidate and it is consistent with
§3.11 and §3.20: a model whose advantage is recognising *which host this is*
would look exactly like this, because host identity does not decay with horizon
while a genuine precursor does.

**This is not established.** It is two cells of fifteen, one seed each, and the
same configuration behaves normally on its other split (`comb → CIC` holdout,
k6/k1 = 0.08). It could be seed noise. What *is* established is that the
reduction is not the explanation and that the predicted states are more
separable than the true ones at every horizon in those cells.

Pre-registering the test, so the criterion is not chosen after seeing the
result: measure the **rollout's terminal drift** — the mean distance between
`S_hat[t+6]` and `S_t`, per class — on the combined model and on a CIC-only
model. If the combined model's predicted states collapse toward two
class-dependent attractors while the CIC-only model's do not, the attractor
account is supported. If drift is comparable and the AP rise persists, it is not,
and the two cells stay unexplained. Either outcome is reportable; the direction
this would push the project is toward §5's trajectory-aware heads and away from
reading rising-with-k AP as forecasting skill.

### 3.27 The pre-registered drift test refuses §3.26's explanation

Run as specified. Fisher-style separation — distance between class means over
mean within-class spread — on 2,400 sampled eval rows (400 positive), computed
identically on the true future state and the deterministic rollout's terminal
state so the comparison is of the states and not of the statistic:

| cell | anomalous? | true `S[t+6]` | predicted `Ŝ[t+6]` | amplification |
|---|---|---|---|---|
| comb → CIC, test | **yes** | 1.177 | 2.435 | 2.07× |
| comb → CTU, holdout | **yes** | 0.911 | 1.813 | 1.99× |
| CIC → CIC, test | no | 1.257 | 2.173 | **1.73×** |
| CTU → CTU, test | no | 0.636 | 0.627 | 0.99× |

The criterion was: the combined model amplifies separation *while the CIC-only
model does not*. **The CIC-only model amplifies it at 1.73× and shows no
anomaly.** The criterion fails. The two anomalous cells do carry the largest
amplification and the ordering is consistent, but 1.73 against 2.07 does not
separate an anomaly from a non-anomaly, and reading it as though it did is
precisely what writing the criterion down beforehand was meant to prevent.

The drift column argues against the account from a second direction. An
attractor would pull trajectories together: shrinking within-class spread,
shrinking distance travelled. The rollout does the opposite — it **over-moves**,
and most on the benign class:

| cell | true benign drift | predicted benign drift | true attack | predicted attack |
|---|---|---|---|---|
| CIC → CIC, test | 3.30 | 7.07 | 9.37 | 14.09 |
| comb → CIC, test | 2.87 | 7.43 | 7.44 | 10.08 |
| comb → CTU, holdout | 5.01 | 8.35 | 8.60 | 12.41 |
| CTU → CTU, test | 3.50 | 7.84 | 5.20 | 8.86 |

Predicted benign hosts travel roughly **twice** as far from their origin as real
ones, in every cell. That is a systematic property of the transition model and it
is not what §3.26 guessed at.

Two things this test taught, one of them about itself:

**The instrument was weakly coupled to the question.** L2 separation between
class means in raw feature space is not what the nonlinear frozen head reads, so
even a clean positive would have been indirect. The right test runs through the
head — compare the head's own scores on predicted against true terminal states,
per class — and that is what a follow-up should do rather than refining this
statistic.

**The phenomenon is now better specified than any explanation for it.** The
rollout inflates class separation *and* inflates benign drift, in three of four
cells, and the one cell where it does neither (CTU → CTU) is also the one where
the deterministic forecast decays fastest (k6/k1 = 0.03). That is a coherent
description of what the transition model does. It is not yet a reason for why
two cells beat their oracle.

§3.24's two cells remain **unexplained**. Recorded as entry 10 in
`RUN9_NEGATIVE_RESULTS.md` — the first pre-registered criterion in this phase to
refuse one of the log's own explanations rather than one of its interventions.

### 3.28 A near-miss, and the guard it bought (§22, §33)

§3.27's follow-up — score predicted against true terminal states *through the
head*, since L2 separation is not what the head reads — produced numbers that
reproduced the benchmark's own per-horizon table on two cells and missed badly
on a third: `comb → CTU` holdout came out at 0.149 where the benchmark recorded
0.431, a 2.9× discrepancy on one of the two cells §3.26 is about.

Chased rather than reported, because a headline number that does not reproduce
is a §33 problem whichever side the error is on. **The error was in the probe.**
The probe loaded `config/combined_eval_ctu.yaml` directly; the queue runs that
config with `--set artifacts.scaler_dir=<the source run's scaler>`. Without the
override the config resolves to the shared `ml/artifacts/scaler`, and that
directory now holds a **different feature regime**: 0 dropped features against
the combined model's 13 (`cross_core`, 32 kept). Every recorded cell is sound —
audited all 20, each used a scaler from its own model's family.

What makes it worth a section is *how* it failed. A dropped feature is **zeroed,
not removed**, so the tensor shapes match, `load_state_dict` succeeds, the
rollout runs, and the AP comes out 2.9× different with nothing raised anywhere.
That is precisely the failure this project's conventions single out — silent
coercion producing a plausible wrong number — and it took a cross-check against
an independently recorded value to notice.

**`nidra/eval/benchmark.py` now refuses it.** `load_models` compares the loaded
scaler's drop set against each checkpoint's recorded `dropped_features` and
raises a message naming the differing features and the checkpoint, rather than
the count alone. Verified against the actual near-miss: the unoverridden config
is now refused, and both legitimate spellings — the queue's `--set` and a run's
own `config.yaml` — still load. Checkpoints predating the `dropped_features`
field pass through, because refusing them would break replaying Run 8's
artifacts, which §29 forbids.

The residual lesson is about the configs, not the guard: `config/cic2ctu.yaml`,
`config/ctu2cic.yaml` and `config/combined_eval_ctu.yaml` are **not
self-contained**. Each is correct only when run with the scaler override, and
the shared directory they otherwise resolve to is rewritten by whatever trained
last. The guard converts that from a silent wrong answer into a refusal, which
is the right first move; making the configs self-contained would be better and
is not done here.

And the k=6 decomposition that started this, on the two cells the probe *did*
reproduce:

| cell | states | AP | mean·attack | mean·benign | benign p99 |
|---|---|---|---|---|---|
| comb → CIC, test (**anomalous**) | true | 0.059 | 0.041 | 0.00055 | 0.0056 |
| | predicted | **0.109** | 0.105 | 0.00255 | **0.592** |
| CIC → CIC, test (control) | true | 0.095 | 0.068 | 0.00097 | 0.211 |
| | predicted | **0.017** | 0.023 | 0.01422 | 0.046 |

In the anomalous cell the rollout raises **everything** — the attack mean 2.6×
and the benign 99th percentile by a factor of 105 — and AP still nearly doubles.
So the gain is not the head separating attacks more cleanly; it is a re-ranking
whose mechanism is still not identified. The control behaves as expected: the
rollout degrades everything and AP falls to a fifth. §3.24's two cells remain
open.

### 3.29 Stage B: the head comparison confirmed on three seeds (§19, §36 item 7)

§19 forbids calling a one-seed screening result final. The confirmation ran on
CTU-13, three head-training seeds per variant, all six cells initialised from the
**same** `ctu_dyn` checkpoint so the encoder and transition are byte-identical
and the only thing that changes is what the head reads:

| head | seeds | mean val AP | sd | individual seeds |
|---|---|---|---|---|
| `state` (Run 8 architecture) | 3 | 0.3212 | 0.0468 | 0.3751, 0.2975, 0.2910 |
| `state+hidden` | 3 | **0.4931** | **0.0083** | 0.4877, 0.5027, 0.4889 |

**+0.172 mean, and the distributions do not overlap** — the worst history-aware
seed (0.4877) is 0.11 above the best per-state seed (0.3751). Against the Stage A
screening's +0.1363 on one seed, the effect replicates and is if anything larger.
This is the phase's central hypothesis and on the selection metric it holds.

The second column was not expected and matters on its own. **The history-aware
head is 5.6× more stable across seeds** (sd 0.0083 against 0.0468). With the
transition model held fixed, that 0.0468 is *entirely* head-training noise — 15%
of the per-state head's own mean. §32 ranks reproducibility ninth, and this is a
point for the same head that §3.23 and §3.20 counted two points against.

It also recalibrates how every Stage A number in this log should be read. §3.12's
one-seed cells carry roughly ±0.05 of head-training noise on the `state` arm; its
CTU row (0.3531) sits within one sd of the confirmed mean (0.3212). The gaps it
reported are several times that noise, so its conclusions stand — but a Stage A
difference smaller than about 0.1 on a `state` arm should not have been believed,
and none was relied on.

Three things this does **not** say, all of which the phase has to keep straight:

1. **The seeds vary head training, not dynamics.** One `ctu_dyn` checkpoint
   underlies all six cells. That is what §4 requires for a controlled head
   comparison, and it means the sd is head-training variance, not end-to-end.
   `ctu_dyn_s1` and `ctu_dyn_s2` exist for the end-to-end question and are not
   consumed here.
2. **This is head-training AP on observed states, not the forecast benchmark.**
   The benchmark is the selection metric and it is running now, validation only —
   test and holdout stay untouched until a winner is frozen.
3. **It is CTU-13.** §3.11 already showed that most of CTU's history-aware gain
   is the head recognising the infected host rather than the moment, while CIC's
   survives the within-host test. A confirmed aggregate on the corpus where the
   decomposition is worst is confirmation that the effect is real, not that it is
   the effect we want.

### 3.30 Six cells of the cross-dataset matrix never ran, and the log said they had

Dry-running the scorecard before the report queue reaches it turned up three
empty cells: `CIC → CTU`, `CTU → CIC` and `CIC+CTU → CTU`, all at
`state+hidden`, across both splits. Six cells of the phase's headline artifact,
serving §31 Q2, Q3 and Q9 directly.

**Cause.** `config/cic2ctu.yaml`, `config/ctu2cic.yaml` and
`config/combined_eval_ctu.yaml` are generic evaluation configs and do not
declare `model.risk_head.components`. So `world_model_from_config` built the
45-wide per-state head and `load_state_dict` refused a 173-wide checkpoint:

```
size mismatch for risk_head.net.0.weight: copying a param with shape
torch.Size([64, 173]) ... the shape in current model is torch.Size([64, 45]).
```

That is a loud, correct, informative failure. It was invisible anyway.

**Why it was invisible.** The queue's `bench()` ends with

```zsh
$PY -m nidra.eval.benchmark ... 2>&1 | tail -16
echo "END $label/$split rc=$? $(date +%H:%M:%S)"
```

`$?` after a pipeline is the **last** command's status — `tail`'s. Every one of
the three crashes was recorded `END … rc=0`, three seconds after its START. A
three-second benchmark next to fourteen-minute neighbours was there to be seen
in the log and nobody was reading the log for that. §3.25's "all 42 exited zero"
was measuring `tail` and has been withdrawn.

**Three fixes, in increasing order of durability.**

1. The six cells are re-queued with `--set model.risk_head.components=[state,hidden]`,
   which was verified to load the checkpoint before queueing rather than after.
2. The recovery queue uses `${pipestatus[1]}` and prints an explicit `FAILED`
   line, so a crash is visible without reading tracebacks. The original queues
   are left as they are: rewriting a running script is worse than the defect.
3. `load_models` now refuses the mismatch by **name** rather than by shape
   (D143), with a message quoting the override that fixes it. A shape error says
   what broke; this says what to do. It also covers the case the shape error
   cannot — two different component sets that happen to produce the same width.

**What this says about the rest of the phase.** The failure needed two
independent slips: a config that could not express the head, and a log that
could not report a failure. Every other recorded cell was audited for the same
pattern in §3.28 and the matrix's own generator already renders a missing cell
as a gap rather than borrowing a neighbour's number — which is the only reason
this surfaced as three dashes instead of three plausible numbers. `q_report`
will render the scorecard before the recovered cells land, so a second pass
regenerates it once they do; the published artifact is the complete one.

### 3.31 The published system was the one system without a confidence interval (§33, §36 item 21)

Reading the Stage B selection benchmark, `world_model_calibrated` had no
`auc_pr_bootstrap`. Ten other systems in the same table did:

| system | AP | 95% CI |
|---|---|---|
| `gru_classifier` | 0.509 | yes |
| `gbdt_current_state` | 0.463 | yes |
| `world_model` (pooled, uncalibrated) | 0.375 | yes |
| `noised_persistence` | 0.375 | yes |
| `persistence` | 0.336 | yes |
| `world_model_deterministic` | 0.318 | yes |
| **`world_model_calibrated`** — *the published system* | **0.366** | **no** |

`BOOTSTRAP_SYSTEMS` listed ten systems and not the one every headline AP in this
phase is quoted from. The consequence is visible in the artifact: the
scorecard's column is headed `AP [95% CI]` and has been rendering a bare AP in
**every cell**, because `_ci()` reads a key that was never written. §33 says
report confidence intervals; the table said it was reporting them.

The uncalibrated `world_model` is not a substitute. Platt is monotone *within* a
horizon, but the composite takes `max` **across** horizons afterwards, so the
two can order rows differently — §3.24 measured exactly that, with calibration
flipping two oracle comparisons.

Fixed, with `world_model_calibrated` first in the list so the omission is hard
to repeat. The cost is one more bootstrap per benchmark, negligible beside the
rollout.

**What this does not retroactively fix.** Every benchmark already on disk was
written without it, and the interval cannot be recovered from a summary — the
per-row scores are not stored. So the 20 recorded cross-evaluation cells keep a
bare AP, and every benchmark from here carries the interval: the six recovered
cells, the stage-balanced run, the decomposable-head arms, and the test/holdout
runs that follow the architecture freeze. The final report's headline numbers
come from that second group, which is the one that matters for §36 item 21 — but
the log should say plainly that the earlier table was not what its header
claimed.

**The twenty recorded cells are queued for re-run, into a separate directory.**
§3.31's fix is not retroactive, so the matrix would otherwise ship with a
`AP [95% CI]` column that is empty for every cell it already has. The re-run
buys the intervals and, because the evaluation code has changed since those
cells were written — the host-identity block, the calibration reader, D142 and
D143's guards, `--force-pooling` — it is simultaneously a reproducibility check
on every published number in the matrix.

Written to `repro_<label>/` rather than over `xeval_<label>/`. Overwriting would
leave this log's figures unmatched by any artifact exactly in the case where the
answer is interesting, which is the opposite of what §33 asks for.
`nidra/scripts/reproduction_check.py` compares them and is strict about what
counts as agreement: a difference in `n_rows` or prevalence is reported as
**different data**, not as small drift, because a matching AP on a different row
set is a worse finding than a moved one. Tolerance is 0.005 AP — a re-run is
seeded but the bootstrap resamples and the rollout draws, and 0.005 is an order
of magnitude below the smallest difference this phase draws a conclusion from.
Queued behind every other queue; it cannot starve anything.

### 3.32 Stage B on the selection metric, and the number that complicates it (§19, §23, §24, §31 Q6/Q7/Q8)

§3.29 confirmed the head comparison on the head-training metric. This is the
same two models on the **forecast benchmark**, which is what actually selects.
CTU-13 validation, identical rows, same pooling rule, three-member ensembles:

| | `state` | `state+hidden` |
|---|---|---|
| AP | 0.3660 | **0.4910** |
| ROC | **0.7752** | 0.7484 |
| precision @ operating point | 0.663 | **0.818** |
| recall | 0.410 | **0.484** |
| F1 | 0.507 | **0.608** |
| **false alarms / hour** | 9.26 | **4.79** |
| within-host ROC | **0.8158** | 0.6967 |
| oracle (frozen head on the true future) | 0.3350 | 0.5352 |
| persistence (frozen head, no transition step) | 0.3364 | 0.4991 |
| GRU sequence classifier, same rows | 0.5093 | 0.5093 |

**On the operating point the history-aware head wins outright.** Half the false
alarms — 4.79 an hour against 9.26 — at *higher* recall and markedly higher
precision. F1 0.608 against 0.507. For a system whose output is an alert, that
is the comparison that matters and it is not close.

Three things in the same table pull the other way, and the phase has to carry
all of them.

**AP rises while ROC falls** (0.366 → 0.491, 0.775 → 0.748). A gain concentrated
at the top of the ranking and a loss in the global ordering. Good for an alerting
threshold, and not the same thing as "the model got better".

**Within-host ROC falls from 0.816 to 0.697.** §3.11 established this as the one
column that separates learning the behaviour from learning the host: on the
infected host alone, where identity is constant, does the system order the attack
windows above that host's own benign ones? The Run 8 per-state head does that
*better*. The history-aware head is worse at the timing question while being far
better in aggregate — which is the host-identity signature, now appearing on the
run that selects the architecture rather than on a probe. (Validation carries one
infected host, so this is one host's ordering; §3.14 and §3.17 apply.)

**Neither head beats persistence.** The paired episode-cluster bootstrap on the
difference:

| head | world model − persistence | 95% CI |
|---|---|---|
| `state` | +0.0388 | [−0.0104, +0.0923] |
| `state+hidden` | **−0.0100** | [−0.0342, +0.0034] |

Both intervals contain zero, and the history-aware head's point estimate is
**negative**. Persistence here is the frozen head applied with the transition
step removed while the encoder still advances — so a history-aware head keeps
its history and the *only* thing taken away is the predicted change. The reading
is direct: on this split, the transition model contributes nothing measurable,
and the whole of the +0.125 is the head reading history that persistence gives
it anyway.

That is the ablation CLAUDE.md names explicitly — "if persistence matches the
model, we report that honestly and diagnose" — landing on the phase's own
candidate architecture.

**And a GRU sequence classifier on the identical rows reaches 0.5093**, ahead of
both. §24 asks whether the world model outperforms a strong sequence classifier;
on this split it does not.

**What the absolute CIs are worth here: very little.** Validation has eight
positive episode clusters, so the bootstrap on an absolute AP spans
[0.001, 0.826] and says nothing. The *paired difference* bootstrap resamples 347
clusters and is informative, which is why the persistence comparison above is
quoted from it and the absolute APs are quoted bare. A single-number AP on this
split should not be treated as measured to three decimals.

**Status.** These are **validation** numbers, which is where the operating point
is selected; test and holdout remain untouched. The architecture decision
therefore rests on: a large, replicated, low-variance head-training gain (§3.29);
a decisive operating-point gain here; against a worse within-host ordering, worse
calibration (§3.23), a lost explanation surface (§3.20), and a transition-model
contribution indistinguishable from zero for the winning head. Under §32's
ranking that is not an obvious call, and it should not be made on CTU validation
alone — the CIC arms and the unseen-family run are still in flight.

### 3.33 Is the persistence ablation even valid for a history-aware head? (§18, §23)

Before diagnosing §3.32's negative result, the obvious escape had to be closed:
persistence keeps the encoder advancing, so perhaps a head reading the hidden
state simply does not notice the ablation, and the null is an artefact of the
control rather than a fact about the transition model. That would have been a
comfortable explanation and it is wrong.

From `WorldModel.rollout`, the hidden is advanced by feeding **the state actually
produced**:

```python
h_t, h = self.encoder(nxt.unsqueeze(1), h)
```

Under `state_source="persist"`, `nxt = anchor` — the GRU is fed S_t six times.
Under `"model"` it is fed the predicted trajectory. The two hidden paths are
therefore different, `mu` is zeroed so `realized_deltas()` is identically zero,
and only `logvar` is shared. **The ablation removes the transition's
contribution from the hidden channel as well as the state channel**, which is
exactly what it claims to do, for a trajectory head as much as a per-state one.

§3.32's null stands. It is a measurement of the transition model, not of the
control.

**The diagnosis CLAUDE.md asks for, then.** §3.20 measured that this head's score
is set by the encoder hidden state and that the origin state contributes almost
nothing — risk moved from 0.0000128 to 1.0000000 behind a byte-identical origin
state. The hidden state entering the rollout already summarises thirty observed
windows. Six further steps, predicted or constant, are appended to that summary;
if the signal the head reads is *what this host has been doing* rather than
*where it is going*, those six steps cannot change much, and the transition's
contribution is small **by construction for this head**.

That account is testable and the test is cheap, so it is pre-registered here
before running: measure the relative divergence between the persistence and
model hidden states, ‖h_persist[k] − h_model[k]‖ / ‖h_model[k]‖, at each k.

- If the hiddens stay close — say under ~10% by k=6 — the head is reading a
  summary the rollout barely moves, the null is explained, and the implication
  is that a history-aware head **cannot** demonstrate transition-model value
  under this ablation no matter how good the transition model is. That would
  make §3.32's null uninformative about the transition model after all, for a
  different reason than the one just ruled out, and would call for a different
  control.
- If they diverge substantially and the AP still does not move, the transition
  model genuinely is not contributing and the null means what it says.

Both outcomes are reportable and they point at different next experiments, which
is the reason to write the criterion down first.

### 3.34 The diagnosis: the head is no more sensitive to the rollout than to noise

§3.33's pre-registered test, run as written. 1,483 CTU validation rows (300
positive), the confirmed `state+hidden` model, deterministic rollouts under
`state_source="model"` and `"persist"`:

| k | hidden relative divergence | state relative divergence | mean risk, model | mean risk, persist |
|---|---|---|---|---|
| 1 | 40.5% | 100.9% | 0.09843 | 0.09902 |
| 3 | 53.4% | 104.7% | 0.09876 | 0.09979 |
| 6 | **62.3%** | 102.2% | 0.09896 | 0.10029 |

**The second branch of the criterion.** The rollout moves the hidden state by
40–62%, far past the ~10% that would have made the ablation uninformative. The
ablation bites. And the head's answer does not move:

- composite correlation between the two arms, all rows: **0.9839**
- **positives only: 0.9910**
- attack composite mean: 0.51833 against 0.51689 — **0.28%**

So §3.32's null means what it says. The transition model is not contributing to
this head's score.

**Why, measured rather than argued.** Feeding the head's first layer the
rollout's actual displacement `h_model[6] − h_persist[6]` gives a gain of
0.0909. The same statistic on a **random** displacement of the same norm gives
0.0845 — a ratio of **1.08×**. The head is no more responsive to the transition
model's displacement than to noise of equal size. (The first hypothesis was
"nearly orthogonal to the readout"; the null says it is not orthogonal either,
just uninformative. Without the null the 0.0909 would have been read as
orthogonality, which is a different and wrong claim.)

**One real effect, invisible to AP.** Scoring both rollouts with the same head
and no calibration, the benign composite mean falls from 0.00412 under
persistence to 0.00201 under the model — roughly halved — while the attack mean
is unchanged. A near-uniform rescaling of one class is monotone enough to leave
a rank metric alone, which is exactly the AP null. It is a genuine effect of the
transition step on the score *scale*.

**What it is not.** The benchmark's table invites a false reading and it should
be named:

| system | AP | FA/h at the common threshold |
|---|---|---|
| `world_model_calibrated` | 0.4910 | 4.79 |
| `persistence` | 0.4991 | 23.64 |
| `world_model` (uncalibrated) | 0.4891 | 122.86 |

Those false-alarm rates are all taken at the **one** threshold selected for the
calibrated world model. Only that system carries the Platt layer, so the same
number means something different on every other row's score scale — §3.18's
error, one table over. **This is not evidence that the transition model reduces
false alarms**; it is evidence that calibration does. A real comparison would
calibrate each arm on its own validation split, and has not been run.

**The structural reading.** The risk head trains on **observed** states and
freezes (CLAUDE.md invariant 1). Its readout is whatever separates attack from
benign among observed hidden states, and there is no mechanism by which it would
become sensitive to where a rollout displaces them — the measurement says it is
not. That invariant is the reason the forecasting claim is falsifiable, and on
this split the falsification came back negative. Not a bug in the head; the
design working, and reporting a null.

This is the sharpest thing the phase has to say about §31 Q13. The limit is not
the head's capacity, the objective, or the corpus. It is that **a frozen
observed-state head and a transition model are only coupled through a channel
the head has no reason to read** — and measurably does not.

### 3.35 The pre-registered stage-balanced objective: criterion 1 fails, and §12 is refuted

§3.13 pre-registered this before running, with the primary criterion fixed: the
risk head's per-stage ROC on `recon` and `c2` must rise **above 0.5** on observed
validation states. The kill condition was stated in the same paragraph — if
forcing the rare stages to own an equal share of the gradient still leaves them
below chance, the pooled objective is not the explanation and the §12 direction
is wrong.

Same `ctu_dyn` checkpoint, same seed, same sampler, same data. The only thing
that moves is the weighting inside the positive class.

**`state+hidden`** (the head the §3.9 baseline was measured on):

| stage | windows | baseline risk-head ROC | stage-balanced | Δ |
|---|---|---|---|---|
| recon | 17 | 0.320 | 0.309 | −0.011 |
| c2 | 24 | 0.434 | 0.451 | +0.017 |
| exfil | 172 | 0.915 | 0.917 | +0.002 |
| *aggregate val AP* | | 0.4894 | 0.4927 | +0.003 |

**`state`** (the Run 8 architecture; baseline diagnostic run for this comparison):

| stage | windows | baseline risk-head ROC | stage-balanced | Δ |
|---|---|---|---|---|
| recon | 17 | 0.126 | 0.167 | +0.041 |
| c2 | 24 | 0.301 | 0.271 | −0.030 |
| exfil | 172 | 0.872 | 0.869 | −0.003 |
| *aggregate val AP* | | 0.3531 | 0.2965 | **−0.057** |

**Criterion 1 fails in every cell.** The best `c2` reaches is 0.451 and the best
`recon` is 0.309; both remain below chance. Criterion 3's guardrail holds (exfil
0.869 and 0.917, both above 0.85) and criterion 2 is moot, because the
pre-registration made criterion 1 decisive on its own.

**The intervention did apply** — checked rather than assumed, because a null that
is really a plumbing failure is the worst outcome available here.
`stage_balanced_positives: true` is in the run's config and absent from the
baseline's, and the `state` arm's aggregate AP moved 16% (0.3531 → 0.2965),
which is the objective visibly changing what the head learned. It changed the
head; it did not change the ranking of the rare stages.

**So the pooled objective is not why the risk head ranks recon and c2 below
chance.** §12's direction — rebalance the positive class so the rare stages own
their share of the gradient — is refuted on its own pre-registered terms. The AP
cost was paid and bought nothing.

**An honest limit on the strength of that.** One seed, and 17 and 24 positive
windows. §3.29 measured the `state` head's head-training variance at sd 0.047 on
aggregate AP, so a per-stage ROC on seventeen windows is noisy and this run
cannot distinguish 0.434 from 0.451. What it can say is that the intervention
produced **no movement toward the criterion** in any of the six cells — the
largest change in the intended direction is +0.041 against a required +0.18 —
and that is what refutes the direction, not a precisely measured null.

**What it points at instead.** The `state+hidden` head barely responded to the
reweighting at all (every Δ ≤ 0.017) while the `state` head moved substantially.
That is consistent with §3.34: the history-aware head's score is set by a
component of the encoder hidden state that encodes *which host this is*, and no
reweighting of the positive class by stage touches that. The two nulls are the
same null seen twice — the thing the head is reading is not the thing these
interventions are moving.

### 3.36 Every recorded cell reproduces exactly (§32 criterion 9, §36 items 9, 21)

The twenty recorded cross-evaluation cells were scored before
`world_model_calibrated` entered `BOOTSTRAP_SYSTEMS` (§3.31) and before the
host-identity block, the calibration reader and D142/D143's guards landed. They
were re-run into `repro_*` rather than over the originals, so that a drift would
leave the published figure still matched to the artifact that produced it.

28 cells re-ran. **All 28 reproduce, and not merely within tolerance — the AP is
bit-identical to the last digit in every one**, on identical row counts and
identical prevalence. `reproduction_check.py` judges at 0.005 AP because a
bootstrap resamples and a rollout draws; neither turned out to matter, because
the point estimate is a deterministic function of pinned seeds and only the
interval resamples.

The re-run also fixes what prompted it: **6 of 28 cells carried a confidence
interval before, 28 of 28 do now.**

Full table in `reports/run9/reproduction_check.md`. This is the §33 line "every
final metric gets a reproducible artifact" discharged by measurement rather than
by assertion.

One validity check worth recording separately, because the matrix would be
meaningless without it: for each evaluation target, every training source scores
**exactly the same rows** — all `*2cic` test cells are n=21,173 / 946 positive /
prevalence 0.00417, all `*2ctu` are n=25,099 / 3,688 / 0.00749. The cross-dataset
comparison varies the training set and nothing else.

### 3.37 The rollout manufactured state in features that carry no gradient (D145)

This is a defect, found while chasing §3.24's oracle anomaly, and it contaminates
every cross-dataset number in this report.

`train/losses.py` masks dropped features out of the transition loss — a feature
the scaler found constant or duplicated is excluded from the NLL and MSE
reductions. That is deliberate and correct: there is nothing to fit. The
consequence was not noticed. **The transition network receives no gradient at all
on those output dimensions**, so whatever it emits there is untrained, and
`rollout()` fed that output back in as the next state, once per step.

Measured on `comb2cic_state+hidden`, 13 dropped of 45:

| k | dropped slots, rms | abs max | kept slots, rms | true kept, rms |
|---|---|---|---|---|
| 1 | 0.581 | 2.24 | 0.643 | 0.712 |
| 3 | 1.245 | 4.10 | 0.650 | 0.716 |
| 6 | **2.035** | **6.21** | 0.695 | 0.709 |

The input holds those slots at exactly 0.0, and so does the true future. By k=6
the forecast carried three times more magnitude in features that do not exist for
this dataset pair than in the features that do — against a state clamp of 10.

It is present in every cell of the matrix, and in every one the phantom exceeds
the real features:

| cell | phantom rms k=6 | abs max | real rms | ratio |
|---|---|---|---|---|
| CIC → CIC `state+hidden`, test | 1.69 | 6.31 | 0.45 | 3.8× |
| CTU → CTU `state+hidden`, test | 2.12 | 8.95 | 0.95 | 2.2× |
| comb → CIC `state+hidden`, test | 2.01 | 5.78 | 0.68 | 2.9× |
| comb → CTU `state+hidden`, holdout | 2.49 | 8.88 | 0.68 | 3.7× |

**Why it reaches the answer.** The frozen risk head reads the 45-dim state. In
training those slots were always exactly zero, so the head's weights on them were
never constrained by data — they are whatever initialisation and weight decay
left. At inference the rollout hands them rms-2.03 inputs. The risk score
therefore contains a per-host projection of untrained weights onto untrained
drift, growing with horizon. The phantom slots also feed back through the GRU, so
they perturb the real features too: masking moves the kept-slot rms at k=6 from
0.695 to 0.603.

**What this does and does not explain.** It is the obvious candidate for two
findings already in this log — §3.24's oracle-beating (the oracle runs
`state_source="truth"`, whose dropped slots are exact zeros, so it alone gets no
contribution through those weights) and the deterministic arm's below-chance ROC
in 12 of 24 cells (the mean path drives the phantom systematically with no
sampling to cancel it). But the magnitude does not order the anomaly: the largest
phantom-to-real ratio, 3.8×, belongs to a cell that does *not* beat its oracle.
**Pre-registered before the re-scores land:** if D145 is the mechanism, masking
removes the oracle-beating in `comb2cic/test` and `comb2ctu/holdout`; if those
cells still beat their oracle with the phantom slots at zero, the mechanism is
refuted and negative result 10 stands unchanged.

**Fix.** `WorldModel.set_feature_mask()` takes the kept-feature mask from the
scaler and `rollout()` applies it to `nxt`, `mu` and `logvar` after the clamp. All
three reach a head — `mu` a `state+delta` head, `logvar` a `state+logvar` head —
and §3.38 below turns on exactly the `state+logvar` arm, so leaving that channel
untrained would have put the defect inside the conclusion. `logvar` is pinned to
the transition's floor rather than zeroed, because a log-variance of 0 asserts
unit variance rather than none; sampling in a dropped slot cannot reach the state
anyway, since the mask is applied after the noise is added. The mask is
not a checkpoint entry; it belongs to the scaler, and `load_models` sets it from
the scaler that D142's guard already forces to match. Passing `None` restores the
old path exactly, so every recorded run stays reproducible — which §3.36 has just
finished demonstrating and which must not be broken by the fix for it.

Twenty-five tests in `tests/test_rollout_dropped_features.py`; suite 762 → 787.

**Four paths hand a head a transition output, and all four needed the contract.**
Found by grepping for every direct `transition()` call and every `WorldModel`
construction rather than by reasoning about which ones mattered:

| path | what it feeds | status |
|---|---|---|
| `WorldModel.rollout` | the forecast, and the state a head reads | masked |
| `WorldModel.observed_context` | the served origin state's head inputs | `logvar` pinned |
| `train/head_context.py` | head **training** inputs | `logvar` pinned |
| `explain/counterfactual.py` | the served "model-internal what-if" | masked |
| `train/losses.py` | the loss itself | already masked — this is the origin of the defect, not a victim of it |

`NidraPredictor` loads its own ensemble rather than going through
`benchmark.load_models`, so without the same call production would have been the
one place the defect went unmeasured. The counterfactual matters for a second
reason: it re-implements the rollout loop so it can re-clamp the intervened
feature every step, and it is one of the six demo-critical paths — a divergence
there appears in front of a viewer rather than in a metric. The mask is applied
*after* the intervention, so a what-if on a dropped feature is inert rather than a
confident answer about a quantity the model has no information on.

**Wiring it at each call site was itself the bug.** The first attempt set the
mask in `benchmark.load_models` and `NidraPredictor` and missed the training
entry point, so re-running §3.38's comparison "under the fix" returned
`state+logvar` = **0.7754417090891179**, bit-identical to the unfixed run,
because `build_head_context` was calling `mask_logvar` on a model whose mask was
still `None`. The run exited zero and wrote a full provenance record; its
apparent conclusion — "§3.38 is unaffected by D145" — is a plausible finding.
Bit-identical to sixteen significant figures is what caught it: two head
trainings with different inputs do not agree that closely.

The mask now comes from `world_model_from_config`, the single construction point,
and both call sites were deleted rather than kept in parallel. `eval/run_eval.py`
and `scripts/fit_calibration.py` go through the same builder, which is safe
rather than a change of policy: the default scaler drops nothing, so their mask
is `None` and the Runs 1–7 figures they exist to reproduce are untouched.
Recorded as negative result 15.

**Blast radius.** 40 of 63 run scalers drop features: every CTU-13 run drops 15,
every combined and `cic_core` run drops 13. The 23 `full`-regime runs drop nothing
and are arithmetically unaffected, so **Run 8 is not touched by this** — and its
artifacts are not re-run regardless (§33).

**The rollout was not the only exposed path.** `observed_context` (serving,
`score_observed`) and `train/head_context.py` (head *training*) both call the
transition directly and take its `logvar`, so a `state+logvar` head was reading
13 untrained log-variances of 45 **while it was being trained** — which is
precisely the number §3.38 turns on. Both now go through the same
`mask_logvar` helper. `state`, `hidden` and `delta` are built from observed data
and were always clean; only `logvar` needed it. This means §3.38's comparison
has to be re-trained, not merely re-scored, and it is: `cic_core_decomp_masked`
re-runs all six arms, with `state` and `state+hidden` included as controls that
do not read `logvar` and should therefore be unchanged.

#### The pre-registered test, and its answer

§3.37 registered this before the re-scores ran: *if D145 is the mechanism behind
the oracle anomaly, masking removes the oracle-beating in `comb2cic/test` and
`comb2ctu/holdout`.*

**It does not. The hypothesis is refuted.**

| system | AP before | AP after | ROC before | ROC after |
|---|---:|---:|---:|---:|
| `world_model_calibrated` | 0.1584 | **0.1990** | 0.889 | 0.950 |
| `world_model` (raw) | 0.1470 | **0.2018** | 0.913 | 0.967 |
| `world_model_deterministic` | 0.0405 | 0.0408 | 0.254 | 0.266 |
| `oracle_true_future` | 0.0519 | 0.0519 | 0.367 | 0.367 |
| `persistence` | 0.0342 | 0.0342 | 0.194 | 0.194 |

Both cells the test named have now run, and both still beat their oracle:

| cell | raw arm − oracle, before | after | |
|---|---:|---:|---|
| comb → CIC, test | +0.0951 | **+0.1499** | the gap *grew* |
| comb → CTU, holdout | +0.0528 | **+0.0435** | shrank by 18%, nowhere near closed |

The phantom accounts for a small part of one gap and none of the other. The
oracle and persistence arms are unchanged in both, exactly as they should be —
their states never carried it. **§3.24's cells stay unexplained and negative
result 10 stands as written.**

#### What the correction did surface

With the phantom removed, the same pattern is visible in both anomalous cells and
it is sharper than anything §3.24, §3.26 or §3.27 had:

| cell | oracle ROC | persistence ROC | raw rollout ROC |
|---|---:|---:|---:|
| comb → CIC, test | 0.367 | 0.194 | **0.967** |
| comb → CTU, holdout | 0.558 | 0.456 | **0.897** |

The frozen head ranks at or below chance on the **true** future states and far
above chance on states its own transition model produced. Both cells are
combined-training cells, which is the condition under which a head is furthest
from the state distribution of whichever single corpus it is then scored on.

The hypothesis this suggests — that the rollout acts as a *projection onto the
model's own learned manifold*, where the frozen head is reliable, and that the
oracle's true states are out-of-distribution for a head trained on a different
mixture — is stated here as a hypothesis and nothing more. §3.27 is the standing
reminder of what happens to an explanation on this question that is not given a
criterion in advance. The test it needs is a distance measurement between the
head-training state distribution and each of the two candidate inputs, on cells
where the model beats its oracle and on cells where it does not; the first
attempt at that measurement is what found D145, so it has not yet been run to
answer the question it was written for.

One thing the table settles in passing: **the deterministic arm's below-chance
ROC is not the phantom either** — 0.254 to 0.266. That remains unexplained too.

One thing it does *not* settle, and a claim drafted from this cell alone had to
be withdrawn when the next two landed. Raw AP by cell:

| cell | before | after | Δ |
|---|---:|---:|---:|
| comb → CIC, test | 0.1470 | 0.2018 | **+0.0548** |
| CTU → CTU, test | 0.3203 | 0.3254 | +0.0050 |
| comb → CTU, test | 0.3697 | 0.3551 | **−0.0147** |

"The phantom was costing accuracy" was true of the first cell and false of the
third. An untrained signal projected through untrained weights helps some cells
and hurts others, which is the reason the correction is being measured across the
whole matrix rather than estimated from a sample. The oracle and persistence arms
are unchanged in every cell, as they must be — their states never carried it.

What the table does point at is something the phantom was obscuring: on this
cell the oracle itself ranks at ROC 0.367 and persistence at 0.194, both *below
chance*, while the sampled rollout reaches 0.967. A frozen head trained on
combined data appears to be anti-correlated on CIC's true state distribution and
strongly correlated on states its own transition model produced. That is a
sharper statement of §3.24's anomaly than anything before it, and it is the next
thing to test.

On a non-anomalous cell the correction is much smaller — CTU → CTU test moves
+0.005 raw, −0.004 calibrated, and the oracle still wins by 0.026. The
correction is not uniform, and the full 28-cell re-score is running.

### 3.38 On CIC, the decomposable head is not the loser it was on CTU (§5, §27, §31 Q13/Q14)

§3.22 recorded that the explainability cost of the history-aware head might not
be a component choice at all — on CTU, `state+logvar` carried only 14% of what
`state+hidden` carried, so there was no decomposable head to retreat to. It also
recorded, as a pre-commitment, that **CIC was untested**: the ablation there had
only ever run `state` and `state+hidden`, and a negative would close the
direction properly rather than leave it standing as an untested suggestion.

Four CIC arms were queued. The result is not the negative that was expected.

| head reads | val AP | input dim | decomposable into the 45 named features? |
|---|---|---:|---|
| `state+hidden` | 0.7825 | 173 | no |
| **`state+logvar`** | **0.7754** | 90 | **yes** |
| `hidden` alone | 0.7553 | 128 | no |
| `state+delta+logvar` | 0.7330 | 135 | yes |
| `state` | 0.6781 | 45 | yes |
| `state+delta` | 0.6607 | 90 | yes |

On CIC, `state+logvar` reaches **99.1% of the history-aware head's AP** — 0.0071
behind, against the ±0.047 head-training seed spread §3.29 measured, so the two
are not distinguishable at one seed. On CTU the same arm carried 14%. The
explainability cost is therefore **dataset-dependent, not intrinsic**, which is
the opposite of what §3.22 was heading towards.

Three things keep this from being a conclusion:

1. **One seed, Stage A.** §19 and §33 both forbid reporting a screening run as
   final, and the gap is far inside the noise §3.29 measured. This selects a
   candidate for Stage B; it does not settle anything.
2. **It is a validation number.** No test or holdout cell has been run for these
   arms, and §3.29/§3.32 are the standing reminder that a val ranking in this
   phase has twice failed to survive the forecast benchmark.
3. **D145 lands directly on it.** `state+logvar` reads 45 log-variances, 13 of
   which were untrained in the `cic_core` regime this ran under. Part of the
   0.7754 may be the head reading structure in slots that carry no information.
   §3.37's fix pins those to the variance floor, so **this comparison has to be
   re-run before it is used for anything** — and it is the single arm most
   exposed to the defect, which is why the fix was extended to `logvar` rather
   than stopping at the state.

Recorded as an open candidate, not a finding. Q13's limit 4 and Q14's direction 4
are updated to say the direction is *open on CIC and closed on CTU*, pending a
re-run under §3.37 and a Stage B confirmation.

### 3.39 The unseen attack family: the world model loses to persistence (§31 Q9, §36 item 15)

Neris is CTU's largest family — scenarios 1, 2 and 9. Withheld from training
entirely; heads and dynamics retrained without it; evaluated on it. This is the
§31 Q9 evidence and nothing else in the phase substitutes for it.

| head | split | n | prev | AP [95% CI] | ROC | persistence | oracle | within-host ROC |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `state` | val | 21,786 | 0.0122 | 0.0116 [0.008, 0.015] | 0.505 | 0.0108 | 0.0125 | 0.612 |
| `state` | test | 22,501 | 0.0079 | 0.0442 [0.015, 0.084] | 0.869 | **0.4448** | 0.4898 | 0.943 |
| `state+hidden` | val | 21,786 | 0.0122 | 0.1156 [0.070, 0.183] | 0.660 | 0.0748 | 0.1244 | 0.553 |
| `state+hidden` | test | 22,501 | 0.0079 | **0.7166** [0.485, 0.860] | 0.950 | **0.7715** | 0.7957 | 0.949 |

**Read the validation rows first, and then discard their operating point.** The
model is at chance on val (ROC 0.505 and 0.660). The threshold and calibration
were nevertheless frozen there, because §33 forbids selecting them anywhere else.
So the F1 and false-alarms-per-hour columns for this experiment describe a
threshold chosen on a split where the model cannot rank — 6,151 and 6,293 false
alarms per hour for `state`, 27.5 for `state+hidden`. Those numbers are not
evidence about either head. AP is the only usable column here.

On that column:

- **`state+hidden` reaches AP 0.717 on a family it has never seen**, with ROC
  0.950 and a within-host ROC of 0.949 across **ten infected hosts** — the first
  genuinely cross-host within-host measurement in the phase, and §3.14's central
  complaint (one infected host in train and in val, the same address) does not
  apply to it.
- **Persistence reaches 0.772.** The world model loses to it, on both heads. The
  intervals are wide and overlapping, so the gap is not established either way —
  but the honest summary of Q9 is that the model transfers to an unseen family
  *no better than repeating the host's current state does*.
- **`state`'s aggregate AP collapses to 0.044 while its within-host ROC is
  0.943.** It orders windows correctly inside each host and cannot compare
  across them — 0.869 aggregate ROC against 0.044 AP is what that looks like at
  0.8% prevalence. This is the same per-host-calibration failure as §3.11, in its
  clearest form yet.

GRU is `nan` in these cells: `gru_classifier.pt` was not copied into the lofo
runs, so the strongest baseline is **missing** from the one experiment where it
would matter most. That is a gap, recorded as such, not an omission of an
unfavourable number — §3.32 already reports the GRU beating both heads on the
within-dataset benchmark, so the expectation is that it would win here too.

### 3.40 Q2 and Q3 at a common readout (§31 Q2/Q3, §36 items 12–14)

The matrix cells each use the pooling the validation split selected for them,
which makes a cross-training-set comparison partly a comparison of readouts.
Re-scored with `--force-pooling 'mean|q=-|max'` on all four, natural prevalence,
episode-cluster intervals:

| cell | AP [95% CI] | ROC | persistence | oracle | calibration matched to the forced readout? |
|---|---|---:|---:|---:|---|
| CIC → CIC | 0.0672 [0.021, 0.239] | 0.575 | 0.0414 | 0.0689 | **no** (selected `integrated`) |
| CIC+CTU → CIC | 0.1584 [0.052, 0.283] | 0.889 | 0.0342 | 0.0519 | yes |
| CTU → CTU | 0.3221 [0.125, 0.583] | 0.680 | 0.3215 | 0.3517 | yes |
| CIC+CTU → CTU | 0.3734 [0.204, 0.586] | 0.795 | 0.3623 | 0.3999 | yes |

n = 21,173 (946 positive, prevalence 0.00417) for both CIC rows and 25,099 (3,688,
0.00749) for both CTU rows — identical rows within each target, so only the
training set varies.

**The baseline arm of Q2 is handicapped.** CIC → CIC is the one cell whose Platt
layer was fitted under a different pooling key than it is scored at. Forcing the
readout leaves its calibration mismatched while its comparison arm's is not,
which is exactly the asymmetry that manufactures an improvement. The uncalibrated
`world_model` column is the check: 0.0380 → 0.1470, ROC 0.438 → 0.913. The gain
survives removing the Platt layer, so it is not a calibration artifact — but the
calibrated figures in the table above overstate it.

**Q2 — does adding CTU-13 improve CIC generalisation?** Point estimate yes, and
substantially (ROC 0.44 → 0.91 uncalibrated). Statistically, the intervals
overlap across most of their range on **15 positive episode clusters**. Supported
as a direction, not as a quantity.

**Q3 — does adding CIC improve CTU?** +0.051 calibrated, +0.049 uncalibrated, on
intervals that overlap almost entirely. **And persistence is level with the model
in both CTU rows** — 0.3215 against 0.3221, and 0.3623 against 0.3734. Whatever
the training mixture does on CTU, it does not lift the model past repeating the
current state. Not supported.

Every number in this section was produced under D145 and needs re-running under
§3.37's fix before it is final.

#### A correction to finding 8

Finding 8 in the navigation table says that every "% of oracle" statement has to
be made against the deterministic arm. **That guidance is withdrawn.** The
deterministic arm scores *below chance* in 12 of 24 recorded cells, ROC as low as
0.009, while the sampled arm on the same rollout reaches 0.8–0.97 — it is not a
cleaner version of the system, it is a differently-behaved one, and §3.37 gives
the likely reason. The uncalibrated sampled arm `world_model` is the reference
that is actually like-for-like with the oracle: same pooling, same sampling, no
Platt layer. On that arm the oracle is beaten in **11 of 28** cells, of which
only four exceed 0.03, and all four are combined-training cells.

### 3.41 Pre-registration: is the rollout a projection onto the training manifold? (§25, §36 item 22)

Written before the measurement, because §3.27 is what happens on this question
when an explanation is not given a criterion in advance, and because the first
attempt at this measurement found D145 instead of answering it.

**The observation to explain.** In both cells that beat their oracle, the frozen
head ranks at or below chance on the *true* future states and far above chance on
states the transition model produced:

| cell | oracle ROC | persistence ROC | raw rollout ROC |
|---|---:|---:|---:|
| comb → CIC, test | 0.367 | 0.194 | 0.967 |
| comb → CTU, holdout | 0.558 | 0.456 | 0.897 |

**The hypothesis.** A frozen head is reliable on states resembling the ones it
trained on. A rolled-out state is generated by the model and therefore lies on
the model's own learned manifold, near its training distribution; the true future
state of a corpus the head was not trained on alone is further away. If so the
rollout acts as a *projection*, and the head does better on the projection than
on the truth — which is what beating the oracle would mean.

**The measurement.** Mean standardised distance from the head-training state
cloud to (a) the true future state at k=6 and (b) the sample-mean rolled-out
state at k=6, on the same rows. Standardisation uses the training cloud's own
per-feature spread, over features with non-degenerate training variance only —
the degenerate ones are what produced distances of 2×10⁶ on the first attempt and
led to D145. Reported as the ratio d(rollout)/d(truth).

**The criterion, fixed now.**

- **Supported** if, on every cell where the raw arm beats its oracle, the ratio is
  below 1 — the rollout's states are closer to the head's training distribution
  than the truth is — **and** on every cell where the oracle wins, it is at or
  above 1.
- **Refuted** if the ratio fails to separate the two groups: if any
  oracle-beating cell has a ratio at or above 1, or any oracle-winning cell has
  one below 1.
- **Not supported either way** if fewer than three cells fall in each group once
  the full matrix is re-scored, in which case the test is underpowered and says
  so rather than reading a two-cell pattern.

A refutation here leaves §3.24 unexplained for the third time, and that is an
acceptable outcome. What is not acceptable is choosing the statistic after
seeing which one separates the groups, which is why the statistic, the horizon,
the standardisation and the threshold are all fixed above.

The measurement runs on the `mask_*` artifacts, after the sweep, so that it is
made on the corrected model rather than on the one with phantom features.

#### Amendment, before the verdict: the exclusion threshold was wrong

The criterion above says "features with non-degenerate training variance only —
the degenerate ones are what produced distances of 2×10⁶ on the first attempt".
The code implemented that as `sd > 1e-6`, and **that does not implement it**. On
the first cell scored, `rst_ratio` had a training sd of 5.33e-05, passed the
threshold, contributed a mean z² of **1,009,400** and produced a rollout distance
of 806 against a truth distance of 4.9 — the exact failure the pre-registration
named, one order of magnitude down.

Two things made it visible rather than plausible. The ratio was 167, which is not
a number a distance ratio takes. And the kept-feature *set* moved with the
training sample size — 31 features at 1,500 samples, 32 at 2,500 — so the
statistic was unstable in a parameter that should be irrelevant.

**What changed and what did not.** The statistic, the horizon, the standardisation
and the ≥1 / <1 threshold are unchanged. Only the numeric definition of
"degenerate" moved, from 1e-6 to 1e-2, set against the scaling convention rather
than against this data: the scaler maps features to roughly unit spread, so a
training sd two orders below that is a constant.

**Because that is still a judgement made after seeing a failure, the verdict is
computed at four thresholds — 3e-3, 1e-2, 3e-2, 1e-1 — and is only a verdict if
all four agree.** If they disagree the probe reports `INCONCLUSIVE` and says so,
because a conclusion that depends on which threshold was chosen is a conclusion
about the choice. This amendment was written before any ratio was read.

### 3.42 Under the D145 fix, the decomposable head is no longer behind (§5, §27, §31 Q13/Q14)

§3.38 found `state+logvar` at 99.1% of the history-aware head on CIC and flagged
it as the arm most exposed to D145, because it reads 45 log-variances of which 13
were untrained. §3.37's fix pins those to the transition's floor in head
*training* as well as at serving, so the comparison was re-trained rather than
re-scored.

**The controls first, because the first attempt at this run was a silent no-op**
(negative result 15). `state`, `state+hidden`, `hidden` and `state+delta` do not
read `logvar` and must therefore be unchanged:

| control | before | after | Δ |
|---|---:|---:|---:|
| `state` | 0.6780771051850328 | 0.6780771051850328 | 0 |
| `state+hidden` | 0.7825444222939472 | 0.7825444222939472 | 0 |
| `hidden` | 0.7553316342825650 | 0.7553316342825650 | 0 |
| `state+delta` | 0.6606590852996989 | 0.6606590852996989 | 0 |

Bit-identical, all four. The two arms that read `logvar` are the two that moved,
which is what says the harness is reporting the fix rather than reporting
nothing.

| head reads | val AP before | val AP after | Δ | input dim | decomposable? |
|---|---:|---:|---:|---:|---|
| **`state+logvar`** | 0.7754 | **0.7869** | **+0.0114** | 90 | **yes** |
| `state+hidden` | 0.7825 | 0.7825 | 0 | 173 | no |
| `state+delta+logvar` | 0.7330 | **0.7817** | **+0.0487** | 135 | **yes** |
| `hidden` | 0.7553 | 0.7553 | 0 | 128 | no |
| `state` | 0.6781 | 0.6781 | 0 | 45 | yes |
| `state+delta` | 0.6607 | 0.6607 | 0 | 90 | yes |

**On CIC the decomposable head now edges ahead of the opaque one** — 0.7869
against 0.7825 — and a second decomposable arm, `state+delta+logvar`, lands
within 0.0008 of it. Both are built entirely from quantities defined over the 45
named features; neither reads the 128-dimensional recurrent summary that §3.20
found impossible to attribute.

That is the first genuinely positive architectural result in the phase, and the
reasons to hold it loosely are the same three as before plus one:

1. **One seed, Stage A, validation only.** The 0.0044 margin is two orders below
   the ±0.047 seed spread §3.29 measured. This selects a candidate for Stage B; it
   settles nothing. §19 and §33 both forbid reporting a screening run as final.
2. **94 positives.** The head-training validation split carries 94 positive rows
   in 893,701. An AP computed on 94 positives is not a precise instrument, and no
   part of this table should be read to more than two decimal places.
3. **No test or holdout cell exists for these arms.** §3.29 and §3.32 are the
   standing reminder that a validation ranking in this phase has twice failed to
   survive the forecast benchmark.
4. **The direction of the correction is itself informative and was not
   predicted.** Removing 13 untrained log-variances *improved* both arms that read
   them, by +0.011 and +0.049. The untrained channel was noise the head had to
   work around, not signal it was exploiting — which is the opposite of the worry
   in §3.38's third caveat, where the concern was that part of the 0.7754 might be
   the head reading structure in slots that carry no information.

**Q13's limit 5 and Q14's direction 4 are revised again.** The claim "the AP gain
and the explainability loss are the same thing" was a CTU statement read as a
general one (negative result 13); on CIC, under the corrected model, it is not
even true as a ranking. What is now open is whether a 90-dimensional head built
from named features can match a 173-dimensional one across seeds and on a split
it was not selected on. That is a Stage B question and it is the strongest
candidate this phase has produced.

### 3.43 The full re-score under D145's fix: 21 cells up, 7 down, and the oracle anomaly got worse (§3.37, §36 items 10–14, 18)

All 28 recorded cells re-scored into `mask_*`, zero failures. Full table in
`reports/run9/mask_correction.md`.

**21 up, 7 down.** Five moved the published arm by at least 0.05 AP. The deltas
are not averaged anywhere, because an untrained signal projected through
untrained weights has no reason to point the same way twice and the split is the
finding. The largest movers:

| cell | AP before | AP after | Δ |
|---|---:|---:|---:|
| comb → CIC `state`, holdout | 0.2243 | 0.3229 | **+0.0986** |
| CTU → CIC `state`, test | 0.0448 | 0.1147 | **+0.0699** |
| CIC → CIC `state+hidden`, test | 0.0642 | 0.1173 | **+0.0531** |
| CTU → CTU `state`, holdout | 0.1884 | 0.2410 | **+0.0526** |
| CIC → CIC `state`, test | 0.1110 | 0.1633 | **+0.0523** |
| CIC → CIC `state`, val | 0.7809 | 0.7491 | **−0.0319** |
| comb → CIC `state+hidden`, val | 0.5381 | 0.5156 | **−0.0225** |

The oracle and persistence arms are unchanged in **all 28 cells**, which is the
check that the re-score differs from the original in the fix and nothing else —
their states hold dropped slots at exactly zero, so the mask cannot touch them,
and `mask_correction.py` refuses to report a delta for any cell where they move.

**The oracle anomaly did not shrink. It grew.**

| | before | after |
|---|---:|---:|
| raw arm beats its oracle | 10 of 28 | **17 of 28** |

Eight cells flipped from losing to their oracle to beating it; one went the other
way. §3.37's pre-registered test had already been refuted on the two cells it
named; the full sweep says the same thing at matrix scale and more loudly. **The
phantom drift was not what made the model beat its oracle — removing it made more
cells do so.** Negative result 10 stands, and §3.24 is now unexplained across a
larger set of cells than when it was written.

**What it does to the answers.** The direction of every published conclusion
survives, and two get slightly weaker:

| question | before | after |
|---|---|---|
| Q2 — does CTU help on CIC? (test) | 0.1584 vs 0.0642, gap +0.0942 | 0.1990 vs 0.1173, gap **+0.0817** |
| Q3 — does CIC help on CTU? (test) | 0.3734 vs 0.3221, gap +0.0513 | 0.3537 vs 0.3186, gap **+0.0351** |

Q2 remains supported as a direction and not as a quantity; Q3 remains
unsupported, and its gap is now smaller than it was. Nothing flips.

**One thing the sweep settles that was open.** §3.37 recorded, from three cells,
that the correction "does not go one way". At 28 cells the split is 21 up and 7
down — so the fix does help more often than it hurts, and the earlier statement
was right to refuse a direction from three cells but would have been right in
spirit if it had guessed one. The reason it is still not summarised as "the fix
improves the model" is that seven cells got worse and two of those are validation
cells, where the operating point is selected.

### 3.44 The scorecard on corrected artifacts: the model beats its best baseline in 3 of 12 cells (§17, §23, §24, §31 Q7/Q8/Q11, §32 criterion 1)

`reports/run9/scorecard_run9.md`, built from the `mask_*` artifacts, `state+hidden`
head, natural prevalence, episode-cluster intervals. This is the table the phase
was designed to produce.

| regime | split | model AP | persistence | best baseline | model wins? | ROC |
|---|---|---:|---:|---|:--:|---:|
| CIC → CIC | test | 0.1173 | 0.0414 | `noised_persistence` 0.0981 | **yes** | 0.760 |
| CIC → CIC | holdout | 0.3869 | 0.3422 | `lr_flattened_history` 0.3815 | **yes** | 0.839 |
| CTU → CTU | test | 0.3186 | 0.3215 | `noised_persistence` 0.3532 | no | 0.691 |
| CTU → CTU | holdout | 0.2273 | 0.1955 | `noised_persistence` 0.2452 | no | 0.720 |
| CIC → CTU | test | 0.0105 | 0.0084 | `noised_persistence` 0.0125 | no | 0.580 |
| CIC → CTU | holdout | 0.0159 | 0.0121 | `noised_persistence` 0.0172 | no | 0.758 |
| CTU → CIC | test | 0.0040 | 0.0021 | `gbdt_current_state` 0.0240 | no | **0.222** |
| CTU → CIC | holdout | 0.0003 | 0.0002 | `gbdt_current_state` 0.0014 | no | **0.133** |
| **comb → CIC** | **test** | **0.1990** | 0.0342 | `noised_persistence` 0.1073 | **yes** | **0.950** |
| comb → CIC | holdout | 0.3392 | 0.3414 | `persistence_rollout` 0.3556 | no | 0.923 |
| comb → CTU | test | 0.3537 | 0.3623 | `persistence` 0.3623 | no | 0.789 |
| comb → CTU | holdout | 0.2400 | 0.2026 | `gbdt_current_state` 0.2992 | no | 0.836 |

**Three of twelve.** That is the §32 criterion-1 answer for the phase, stated as
a fraction rather than as the best cell.

**The one cell that is not close.** `comb → CIC` test: AP 0.199 against a best
baseline of 0.107 and a persistence of 0.034 — nearly double the strongest
baseline, at ROC 0.950, on 946 positives in 21,173 rows at prevalence 0.00417.
It is the best result the phase produced, and it is the cell where training on
both corpora meets the target the head can actually rank. The two CIC → CIC wins
are narrow enough (0.1173 vs 0.0981; 0.3869 vs 0.3815) that on their own they
would not be worth claiming.

**Everything aimed at CTU loses.** In all four CTU-target cells the best baseline
wins, and in three of them the winner is `noised_persistence` — persistence plus
isotropic noise scaled to the model's own predicted variance. That is the §24
question answered directly: on CTU, what the world model contributes over
persistence is reproduced by adding correctly-scaled noise to persistence. The
transition model's *mean* is not carrying the result there.

**`CTU → CIC` is not weak, it is inverted.** ROC 0.222 on test and 0.133 on
holdout — far below chance, on both splits, in the same direction. A model
trained on CTU ranks CIC hosts close to backwards. This is worth separating from
"transfer is hard": a coin flip would score 0.5. Whatever the CTU-trained head
learned is a real ordering that is anti-correlated with CIC's labels, which is a
more specific and more interesting failure than noise, and it is unexplained.

**Calibration.** Five of twelve cells do not beat predicting the base rate and
never moving — both `CIC → CTU` cells, both `CTU → CIC` cells, and `CIC → CIC`
test. The `CIC → CTU` cells also carry 398 and 349 false alarms per hour. Those
four transfer cells should not be displayed to an operator as probabilities at
all, and the scorecard marks each one **no** rather than reporting only AP.

**What this does to Q11 (best configuration under natural prevalence).** The
answer is `CIC+CTU → CIC` with the `state+hidden` head, and the honest form of
that answer is: *the best configuration is the only regime-and-target pair where
the model clearly beats its baselines, and it does not generalise to the other
five.* A configuration that wins one cell of twelve is a finding about that cell.

### 3.45 §3.41's verdict: REFUTED, at every threshold, in the opposite direction (§25, §36 item 22)

The pre-registered criterion, run as amended, on the `mask_*` artifacts. 24 cells,
17 of which beat their oracle and 7 of which do not — both groups above the
minimum of 3, so the test is powered.

**REFUTED, and the same at all four thresholds** (3e-3, 1e-2, 3e-2, 1e-1), so the
amendment to the exclusion threshold makes no difference to the answer.

It fails in both of the ways it could:

**The direction is wrong everywhere.** The hypothesis needed rolled-out states to
be *closer* to the head's training distribution than the true future is. In
**0 of 24 cells** are they closer. Not in the oracle-beating cells, not in the
others, not at any threshold.

**And the statistic does not separate the groups.**

| group | n | min | median | max |
|---|---:|---:|---:|---:|
| beats its oracle | 17 | 1.143 | 1.424 | 2.211 |
| oracle wins | 7 | 1.141 | 1.298 | 1.941 |

The ranges are almost the same range. Even if the ≥1/<1 threshold were moved to
wherever it separated them best, there is nothing here to separate.

**So §3.24 is unexplained after four attempts**, three of them against criteria
fixed in advance:

| attempt | where | outcome |
|---|---|---|
| the horizon reduction | §3.26 | eliminated (6 of 6 individual horizons) |
| combined-model drift amplification | §3.27 | criterion failed — the CIC-only model amplifies too |
| D145's phantom features | §3.37 | refuted; masking made *more* cells beat their oracle, 10 → 17 |
| projection onto the training manifold | §3.45 | refuted at every threshold, direction inverted |

**What the refutation leaves is a sharper question than the one it answered.**
The frozen head ranks *better* on states that are measurably *further* from its
own training distribution — ratio median 1.42, and it holds in every cell. That
is the opposite of how a frozen readout is supposed to behave, and it is now the
most specific unexplained thing in the project.

It is worth being precise about what is and is not surprising. A rollout drifting
away from the training cloud is ordinary: six recursive steps with a learned mean
will do that. What is not ordinary is that the head's *ranking quality* improves
along that drift, while the same head applied to the true future — which is
nearer its training data — ranks at or below chance in the cells where this is
sharpest (oracle ROC 0.367 and 0.558, §3.37).

No fifth hypothesis is offered here. §3.27 and this section are two cases of an
explanation for §3.24 being written down and then refused by its own criterion,
and a third guess without a new measurement behind it would be the thing those
two sections exist to discourage.

### 3.46 §3.34's diagnosis re-measured under the D145 fix (D144 revised)

§3.34 is the phase's central mechanical result and it was measured on a rollout
that was manufacturing state in 15 of 45 feature slots. The phantom compounds
with horizon and feeds back through the GRU, so it could have been inflating the
very quantity the section turns on — how far the rollout moves the hidden state.
Re-measured on identical rows (`ctu_heads__state+hidden`, val, n=1,483).

| quantity | before the fix | after | verdict |
|---|---|---|---|
| hidden relative divergence, k=1 → k=6 | 40% → 62% | **39% → 54%** | the phantom was inflating the far end |
| head's answer, positives (composite mean) | 0.28% | **0.36%** | unchanged in kind |
| composite correlation, positives only | 0.9910 | **0.9817** | unchanged in kind |
| head-layer gain on the rollout's displacement | 0.0909 | **0.0962** | |
| the same on a random displacement of equal norm | 0.0845 | **0.0835** | |
| **ratio** | **1.08×** | **1.15×** | still the null |

**The conclusion is unchanged and two of its numbers are not.** The rollout moves
the hidden state by about half, the head's answer moves by a third of a percent,
and the head's first layer is no more sensitive to that displacement than to
noise of the same size — 1.15× rather than 1.08×, which is a larger margin over
the null but nowhere near a coupling. The structural reason given in §3.34 stands:
a head that trains on observed states and freezes has no mechanism to become
sensitive to where a rollout displaces them.

**One observation in D144 does not survive and is withdrawn.** D144 recorded that
"the benign composite mean halves under the model rollout (0.00412 → 0.00201)
while the attack mean is unchanged", offered as a real effect that AP cannot see.
Under the fix the benign means are **0.00672 against 0.00796** — a 19% difference,
in the opposite direction, and not a halving. That observation was an artifact of
the phantom features and should not have been recorded as an effect of the
transition model. It was already flagged as "NOT to be reported as a false-alarm
improvement"; it is now withdrawn entirely.

The upper end of the divergence moving 62% → 54% is worth keeping in view for a
different reason: it is the clearest single illustration that D145 inflated a
quantity nobody was looking at it to inflate. Nothing in §3.33's pre-registered
criterion turned on the difference — the criterion was 10% and both numbers clear
it by a factor of four — so the pre-registered conclusion is unaffected. It would
not have been if the criterion had been set at 55%.

### 3.47 On the forecast benchmark, the per-state head beats the history-aware one (§36 item 7, §31 Q4/Q6, §32)

§2.1 of the final report leads with the history-aware head as "a real and
replicated improvement over Run 8's per-state head", on three training regimes
and three seeds. Every number behind that claim is a **head-training validation
AP**. §3.29 and §3.32 both warned that a validation ranking in this phase has
twice failed to survive the forecast benchmark. With the matrix re-scored under
the D145 fix, the two heads can be compared on the benchmark directly, on
identical rows, each at the operating point its own validation split selected.

| regime | split | `state` AP | `state+hidden` AP | `state` ROC | `state+hidden` ROC |
|---|---|---:|---:|---:|---:|
| CIC → CIC | test | **0.1633** | 0.1173 | **0.978** | 0.760 |
| CIC → CIC | holdout | 0.3370 | **0.3869** | **0.989** | 0.839 |
| CTU → CTU | test | 0.1916 | **0.3186** | **0.781** | 0.691 |
| CTU → CTU | holdout | **0.2410** | 0.2273 | **0.955** | 0.720 |
| CIC → CTU | test | **0.0127** | 0.0105 | **0.713** | 0.580 |
| CIC → CTU | holdout | **0.0373** | 0.0159 | **0.921** | 0.758 |
| CTU → CIC | test | **0.1147** | 0.0040 | **0.919** | 0.222 |
| CTU → CIC | holdout | **0.0048** | 0.0003 | **0.713** | 0.133 |
| comb → CIC | test | 0.1759 | **0.1990** | **0.975** | 0.950 |
| comb → CIC | holdout | 0.3229 | **0.3392** | **0.975** | 0.923 |
| comb → CTU | test | 0.1848 | **0.3537** | 0.777 | **0.789** |
| comb → CTU | holdout | **0.2826** | 0.2400 | **0.969** | 0.836 |

| | `state` | `state+hidden` |
|---|---:|---:|
| cells won on AP | **7** of 12 | 5 of 12 |
| cells won on ROC | **11** of 12 | 1 of 12 |
| median ROC | **0.938** | 0.759 |
| median AP | 0.1804 | **0.2132** |
| **beats its own best baseline** | **7** of 12 | **3** of 12 |

**The per-state head — Run 8's architecture, the one this phase set out to
improve on — beats the history-aware head on the forecast benchmark.** Not
marginally: it ranks better in 11 of 12 cells, with a median ROC of 0.938 against
0.759, and it clears its own baselines in more than twice as many cells.

The two metrics disagree in an informative way. `state+hidden` has the higher
median AP, carried by two large wins on CTU targets (+0.127 and +0.169), and the
lower ROC almost everywhere. That is the §3.11 signature again at matrix scale:
the history-aware head concentrates its score on a few hosts it recognises, which
lifts average precision where those hosts are the positives and damages the
global ordering. §3.32 saw the same thing on one run — aggregate AP up, within-host
ROC down.

**`CTU → CIC` is where it is starkest.** `state` reaches ROC 0.919; `state+hidden`
reaches **0.222**, far below chance. The inverted transfer reported in §3.44 is
not a property of the model — it is a property of the *history-aware head*. The
per-state head transfers from CTU to CIC with a usable ordering and merely poor
precision; the history-aware head transfers backwards.

**What this does to item 7.** The architecture call is not "confirm `state+hidden`
across seeds". It is:

- `state+hidden` is better on **head-training validation AP**, replicated across
  three regimes and three seeds (§2.1, §3.29). That result stands.
- `state` is better on **the forecast benchmark at natural prevalence**, on 12
  corrected cells, by ROC and by baseline-clearing (this section).
- These are not in conflict. They are a measurement of how far the validation
  objective the head is selected on has drifted from the thing the phase is
  trying to do.

**Caveat, load-bearing:** one seed per cell. A one-seed benchmark cell cannot
reverse a three-seed headline, and §3.29's headline was three seeds.

**So it is being run.** §3.29's own weights — `ctu_confirm__state` and
`ctu_confirm__state+hidden`, three seeds each off the same `ctu_dyn` checkpoint —
are being scored on the forecast benchmark at test and holdout. That asks §3.29's
question on §3.47's metric with nothing else changed: same data, same dynamics,
same seeds, only the evaluation differs. If `state+hidden` wins there, §2.1 stands
and this section is a one-seed artifact. If `state` wins, the phase's headline
claim is measured on the wrong objective and §2.1 has to be rewritten.

Pre-registered before the run, since the outcome is a headline either way:
**`state+hidden` is confirmed if it wins the mean benchmark AP at both splits and
its seed distribution does not overlap `state`'s at either** — the same bar §3.29
cleared on validation. Anything less is a failure to replicate on the benchmark,
and will be reported as one rather than as a tie.

### 3.48 Stage B on the decomposable head: it wins on every seed, on the metric §3.47 just discredited

§3.42 put `state+logvar` ahead of `state+hidden` on one seed and queued Stage B.
Three seeds, CIC, each variant initialised from the same per-seed `cic_core_dyn`
checkpoint so the encoder and transition are byte-identical within a seed.

| head reads | dim | seed 0 | seed 1 | seed 2 | mean | sd | decomposable? |
|---|---:|---:|---:|---:|---:|---:|---|
| `state+delta+logvar` | 135 | 0.7817 | 0.6890 | 0.7636 | **0.7448** | 0.0492 | **yes** |
| **`state+logvar`** | 90 | 0.7869 | 0.6884 | 0.7542 | **0.7432** | 0.0501 | **yes** |
| `state+hidden` | 173 | 0.7825 | 0.6711 | 0.6777 | 0.7105 | 0.0625 | no |

The unpaired ranges overlap heavily, and the unpaired comparison is the wrong
one: the three variants share a dynamics checkpoint within each seed, so the
seed effect is common and the paired difference is what the design licenses.

| head, paired against `state+hidden` | s0 | s1 | s2 | mean | wins | worst case |
|---|---:|---:|---:|---:|:--:|---:|
| **`state+logvar`** | +0.0043 | +0.0173 | +0.0765 | **+0.0327** | **3/3** | **+0.0043** |
| `state+delta+logvar` | −0.0008 | +0.0178 | +0.0859 | +0.0343 | 2/3 | −0.0008 |

**`state+logvar` beats `state+hidden` on every seed**, with a worst case that is
still positive. A 90-dimensional head built entirely from the 45 named features
beats the 173-dimensional head that reads the opaque recurrent summary, three
times out of three. §3.42's one-seed result replicates.

**And it replicates on head-training validation AP, which §3.47 has just shown is
the wrong objective for choosing a head.** These two sections landed within hours
of each other and they have to be read together:

- §3.29: `state+hidden` beats `state` on validation AP, three seeds, distributions
  not overlapping.
- §3.47: `state` beats `state+hidden` on the forecast benchmark — ROC in 11 of 12
  cells, baseline-clearing 7 of 12 against 3 of 12.
- §3.48: `state+logvar` beats `state+hidden` on validation AP, three seeds, paired
  3/3.

The third result is the same kind of evidence as the first, and the first did not
survive the second. **So §3.48 does not select an architecture.** It establishes
that a decomposable head is *not worse* on the objective heads are trained
against, which was the open question from §3.20's explainability cost — and it
says nothing yet about the benchmark.

What would settle it is `state+logvar` on the forecast benchmark against `state`
and `state+hidden`, three seeds. That needs three head-training seeds for a
variant that currently has them only on CIC, plus twelve benchmark cells. It is
the obvious next run and it is **not** run here; the phase's remaining compute is
committed to re-asking §3.29's own question on the benchmark (§3.47), which is
the prior claim and the one currently in the final report's §2.1.
