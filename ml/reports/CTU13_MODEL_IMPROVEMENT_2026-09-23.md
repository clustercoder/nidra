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

* the **only** variant whose margin over `persistence_rollout` excludes zero
  (+0.033 [+0.0002, +0.055]). That is the strict ablation: the same head, the
  same encoder advance, only the predicted change removed. It is the first
  interval in this project that isolates the transition model's own
  contribution to risk and does not contain zero. It just barely excludes it,
  on one seed, and is reported as a screening result until §3.6 confirms it;
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
for "does adding CTU help the dynamics" — the latter needs the pooling held
fixed, which is a separate, cheaper run and is not in this matrix.

This is also the mechanism behind §3.18's 208 false alarms an hour: a threshold
of 0.501969 chosen against CIC's score distribution means something entirely
different against CTU's.
