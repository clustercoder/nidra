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
group. Fitting on captures 1–3 and scoring capture 4 makes it cross-host and
cross-capture by construction, so it cannot answer with host identity — it asks
exactly what the model is asked. It reads labels the model does not, so it is an
upper bound in the same sense the oracle is: a diagnostic, never a system, and
nothing in the pipeline reads it.

| group | positives | positive hosts | probe AP | probe ROC | world model ROC | verdict |
|---|---|---|---|---|---|---|
| ctu_6:exfil (Menti) | 122 | 1 | 0.814 | 0.992 | 1.000 | model at the ceiling |
| ctu_4:exfil (Rbot) | 50 | 1 | 0.091 | 0.829 | 0.704 | partial gap |
| ctu_4:recon (Rbot) | 17 | 1 | 0.033 | **0.970** | ~0.42 | **model failure** |
| ctu_4:c2 (Rbot) | 23 | 1 | 0.008 | **0.443** | ~0.47 | **not separable at all** |

The two Rbot failures that looked identical in the benchmark have opposite causes.

**Rbot C2 is not there to be found.** A probe handed the C2 label from the
training captures ranks capture 4's C2 windows *below* chance (ROC 0.443). Rbot's
command-and-control on this host does not resemble Rbot's command-and-control on
the training hosts in 32 flow features at Δ=60 s. No head architecture fixes
that; it is a representation limit, and the honest options are a finer Δ, features
the flow record does not currently carry, or accepting it.

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
