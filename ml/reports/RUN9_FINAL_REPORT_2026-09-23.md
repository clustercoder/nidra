# NIDRA — Run 9 final report: CTU-13 integration and model improvement

**Branch** `research/run9-ctu13`. **Baseline** Run 8, frozen at tag `baseline-run8`
(`ad1430c`). No Run 8 artifact, metric or report is modified by anything in this phase.

This is the synthesis. The append-only log it is built from is
`CTU13_MODEL_IMPROVEMENT_2026-09-23.md`; the fourteen roadmap questions are answered
in `RUN9_QUESTIONS.md`; what failed is in `RUN9_NEGATIVE_RESULTS.md`; per-item
status is in `RUN9_DELIVERABLE_STATUS.md`.

> **Status: in progress.** Sections 1–4 are written from results that are on disk
> and will not change: the architecture comparison, the coupling measurement, and
> the evaluation-apparatus corrections. Section 5 lists what is still running and
> names, for each, the conclusion it could overturn. Nothing here is stated more
> firmly than its evidence supports, and anything a pending arm could change is
> marked.

---

## 1. What this phase set out to do

Run 8 ended with a specific diagnosis: the transition model learns useful network
dynamics — state-forecast skill over persistence 0.587 on test, 0.616 on holdout —
while the **per-state risk head** is the ceiling. On Friday it ranked Bot-C2 windows
below silence (ROC 0.37) where a GRU sequence classifier on the identical rows
reached 0.976.

Two questions followed, and neither could be answered inside CIC-IDS2017: whether
that failure is a property of the model or of one dataset, and whether the project
has anything to say about advance warning. CTU-13 was added to answer both, and to
make "generalises to unseen attack families" a cross-dataset claim.

---

## 2. What NIDRA has demonstrated

Each of these is measured, reproducible from a recorded artifact, and does not
depend on an arm still running.

**2.1 The history-aware risk head is a real and replicated improvement over Run 8's
per-state head.** Three training regimes, each with its own frozen encoder and
transition so the head is the only thing that changes (§3.12), then three seeds off
one shared checkpoint (§3.29):

| | validation AP |
|---|---|
| Stage A, CTU-only | 0.3531 → 0.4894 |
| Stage A, CIC-only | 0.6781 → 0.7825 |
| Stage A, combined | 0.3850 → 0.5130 |
| **Stage B, CTU, 3 seeds** | **0.3212 ± 0.0468 → 0.4931 ± 0.0083** |

Stage B's distributions do not overlap: the worst history-aware seed is 0.11 above
the best per-state one. The history-aware head is also **5.6× more stable across
seeds**, which with the transition model held fixed is purely head-training
variance.

**2.2 On the operating point it produces a materially better alert.** CTU
validation, three-member ensembles, identical rows (§3.32): F1 **0.608** against
0.507, false alarms **4.79/h** against 9.26/h, at *higher* recall (0.484 vs 0.410)
and higher precision (0.818 vs 0.663).

**2.3 The transition model and the frozen risk head are barely coupled, and this is
measured rather than inferred.** This is the phase's most important result (§3.33,
§3.34, D144). Over a six-step rollout the encoder hidden state moves **40–62%**
relative — so the persistence ablation genuinely bites — and the head's answer moves
**0.28%** on the positives (composite correlation 0.9910). The head's first layer
responds to the rollout's actual displacement with a gain of 0.0909, and to a
**random** displacement of the same norm with 0.0845: a ratio of **1.08×**. The head
is no more sensitive to the transition model than to noise of equal size.

The reason is structural. The risk head trains on *observed* states and freezes, so
its readout separates observed hidden states and has no mechanism to become
sensitive to where a rollout displaces them. That invariant exists to make the
forecasting claim falsifiable; here it returned a negative.

**2.4 CTU-13 cannot support an advance-warning claim, and the reason is in the
data.** A fifth of CTU's forecast positives sit in the silence floor — state vectors
bit-identical to 47,650 negatives (§3.10). The infected host is silent for minutes
before onset, so no window size recovers a precursor, which killed the Δ=15
direction for the cost of one query (§3.15). CIC's pre-onset windows are non-empty
and carry ROC 0.75 of separable signal against a model at the floor — a small
measured gap, not unlocked advance warning.

**2.5 On CTU, the history-aware head's advantage is substantially host recognition
rather than timing.** The floor-stratum probe found its 58× lift over the state-only
ceiling is entirely host identity: host-mean ROC 0.9993, within-host lift 0.98×
(§3.11). Stage B reproduces the signature on the run that selects the architecture:
aggregate AP up (+0.125) while **within-host ROC falls, 0.816 → 0.697** (§3.32). On
CIC the gain survives the within-host test; on CTU it largely does not.

**2.6 Evaluation-apparatus defects found and fixed, several of which were producing
or hiding wrong numbers.** These are results in their own right — each was a way the
phase could have reported something false.

| defect | symptom | §/D |
|---|---|---|
| The scorecard's FA/h column reported a per-row fraction | every regime looked silent; hid a 208 alarms/hour transfer failure | §3.18 |
| The published system was absent from `BOOTSTRAP_SYSTEMS` | the `AP [95% CI]` column rendered a bare AP in **every** cell | §3.31 |
| The oracle was compared against a differently-read system | "% of oracle" exceeded 1 in 7 of 14 cells | §3.24 |
| A scaler from a different feature regime loads silently | dropped features are zeroed, not removed; AP came out 2.9× off | §3.28, D142 |
| Transfer configs cannot express the head variant | six matrix cells never ran | §3.30, D143 |
| Queues wrote `rc=$?` after a pipeline | three crashes logged `rc=0`; "all 42 exited zero" withdrawn | §3.30 |

**2.7 Serving.** The history-aware head costs nothing measurable to serve: 68.2 ms
against 70.4 ms median, ratio 1.033×, and a decomposition put the two at 71.5 and
71.0 ms — the wider head nominally faster, which is the clearest way to say the
difference is noise (§3.21). Before this phase `NidraPredictor` could not run it at
all: all four of `forecast`, `forecast_batch`, `counterfactual` and `explain` raised
(§3.20, D141).

---

## 3. What remains unproven

**3.1 That the world model forecasts better than not forecasting.** The paired
episode-cluster bootstrap of world model minus persistence, CTU validation, three
seeds: `state` **+0.0388 [−0.0104, +0.0923]**, `state+hidden` **−0.0100 [−0.0342,
+0.0034]**. Both contain zero; the selected head's point estimate is negative.
Persistence removes the transition step while the encoder still advances, so the
entire +0.125 AP gain is head-side (§3.32). §2.3 explains why.

**3.2 That the world model beats a strong sequence classifier.** A GRU classifier on
identical rows reaches **0.5093** against `state+hidden`'s 0.4910 and `state`'s
0.3660. The absolute CIs on that split are worthless — eight positive episode
clusters, intervals spanning [0.001, 0.84] — so this is not a *measured* gap. What is
measured is that the world model does not clear it. The GRU cannot roll a state
forward, produce a horizon or support a counterfactual, so a tie on AP is not a tie
on the claim; that distinction is the project's to make honestly, not to hide behind.

**3.3 Genuine advance warning, on either corpus.** See §2.4. CTU cannot support it;
CIC's measured headroom is modest and mostly bare activity.

**3.4 That the history-aware head is the right *deployment* choice.** It wins AP
(§2.1) and the alert (§2.2) and reproducibility, and loses on within-host ordering
(§2.5), calibration — the only within-dataset arm whose calibrated score loses to
predicting the prevalence and never moving (§3.23) — and explainability: SHAP over
all 45 named features attributes a served risk of 1.00 to nothing larger than 0.0003,
because the origin state is not what moved the score (§3.20). Under §32's ranking
this is not an obvious call.

**3.5 Causality.** Unchanged and not attempted. The counterfactual endpoint returns
and displays "model-internal what-if".

---

## 4. What the phase learned about how to run the phase

Three of the six killed interventions were killed by a measurement costing minutes
rather than by building the thing. The two apparatus defects that mattered most
(D142, D143) were both found by **cross-checking a number against an independently
recorded one**, not by a test — and the tests that now exist would not have caught
either, because neither was a logic error in a function.

Two pre-registered criteria were written before their measurements. One (§3.13) is
still queued. The other (§3.26) **refused its own author's explanation**: the
attractor account required a CIC-only model not to amplify class separation, and it
amplifies at 1.73×. §3.24's two anomalous cells are recorded as unexplained rather
than kept alive on a partial match.

---

## 5. Still running, and what each could change

| in flight | could overturn |
|---|---|
| Six recovered `state+hidden` transfer cells | §2.5 and §3.2 — the cross-dataset half of the head comparison is currently CIC/CTU-within-dataset only |
| `lofo_without_neris` (unseen attack family) | §2.1's generality; this is the §31 Q9 evidence and nothing here substitutes for it |
| The pre-registered stage-balanced risk objective (§3.13) | §2.5 — if it lifts the below-chance stages, the pooled objective was the explanation |
| CIC decomposable-head arms (`state+logvar`, `state+delta`) | §3.4's explainability cost, if a decomposable head keeps most of the gain |
| Fixed-pooling cells | Q2/Q3 under a common readout |
| Re-run of the 20 recorded cells into `repro_*` | everything in §2 that quotes the matrix, if the numbers do not reproduce |

The last row is the one to watch: it is simultaneously the missing confidence
intervals and a reproducibility check on every published number in the matrix, and
it is written to a separate directory precisely so a disagreement is visible rather
than overwritten.
