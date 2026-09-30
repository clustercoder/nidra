# NIDRA — Run 9 final report: CTU-13 integration and model improvement

**Branch** `research/run9-ctu13`. **Baseline** Run 8, frozen at tag `baseline-run8`
(`ad1430c`). No Run 8 artifact, metric or report is modified by anything in this phase.

This is the synthesis. The append-only log it is built from is
`CTU13_MODEL_IMPROVEMENT_2026-09-23.md`; the fourteen roadmap questions are answered
in `RUN9_QUESTIONS.md`; what failed is in `RUN9_NEGATIVE_RESULTS.md`; per-item
status is in `RUN9_DELIVERABLE_STATUS.md`.

> **Status: final for the phase, 2026-09-24.** Nothing is running. Every
> cross-dataset number is on D145-corrected artifacts, every final metric carries
> an interval, and the two questions that stayed open — advance warning and the
> oracle gap — are recorded as open rather than closed with a story. Section 5 is
> what landed after the core was drafted and what each landing changed.

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

Each of these is measured, reproducible from a recorded artifact, and scored on
D145-corrected weights. Nothing here is waiting on an arm.

**2.1 The history-aware risk head is a real and replicated improvement over Run 8's
per-state head — on the objective heads are trained against, and not on the one the
system is evaluated against.** This claim was the phase's headline and it has been
narrowed by its own evidence; §2.1b states what happened. Three training regimes, each
with its own frozen encoder and transition so the head is the only thing that changes
(§3.12), then three seeds off one shared checkpoint (§3.29):

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

**2.1b And it does not survive the forecast benchmark.** §3.29's own three-seed
weights, scored on the benchmark instead of on head-training validation — same data,
same dynamics, same seeds, only the evaluation differs — against a bar pre-registered
before the run (win mean AP at both splits, no seed-distribution overlap at either,
the same standard §3.29 cleared on validation):

| split | `state` mean AP | `state+hidden` mean AP | paired | `state` ROC | `state+hidden` ROC |
|---|---:|---:|:--:|---:|---:|
| test | 0.1937 | **0.3624** | 3/3, no overlap | **0.797** | 0.693 |
| holdout | **0.2273** | 0.2175 | 1/3, overlapping | **0.912** | **0.587** |

**The bar is not met** — one split won decisively, the other lost narrowly — and this
is recorded as a failure to replicate rather than as a tie, as pre-registered. On ROC
the per-state head wins both splits, and on holdout it wins by 0.33: two systems with
almost identical average precision, one of which ranks near chance. Across the wider
one-seed matrix the same pattern holds — `state` ahead on ROC in **11 of 12** cells and
clearing its own baselines in **7 of 12** against `state+hidden`'s 3 of 12 (§3.47,
§3.49).

So 2.1 is true as written and is a statement about head-training validation AP. The
architecture the evidence supports under §32's ranking is **`state`, the Run 8
architecture this phase set out to improve on.**

**2.2 On the operating point it produces a materially better alert.** CTU
validation, three-member ensembles, identical rows (§3.32): F1 **0.608** against
0.507, false alarms **4.79/h** against 9.26/h, at *higher* recall (0.484 vs 0.410)
and higher precision (0.818 vs 0.663).

**2.3 The transition model and the frozen risk head are barely coupled, and this is
measured rather than inferred.** This is the phase's most important result (§3.33,
§3.34, §3.46, D144). Numbers below are re-measured under the D145 fix, because the
original was taken on a rollout manufacturing state in 15 of 45 slots. Over a
six-step rollout the encoder hidden state moves **39–54%** relative — so the
persistence ablation genuinely bites — and the head's answer moves **0.36%** on the
positives (composite correlation 0.9817). The head's first layer responds to the
rollout's actual displacement with a gain of 0.0962, and to a **random**
displacement of the same norm with 0.0835: a ratio of **1.15×**. The head is no
more sensitive to the transition model than to noise of equal size.

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
| Queues wrote `rc=$?` after a pipeline | three crashes logged `rc=0`; "all 42 exited zero" withdrawn — **and it recurred** in a script written after this was documented | §3.30, NR 17 |
| The rollout manufactured state in features carrying no gradient | phantom rms 2.03 by k=6 against real features at 0.69, reaching the frozen head and head *training*; every CTU and cross-dataset number was produced under it | §3.37, D145 |
| A fix wired per call site instead of at the constructor | the re-run returned a bit-identical number and read as "unaffected by D145" | §3.37, NR 15 |

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

## 5. Landed since the core was drafted

Every row of the previous version of this section has now landed. What they changed:

| result | what it did to the report |
|---|---|
| Six recovered `state+hidden` transfer cells | completed the cross-dataset half of the head comparison (§3.30) |
| `lofo_without_neris` (unseen attack family) | **Q9 answered.** AP 0.717 [0.485, 0.860] on a withheld family, within-host ROC 0.949 across ten infected hosts — and persistence reaches 0.772. The model transfers, and not better than persistence does (§3.39) |
| Pre-registered stage-balanced risk objective | criterion 1 **failed**; the pooled objective is not the explanation, and §12 is refuted (§3.35, §3.27) |
| CIC decomposable-head arms | **reversed the CTU verdict, then went further.** Re-trained under the D145 fix, `state+logvar` (90 dims, all from the 45 named features) reaches **0.7869 against `state+hidden`'s 0.7825** — ahead, not merely close — with every non-`logvar` control bit-identical (§3.38, §3.42). **Stage B then confirmed it at three seeds, 3/3 paired** (§3.48). It has no forecast-benchmark cell, which after §3.49 is the evaluation that matters and the one that reversed the other head's validation win |
| Fixed-pooling cells | Q2 supported as a direction, Q3 not supported; persistence is level with the model on both CTU rows (§3.40) |
| Re-run of the recorded cells into `repro_*` | **all 28 reproduce bit-identically**, and CI coverage went 6/28 → 28/28 (§3.36) |
| D145 re-score of all 28 cells into `mask_*` | **21 up, 7 down**, no published direction flipped, and the oracle anomaly got *worse* — 10 of 28 cells beating their oracle became 17 of 28 (§3.44) |
| §3.29's three-seed weights on the forecast benchmark | **the phase's headline claim did not replicate.** The pre-registered both-splits bar is not met, and the per-state head wins ROC on both splits. §36 item 7 is answered `state` (§3.49, D146) |
| Paired intervals on the published arm | **one cell of 28 has a supported margin with room to spare** — `CIC+CTU → CIC`, `state+hidden`, test, +0.0660 [+0.0235, +0.1296]. Three more clear zero by ~1e-05 and are flagged, three are negative, 22 span it (§3.50) |

## 6. The defect found on the last day

While investigating the oracle anomaly, the rollout was found to be manufacturing
state in the 13–15 feature slots that the transition loss masks — slots where the
network receives no gradient, so its output there is untrained, compounding to rms
2.03 by k=6 against real features at 0.69. It reached the frozen head, whose
weights on those slots were never constrained by data either. It was also present
in head *training*, through `head_context`'s unmasked `logvar`.

Three things follow, and they pull in different directions:

1. **Every CTU and cross-dataset number in this phase was provisional, and has
   since been re-scored.** All 28 cells were re-run under the fix into `mask_*`:
   21 up, 7 down, no published direction flipped (§3.44). Run 8 is unaffected —
   its regime drops nothing — and its artifacts are untouched.
2. **The correction does not go one way.** Raw AP by cell: comb → CIC test
   **+0.055** (0.147 → 0.202, ROC 0.913 → 0.967), CTU → CTU test +0.005, comb →
   CTU test **−0.015**. The first cell alone would have supported "the defect was
   costing accuracy"; the third refutes it. The phantom helped some cells and hurt
   others, which is what an untrained signal projected through untrained weights
   should do, and it is why the whole matrix is being re-scored rather than the
   correction being estimated from a sample. The oracle and persistence arms are
   unchanged in every cell, as they must be — their states never carried it.
3. **It explains neither of the two anomalies it looked like it would.** A test
   was pre-registered before the re-scores ran, and it came back negative: the
   model beats its oracle by *more* after masking (+0.150 against +0.095), and
   the deterministic arm is still below chance. §3.24's cells remain unexplained.

What the correction did surface is a sharper version of the anomaly. On
`comb2cic/test` the oracle ranks at ROC 0.367 and persistence at 0.194 — both
below chance — while the sampled rollout reaches 0.967. A frozen head trained on
combined data appears anti-correlated on CIC's true state distribution and
strongly correlated on states its own transition model produced. If that survives
the full re-score it is the most interesting open question in the phase, and it is
a question about what the rollout does to a state, not about accuracy.

## 7. What the evidence says about where to go next

Two results in this phase point the same way, and it is not the way the roadmap's
§14 assumed.

**§3.34:** the frozen head's answer moves 0.28% when the rollout moves the hidden
state by 40–62%, and its first-layer response to that displacement is 1.08× its
response to a random displacement of the same size. The head is barely coupled to
the transition model.

**§3.42:** a 90-dimensional head built from named features matches — on one seed,
slightly beats — a 173-dimensional one that reads the full recurrent summary.

Together these say the binding constraint is **not head capacity**. §14 proposed
a transformer or TCN once the simpler history-aware heads had been tested; they
have now been tested, and the result is that the simplest competitive head is
already competitive. Spending the next phase on a larger sequence model would be
adding capacity to the component that is not short of it.

What both results point at instead is the **coupling** between a frozen head and
a rollout it never saw. That coupling is weak by construction: invariant 1 freezes
the heads on observed states precisely so that any forward-looking capability has
to originate in the transition model, which is what makes the forecasting claim
falsifiable. §3.34 measured the cost of that design and it is most of the gap.

This is the uncomfortable shape of the finding. The invariant that makes the
claim testable is the same invariant that caps it, and the honest options are
narrow: keep the invariant and accept the cap, or change the architecture so the
head reads something the rollout genuinely moves — without unfreezing it on
predicted states, which would close the gap and simultaneously remove the reason
the result means anything.

## 8. The honest summary

NIDRA's transition model learns network dynamics that are real and measurable, and
this phase added a second corpus, a cross-dataset matrix, a history-aware head, an
unseen-family test, a defect fix and a reproducibility guarantee to the evidence for
that. Every recorded number reproduces bit-identically, every final metric has an
interval, and the phase found and reported six ways it could have published something
false.

**It did not turn those dynamics into attack forecasting that beats persistence.**
That is the finding, and it held under every way the phase found to ask it:

- On the **unseen attack family**, 0.717 against persistence's 0.772 (§3.39).
- On both **CTU rows** of the fixed-readout comparison, 0.322 against 0.322 and
  0.373 against 0.362 (§3.40).
- Across the **corrected cross-dataset scorecard**, the model beats its best baseline
  in **3 of 12 cells**, and three of the four CTU-target losses are to persistence
  plus noise scaled to the model's own predicted variance (§3.44).
- And the **head that was supposed to fix it** wins the objective it is trained on and
  loses the one it is evaluated on (§3.47, §3.49).

The mechanism is measured, not inferred. The frozen head's answer moves **0.36%** when
the rollout moves the hidden state **39–54%**, and its first layer responds to that
displacement **1.15×** as strongly as to noise of equal size (§3.46). The invariant
that makes the forecasting claim falsifiable — heads train on observed states and
freeze — is the same invariant that caps it. The phase returned the negative rather
than removing the invariant, which is the only choice under which the result means
anything.

**One cell is genuinely good and should not be lost in the summary.** `CIC+CTU → CIC`
on test reaches AP 0.199 against a best baseline of 0.107, at ROC 0.950, natural
prevalence 0.00417. It is also **the only cell in the matrix whose margin over its
strongest baseline survives a paired episode-cluster bootstrap with room to spare** —
+0.0660 [+0.0235, +0.1296] over persistence-plus-noise, across 1,742 clusters, on the
arm the project actually publishes (§3.50). Three further cells clear zero by about
1e-05 and are flagged rather than claimed; three are negative; 22 span zero. Training on
both corpora and evaluating on CIC is where this architecture works. It does not
transfer to the other five regimes, and §32's ranking puts cross-dataset generalisation
above a single aggregate, so this is a finding about that cell rather than a system to
ship.

**What is still unexplained, after four attempts and three pre-registered criteria:**
the model beats its own oracle in 17 of 28 cells, and the frozen head ranks *better*
on states measurably *further* from its training distribution — median ratio 1.42, in
every cell measured (§3.45). No fifth hypothesis is offered here without a measurement
behind it.

**What the next phase should not be** is a larger sequence model. §14 proposed one
after the simpler history-aware heads were tested; they have been, and a 90-dimensional
head built from named features matches or beats the 173-dimensional one that reads the
full recurrent summary (§3.48). The constraint is not head capacity. It is the coupling
between a frozen head and a rollout it never saw, and that is a question about the
architecture's shape rather than its size.
