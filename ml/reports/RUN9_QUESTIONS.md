# Run 9 — the fourteen questions, answered as the evidence arrives

Living document. Every answer cites the section of
`CTU13_MODEL_IMPROVEMENT_2026-09-23.md` and the decision record it rests on, and
carries a status:

- **answered** — measured, with the artifact committed.
- **partial** — measured on validation only, or on one seed, or under a caveat
  that a later experiment could lift.
- **pending** — no evidence yet; what will settle it is named.

**Final for the phase, 2026-09-24.** Nothing is running. Every cross-dataset cell
is scored on D145-corrected artifacts, and where an answer cites validation rather
than test or holdout it says so, because on this project the two have disagreed.

---

## Q1 — Does CTU-13 contain enough temporal precursor information to make genuine advance warning feasible?

**Answered: no.** Two independent measurements say so. The infected host has
emitted nothing for a median of 4.5 minutes (validation) or 47 minutes (train)
before an attack begins, so no finer window recovers a precursor — the Δ=15
idea is closed by measurement. And a probe fit on that host's own training rows
and scored on its validation rows returns AP equal to the base rate and ROC of
exactly **0.5000**: a constant, which is all any function can produce when every
row it scores is the same vector. *(§3.15.)*

CIC is not the same and must not be answered with it: in 83% of CIC validation
onsets the host is active in the window immediately before the attack, and the
same probe reaches **ROC 0.7500** there against a model at the floor. Most of
that is `is_active` alone (0.6783), so the genuine precursor content is the gap
between the two — real, and modest.

The detail behind the CTU answer: Every pre-onset positive on CTU is a
silent window — 71 of 71 on train, 84 of 84 on validation at `is_active == 0` —
and most are at the exact silence floor, bit-identical to 1,026,383 and 47,650
negatives respectively. Inside that stratum the best AP any function of the
state can reach is the stratum's own prevalence, 0.001216 on validation, and
the published state-only head emits exactly one distinct score across all
47,708 rows and lands on it. Twenty percent of CTU validation's positives are
in that stratum. History is the only thing that can distinguish them, and §3.10
shows what history actually does with them: all 58 floor positives sit on one
host, the host-mean collapse reproduces the ranking at ROC 0.9993, and within
that host the head scores 0.98× lift — worse than a constant.
*(§3.10, D129.)*

CIC has the same structure more weakly: its pre-onset windows are silent 24 of
45 times on train and 9 of 22 on validation, so this is not purely a CTU
artifact, but CTU's advance-warning signal specifically is a silence
phenomenon.

## Q2 — Does adding CTU-13 improve CIC-IDS2017 generalization?

**Yes as a direction, not as a quantity.** At a forced common readout
(`mean|q=-|max`), on identical rows (n=21,173, 946 positive, prevalence 0.00417).
The table below is the pre-fix measurement; §3.44 re-scored it under D145's fix
and the gap narrowed from +0.0942 to **+0.0817** without changing sign:

| training set | AP [95% CI] | ROC | uncalibrated AP | uncalibrated ROC |
|---|---|---:|---:|---:|
| CIC only | 0.0672 [0.021, 0.239] | 0.575 | 0.0380 | 0.438 |
| CIC + CTU | 0.1584 [0.052, 0.283] | 0.889 | 0.1470 | 0.913 |

The uncalibrated column matters here: the CIC-only arm is the one cell in the
matrix whose Platt layer was fitted under a different pooling key than it is
scored at, so the calibrated comparison is asymmetric in the improvement's
favour. Removing the Platt layer entirely leaves the gain intact (ROC 0.44 →
0.91), so it is not a calibration artifact — but the intervals overlap across
most of their range on **15 positive episode clusters**, so the size of the
effect is not established. *(§3.40.)*

## Q3 — Does training on CIC + CTU improve CTU performance?

**Not supported.** Same discipline, identical rows (n=25,099, 3,688 positive,
prevalence 0.00749):

| training set | AP [95% CI] | ROC | persistence |
|---|---|---:|---:|
| CTU only | 0.3221 [0.125, 0.583] | 0.680 | 0.3215 |
| CIC + CTU | 0.3734 [0.204, 0.586] | 0.795 | 0.3623 |

+0.051 calibrated, +0.049 uncalibrated, on intervals that overlap almost
entirely. The more important column is the last one: **persistence is level with
the model in both rows** — 0.3215 against 0.3221, and 0.3623 against 0.3734.
Whatever the training mixture does on CTU, it does not lift the model past
repeating the host's current state. *(§3.40.)*

Head-training validation AP is available and is *not* an answer: the combined
regime's 0.3850 / 0.5130 and CTU-only's 0.3531 / 0.4894 are computed on
different validation sets with different prevalences, so they are not
comparable. *(§3.12, D131.)*

**Both answers were re-measured under D145's fix and both survived.** Every cell
above was first scored while the rollout was writing untrained values into 13 of 45
feature slots. All 28 were re-run (§3.44): Q2's gap went +0.0942 → **+0.0817** and
Q3's +0.0513 → **+0.0351**, each keeping its sign and each getting smaller. Q2 is
still a direction rather than a quantity; Q3 is still unsupported.

## Q4 — Does the history-aware risk head fix the Run 8 failure?

**Answered: no.** It fixes the objective heads are trained against and not the one
the system is evaluated against. The paragraphs below are the phase's reasoning in
the order it was measured; the last one is the answer.

The aggregate says yes three times over: `state` → `state+hidden` moves
validation AP 0.3531 → 0.4894 (CTU), 0.6781 → 0.7825 (CIC), 0.3850 → 0.5130
(combined), each with its own frozen encoder and transition so the head is the
only thing that changes. *(§3.12, D131.)*

The decomposition disagrees across datasets. On CTU both heads score *below* a
host-level constant (host-mean AP 0.7579 against 0.3531 and 0.4894) and reach
within-host ROC 0.6559 and 0.6621. On CIC both beat it and the history-aware
head reaches within-host ROC 0.9245 against 0.8256 — the gain is reproduced
inside the host, where identity is constant and only timing is left.
*(§3.11, D130.)*

**Stage B, three seeds on the forecast benchmark, does not settle it either way
and adds a third reading.** The head-training gain replicates cleanly (+0.172,
no overlap, 5.6× more stable — §3.29) and the operating point improves
decisively (§3.32). But on that same run the within-host ROC *falls*, 0.816 →
0.697, and the world model's margin over persistence goes to **−0.0100
[−0.0342, +0.0034]** — so whatever the history-aware head fixed, it did not fix
it by forecasting. The transition step contributes nothing measurable for it.

Run 8's failure was a per-state head that could not read what a GRU could. The
history-aware head reads it. What it does with it, on CTU, is recognise the host
better rather than the moment — and a GRU classifier on the identical rows still
scores higher than either (Q7).

Two limits on that. Both figures are validation, where each corpus has exactly
one infected host, so neither can speak to the ten-host test splits. And on CIC
that one host is `172.16.0.1`, the external attack machine, not a compromised
internal workstation. *(§3.14, D133.)*

**And the forecast benchmark settles it, against the head.** §3.47 pre-registered
the bar — win mean benchmark AP at both splits with no seed-distribution overlap at
either, the standard the validation claim had already cleared — and §3.49 ran it on
§3.29's own three-seed weights. `state+hidden` wins CTU test 3/3 with no overlap
(0.3624 against 0.1937) and loses CTU holdout 1/3 with overlap (0.2175 against
0.2273). The bar required both. On ROC the per-state head wins both splits, on
holdout **0.912 against 0.587** — near chance, at almost the same AP. Across the
one-seed matrix `state` leads on ROC in 11 of 12 cells and clears its own baselines
in 7 of 12 against 3 of 12.

So Run 8's failure is not fixed. The history-aware head reads what the per-state head
could not, wins the objective it is trained on, produces a better alert — and does not
rank hosts better on the evaluation that decides the project's claim. §36 item 7 is
answered `state` on that basis, and D146 records why. *(§3.47, §3.49, D146.)*

## Q5 — Does uncertainty improve attack-risk forecasting?

**Answered: no, not as a head input.** The `logvar` component was included in
the head-component ablation and did not earn its place. *(§3.4.)* The related
`context_noise` intervention — built to fix the diagnosed exposure mismatch
between observed and rollout hidden states — closed the gap and *erased* the
transition model's measured contribution, and was recorded as a negative result
rather than adopted. *(§3.6, D126.)*

## Q6 — Does trajectory-aware risk outperform state-only risk?

**Answered: on the training objective yes, on the forecast benchmark no** — the same
answer as Q4, since this is the same experiment read a different way, and it now has
the same three-seed benchmark behind it (§3.49). Aggregate validation: yes, in all
three regimes. Within-host: yes clearly
on CIC, marginally on CTU. On the rows where the state is provably
uninformative, the history-aware head reaches 58× the state-only ceiling and
every unit of that is host recognition. *(§3.10, §3.11, §3.12.)*

**Stage B sharpens this into a specific trade, measured on one split.** On the
forecast benchmark at three seeds, CTU validation, the history-aware head
delivers the better *alert* — F1 0.608 against 0.507, false alarms halved from
9.26/h to 4.79/h at higher recall and precision — and the worse *ordering*: ROC
0.748 against 0.775, and within-host ROC **0.697 against 0.816**. The Run 8
per-state head answers the timing question better on the one host validation
has. *(§3.32.)*

So "outperform" depends on the axis, and the two axes disagree in a way that is
not noise: aggregate up, within-host down is the host-identity signature, and it
now appears on the run that selects the architecture rather than on a probe.

**§3.49 extends the trade to the held-out split and keeps its shape.** Benchmark ROC
0.912 for `state` against 0.587 for `state+hidden` on holdout, at AP 0.2273 against
0.2175 — the ordering collapses while the aggregate does not move. Two systems with
the same average precision and a 0.33 gap in ranking quality is the sharpest form of
this signature the phase produced, and it is why the architecture call goes to the
per-state head.

## Q7 — Does the world model outperform a strong GRU sequence classifier?

**No, on the one split where the confirmed architecture has been scored.** On
CTU validation, three-member ensembles, identical rows: the GRU sequence
classifier reaches AP **0.5093** against `state+hidden`'s **0.4910** and
`state`'s **0.3660**. *(§3.32.)*

Two caveats, neither of which rescues the world model. The absolute CIs on this
split are worthless — eight positive episode clusters, intervals spanning
[0.001, 0.84] — so 0.509 against 0.491 is not a *measured* difference; what is
measured is that the world model does not clear it. And the GRU is a classifier,
not a forecaster: it answers "is this host compromised" and cannot roll a state
forward, produce a horizon or support a counterfactual, so a tie on AP is not a
tie on what the project claims. That distinction is the project's to make
honestly, not to hide behind: on the metric both systems are scored by, the
simpler model is not behind.

The matrix will add the CIC and transfer arms. Answer stands as **no** until one
of them shows otherwise.

## Q8 — Does the world model provide additional value over a history classifier?

**Partial.** The strict form of this question is the `persistence_rollout`
ablation — the same forward simulation with only the next state swapped, which
isolates the transition model's own contribution. On CTU validation it gave
+0.0331 AP [+0.000203, +0.0551], the first interval in the project excluding
zero. *(§3.5.)* The first cross-dataset benchmark to land (`cic2cic_state`, val)
puts the world model at AP 0.7911 against persistence at 0.7715, and reproduces
that ordering within-host (0.8499 against 0.8162), which is the first evidence
the margin is not host identity. One seed, one split.

**Stage B contradicts that on its own split, for the selected head.** The paired
episode-cluster bootstrap of world model minus persistence on CTU validation,
three-member ensembles: `state` **+0.0388 [−0.0104, +0.0923]**, `state+hidden`
**−0.0100 [−0.0342, +0.0034]**. Both contain zero and the history-aware head's
point estimate is negative. Persistence removes the transition step while the
encoder still advances, so the history-aware head keeps its history and only the
predicted change is taken away — which means the entire +0.125 AP gain over
`state` is head-side, not forecast-side. *(§3.32.)*

The §3.5 result was `persistence_rollout` on the per-state head at one seed;
this is the same comparison at three seeds on the head the phase now prefers.
They do not conflict so much as answer about different heads, and the honest
composite is: **the transition model's contribution is not distinguishable from
zero for either head at three seeds, and is a positive point estimate only for
the head that loses on every other axis.**

## Q9 — Does the model generalize to completely unseen attack families/scenarios?

**Answered: it transfers, and it transfers no better than persistence does.**

Neris — CTU's largest family, scenarios 1, 2 and 9 — was withheld from training
entirely and the model retrained without it. On the withheld family the
history-aware head reaches **AP 0.717 [0.485, 0.860]**, ROC 0.950, and a
within-host ROC of **0.949 across ten infected hosts**: the first genuinely
cross-host within-host measurement in the phase, and a good one. **Persistence
on the same rows reaches 0.772.** The intervals are wide and overlapping, so
neither is established as better — but the model does not beat repeating the
host's current state, and that is the answer to Q9 as asked.

The state-only head shows the failure mode in its clearest form: within-host ROC
0.943, aggregate AP 0.044. It ranks windows correctly inside each host and cannot
compare across them.

Two caveats that are not decoration. The operating point was frozen on a
validation split where the model is at chance (ROC 0.505 / 0.660), so the F1 and
false-alarm columns of that experiment describe nothing; AP is the only usable
metric there. And the GRU baseline is **missing** from these cells — the
classifier was not copied into the lofo runs — which is the one experiment where
the strongest baseline would have mattered most. *(§3.39.)*

Still outstanding on this question: the two transfer arms `cic2ctu` and
`ctu2cic`, both of which must be re-run under D145's fix.

What has been established is where the question *can* be asked. CTU-13 reuses
the same infected address (`147.32.84.165`) across scenarios 1–4 and 6, and
CIC's train and validation share `172.16.0.1`, so **no validation number in
this phase is cross-host** — validation tests unseen captures and unseen
families on a machine the model has already seen infected. Both test splits
contain nine hosts never infected during training; both holdouts contain one.
Those are the only genuinely cross-host evaluations available. *(§3.17, D136.)*

## Q10 — Does it produce genuine advance warning?

**Answered for CTU: no, and the reason is in the data. On CIC: not yet, and
there is a measured ceiling on how much it could.** CIC's pre-onset windows are
non-empty and weakly separable (probe ROC 0.75, of which `is_active` alone
gives 0.68) while the model sits at the prevalence floor — a quantified,
actionable gap, and a small one. *(§3.15.)*

For CTU, see Q1. The pre-onset
rows carry no state information and the history-aware head answers them with
the infected host's identity rather than the moment. *(§3.10, D129.)* Run 8
reported Task B near the prevalence floor on CIC and attributed it to a scarcity
of same-host precursors; CTU *has* precursors and they are empty. Two datasets,
two mechanisms, the same conclusion — and this corpus family cannot support a
strong advance-warning claim.

## Q11 — What is the best configuration under natural prevalence?

**`CIC+CTU → CIC` with the `state+hidden` head — and the honest form of the
answer is that it is the only one of twelve cells where the model clearly beats
its baselines.**

On the corrected artifacts, that cell reaches AP **0.199** against a best
baseline of 0.107 and a persistence of 0.034, at ROC 0.950, on 946 positives in
21,173 rows at prevalence 0.00417. Nearly double the strongest baseline, and the
best result the phase produced.

**Across the whole scorecard the model beats its best baseline in 3 of 12
cells** (§3.44). The other two wins are narrow enough — 0.1173 vs 0.0981 and
0.3869 vs 0.3815 — that they would not be worth claiming alone. All four
CTU-target cells lose, three of them to `noised_persistence`.

So "best configuration" here names a cell, not a system. A configuration that
wins one cell of twelve is a finding about that cell, and §32's ranking is
explicit that cross-dataset generalisation outranks a single aggregate — by that
ranking this configuration does not win, because it does not transfer. *(§3.44.)*

## Q12 — Which improvements are statistically supported rather than noise?

**Answered: one, and it is not the one this phase spent its time on.**

Every cell now carries a **paired** episode-cluster interval on the margin
between the *published* arm and the strongest baseline in that cell — not the
uncalibrated `world_model` the benchmark's own attribution block pairs, and not
a marginal interval on each system separately (§3.50). Across 28 cells:

| | cells |
|---|---:|
| margin interval entirely **above** zero | 3 |
| margin interval entirely **below** zero | 3 |
| interval spans zero | 22 |

Five of the six counted cells clear zero by less than 0.001 and three by about
1e-05, which at 300 resamples is below the resolution of the order statistic that
produced it. They are counted — the rule was fixed before the data was read — and
flagged.

**One cell has a supported margin with room to spare:** `CIC+CTU → CIC`,
`state+hidden`, test. AP **0.1654** against `noised_persistence`'s 0.0994, margin
**+0.0660 [+0.0235, +0.1296]** over 1,742 episode clusters at natural prevalence
0.00417. It is the same cell §3.44 singled out on point estimates, and it is the
only one in the matrix that survives a paired interval.

The previous answer here — +0.0331 [+0.000203, +0.0551] over `persistence_rollout`
(§3.5) — still stands as measured but is superseded as *the* answer: it is CTU
**validation**, which is where the operating point is chosen, and it is the
uncalibrated arm.

Everything else in the phase is a point estimate, a one-seed cell, or an interval
that includes zero. The head ablation's three-regime replication (§3.12) and
§3.29's three seeds are consistency evidence, not confidence intervals, and
§3.49 is the case where consistency on one objective did not carry to the other.

## Q13 — What is still limiting NIDRA?

**Answered, and the answer changed during the phase.** Three limits, in the
order they bind:

1. **The single-infected-host structure of both corpora.** Training and
   validation have exactly one infected host each — not one per attack group,
   one per split. No architecture, objective or curriculum touches that.
   *(§3.14, D133.)*
2. **The risk head's pooled objective.** It ranks recon and c2 *below chance*
   (0.320 and 0.434) where the frozen stage head, trained on the same states
   with class weights, ranks them at 0.638 and 0.854 — so the signal is inside
   the model and the published composite does not use it. No scalar fusion of
   the two recovers it, because one pooled ranking cannot hold both orderings.
   *(§3.9, D128.)* The stage-balanced objective is pre-registered and queued.
   *(§3.13, D132.)*
3. **The label's pre-onset windows are empty on CTU.** See Q1 and Q10.
4. **The frozen head and the transition model are barely coupled — measured, not
   inferred.** The rollout displaces the encoder hidden state by 40–62% over six
   steps, and the head's answer moves 0.28% on the positives (composite
   correlation 0.9910). Its first layer responds to that displacement with a
   gain of 0.0909 against 0.0845 for a *random* displacement of the same norm:
   **1.08×**. The head is no more sensitive to the transition model than to
   noise.

   This is structural rather than a tuning failure. The risk head trains on
   observed states and freezes, so its readout separates observed hidden states
   and has no mechanism to become sensitive to where a rollout puts them.
   Invariant 1 exists to make the forecasting claim falsifiable; on this split it
   returned a negative, which is the invariant doing its job. Unfreezing the head
   on predicted states would close the gap and destroy the reason the result
   would mean anything. *(§3.33, §3.34, D144.)*

   This is probably the binding limit on the *model*, where limit 1 is the
   binding limit on what can be *measured*. They are different kinds of
   constraint and neither substitutes for the other.
5. **The winning head is the least explainable one.** The `state+hidden` head's
   score is set by the encoder hidden state, not the origin state: behind a
   byte-identical origin state, swapping in a busier history moves the served
   risk from 0.0000128 to 1.0000000. KernelSHAP over all 45 named features then
   attributes that 1.00 to nothing larger than 0.0003. The attribution is
   correct — the state genuinely is not what moved the score — and the project's
   whole explanation surface is 45 named features. This is a limit on what the
   improvement can be *shipped* as, not on its accuracy. *(§3.20, D141.)*

   **Revised: this limit is dataset-dependent, not intrinsic.** On CTU the
   decomposable `state+logvar` head carried 14% of the history-aware head's
   gain, which read as "there is nothing to retreat to". On CIC the same arm
   reaches **99.1%** — 0.7754 against 0.7825, well inside the ±0.047 seed
   spread. A head that keeps essentially all of the gain *and* decomposes into
   the 45 named features may exist on CIC. **Under the D145 fix it no longer
   merely keeps most of the gain — it edges ahead, 0.7869 against 0.7825**, with
   every non-`logvar` control bit-identical. One seed, validation only, 94
   positives: an open candidate rather than a finding, and Stage B is queued.
   *(§3.38, §3.42.)*

6. **The rollout was writing untrained values into the state it forecasts.**
   Not a limit of the approach — a defect, found late, now fixed. The transition
   loss masks dropped features, so the network gets no gradient there; the
   rollout fed its untrained output back and it compounded to rms 2.03 by k=6,
   in slots that are exactly zero in the input and in the truth, carrying more
   magnitude than the real features. Every CTU and cross-dataset number in this
   phase was produced under it, and all 28 cells have since been re-scored: 21 up,
   7 down, no published direction flipped. Run 8 is unaffected — its regime drops
   nothing. *(§3.37, §3.44, D145.)*

What is *not* the limit, contrary to how §3.8 originally read it: attack-group
separability. `ctu_4:c2` was written up as unreachable on a probe's ROC of
0.443 and NIDRA's own stage head then reached 0.864 on the same windows under
the same cross-host construction. That conclusion is withdrawn and the probe's
verdict is now documented as one-sided. *(§3.8 correction, D128.)*

## Q14 — What should the next research phase be?

**Partial — three directions are already evidence-backed, and the matrix may add
a fourth.**

1. **A corpus with two infected hosts in the same attack stage on the same
   split.** This is the binding limitation and no modelling change addresses it.
   Until then, every per-group number in this project needs the within-host
   decomposition attached, which is now what the benchmark and the scorecard do.
2. **The stage-aware risk objective**, pre-registered at §3.13. If criterion 1
   fails, the pooled objective is not the explanation and this direction closes
   — which is itself worth knowing.
3. **Advance warning, pursued on CIC rather than CTU.** This replaces the Δ=15
   idea, which was tested and withdrawn the same day it was written: CTU's host
   is silent for minutes before onset, so no window size recovers a precursor.
   CIC's pre-onset windows *are* non-empty and carry ROC 0.75 of separable
   signal against a model at the floor. The ceiling is modest and most of it is
   bare activity, so the honest framing of the work is closing a small measured
   gap, not unlocking advance warning. *(§3.15.)*

4. **Trajectory components in place of the raw hidden state** — `delta` and
   `logvar` rather than `hidden`. Motivated by Q13's fourth limit rather than by
   an accuracy hypothesis: they decompose into the 45 named features in a way a
   128-dimensional hidden state does not, so a variant keeping most of the AP
   gain while staying explainable would be worth a lower AP under §32's ranking.

   **The CTU evidence is against it and was already on disk** (§4 component
   table): `state+logvar` reaches 0.373 against `state`'s 0.353 and `hidden`
   alone's 0.496 — the decomposable component carries about a seventh of what the
   hidden state carries, and adding `logvar` on top of `hidden` makes things
   worse. Whatever the head is reading lives in the recurrent summary, not in the
   per-feature quantities. So Q13's fourth limit looks intrinsic rather than a
   component choice: **the AP gain and the explainability loss may be the same
   thing.**

   **Reopened on CIC.** §3.22 committed to testing the four missing CIC arms
   precisely so this direction would close on evidence rather than on an
   untested suggestion. It did not close. On CIC, `state+logvar` reaches
   **0.7754 against `state+hidden`'s 0.7825** — 99.1%, inside the seed spread —
   where on CTU it reached 14%. The same component that looks useless on one
   dataset is nearly sufficient on the other, so "the AP gain and the
   explainability loss are the same thing" is a CTU statement, not a general
   one. What makes it worth pursuing rather than merely noting: `state+logvar`
   is 90-dimensional and decomposes entirely into the 45 named features, so it
   is shippable through the existing SHAP surface in a way `state+hidden` is
   not.

   **The D145 re-run is done and it moved further in this direction, not less.**
   Pinning the 13 untrained log-variances *improved* the arms that read them:
   `state+logvar` 0.7754 → **0.7869**, `state+delta+logvar` 0.7330 → **0.7817**,
   while all four controls that do not read `logvar` came back bit-identical. So
   the decomposable head now edges *past* `state+hidden`'s 0.7825, and a second
   decomposable arm lands within 0.0008 of it. The untrained channel was noise
   the head worked around, not signal it exploited. *(§3.42.)*

   One condition remains before it becomes a recommendation: **Stage B across
   seeds**, queued. A 0.0044 margin against a ±0.047 seed spread is not a margin,
   and the validation split carries 94 positives in 893,701 rows. No test or
   holdout cell exists for these arms yet.

5. **Not a bigger sequence model.** §14 proposed a transformer or TCN once the
   simpler history-aware heads had been tested. They have now been tested, and
   two results point away from it. §3.34: the frozen head's answer moves 0.28%
   when the rollout moves the hidden state 40–62%, and its first-layer response
   to that displacement is 1.08× its response to noise. §3.42: a 90-dimensional
   head built from named features matches a 173-dimensional one that reads the
   full recurrent summary. The binding constraint is not head capacity, and
   adding capacity to the component that is not short of it is the predictable
   way to spend a phase and learn nothing.

   What both point at is the **coupling** between a frozen head and a rollout it
   never saw — weak by construction, because invariant 1 exists to make the
   forecasting claim falsifiable. The honest options are narrow: keep the
   invariant and accept the cap, or find an architecture where the head reads
   something the rollout genuinely moves, without unfreezing it on predicted
   states. Unfreezing would close the gap and remove the reason the result means
   anything, so it is not one of the options. *(§3.34, §3.42.)*

6. **Re-establish the cross-dataset matrix under the D145 fix.** Not a research
   direction so much as a debt: every CTU and cross-dataset number in this phase
   was produced while the rollout was writing untrained values into 13–15 of the
   45 feature slots, at a magnitude exceeding the real features. The fix is in
   and tested; the matrix has to be re-scored before Q2, Q3, Q6, Q7, Q8 or Q9
   are quoted as final. Roughly three hours of compute for the 28 cells.
   *(§3.37, D145.)*

   It is untested on CIC, where the head ablation ran `state` and `state+hidden`
   only and where §3.11 found the gain survives the within-host test that CTU's
   does not. That is a real difference between the two corpora and it is cheap to
   settle, so the CIC arms are queued rather than assumed to match CTU. The
   direction stays on this list because a negative on CIC closes it properly.

5. **Train the transition model with the frozen head in the loop.** Directly
   motivated by Q13's fourth limit and, unusually for this list, it does not
   violate invariant 1. That invariant constrains *heads*: they train on observed
   states and freeze. It says nothing about the transition model. A third stage —
   dynamics, then heads (frozen), then dynamics again under a loss that rewards
   the frozen head for scoring the *predicted* states well — keeps every head
   trained on observed data only, introduces no leakage, and is not circular,
   because the head was never fit to dynamics outputs in the first place.

   It is the one direction the measurement actually points at: if the two halves
   are coupled through a channel the head does not read, the fix is to move the
   transition's output into a channel it does, rather than to teach the head to
   read a new one. Untested, and it would need its own pre-registration —
   including the obvious failure mode, that the transition collapses onto
   whatever the head already likes and stops being a state forecaster. §36's
   state-forecast metrics are the control for that and would have to be reported
   alongside.

Ruled out by measurement rather than opinion: score fusion of the two frozen
heads (§3.9), `context_noise` as a fix for the exposure mismatch (§3.6, D126),
uncertainty as a head input (§3.4), and a finer Δ for CTU precursors (§3.15).
