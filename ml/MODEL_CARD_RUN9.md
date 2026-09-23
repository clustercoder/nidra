# Model Card — NIDRA Run 9 (CTU-13 integration + head research, Δ=60 s, 2026-09-23)

**Status: research phase, not a release.** This card describes what Run 9 measured and
what it established. It does **not** designate a shipped configuration. §36 item 7's
architecture question has an answer — the evidence favours `state`, the Run 8
architecture the phase set out to improve on — and an answer of that shape is a reason
not to ship something new, not a release decision. Every CTU-13 and cross-dataset number
below is post-D145-correction (see *Known defects*).

`MODEL_CARD.md` (Run 8, Δ=60 s, 2026-09-20) remains the card for the **shipped**
artifacts under `ml/artifacts/`. Run 9 did not retrain them, did not overwrite them and
does not supersede that card. Nothing in Run 9 changes a Run 8 number: Run 8's feature
regime drops no features, so D145 cannot reach it.

---

## What Run 9 added

| | |
|---|---|
| **A second corpus** | CTU-13, 13 Argus captures, 19,976,700 flows, integrated on a principled common schema rather than concatenated. Scenarios 5, 7 and 11 are too short to window; `ctu_5` produces zero rows. |
| **A cross-dataset matrix** | Five regimes — CIC→CIC, CTU→CTU, CIC→CTU, CTU→CIC, CIC+CTU→both — on identical rows per evaluation target, so only the training set varies. |
| **A history-aware risk head** | `TrajectoryRiskHead` over subsets of {state, hidden, delta, logvar}, served end to end (D141) — before which the selected architecture could not be run in production at all. |
| **An unseen-family test** | Neris (CTU's largest family, scenarios 1/2/9) withheld from training entirely. |
| **A reproducibility guarantee** | All 28 recorded cells re-scored into a separate directory; all reproduce bit-identically, and interval coverage went from 6/28 to 28/28. |

---

## Intended use and out-of-scope use

Unchanged from Run 8, and one addition. This is a research prototype for NTRO PS 26153:
short-horizon (≤ 6 min) per-host forecasting of attack likelihood and projected stage
from flow and packet telemetry, offline or replayed. Not validated for production
security operations; not a detector; not attribution; not a causal model of attacker
behaviour.

**Added by Run 9:** it should not be relied on to beat a persistence baseline. On the
unseen family it does not (0.717 against 0.772), and on both CTU rows of the
fixed-readout comparison it does not (0.322 against 0.322; 0.373 against 0.362). An
operator choosing between this model and "assume the host stays as it is" has no
measured reason to prefer the model on those slices.

---

## Evaluation protocol

Run 8's protocol, unchanged and enforced, plus what Run 9 added to it.

- **Natural prevalence everywhere.** Test and holdout are never rebalanced.
- **The operating point is selected on validation only** and frozen. `benchmark.py`
  raises if asked to select it on any other split.
- **Episode-cluster bootstrap intervals**, reported with the positive-cluster count,
  because several cells rest on 15 positive clusters and the interval is the honest
  width at that count.
- **Within-host decomposition attached to every per-group number**, because both
  corpora reuse infected hosts across splits and an aggregate can be host identity.
- **Added in Run 9:** the published system is in `BOOTSTRAP_SYSTEMS` (it was not, §3.31);
  a scaler/checkpoint feature-regime guard (D142) and a risk-head-components guard
  (D143), both of which caught real misconfigurations in production.

---

## Known defects

**D145 — the rollout manufactured state in features that carry no gradient.**
`train/losses.py` masks dropped features out of the transition loss, so the transition
network is untrained on those outputs; `rollout()` fed that output back in and it
compounded to rms 2.03 by k=6, against real features at 0.69, in slots that are exactly
zero in both the input and the truth. It reached the frozen head, whose weights there
were never constrained by data either, and it was also present in head *training*
through `head_context`'s unmasked `logvar`.

Fixed across all four live paths (`rollout`, `observed_context`, `head_context`,
`counterfactual`), from a single construction point. It costs nothing measurable at
serving (masked is slower in 19 of 40 paired calls).

**Scope:** 40 of 63 run scalers drop features — every CTU-13 run (15 dropped) and every
combined and `cic_core` run (13). The 23 `full`-regime runs, including Run 8's lineage,
drop nothing and are arithmetically unaffected.

**Consequence:** every CTU-13 and cross-dataset number in Run 9 was first produced under
the defect, and all 28 cells have since been re-scored under the fix. The correction does
not go one way — **21 cells up, 7 down** — so no number may be adjusted by an assumed
direction, only re-measured. Every published direction survived: Q2's cross-dataset gap
narrowed from +0.094 to +0.082 and Q3's from +0.051 to +0.035, and both kept their sign.
The one thing the correction changed materially is the oracle anomaly, which it made
*worse*: cells where the model beats its own oracle went from 10 of 28 to 17 of 28.

---

## What Run 9 has established

1. **The transition model learns dynamics that are real and measurable**, and this holds
   on a second corpus. That was Run 8's central claim and it survived the addition of
   CTU-13.
2. **The history-aware head wins the objective it is trained against and not the one
   the system is evaluated against.** On head-training validation AP it beats every
   per-state head across three regimes and three seeds. On the forecast benchmark, at
   the same three seeds, it fails a bar pre-registered before the run: it wins CTU test
   AP 3/3 with no overlap and loses CTU holdout 1/3, and on ROC it loses both splits
   (0.693 vs 0.797 test, 0.587 vs 0.912 holdout). It halves false alarms and wins F1 at
   the operating point, and its margin over persistence is −0.0100 [−0.0342, +0.0034]. A
   GRU sequence classifier on identical rows beats both.
3. **The model transfers to an unseen attack family** — AP 0.717 [0.485, 0.860], ROC
   0.950, within-host ROC 0.949 across ten infected hosts, the first genuinely
   cross-host within-host measurement in the project. **Persistence reaches 0.772.**
4. **Adding CTU-13 helps on CIC as a direction, not as a quantity** (ROC 0.44 → 0.91
   uncalibrated, intervals overlapping on 15 positive clusters). **Adding CIC does not
   help on CTU.**
5. **Every recorded number reproduces bit-identically** under pinned seeds, and every
   cell carries both a marginal and a **paired** episode-cluster interval — the latter
   on the arm the project publishes, which had never been paired-bootstrapped before
   (§3.50).
6. **On CIC a fully decomposable risk head matches the opaque one.** Under the
   D145 fix, `state+logvar` (90 dims, all derived from the 45 named features)
   beats `state+hidden` (173 dims, reading the recurrent summary) on validation AP
   at **3/3 paired seeds** in Stage B. It matters because §3.20 found the winning
   head's score impossible to attribute through the project's 45-feature explanation
   surface, and a decomposable head would not have that problem. It has **no forecast-
   benchmark evidence at all** — the benchmark is the evaluation that reversed the
   history-aware head's validation win, so this is a validated candidate and not a
   selection.
7. **One cell has a margin over its strongest baseline that survives that interval with
   room to spare:** `CIC+CTU → CIC`, `state+hidden`, test — AP 0.1654 against
   persistence-plus-noise's 0.0994, margin **+0.0660 [+0.0235, +0.1296]** across 1,742
   episode clusters at natural prevalence 0.00417. It is the only one of 28. Three more
   clear zero by about 1e-05 and are flagged rather than claimed, three are negative,
   and 22 span zero.

## What Run 9 has not established

- **That the forecast beats persistence.** It does not, on the slices where the question
  is sharpest: the unseen family (0.717 against 0.772), both CTU rows of the fixed
  readout, and 9 of 12 cells of the corrected cross-dataset scorecard. §3.34, re-measured
  under the D145 fix in §3.46, gives the mechanism: the frozen head's answer moves 0.36%
  when the rollout moves the hidden state 39–54%, and its first-layer response to that
  displacement is 1.15× its response to noise. The invariant that makes the forecasting
  claim falsifiable is the same one that caps it, and Run 9 reported the negative rather
  than removing the invariant.
- **Genuine advance warning.** CTU's pre-onset windows carry no state information; CIC's
  are weakly separable (probe ROC 0.75, of which bare activity gives 0.68) against a
  model at the prevalence floor. Neither corpus supports the claim.
- **Why the model beats its oracle in 17 of 28 cells.** Four explanations have been
  tested and refuted, three against criteria fixed in advance: horizon reduction,
  drift amplification (§3.27), D145 itself (§3.37 — masking made *more* cells beat the
  oracle) and projection onto the training manifold (§3.45 — refuted at all four
  thresholds, with the direction inverted in 24 of 24 cells). What is left is a
  restatement rather than an answer: the frozen head ranks better on states measurably
  further from its training distribution, median ratio 1.42.
- **A shipped configuration.** The architecture question is answered — under §32's
  ranking the evidence favours `state`, which is what Run 8 already ships — and no
  Run 9 artifact is proposed for release. This card does not designate one.

---

## Claims discipline

Unchanged from Run 8 and binding here. This system performs **learned dynamics** and
**temporal forecasting**, not causal inference. Counterfactual outputs are labelled
`"model-internal what-if"` everywhere, and under D145's fix a what-if on a dropped
feature is inert rather than a confident answer about a quantity the model has no
information on.

Run 9 added one discipline worth naming: **pre-registration**. Six criteria were written
down before the measurement that would decide them, and **five refused the hypothesis
they were written for** — the roadmap's §12 stage-balanced objective, three successive
explanations for the oracle gap, and this phase's own headline claim. They are in
`reports/RUN9_NEGATIVE_RESULTS.md`, which has 18 entries — the largest category of
which is not a failed idea but a measurement apparatus that produced a plausible number.

---

## Reproduction

```bash
cd ml
python -m nidra.scripts.reproduction_check --runs experiments/runs --out reports/run9/reproduction_check.md
python -m nidra.scripts.mask_correction    --runs experiments/runs --out reports/run9/mask_correction.md
python -m nidra.scripts.cross_dataset_scorecard --runs experiments/runs --out reports/run9/scorecard.md
python -m nidra.scripts.margin_intervals   --runs experiments/runs --out reports/run9/margin_intervals.md
```

Every experiment carries a provenance record under `experiments/runs/<label>/`. The full
log is `reports/CTU13_MODEL_IMPROVEMENT_2026-09-23.md`; the questions are answered in
`reports/RUN9_QUESTIONS.md`; the deliverable checklist is
`reports/RUN9_DELIVERABLE_STATUS.md`.
