# Model Card — NIDRA Run 9 (CTU-13 integration + head research, Δ=60 s, 2026-09-23)

**Status: research phase, not a release.** This card describes what Run 9 measured and
what it has so far established. It does **not** designate a shipped configuration —
§36 item 7's architecture selection is not final, and every CTU-13 and cross-dataset
number below is provisional pending the D145 re-score (see *Known defects*).

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

**Consequence:** every CTU-13 and cross-dataset number in Run 9 was produced under the
defect. The re-score is in progress. The correction does not go one way — it has helped
some cells and hurt others — so no number here should be adjusted by an assumed
direction.

---

## What Run 9 has established

1. **The transition model learns dynamics that are real and measurable**, and this holds
   on a second corpus. That was Run 8's central claim and it survived the addition of
   CTU-13.
2. **The history-aware head wins the alert and not the forecast.** It halves false
   alarms and wins F1, and its within-host ROC *falls* (0.816 → 0.697) while its margin
   over persistence is −0.0100 [−0.0342, +0.0034]. A GRU sequence classifier on identical
   rows beats both.
3. **The model transfers to an unseen attack family** — AP 0.717 [0.485, 0.860], ROC
   0.950, within-host ROC 0.949 across ten infected hosts, the first genuinely
   cross-host within-host measurement in the project. **Persistence reaches 0.772.**
4. **Adding CTU-13 helps on CIC as a direction, not as a quantity** (ROC 0.44 → 0.91
   uncalibrated, intervals overlapping on 15 positive clusters). **Adding CIC does not
   help on CTU.**
5. **Every recorded number reproduces bit-identically** under pinned seeds.
6. **On CIC a fully decomposable risk head matches the opaque one.** Under the
   D145 fix, `state+logvar` (90 dims, all derived from the 45 named features)
   reaches 0.7869 against `state+hidden`'s 0.7825 on validation, with every
   control that does not read `logvar` bit-identical. One seed, 94 validation
   positives, no test cell yet — a candidate for Stage B, not a conclusion. It
   matters because §3.20 found the winning head's score impossible to attribute
   through the project's 45-feature explanation surface, and a decomposable head
   would not have that problem.

## What Run 9 has not established

- **That the forecast beats persistence.** It does not, on the slices where the question
  is sharpest. §3.34 gives the mechanism: the frozen head's answer moves 0.28% when the
  rollout moves the hidden state 40–62%, and its first-layer response to that
  displacement is 1.08× its response to noise. The invariant that makes the forecasting
  claim falsifiable is the same one that caps it, and Run 9 reported the negative rather
  than removing the invariant.
- **Genuine advance warning.** CTU's pre-onset windows carry no state information; CIC's
  are weakly separable (probe ROC 0.75, of which bare activity gives 0.68) against a
  model at the prevalence floor. Neither corpus supports the claim.
- **Why the model beats its oracle in 11 of 28 cells.** Three explanations have now been
  tested and refuted, two of them against criteria fixed in advance. §3.41 registers the
  fourth.
- **A shipped configuration.** Selection is not final and this card does not make one.

---

## Claims discipline

Unchanged from Run 8 and binding here. This system performs **learned dynamics** and
**temporal forecasting**, not causal inference. Counterfactual outputs are labelled
`"model-internal what-if"` everywhere, and under D145's fix a what-if on a dropped
feature is inert rather than a confident answer about a quantity the model has no
information on.

Run 9 added one discipline worth naming: **pre-registration**. Five criteria were
written down before the measurement that would decide them. Three refused the
hypothesis they were written for, including two of this log's own explanations. They are
in `reports/RUN9_NEGATIVE_RESULTS.md`, which has 15 entries.

---

## Reproduction

```bash
cd ml
python -m nidra.scripts.reproduction_check --runs experiments/runs --out reports/run9/reproduction_check.md
python -m nidra.scripts.mask_correction    --runs experiments/runs --out reports/run9/mask_correction.md
python -m nidra.scripts.cross_dataset_scorecard --runs experiments/runs --out reports/run9/scorecard.md
```

Every experiment carries a provenance record under `experiments/runs/<label>/`. The full
log is `reports/CTU13_MODEL_IMPROVEMENT_2026-09-23.md`; the questions are answered in
`reports/RUN9_QUESTIONS.md`; the deliverable checklist is
`reports/RUN9_DELIVERABLE_STATUS.md`.
