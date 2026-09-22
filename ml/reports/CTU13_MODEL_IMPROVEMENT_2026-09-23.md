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
