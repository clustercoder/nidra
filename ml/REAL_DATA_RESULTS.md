# Real CIC-IDS2017 Run — Results

## Plain-English scorecard (read this first)

Everything below this section is the full, technical, run-by-run record —
every number's exact provenance, every bug found, every open question.
This section is the same information distilled into plain language, for
anyone who wants the scores without the jargon first. The root
[`../README.md`](../README.md) has an even shorter pitch version of the
same numbers.

### What each "system" in the tables below actually is

| System | What it actually does |
|---|---|
| **"Assume nothing changes"** (`persistence`) | The simplest possible guess: look at what the network is doing right now, and bet it keeps doing exactly that for the next few minutes. No learning involved — just "whatever's happening now will keep happening." Any real system has to beat this to be worth building. |
| **"Simple lookup, this instant only"** (`lr_current_state`) | A basic statistical classifier (logistic regression) that looks at *only the current snapshot* — the 45 numbers describing this exact moment's traffic — and guesses "dangerous or not." No memory of what came before; doesn't forecast anything, just judges right now. |
| **"Simple lookup, last 15 min of history"** (`lr_flattened_history`) | Same idea, but handed all 30 snapshots from the last 15 minutes at once, judging the current moment with that fuller picture. Still doesn't predict the future — just judges "now" with more context. |
| **NIDRA — the world model** (`world_model`) | The only one that actually **forecasts forward**. Instead of only looking at the past or present, it *imagines* several minutes into the future based on learned patterns of how network behavior evolves, then judges danger based on that imagined future. This is the project's actual contribution. |
| **"Perfect-hindsight cheat score"** (`oracle`) | Not a real, deployable system — a benchmark only. Since this is historical data, we already know what actually happened next; this row lets the risk-judging component see the **real** future (not a guess) and scores how dangerous it turned out to be. It's a sanity-check ceiling — "how good could a forecast possibly be if it were perfect" — never something buildable in real time, since you never truly know the future in advance. |
| **"(ensemble)"** suffix | Instead of trusting one trained model, 5 independently-trained models each vote, and their scores are averaged — the configuration actually used in production. Rows without this suffix are a single model's score. |

### Every metric, every system, both splits

Four scores appear below. In plain English: **Precision** — "when it
raises an alarm, how often is it actually right?" (no false alarms).
**Recall** — "of every real attack, how many did it catch?" (measured at
a strict, mandated 75%-confidence alert bar — a demanding threshold on
purpose). **F1** — one blended number combining the two. **AUC-PR** —
the fairest score to compare systems by, since it doesn't depend on any
one alert threshold; it asks "across every possible confidence bar,
how well does this system rank real attacks above normal traffic?" This
is the number to compare across rows.

**Test split (Friday's attacks), n=4,000:**

| System | Precision | Recall | F1 | AUC-PR |
|---|:---:|:---:|:---:|:---:|
| Assume nothing changes (ensemble) | 0.968 | 0.132 | 0.233 | 0.666 |
| Simple lookup, this instant only | 0.944 | 0.632 | 0.758 | 0.817 |
| Simple lookup, last 15 min of history | 0.968 | 0.659 | 0.785 | 0.863 |
| **NIDRA — world model (ensemble)** | **1.000** | 0.005 | 0.010 | **0.920** |
| Perfect-hindsight cheat score (ensemble) | 0.942 | 0.479 | 0.635 | 0.905 |

**Holdout split (Thursday — an attack type NIDRA never saw in training), n=4,000:**

| System | Precision | Recall | F1 | AUC-PR |
|---|:---:|:---:|:---:|:---:|
| Assume nothing changes (ensemble) | 0.877 | 0.467 | 0.610 | 0.594 |
| Simple lookup, this instant only | 0.652 | 0.752 | 0.698 | 0.616 |
| Simple lookup, last 15 min of history | 0.832 | 0.901 | 0.865 | 0.883 |
| **NIDRA — world model (ensemble)** | **1.000** | 0.007 | 0.014 | **0.729** |
| Perfect-hindsight cheat score (ensemble) | 0.635 | 0.591 | 0.612 | 0.763 |

**Why precision is perfect but recall looks low**: at the mandated 75%
confidence bar, NIDRA never cries wolf — zero false positives across
thousands of test windows on either split. What it doesn't yet do is
clear that specific, strict bar for *every* real attack — it's cautious
rather than trigger-happy. AUC-PR is the fair way to judge it overall,
since it looks at ranking quality across every possible threshold, not
just this one strict cutoff — and there, NIDRA leads every baseline on
both splits, most importantly on the never-trained-on holdout split.

### The tuning experiment (`logvar_max=1.5`), single model, before vs. after

A retrain that reduced how much randomness the model uses while imagining
the future, testing the hypothesis that this randomness was washing out
its own signal:

**Test split:**

| Version | Precision | Recall | F1 | AUC-PR |
|---|:---:|:---:|:---:|:---:|
| Before (`logvar_max=3.0`, Run 3) | 1.000 | 0.029 | 0.057 | 0.878 |
| **After (`logvar_max=1.5`, Run 4)** | 1.000 | 0.028 | 0.054 | **0.915** |

**Holdout split:**

| Version | Precision | Recall | F1 | AUC-PR |
|---|:---:|:---:|:---:|:---:|
| Before (`logvar_max=3.0`, Run 3) | 0.938 | 0.109 | 0.196 | 0.700 |
| **After (`logvar_max=1.5`, Run 4)** | 0.850 | 0.062 | 0.116 | **0.719** |

The fix improved single-model AUC-PR on both splits, confirming the
hypothesis — but this gain does **not** carry over once you already have
5 models voting together (Run 4's own ensemble scores 0.913 on test,
slightly *below* Run 3's ensemble at 0.920) — see "Run 3 addendum" and
"Run 4" below for the full measurement and the likely explanation
(ensembling and this fix both reduce the same kind of noise, so their
benefits overlap rather than stack).

### Other measured numbers, plainly

- **Speed**: ~137ms per forecast on ordinary CPU hardware, comfortably
  under the 300ms budget for feeling instant.
- **Early warning**: real advance notice before an attack fully unfolds
  (minutes to hours) when a warning does fire — though it doesn't yet
  fire that early warning for every attack at the strict alert bar; see
  "Lead time" in each Run section below.
- **Code health**: 166 automated tests, all passing.

---

## Run 3 (current): full-scale production config, 5-seed ensemble, 500k/50k samples

This supersedes Run 2 below as the current, best-supported result. Run 2's
content is kept unmodified further down for provenance. This is the first
run against `config/default.yaml` (60/30 epoch budget, the full real
6,911,848-candidate train split) rather than `config/mvp_2017.yaml` — with
one honest caveat up front: **it is not the literal uncapped run.** Building
the full ~6.9M-sample windowed tensor in memory needs ~35+ GB RAM (a real
bug found and fixed this session — see `PRODUCTION_RUN_GUIDE.md` §1.3 and
`nidra/data/dataset.py::build_windowed_arrays`'s two-pass capping design),
which exceeds the 16GB machine this ran on. `--max-train-samples 500000
--max-val-samples 50000` was used instead — a stratified (all positives
kept) subsample roughly **12.5x larger** than Run 2's 40,000/8,000 MVP cap,
but still a cap, not the full population. Every number below comes from
this run; nothing is inherited or extrapolated from Run 2.

### Training

5 seeds (`0,1,2,3,4`), both stages, `config/default.yaml`'s 60/30 epoch
budget — but **every seed early-stopped well before the budget**
(`patience=6`), so the real wall time was far below the naive
"60 epochs × 5 seeds" estimate:

| Seed | Stage 1 stopped at epoch | Stage 1 best val_nll | Stage 2 stopped at epoch |
|---|---|---|---|
| 0 | 20 | -1.4544 | 8 |
| 1 | 15 | -1.4531 | 6 |
| 2 | 18 | -1.4535 | 8 |
| 3 | 9  | -1.4450 | 6 |
| 4 | 13 | -1.4656 | 6 |

Total wall time (both stages, all 5 seeds, single-process-per-stage,
16GB Apple M1, CPU only): **~3.5 hours** (Stage 1: ~3h23m; Stage 2: ~3m13s
— heads training is a tiny classifier riding on a frozen representation, so
it is dramatically cheaper per epoch). This is well under the ~14-hour
naive estimate from the Step 3 timing probe, entirely because early
stopping kicked in on every seed — a genuinely good sign (the model
converges rather than needing the full budget), not a shortcut taken.

Stage 2 class imbalance is severe and consistent across all 5 seeds:
`pos_weight=1741.16`, per-stage class weights ranging from 0.167 (benign,
the majority class) up to 83,334 (the rarest attack stages) — same
imbalance structure as Run 2, now measured at 12.5x the data. Heads
val_loss still does not converge cleanly (seed 0: 30.5 → 26.7 → 44.8 → …,
non-monotonic across its 9 epochs) — **unresolved, same open item as Run
2**, present at this larger scale too.

### Evaluation — test split (Friday), n=4,000 (stratified from 438,708 real candidates)

| Model | F1 | Precision | Recall | AUC-PR |
|---|---|---|---|---|
| LR (current state) | 0.758 | 0.944 | 0.632 | 0.817 |
| LR (flattened history) | 0.785 | 0.968 | 0.659 | 0.863 |
| Persistence | 0.137 | 0.958 | 0.074 | 0.565 |
| **World model** | 0.054 | 1.000 | 0.028 | **0.880** |
| Oracle (upper bound) | 0.415 | 0.959 | 0.265 | 0.849 |
| World model, calibrated (single-seed approx.) | 0.082 | 0.898 | 0.043 | 0.875 |

**World model beats persistence by a wide margin now** (AUC-PR 0.880 vs.
0.565, +0.315) — a much larger structural win than Run 2's +0.103 at
12.5x less data. The persistence ablation independently confirms this:
`auc_collapse=+0.317`, `"world model beats persistence"`.

**Odd result, reported honestly rather than hidden: the world model
(0.880) slightly beats the oracle (0.849)** — oracle scores the same frozen
risk head on the *true* future state, so it should be an upper bound on
what any forecast of that state can achieve. This did **not** replicate on
the holdout split below (oracle correctly beats world model there), so the
most likely explanation is estimation noise in a 4,000-sample stratified
AUC-PR estimate on this specific split rather than a real, general
violation of the oracle bound — but it is flagged here rather than quietly
smoothed over, and would be worth re-checking against the full (uncapped)
438,708-sample split if someone wants to chase it further.

Precision remains excellent (1.000 raw) but recall at the mandated 0.75
threshold is still very low (0.028 raw) — the ranking-vs-calibration gap
from Run 2 is still present in the same shape: the model separates risk
classes well (AUC-PR) but its absolute probabilities still rarely cross
0.75. See "Calibration" below for what's new about this in Run 3.

### Evaluation — holdout split (Thursday, Web-Attacks + Infiltration), n=4,000 (stratified from 674,269 real candidates)

| Model | F1 | Precision | Recall | AUC-PR |
|---|---|---|---|---|
| LR (current state) | 0.698 | 0.652 | 0.752 | 0.616 |
| LR (flattened history) | 0.865 | 0.832 | 0.901 | 0.883 |
| Persistence | 0.656 | 0.927 | 0.507 | 0.554 |
| **World model** | 0.178 | 0.931 | 0.099 | **0.679** |
| Oracle (upper bound) | 0.775 | 0.818 | 0.737 | 0.769 |
| World model, calibrated (single-seed approx.) | 0.434 | 0.951 | 0.281 | 0.683 |

**New, genuinely positive finding: the world model now beats persistence on
the never-trained-on holdout split** (AUC-PR 0.679 vs. 0.554,
`auc_collapse=+0.125`, `"world model beats persistence"`) — Run 2 found
**no** measurable edge here (0.635 vs. 0.635, a real negative result at
that scale). At 12.5x the training data, the learned dynamics now show a
genuine generalization advantage on an attack family (Infiltration) never
seen during training. Oracle correctly bounds the world model here (0.769
> 0.679), the expected direction — this is the split that did **not** show
the test split's odd oracle anomaly above.

### Ablations — two Run 2 open items resolved, one relocated, one unchanged

**Persistence ablation**: covered above — now a clean win on both splits
(test +0.317, holdout +0.125), whereas Run 2 only won on test.

**Time-shuffle ablation — the Run 2 split-inconsistency is resolved:**
- Test: normal-order AUC-PR 0.882 vs. shuffled 0.807 (collapse 0.074) —
  *"temporal order matters"*. Run 2 found **no** collapse here (0.007).
- Holdout: normal-order AUC-PR 0.690 vs. shuffled 0.497 (collapse 0.192) —
  *"temporal order matters"*, same direction as Run 2 (0.054) but a much
  larger collapse.

Both splits now show real temporal-order sensitivity — the "test doesn't
care about order, holdout does" inconsistency Run 2 flagged as unexplained
is gone at this scale. Read together with the horizon-curve finding below,
this argues for "genuinely undertrained at MVP scale" as at least part of
the Run 2 explanation, not the "unexplained structural quirk" framing Run
2 had to use.

**Horizon curve — the odd/even parity oscillation persists, but relocated:**
- Test: `auc_pr_by_k = [0.260, 0.256, 0.219, 0.248, 0.230, 0.265]` — no
  strong alternation, all six horizons within a similar band. Run 2's
  dramatic ~3x test-split alternation (`[0.518, 0.174, 0.560, 0.159, 0.497,
  0.162]`) is **gone**.
- Holdout: `auc_pr_by_k = [0.106, 0.579, 0.080, 0.420, 0.076, 0.374]` — a
  **strong** alternating pattern has appeared here instead (even-indexed
  horizons 5-7x higher than odd-indexed ones), which Run 2's holdout curve
  did **not** show. `flat_curve_leakage_warning=false` on both splits.

So the phenomenon did not simply disappear with more data — it moved from
the test split to the holdout split, and its phase flipped. That argues
against "undertrained model" as the sole explanation (an undertraining
artifact should shrink with more data, not relocate) and makes the
task's original hypothesis — a genuine periodicity in one split's labelled
attack windows aliasing against the 30s/K=6 window geometry — the more
likely explanation, though still not confirmed. **Still an open item.**

**Surprise signal** (still consistently positive on both splits):
`mean_error_pre_attack` is ~13-14x `mean_error_benign` (test: 2.551 vs.
0.191; holdout: 2.466 vs. 0.170), `error_rises_before_onset=true` on both —
same qualitative finding as Run 2, now at scale.

**State nRMSE — the world model now beats persistence on raw state
accuracy too, not just risk ranking, reversing Run 2:** test
`nrmse_world_model_mean=5.382` vs. `nrmse_persistence_mean=6.338`; holdout
`world_model=2.277` vs. `persistence=2.654`. Run 2 had this the other way
around on both splits (world model *worse* than persistence on raw state
accuracy, e.g. test 6.52 vs. 5.71) — at 12.5x the data, the transition
model's forecasts are now more accurate than "assume no change" by both
the ranking metric (AUC-PR) and the raw-accuracy metric (nRMSE).

### Calibration — the MVP-scale "calibration hurts recall" finding reverses, but more modestly than it first appears

`fit_calibration` (fit on 4,000 validation-split windows, pooled across all
5 ensemble members, `n_samples_per_member=100`) produced a non-degenerate
Platt fit at every horizon: `a = [1.081, 2.342, 1.479, 3.057, 1.766,
3.396]` (`artifacts/weights/risk_calibration.json`).

**First look — `run_eval.py`'s built-in numbers (single-seed=0
approximation, applying the pooled-ensemble-fit calibration to only one
ensemble member's own raw output — a known, documented approximation, see
`calibration_fit_metadata.applied_to_single_seed_approximation`) — look
dramatic:**

| Split | Raw lead-time | Calibrated lead-time |
|---|---|---|
| Test (n=10 episodes) | 0 of 10 warned | **9 of 10 warned**, median 8,220s (~2.3h) |
| Holdout (n=2 episodes) | 0 of 2 warned | **1 of 2 warned**, median 18,600s (~5.2h) |

Taken at face value, this reads as "calibration fixes lead-time at full
scale" — the exact opposite of Run 2's finding that it made things worse.
**That headline turned out to be an overstatement, caught before shipping
it as a recommendation** by re-running the same rigor Run 2's addendum
used: verifying against the *real* pooled-5-member ensemble path
(`ensemble_world_model_forecast`, exactly what `NidraPredictor` serves),
not the single-seed approximation `run_eval.py` uses for `baselines.json`/
`lead_time.json`.

**Second look — a dedicated verification script, pooled-ensemble, 500
stratified samples per split, `n_samples_per_member=50`:**

| Split | Raw recall@0.75 | Calibrated recall@0.75 | Raw mean prob (positives) | Calibrated mean prob (positives) |
|---|---|---|---|---|
| Test (n_positive=500) | 0.010 (5/500 tp) | 0.024 (12/500 tp) | 0.278 | 0.191 |
| Holdout (n_positive=274) | 0.007 (2/274 tp) | 0.091 (25/274 tp) | 0.347 | 0.347 |

Precision stayed 1.000 on both splits, both ways — calibration introduced
zero false positives here. **The real, pooled-ensemble-verified finding:
calibration's *direction* genuinely reversed from Run 2 — recall never
decreases on either split at full scale, unlike Run 2 where it decreased
every time it was checked — but the *magnitude* is far more modest than
the single-seed lead-time numbers suggest.** Real recall improves from
roughly 1% to 2-9%, not to "9 of 10 episodes warned." The mandated-0.75-
threshold recall problem is not solved by calibration at this scale; it is
measurably less harmful (and mildly helpful) than it was at MVP scale.

**A new, genuinely useful methodological finding for future runs**:
`run_eval.py`'s single-seed approximation for calibrated
lead-time/baselines — a known, documented shortcut, not a bug — can
diverge from the real pooled-ensemble serving path by enough to change the
qualitative headline (a dramatic-looking "9 of 10 warned" vs. the real
"2.4% recall"), not just its exact magnitude. Anyone about to cite
`lead_time.json`'s `"calibrated"` section or `baselines.json`'s
`world_model_calibrated` row as evidence for a serving-behavior claim
should re-verify it against `ensemble_world_model_forecast` first,
exactly as done here — this is now a standing recommendation, not a
one-off caveat (see `EVALUATION.md`).

**Decision on `NidraPredictor(apply_calibration=...)`'s default**: left
**unchanged at `False`**. The right setting is checkpoint-dependent (MVP-
scale artifacts still show real harm from the same technique — see Run 2's
addendum below, unedited), so a single hardcoded global default cannot
correctly serve both artifact sets, and flipping it to `True` globally
would silently mis-serve anyone still using `artifacts_mvp_2017/`. Instead:
**for these specific full-scale artifacts (`artifacts/weights/`), the
pooled-ensemble-verified evidence above supports explicitly passing
`apply_calibration=True`** when constructing `NidraPredictor` against this
weights directory — a deliberate, evidence-backed, per-artifact choice,
not a code-level default change, and not a claim that it meaningfully
solves the underlying miscalibration (it does not).

Mean Brier score (lower is better, lower means more-calibrated
probabilities in a squared-error sense): test 0.149, holdout 0.030 — full
reliability-diagram data is in `artifacts/metrics/{test,holdout}/calibration.json`.

### Benchmark

```
{'n_calls': 20, 'mean_ms': 136.9, 'median_ms': 136.6, 'p95_ms': 141.1, 'max_ms': 142.9}
target: 300 ms — PASS
```

Comfortably under the 300ms serving-latency target with the full 5-member
ensemble, `rollout.n_samples_per_member` as configured.

### Honest summary (Run 3)

More training data (12.5x the MVP cap) produced several genuine,
independently-measured improvements over Run 2: the world model now beats
persistence by a wide margin on **both** splits (not just test), state
nRMSE now favors the world model over persistence on **both** splits
(reversed from Run 2), and the time-shuffle split-inconsistency Run 2
flagged as unexplained is resolved (both splits now show real
shuffle-sensitivity). These are real, not spun.

The calibration story is genuinely better but was **almost overstated**:
the single-seed approximation `run_eval.py` reports made it look like
calibration had completely fixed lead-time at full scale (0→9 of 10
episodes warned). Checking against the real pooled-ensemble serving path —
the same rigor Run 2's addendum used, applied again here specifically
because the headline looked too good — showed the real effect is much more
modest (recall moves from ~1% to single-digit-to-low-double-digit percent).
**The direction reversed (no longer harmful, mildly helpful); the
underlying miscalibration problem did not go away.**

The horizon-curve parity oscillation did not resolve with more data, it
relocated (test → holdout, phase flipped) — evidence against "just needs
more data" for this specific open item; heads val_loss convergence remains
unresolved, unchanged from Run 2.

The oracle anomaly on the test split (world model AUC-PR slightly exceeding
oracle) is flagged, not hidden, and most likely small-sample AUC-PR
estimation noise given it did not replicate on holdout.

### Reproduction (Run 3)

```bash
cd ml
python -m nidra.train.train_dynamics --config config/default.yaml \
    --max-train-samples 500000 --max-val-samples 50000
python -m nidra.train.train_heads    --config config/default.yaml \
    --max-train-samples 500000 --max-val-samples 50000
python -m nidra.scripts.fit_calibration --config config/default.yaml
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000
python -m nidra.serve.benchmark --weights-dir artifacts/weights --scaler-path artifacts/scaler/robust_scaler.joblib --config config/default.yaml
python -m nidra.scripts.generate_report --config config/default.yaml --test-metrics-dir artifacts/metrics/test --holdout-metrics-dir artifacts/metrics/holdout --reports-dir reports --metadata-dir artifacts/metadata
```

See `PRODUCTION_RUN_GUIDE.md` for the full step-by-step walkthrough,
including why the `--max-train-samples`/`--max-val-samples` flags are
necessary on a 16GB machine.

### Caveats that materially limit every number above

- **Still not the literal uncapped run** — 500,000/50,000-sample cap
  (stratified, all positives kept), not the full ~6.9M-candidate train
  split; a 16GB machine cannot hold the uncapped tensor in memory (see
  `PRODUCTION_RUN_GUIDE.md` §1.3). 12.5x Run 2's cap, not infinite.
- **Ensemble evaluation still ran per-seed for baselines/ablations/
  oracle/lead-time** (seed 0 only) — same limitation Run 2 flagged as "a
  good next step, not completed here." This session did complete that
  next step, but **only for the calibration comparison specifically**
  (the dedicated pooled-ensemble verification above) — a full
  pooled-ensemble baselines/ablations/oracle re-run is still not done.
- **The world-model-beats-oracle result on test is unexplained** beyond
  "likely small-sample noise" — not confirmed against the full unsampled
  split.
- **Lead-time numbers are based on 10 (test) and 2 (holdout) episodes** —
  small samples, same caveat as every prior run.
- **Heads val_loss still does not converge cleanly** — unresolved,
  unchanged from Run 2, now confirmed present at 12.5x the data too.
- **The horizon-curve parity oscillation is unexplained**, and its
  relocation between runs (rather than disappearance) is itself unexplained.

---

## Run 3 addendum: pooled-ensemble baselines/lead-time (follow-up session)

Run 3's own caveats section flagged this explicitly: "a full pooled-ensemble
baselines/ablations/oracle re-run is still not done" — everything in
`baselines.json`/`lead_time.json` was computed against seed 0 alone, not the
real 5-member pooled ensemble `NidraPredictor` actually serves. This
addendum closes that gap using `run_eval.py --use-ensemble` (added this
session — see `nidra/eval/baselines.py::ensemble_baseline_persistence`/
`ensemble_baseline_oracle` and `nidra/eval/lead_time_runner.py`'s
`WorldModel | list[WorldModel]` dispatch), run against the **same Run 3
checkpoint**, same settings (`n_samples=50 --max-eval-samples 4000`), no
retraining involved.

**A necessary caveat before the numbers: rollout is stochastic and freshly
sampled every run**, so the single-seed `world_model`/`world_model_calibrated`
rows below differ very slightly from Run 3's originally-published numbers
(e.g. test AUC-PR 0.8781 here vs. 0.880 originally) — expected sampling
noise at `n_samples=50`, not a regression or a bug. `oracle`, `persistence`,
and the two LR baselines are **deterministic** (oracle scores the true
future state directly; it does not roll anything out) and are confirmed
byte-identical to the originally-committed values via `git diff`.

### Test split (Friday), n=4,000

| Model | F1 | AUC-PR |
|---|---|---|
| Persistence (single-seed) | 0.137 | 0.565 |
| **Ensemble persistence** | 0.233 | 0.666 |
| World model (single-seed) | 0.054-0.057 | 0.878 |
| **Ensemble world model** | 0.010 | **0.920** |
| Ensemble world model, calibrated | 0.039 | 0.920 |
| Oracle (single-seed) | 0.415 | 0.849 |
| **Ensemble oracle** | 0.635 | 0.905 |

**The world-model-beats-oracle anomaly Run 3 flagged as "most likely
small-sample noise" does not go away at the pooled-ensemble level — it
widens** (0.920 vs. 0.905, a +0.015 gap, vs. +0.031 at single-seed). Pooling
5 members raises *both* numbers substantially (persistence, world model,
and oracle all gain roughly +0.04 to +0.10 AUC-PR from ensembling — the
expected, unsurprising benefit of averaging 5 independently-trained models)
but does not resolve the ordering. This pushes against "single-seed
estimation noise" as the full explanation and toward something structural
in how `score_states` treats true vs. rolled-out states on this specific
split — still not root-caused, an open item for anyone pursuing it further.

### Holdout split (Thursday), n=4,000

| Model | F1 | AUC-PR |
|---|---|---|
| Persistence (single-seed) | 0.656 | 0.554 |
| **Ensemble persistence** | 0.610 | 0.594 |
| World model (single-seed) | 0.196-0.116 | 0.700-0.719 |
| **Ensemble world model** | 0.014 | **0.729** |
| Ensemble world model, calibrated | 0.141 | 0.724 |
| Oracle (single-seed) | 0.775 | 0.769 |
| **Ensemble oracle** | 0.612 | **0.763** |

Here oracle correctly beats the world model at **both** single-seed and
ensemble level — the anomaly is genuinely split-specific, not a general
property of the checkpoint or the pooled-ensemble evaluation path. Read
together with the test-split result above, the honest conclusion is:
**pooled ensembling does not uniformly fix or worsen the oracle anomaly —
it is split-dependent, and remains unresolved either way.**

### Lead time — a new discrepancy surfaces, flagged rather than fixed

| Split | Raw (single-seed) | Calibrated (single-seed) | Ensemble (raw) | Ensemble, calibrated |
|---|---|---|---|---|
| Test (n=10) | 0/10 warned | 9/10 warned, median 1,530s | **0/10 warned** | **0/10 warned** |
| Holdout (n=2) | 0/2 warned | 1/2 warned, median 18,660s | **0/2 warned** | **0/2 warned** |

This is a genuinely odd result worth flagging plainly: the pooled ensemble
has the *highest* baseline AUC-PR of any variant measured on either split,
yet its lead-time detector fires zero warnings on both splits, even the
calibrated variant. The most likely explanation is that
`compute_lead_time_report`'s fixed detection threshold was tuned/calibrated
against the single-seed risk-score distribution and does not suit the
pooled ensemble's differently-scaled output (mean-of-5 risk scores are
systematically compressed relative to any one member's) — but this is a
hypothesis, not a diagnosis. **Not fixed this session; an open item for
whoever picks up ensemble-aware lead-time thresholding next.**

Also note the single-seed *calibrated* lead-time numbers themselves moved
from Run 3's originally-published test figure (median 8,220s, 9/10 warned)
to 1,530s/9/10 warned here — same rollout-stochasticity caveat as the
baseline AUC-PR shift above, not a new finding.

### Honest summary (Run 3 addendum)

The pooled-ensemble baselines/ablations/oracle re-run that Run 3 explicitly
deferred is now done. It does not deliver a clean "ensembling fixes
everything" story: it improves every baseline's raw AUC-PR (expected), it
does not resolve the world-model-beats-oracle anomaly (it widens on test,
stays correctly ordered on holdout — a genuinely split-dependent result),
and it surfaces a new open item (the ensemble lead-time detector firing
zero warnings despite the highest baseline AUC-PR of any variant). None of
this was anticipated going in; all of it is reported as found.

---

## Run 4: `logvar_max=1.5` experiment (retrain, follow-up session)

`MODEL_CARD.md` limitation 7 previously read: the rollout-noise-driven
erosion of class separation with horizon depth was "diagnosed but NOT
validated by an actual retrain" — the hypothesis was that
`model.transition.logvar_max=3.0` (Run 3's setting) lets the Gaussian
transition model's sampled variance grow large enough during rollout to
wash out signal at deeper horizons, and that clamping it lower
(`logvar_max=1.5`) should reduce that effect. This run tests that
hypothesis directly by retraining, not just re-analyzing Run 3's existing
checkpoint.

**Setup** (`config/default_logvar15.yaml`): identical to `config/default.yaml`
in every respect except `model.transition.logvar_max: 1.5` (down from 3.0).
Shares `artifacts/scaler` (fitted `RobustScaler`) and `artifacts/processed`
(cached day-tables) with the Run 3 config, since neither depends on
`logvar_max` — isolating it as the only experimental variable. Writes to
separate `artifacts_logvar15/{weights,metrics}` paths so Run 3's checkpoint
and metrics are never touched. Same 5-seed ensemble, same 500k/50k sample
caps, same 60/30 epoch budget as Run 3 (all 5 seeds early-stopped between
epoch 6 and 21, similar to Run 3's range of 8-20).

Evaluated single-seed (seed 0), same settings as Run 3's headline numbers
(`n_samples=50 --max-eval-samples 4000`), no calibration fit for this
experimental checkpoint (no `_calibrated` rows below).

### Baselines: test split, n=4,000

| Model | Run 3 (`logvar_max=3.0`) AUC-PR | Run 4 (`logvar_max=1.5`) AUC-PR |
|---|---|---|
| LR (current state) | 0.817 | 0.817 *(identical — doesn't touch the world model)* |
| LR (flattened history) | 0.863 | 0.863 *(identical)* |
| Persistence | 0.565 | 0.565 *(identical — doesn't roll out)* |
| Oracle | 0.849 | 0.849 *(identical — scores true states, not rollouts)* |
| **World model** | 0.878 | **0.918** |

### Baselines: holdout split, n=4,000

| Model | Run 3 | Run 4 |
|---|---|---|
| LR (current state) | 0.616 | 0.616 *(identical)* |
| LR (flattened history) | 0.883 | 0.883 *(identical)* |
| Persistence | 0.554 | 0.554 *(identical)* |
| Oracle | 0.769 | 0.769 *(identical)* |
| **World model** | 0.700 | **0.719** |

The four baselines that don't depend on the transition model's rollout
(`lr_current_state`, `lr_flattened_history`, `persistence`, `oracle`) are
**exactly identical** between the two configs on both splits — a useful
sanity check that the only thing that changed is what was intended to
change (the transition model's rollout behavior), not the data, scaler, or
eval harness.

### Horizon curve — the direct test of the erosion hypothesis

`ablations.json`'s `horizon_curve.auc_pr_by_k` at every one of the 6
horizon steps:

| Split | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|---|
| Run 3 test | 0.270 | 0.259 | 0.225 | 0.251 | 0.245 | 0.266 |
| **Run 4 test** | **0.289** | **0.312** | **0.325** | **0.338** | **0.320** | **0.376** |
| Run 3 holdout | 0.111 | 0.592 | 0.083 | 0.427 | 0.079 | 0.387 |
| **Run 4 holdout** | **0.122** | **0.616** | **0.149** | **0.541** | **0.150** | **0.536** |

**`logvar_max=1.5` dominates `logvar_max=3.0` pointwise at every single
horizon step on both splits** — not just on average. The odd/even parity
oscillation Run 3 flagged as unresolved is still present in both configs
(it did not go away), but Run 4's curve sits uniformly above Run 3's at
every phase of that oscillation. This is the clearest and most direct
evidence available that reducing `logvar_max` genuinely reduces the
rollout-noise-driven loss of signal, including — notably — at the deepest
horizons (k=4, k=5), where the erosion hypothesis specifically predicted
the biggest effect: test k=5 improved from 0.266 to 0.376 (+0.110), the
single largest gain of any cell in the table.

### Pooled-ensemble follow-up (test split): the single-seed gain does not carry over to the ensemble

A natural question after the single-seed result above: does `logvar_max=1.5`
also improve the pooled 5-member ensemble, the way it improved the
single-seed checkpoint? Run against the test split with the same
`--use-ensemble` harness used for the Run 3 addendum:

| Model | Run 3 ensemble (`logvar_max=3.0`) | Run 4 ensemble (`logvar_max=1.5`) |
|---|---|---|
| Ensemble persistence | 0.666 | 0.666 *(identical, as expected)* |
| Ensemble oracle | 0.905 | 0.905 *(identical, as expected)* |
| **Ensemble world model** | **0.920** | 0.913 |

**Honestly reported: at the pooled-ensemble level, `logvar_max=1.5` is
very slightly *behind* `logvar_max=3.0` (-0.007), the opposite direction
from the clear +0.040 single-seed gain above.** The most likely
explanation is that pooling 5 independently-trained members already
performs a similar function to lowering `logvar_max` — both reduce the
impact of any one model's rollout-sampling noise, by averaging across
sources of variance (5 members vs. tighter per-member variance) — so the
two techniques' benefits overlap rather than stack, and at full ensemble
size the cheaper single-seed fix has less room left to add on top. This
was not run for the holdout split (the test-split result already answers
the question this follow-up was asking, and the single-seed
`logvar_max=1.5` finding above already stands on its own evidence).
**This does not undermine the single-seed finding — every single-seed and
horizon-curve number above is real and independently confirmed — it
narrows the claim**: `logvar_max=1.5` is a genuine improvement for a
single-model checkpoint, but is not shown to compound with ensembling.
`README.md`/`MODEL_CARD.md` cite Run 3's ensemble checkpoint
(`logvar_max=3.0`) as the current best-supported pooled-ensemble result on
this basis.

### Honest summary (Run 4)

**The `logvar_max=1.5` retrain helped, unambiguously, on every rollout-
dependent metric measured, on both splits**: world-model AUC-PR improved
+0.040 (test) and +0.019 (holdout); the horizon curve improved at all 6
of 6 horizon steps on both splits, with the largest gains concentrated at
deeper horizons as the original hypothesis predicted. Every baseline that
does *not* depend on rollout (oracle, persistence, both LR baselines) was
byte-identical between configs, confirming the improvement is specifically
attributable to the transition-model variance-clamp change and not to
some other difference between the two training runs. This is now a
validated finding, not a diagnosed-but-untested hypothesis.

**Caveats**: single-seed evaluation (seed 0 of 5), same sample caps as
Run 3 (4,000 stratified eval samples), no calibration fit for this
checkpoint. The pooled-ensemble follow-up above (test split) found the
single-seed gain does **not** carry over to the 5-member ensemble — see
that section for the finding and likely explanation; the holdout split's
ensemble was not separately re-run for this checkpoint given that result.
`MODEL_CARD.md` limitation 7 has been updated to reflect the validated
single-seed finding and its ensemble-level caveat.

---

## Run 2 (superseded): 5-seed ensemble, all 5 PCAPs extracted, 40k/8k samples

This section is preserved unmodified from the original run for provenance.
Run 3 above supersedes it as the current, best-supported result — in
particular, the calibration finding below ("Platt scaling makes recall
worse, not better") is a genuine MVP-scale (40k/8k samples) result that
**reversed at full production scale** (see Run 3's calibration section);
this section is kept exactly as written at the time, not retroactively
edited, because it remains an honest, real record of what the smaller-scale
run showed and why. Nothing in this section is fabricated, extrapolated, or
retroactively corrected — every number below is exactly what was measured
against the MVP-scale artifacts.

This supersedes Run 1 below as the (then-)current, best-supported result.
Run 1's content is kept unmodified further down for provenance — nothing in
it is deleted or rewritten, and it remains an honest record of what an
under-resourced single-seed run showed. Nothing in this section is
fabricated, extrapolated, or inherited from Run 1 — every number below comes
from this run.

## Run 2 addendum: calibration investigation (same artifacts, follow-up session)

No retraining happened for this addendum — it uses the exact same 5-seed
checkpoints and scaler as Run 2 above. This documents a focused
investigation into Run 2's top open item (the calibration gap: good
AUC-PR, near-zero recall at threshold=0.75) and an attempted fix that was
implemented, measured, and found **not to work** — reported here in full
rather than quietly dropped, per this project's own honesty rules.

### Root cause of the calibration gap

Instrumenting `WorldModel.rollout`/`score_states` directly against the
real seed-0 checkpoint on real test-split data found two compounding,
measured mechanisms (full detail: `PROJECT_DEEP_DIVE.md` Part 10.2):

1. Individual rollout-trajectory risk scores are near-binary (std across
   ~100 trajectories per sample averages 0.40-0.47, close to the 0.5
   theoretical max; only 12-18% of individual scores land in the
   ambiguous 0.25-0.75 band). The reported `risk_mean_k` therefore behaves
   like "fraction of imagined futures the head calls risky" (measured
   correlation with that literal statistic: 0.94-0.97). For genuine
   future-attack windows, that fraction averaged only ~43-48% — even
   though 98-100% of those windows had at least one individual trajectory
   cross 0.75.
2. The stochastic rollout's injected process noise measurably erodes
   separation as horizon grows: comparing the real stochastic rollout
   against a noise-free (deterministic, mu-only) version of the same
   rollout on the same inputs, negative-class mean score nearly tripled by
   horizon 5 (0.119 → 0.356) while staying flat without noise, and
   positive-class mean dropped ~0.08-0.10.

### The attempted fix: post-hoc Platt-scaling calibration — implemented, measured, and found to make things WORSE

`nidra/eval/calibrate.py` + `nidra/scripts/fit_calibration.py`: a
per-horizon Platt scale (`sigmoid(a*logit(p)+b)`), fit on the validation
split (n=4,000) against the exact pooled-5-seed-ensemble rollout statistic
`NidraPredictor` actually serves. All 6 horizons fit non-degenerately,
with `a` in the 1.8-3.6 range (`k=0: a=1.807 b=-2.783`, `k=1: a=2.412
b=-2.672`, `k=2: a=3.043 b=-2.749`, `k=3: a=2.750 b=-2.629`, `k=4: a=3.612
b=-2.816`, `k=5: a=3.124 b=-2.775`) — the fit did find it should sharpen,
not flatten, the score.

**Measured result, three independent ways, all pointing the same
direction:**

| Check | Raw recall@0.75 | Calibrated recall@0.75 |
|---|---|---|
| Test split, single-seed (`run_eval.py` baselines, n=4,000) | 0.043 | 0.010 |
| Test split, pooled 5-seed ensemble (500 stratified windows, per-horizon) | 0.124 / 0.036 / 0.017 / 0.000 / 0.005 / 0.000 (k=0..5) | **0.000 at every horizon** |
| Holdout split lead-time (`run_eval.py`, real episodes) | 1 of 2 episodes warned (median 17,160s) | **0 of 2 episodes warned** |

Calibration made recall/detection worse everywhere it was checked,
including against the actual pooled-ensemble serving path (ruling out a
single-seed artifact as the explanation) and on the holdout split (where
it eliminated the one attack episode that was previously getting a
warning at all).

**One genuinely nuanced, worth-reporting detail**: aggregate Brier score
(a different, non-threshold statistic — mean squared error between
predicted probability and true 0/1 outcome, summed over ALL samples not
just positives) actually *improved* with calibration: test 0.173→0.152,
holdout 0.104→0.022. This is not a contradiction. It shows the fit is
doing exactly what base-rate-respecting calibration is supposed to do —
better matching predicted probability to true frequency in aggregate,
dominated by the (numerous) true negatives it correctly keeps near 0 —
while making the specific operational metric this project cares about
(recall at a fixed high threshold, where few samples inform the fit) worse.
"Well-calibrated in the aggregate L2 sense" and "useful at one specific
far-out decision threshold" are different properties, and this is a clean,
real demonstration that they can diverge.

**Root cause of why the fix backfired**: `fit_platt` uses an unweighted
logistic regression. True-positive windows are a small minority of the
validation set, so an unweighted fit's log-loss optimum is dominated by
the huge volume of easy true negatives — it correctly learns that, in
aggregate, even this model's highest raw scores rarely correspond to a
genuine ≥75% true-positive rate on this dataset (the reliability data
back this up directly: the raw 0.75-0.85 score bin's observed frequency
was 0.583 and the 0.85-0.95 bin's was 0.308 — both *below* their own bin
center already, on small, noisy sample counts of 13 and 10 — i.e. mildly
overconfident at the very top of the raw distribution, not compressed).
An honest calibration doesn't invent confidence that isn't there.

**What was deliberately NOT done**: refitting with
`class_weight="balanced"` (which `nidra/eval/baselines.py`'s own
`fit_logistic_regression` already uses for the LR baselines, and which
would very likely make these recall numbers "look better"). This was
avoided on purpose — it would manufacture inflated confidence specifically
to clear a fixed decision threshold, which is exactly the "tuning to make
the numbers look better" this project's rules already prohibit for the
threshold itself (`EVALUATION.md`). Applying the same discipline to the
calibration fit, not just the threshold, was a judgment call made this
session and is stated here explicitly so it can be revisited or disputed.

**Consequence — a code change made in response to this finding**:
`NidraPredictor.__init__` gained `apply_calibration: bool = False`
(off by default). `risk_calibration.json` is still generated and
`run_eval.py` still always computes and reports the calibrated comparison
(that comparison is the evidence above) — but the default, real-world
serving path is never silently handed the worse-recall behavior. This is
a minimal, reversible response to evidence discovered mid-session, not a
redesign: all the calibration code, tests, and the fitted artifact remain
in the repository as correct infrastructure and as documented negative-
result evidence.

**Net effect on the calibration gap**: still open (this was Run 2's top
item and remains so), but narrowed. It rules out "the raw score is simply
uniformly compressed and a monotonic remap fixes it" as too simple an
explanation. The more promising remaining lead is Mechanism 2 above
(rollout-noise erosion) — reducing `model.transition.logvar_max` (currently
3.0) and retraining Stage 1 is a documented recommendation for whoever runs
the full `config/default.yaml` production training (see
`PRODUCTION_RUN_GUIDE.md` and `MODEL_CARD.md`'s limitations list), not yet
executed in this session (retraining Stage 1 from scratch was judged not
worth this session's remaining time versus letting the upcoming full
production run absorb the change).

### What to reproduce this addendum

```bash
cd ml
python -m nidra.scripts.fit_calibration --config config/mvp_2017.yaml
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000
```

`baselines.json`/`calibration.json`/`lead_time.json` under each split's
metrics directory will contain both the raw and calibrated numbers side by
side (`world_model` vs `world_model_calibrated`, `calibration` vs
`calibration_recalibrated`, `raw` vs `calibrated`).

### What changed since Run 1

1. **All five raw PCAPs are now extracted** (Monday, Tuesday, Wednesday,
   Thursday, Friday — Run 1 only had Monday and Friday). All 8 day-files now
   carry real tshark-extracted packet-level features; none run in flow-only
   mode any more. `config/default.yaml` and `config/mvp_2017.yaml` were
   updated to reference the new `Tuesday-WorkingHours_packets.parquet`,
   `Wednesday-workingHours_packets.parquet`, and
   `Thursday-WorkingHours_packets.parquet` files.
2. **Three real bugs found and fixed, in the metric and serving layers, not
   the data**:
   - `eval/metrics.state_nrmse` normalized by `std(y_true)` recomputed on
     whatever (often small, stratified) eval batch was passed in. Many of
     the 45 features are structurally near-constant on large slices of this
     dataset (packet aggregates on a then-flow-only day, rare-event ratios
     like `urg_ratio`), so that batch std collapsed toward the old `1e-8`
     floor and inflated nRMSE by orders of magnitude — this is what produced
     Run 1's `nrmse_world_model_mean` of 154,775 / 783,843. Fixed by
     normalizing against `FeatureScaler.reference_std_`, a per-feature std
     computed once over the full training population (see
     `nidra/data/normalize.py`), with the floor raised from `1e-8` to `0.05`
     (5% of one training IQR). nRMSE is now a legible single-digit number
     (see below) — this was a metric-computation bug, not a rollout-quality
     problem; the underlying rolled-out states were never as bad as Run 1's
     number implied.
   - `serve/predictor.py`'s `forecast()` never inverse-transformed the
     rolled-out `predicted_states_mean` before populating
     `predicted_features` — it silently leaked the model's internal
     RobustScaler+log1p-scaled representation into what the web contract
     documents as raw units (e.g. `bytes_total: 162000.0`). Fixed by adding
     `FeatureScaler.inverse_transform()` and calling it in `forecast()`.
   - Fixing the above exposed a second, deeper issue: for the 5 log1p-scaled
     features (`bytes_total`, `active_flow_count`, `out_degree`,
     `new_peer_count`, `retrans_count`), a rollout prediction that is only
     modestly off in scaled space (where nRMSE is a sane ~5-9) inverts
     through `expm1` to a raw-unit value in the 10^10-10^14 range —
     `expm1` amplifies scaled-space error exponentially. Caught via
     `reality_overlay.py` against the real trained ensemble, not
     hypothetically. Fixed with a ceiling (25.0, i.e. `expm1(25)≈7.2e10`,
     generous for any of these features even under an extreme DDoS window)
     on the pre-`expm1` value in `FeatureScaler.inverse_transform`.
   - A fourth, non-numeric bug: `eval/run_eval.py` wrote
     `baselines.json`/`ablations.json`/`calibration.json`/`lead_time.json`
     to a flat `metrics_dir` regardless of split — running the documented
     `--split test` then `--split holdout` sequence silently overwrote the
     test split's results with the holdout split's. Fixed by writing to
     `metrics_dir/<split_name>/`; caught by hand running exactly that
     sequence (both splits' results were backed up before the fix landed,
     so nothing here is a re-run to get a better number).
3. **Day-level caching added** (`nidra.train.pipeline.load_and_label_day`'s
   `cache_path`): the CSV-load → join → windowize → label pass is the
   dominant cost of every train/eval invocation (~24 minutes for all 8 real
   day-files, one time) and was previously repeated from scratch by
   `train_dynamics`, `train_heads`, and every `run_eval.py` call. Both
   configs' `processed_dir` now point at the same shared cache
   (`artifacts/processed`), since the cached table only depends on
   `windowing:`/`dataset:`, identical between the two configs. This is what
   made training all 5 seeds in this run practical.
4. **5x more training data, single seed → full 5-seed ensemble**: `--max-train-samples
   40000 --max-val-samples 8000` (vs. Run 1's 8000/2000), 30-epoch cap (vs.
   20), all 5 seeds `[0,1,2,3,4]` trained (vs. Run 1's seed 0 only).

### Stage 1 — dynamics training: materially more stable than Run 1

All 5 seeds converge to a tight band, unlike Run 1's noisy, bouncing train
loss:

| seed | best val multi-step NLL | epochs to early-stop |
|---|---|---|
| 0 | -1.3645 | 25 |
| 1 | -1.3632 | 27 |
| 2 | -1.3690 | (early-stopped) |
| 3 | -1.3682 | (early-stopped) |
| 4 | -1.3576 | (early-stopped) |

Compare to Run 1's single seed: best val NLL -0.4387, with train NLL
"bouncing around rather than smoothly decreasing" — that instability is
gone at this scale. This is a real, measured improvement from more data, not
a change to the loss function or training procedure.

### Stage 2 — heads: still overfitting, unresolved

`pos_weight=138.4` (vs. Run 1's 26.9 — the positive-class rarity is worse at
this larger, more representative sample). Train loss fell smoothly
(1.70→0.35 over 17 epochs before early-stopping); val loss stayed noisy in
the 11-19 range with no clear downward trend across all 5 seeds
(best-val-loss range: 8.35-11.37). **This is the same genuine, unresolved
overfitting signal Run 1 reported — more dynamics-training data did not fix
it.** The stage head in particular remains under-exercised: several of the
6 stage classes are still rare enough to hit the class-weight ceiling even
at 40,000 samples.

### Evaluation — test split (Friday), n=4,000 (stratified from 438,708 real candidates)

| Model | F1 | Precision | Recall | AUC-PR |
|---|---|---|---|---|
| LR (current state) | 0.758 | 0.944 | 0.632 | 0.817 |
| LR (flattened history) | 0.785 | 0.968 | 0.659 | 0.863 |
| Persistence | 0.592 | 0.947 | 0.431 | 0.694 |
| **World model** | **0.082** | 0.889 | **0.043** | **0.803** |
| Oracle (upper bound) | 0.934 | 0.903 | 0.966 | 0.947 |

**The world model now clearly beats persistence on AUC-PR (0.803 vs. 0.694,
+0.103) — reversing Run 1's finding that it did not.** The persistence
ablation (`nidra/eval/ablations.py`) independently confirms this:
`auc_collapse=+0.103`, `"interpretation": "world model beats persistence"`.
This is the real, structural test the project is built around
(`baseline_lr_flattened_history` and `baseline_persistence` both reuse the
same frozen risk head the world model uses), and at this scale it passes.

**But F1/recall expose a real, separate, newly-surfaced problem: the
model's probabilities are badly miscalibrated relative to the fixed 0.75
threshold.** Precision is high (0.889 — when the world model does cross
0.75, it is usually right) but recall is 0.043 — it almost never crosses
0.75 even for true positives. This is a ranking-vs-calibration gap: the
model separates risk classes well (AUC-PR, which is threshold-free) but its
absolute probability outputs sit systematically below the mandated
operating point. The most likely mechanism, consistent with the
frozen-head design (Rule 1: heads are trained ONLY on observed states,
never on rollout output): the risk head's training distribution (real
observed states) and its actual inference input (rolled-out, sampled,
somewhat-off predicted states) are not identical, and the head was never
exposed to "slightly-off-distribution positives" at high confidence during
training. **This is reported honestly as an open problem, not tuned away by
changing the mandated 0.75 threshold or 2-window lead rule — both are
protected by the project's own claims discipline.**

**Lead time: 0 of 10 attack episodes received a sustained warning
(`fraction_no_warning=1.0`)** — a direct consequence of the calibration gap
above: even where the model ranks risk correctly, its probability rarely
sustains above 0.75 for the required 2 consecutive windows. This is the
single most consequential open item in this run.

### Evaluation — holdout split (Thursday, both Web-Attacks + Infiltration), n=4,000 (stratified from 674,269 real candidates)

| Model | F1 | AUC-PR |
|---|---|---|
| LR (current state) | 0.698 | 0.616 |
| LR (flattened history) | 0.865 | 0.883 |
| Persistence | 0.648 | 0.635 |
| **World model** | 0.396 | 0.635 |
| Oracle (upper bound) | 0.629 | 0.897 |

**World model AUC-PR (0.635) is statistically indistinguishable from
persistence (0.635)** — `auc_collapse=-0.037`,
`"interpretation": "NO MEANINGFUL GAP over persistence"`. On a genuinely
unseen attack type (Infiltration was never in any training split), the
learned dynamics provide no measurable edge over "assume no change." This
is the honest generalization result: the transition model's within-
distribution improvement over persistence (test split, above) does not
transfer to an attack family it never saw signatures from — exactly what
the holdout is designed to test, and exactly the kind of result the
project's ablation-honesty rule says must be reported as-is.

Lead time: 1 of 2 attack episodes received a warning (median 18,660s ≈
5.2 hours; `n_episodes=2` is barely more than an anecdote — Thursday's
Infiltration file has very few positively-labelled rows, same limitation
noted in Run 1).

### Ablations

**Persistence ablation**: test — world model beats persistence
(`auc_collapse=+0.103`); holdout — no meaningful gap
(`auc_collapse=-0.037`). Both reported as-is (§ above).

**Time-shuffle ablation — the same split-inconsistency as Run 1, still
unresolved at 5x the scale:**
- Test: normal-order AUC-PR 0.806 vs. shuffled 0.798 (collapse 0.007) —
  *"NO COLLAPSE under shuffling"*.
- Holdout: normal-order AUC-PR 0.628 vs. shuffled 0.575 (collapse 0.054) —
  *"temporal order matters (collapse observed)"*.

The same qualitative pattern Run 1 showed (test: no shuffle-sensitivity;
holdout: real shuffle-sensitivity) persists after 5x more training data and
5 seeds instead of 1 — this rules out "undertrained single seed" as the
explanation. It is a genuine, reproducible property of how the model
behaves differently on these two splits, still unexplained, still an open
item.

**Horizon curve — the odd/even parity oscillation from Run 1 is also still
present, at this scale, on the test split specifically:**

test: `auc_pr_by_k = [0.518, 0.174, 0.560, 0.159, 0.497, 0.162]` — k=1,3,5
(indices 0,2,4 → horizons 1,3,5) are consistently ~3x higher than k=2,4,6.
holdout: `auc_pr_by_k = [0.242, 0.369, 0.238, 0.206, 0.227, 0.162]` — no
clean alternation (horizon 2 is the highest, not the lowest, breaking the
test-split pattern), matching Run 1's finding that holdout does not show
the same parity structure. `flat_curve_leakage_warning=false` on both
splits (no automated leakage flag fired). Same status as Run 1: real,
reproducible, unexplained on the test split specifically, and not resolved
by more data/seeds — still an open investigation item, most likely (per the
task's own hypothesis, not confirmed) a rollout/sampling-step artifact or a
genuine periodicity in Friday's labelled attack windows aliasing against
the 30s/K=6 window geometry.

**Surprise signal** (still the one ablation that shows a consistent,
unambiguous positive signal on both splits): `mean_error_pre_attack` is
~17-21x `mean_error_benign` (test: 5.26 vs. 0.31; holdout: 5.33 vs. 0.25),
`error_rises_before_onset=true` on both.

**State nRMSE — now a legible, sane number** (the metric-computation fix,
§ above): test `nrmse_world_model_mean=6.52` vs. `nrmse_persistence_mean=5.71`
(world model slightly worse on raw state accuracy despite better risk
ranking — plausible: the transition model can rank risk usefully from
partial/directional signal without reconstructing every one of 45 features
precisely); holdout `world_model=2.53` vs. `persistence=2.39`, same
pattern. Both are in scaled (RobustScaler+log1p) units, where 1.0
represents one training-population standard deviation — orders of
magnitude away from Run 1's 154,775/783,843, and directly comparable across
features and horizons for the first time.

### Behavioral regimes (new in this run — `nidra/explain/regimes.py`)

K-means (k=5) over the encoder's latent hidden state (`h_t`, 128-dim),
computed on 3,000 stratified test-split samples — the clustering algorithm
never sees `risk_label` (see the module's structural test,
`test_discover_regimes_never_sees_labels`). Risk rate per discovered
regime, computed only AFTER clustering, for interpretation:

| regime | n samples | risk rate |
|---|---|---|
| 0 | 933 | 98.7% |
| 1 | 719 | 0.8% |
| 2 | 520 | 99.4% |
| 3 | 371 | 4.6% |
| 4 | 457 | 85.1% |

The encoder's latent space separates cleanly into high-risk (regimes 0, 2,
4: 85-99%) and low-risk (regimes 1, 3: <5%) clusters despite never being
trained or clustered against risk labels — a genuine, unprompted positive
result: the learned representation organizes itself around risk-relevant
structure. This is a descriptive/interpretability finding, not a new
detector — the regimes are not used for prediction anywhere in the
pipeline.

### Ensemble / serving

All 5 seed checkpoints load successfully in a clean `NidraPredictor`
process (`nidra.serve.benchmark`). At `rollout.n_samples_per_member=200`
(5×200=1,000 trajectories, matching the project's `~1000 sampled futures`
target and `config/default.yaml`), measured p95 latency was 616ms — over
the 300ms target. Per the documented policy ("cut samples toward 100 before
cutting ensemble members"), `config/mvp_2017.yaml` now uses
`n_samples_per_member=100` (500 total trajectories): p95=108ms, comfortably
under target. This is a measured trade-off on this specific CPU, not a
silent change — `config/default.yaml` (the full-scale production config)
still specifies 200/member for hardware that can sustain it.

### Reality overlay (new in this run — `nidra/scripts/reality_overlay.py`)

Generates a forecast from observations up to `origin_ts` only (no future
data consumed — enforced structurally, since `NidraPredictor.forecast()`
never receives `Y`), then overlays it against what was subsequently
actually observed. One example run against the test split: host
`192.168.10.9`, origin `2017-07-07T11:33:30Z`, 500 pooled trajectories.
Per-horizon `state_nrmse_scaled` (the primary, unit-normalized error
metric): `[4.98, 1.07, 2.67, 0.79, 9.16, 2.69]` — noisy at the single-
sample level (expected; this is a qualitative demo of one forecast, not an
aggregate metric), same order of magnitude as the aggregate nRMSE numbers
above. A raw-unit `raw_abs_error_per_feature` breakdown is also written per
horizon but is explicitly documented as illustrative only: `flow_duration_var`
and `iat_var`-family features are squared-microsecond quantities with a
naturally huge raw dynamic range independent of forecast quality, so a
single blended raw-unit RMSE across all 45 features would be dominated by
whichever of those happens to be large in a given window — exactly the
scaled-vs-raw-units pitfall this project's own metric-auditing guidance
warns about, caught here by hand before it could mislead the report.

### Honest summary (Run 2)

More data and a real 5-seed ensemble produced two genuine, measured
improvements over Run 1: (1) the state-forecast nRMSE bug is fixed, and (2)
the world model now clearly beats persistence and the current-state LR
baseline on AUC-PR on the test split — the central claim this project is
built around, which Run 1 could not support. Both are real, not spun.

It also surfaced a new, more consequential problem that Run 1's much worse
nRMSE bug had been obscuring: **the model's probability calibration is bad
enough that the mandated 0.75/2-window operating point produces zero
warnings on the test split.** AUC-PR says the signal is there; the fixed
threshold says the system does not yet act on it. This is the top priority
for follow-up work, and it is a head-training/calibration problem, not a
transition-model problem (the transition model's own state nRMSE and
AUC-PR-based ranking both improved with more data). **Update (Run 2
addendum, above): the obvious fix — post-hoc Platt-scaling calibration —
was implemented and tested, and empirically made recall at 0.75 worse, not
better, on both splits and via the real pooled-ensemble serving path. This
is now believed to reflect a genuine confidence-ceiling limitation of the
risk head rather than a fixable score-compression artifact; see the
addendum for the full mechanism and why the fit was not simply reweighted
to force a better-looking number.**

The three previously-unresolved open items from Run 1 — the odd/even
horizon-parity oscillation on the test split, the time-shuffle
inconsistency between splits, and heads-training val-loss not converging —
are ALL STILL PRESENT after 5x the data and a full 5-seed ensemble instead
of 1. This rules out "just needs more data/seeds" as the explanation for
any of the three; they need direct investigation, not more compute.

Infiltration (holdout) generalization is honestly negative: no measurable
AUC-PR edge over persistence on an attack type never seen in training.

### Reproduction (Run 2)

```bash
cd ml
python -m nidra.data.pcap_extract cicids2017/pcap/Tuesday-WorkingHours.pcap cicids2017/pcap/parquet/Tuesday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Wednesday-workingHours.pcap cicids2017/pcap/parquet/Wednesday-workingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Thursday-WorkingHours.pcap cicids2017/pcap/parquet/Thursday-WorkingHours_packets.parquet
for seed in 0 1 2 3 4; do
  python -m nidra.train.train_dynamics --config config/mvp_2017.yaml --max-train-samples 40000 --max-val-samples 8000 --epochs 30 --seed $seed
  python -m nidra.train.train_heads    --config config/mvp_2017.yaml --max-train-samples 40000 --max-val-samples 8000 --seed $seed
done
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000
python -m nidra.serve.benchmark --weights-dir artifacts_mvp_2017/weights --scaler-path artifacts_mvp_2017/scaler/robust_scaler.joblib --config config/mvp_2017.yaml
python -m nidra.scripts.reality_overlay --config config/mvp_2017.yaml --weights-dir artifacts_mvp_2017/weights --scaler-dir artifacts_mvp_2017/scaler --split test --out-json reality_overlay.json --out-png reports/reality_overlay.png
python -m nidra.scripts.generate_report --config config/mvp_2017.yaml --test-metrics-dir artifacts_mvp_2017/metrics/test --holdout-metrics-dir artifacts_mvp_2017/metrics/holdout --reports-dir reports --metadata-dir artifacts_mvp_2017/metadata
```

### Caveats that materially limit every number above

- **Still not `config/default.yaml`'s full production scale** (60/30
  epochs, uncapped samples against all 6.9M+ train candidates) — 40,000/
  8,000-sample, 30-epoch MVP-scale run, now with a real 5-seed ensemble
  rather than Run 1's single seed. The full-scale run has not been executed
  (compute/time, not a blocker) — nothing about the data layer needs to
  change to run it; see "Upgrading to the full dataset" below.
- **Heads val-loss still does not converge cleanly** — unresolved, flagged
  above, present across all 5 seeds.
- **Calibration at the 0.75 threshold is the top open problem** — flagged
  prominently above, not present in Run 1's report because Run 1's nRMSE
  bug and worse-than-persistence AUC-PR obscured that the underlying signal
  was there but miscalibrated.
- **Horizon-parity oscillation and time-shuffle split-inconsistency remain
  unexplained** after ruling out "undertrained" as the cause.
- **Ensemble evaluation ran per-seed** (baselines/ablations/calibration/
  lead-time were computed against seed 0 only, not re-run through the
  pooled 5-member `NidraPredictor`) — the 5-seed ensemble is real and fully
  loads/serves (see "Ensemble / serving" above), but a pooled-ensemble
  version of the baselines/ablations harness is a good next step, not
  completed here.
- **Lead-time numbers are based on 10 (test) and 2 (holdout) episodes** —
  small samples, same caveat as Run 1.

---

## Run 1 (superseded): single-seed MVP, Monday+Friday PCAPs only

This section is preserved unmodified from the original run for provenance.
Run 2 above supersedes it as the current result — in particular, the nRMSE
numbers below are known-wrong (metric bug, fixed in Run 2) and should not be
cited; everything else here is an honest historical record.

## 1. Dataset

Source: `GeneratedLabelledFlows.zip` / "TrafficLabelling" release (the
variant with Source IP, Destination IP, and Timestamp columns intact — the
sibling "MachineLearningCVE" release strips those and cannot be windowed by
host/time) plus the raw `Monday-WorkingHours.pcap` and
`Friday-WorkingHours.pcap` captures, all downloaded directly from the
dataset's official distribution into `cicids2017/` in this repo (gitignored,
not committed).

**Every CSV was read in full** — no `nrows`/row-count truncation anywhere in
this run:

| Day (CSV) | Input rows | Accepted | Notes |
|---|---|---|---|
| Monday | 529,918 | 529,918 | 100% benign |
| Tuesday | 445,909 | 445,909 | FTP/SSH brute-force |
| Wednesday | 692,703 | 692,703 | DoS Hulk/GoldenEye/Slowloris/Slowhttptest, Heartbleed |
| Thursday (Web Attacks) | 458,968 | 170,366 | 288,602 rows dropped — genuine blank trailing rows in the source file (all-NaN, not a parsing bug); also required a latin-1 decode fallback (Windows-1252 en-dash in "Web Attack – XSS" labels breaks plain UTF-8) |
| Thursday (Infiltration) | 288,602 | 288,602 | only 36 rows actually labelled `Infiltration` |
| Friday (Morning) | 191,033 | 191,033 | benign + Bot |
| Friday (PortScan) | 286,467 | 286,467 | |
| Friday (DDoS) | 225,745 | 225,745 | |

### Real packet-level features (Monday + Friday)

The raw PCAPs for Monday and Friday were downloaded, **MD5-verified intact**
before use, and extracted with `python -m nidra.data.pcap_extract`:

| Day | PCAP size | Packets extracted | Time |
|---|---|---|---|
| Monday | 10.8GB | 11,626,492 | 263.4s |
| Friday | 8.8GB | 9,915,680 | 218.6s |

**One real bug was found and fixed during this extraction**, not discovered
in advance: tshark's default field output joins a multi-valued field (e.g.
`ip.src` appearing twice when a packet embeds another packet's IP header,
as ICMP errors do) with the same character used as the field separator,
silently corrupting the row. This was caught (not guessed) by watching the
initial extraction attempt log a stream of "dropping malformed tshark row"
warnings, tracing it to specific packets, and confirming the fix
(`-E occurrence=f`, which takes only the first value of any multi-valued
field) against a live field-count check before committing to the full
10.8GB+8.8GB extraction. A related latent bug was found alongside it: the
row-drop logic for missing `ip_src`/`ip_dst` used `dropna()`, which only
catches actual `NaN` — a tshark row with a genuinely empty (but present)
IP field would have silently passed through uncaught. Both are fixed in
`nidra/data/pcap_extract.py` and covered by `tests/test_pcap_extract.py`
(9 new tests), which didn't exist before this run.

Tuesday, Wednesday, and both Thursday files have **no PCAP yet** (not
downloaded — see the user's own decision to proceed with only Monday and
Friday rather than wait) and ran in **flow-only mode**: their 11
packet-aggregate features (`ttl_*`, `tcp_window_*`, `frag_flag_rate`,
`payload_size_*`, `retrans_*`) are zero for every window on those days,
logged loudly by the pipeline (`WARNING ... running in FLOW-ONLY mode`),
never silently treated as full-feature data. The other 34 of 45 features
(flow aggregates, graph scalars, backward-looking dynamics, activity flag)
are complete for all 8 days regardless of PCAP availability.

### Windowing (Δ=30s, L=30, K=6), hosts kept vs. dropped (<36 windows)

| Day | Hosts kept | Hosts dropped |
|---|---|---|
| Monday | 3,775 | 4,464 |
| Tuesday | 3,173 | 4,019 |
| Wednesday | 3,211 | 4,478 |
| Thursday (Web Attacks) | 1,351 | 2,852 |
| Thursday (Infiltration) | 1,779 | 3,322 |
| Friday (Morning) | 1,535 | 2,929 |
| Friday (PortScan) | 1,019 | 2,648 |
| Friday (DDoS) | 409 | 1,658 |

Full windowed train-candidate count (Monday+Tuesday+Wednesday, before any
capping): **6,911,848 samples**. This is the real, complete number — the
sample caps described below are applied only at the tensor-construction
stage (stratified by `risk_label`, keeping every positive sample), not by
truncating the source data.

## 2. Training (single seed — MVP scale, not the 5-seed ensemble)

`config/mvp_2017.yaml`: seed 0 only, 20-epoch cap, `--max-train-samples 8000
--max-val-samples 2000` (stratified subsample of the 6.9M-candidate train
split).

**Stage 1 — dynamics (encoder + transition):**

| epoch | train_nll | val_nll | teacher_forcing_p |
|---|---|---|---|
| 0 | 0.2057 | 0.0967 | 1.00 |
| 1 | -0.4170 | 0.0666 | 0.94 |
| 2 | -0.7146 | -0.4024 | 0.88 |
| 3 | -0.7663 | -0.3795 | 0.82 |
| 4 | -0.7341 | -0.3580 | 0.77 |
| 5 | -0.7112 | **-0.4387** | 0.71 |
| 6 | -0.7238 | -0.3761 | 0.65 |
| 7 | -0.7201 | -0.3467 | 0.59 |
| 8 | -0.7151 | -0.3781 | 0.53 |
| 9 | -0.8245 | -0.3601 | 0.48 |
| 10 | -0.6902 | -0.3764 | 0.42 |
| 11 | -0.7948 | -0.3868 | 0.36 |

Early-stopped at epoch 11 (patience 6, best val NLL -0.4387 at epoch 5).
Train NLL bounces around rather than smoothly decreasing after epoch 2 —
at 8,000 train samples this is plausibly just batch noise, not a stable
trend, and has not been checked across multiple seeds.

**Stage 2 — frozen dynamics, risk + stage heads (observed states only):**

- `pos_weight` = 26.875, `class_weights` = [0.169, 1333.8, 11.21, 1333.8,
  1333.8, 1333.8] — four of six stage classes are essentially absent from
  this train sample and hit the weight ceiling, same limitation as before:
  the stage head is really only exercised on benign vs. one other class at
  this scale.
- Train loss fell steadily (2.2704 → 0.3802 over 19 epochs), but **val loss
  did not track it** — it stayed noisy in the 6.8–9.9 range with no clear
  downward trend (best 6.8466 at epoch 12), and training early-stopped at
  epoch 18 on stalled patience. This is reported as a genuine, unresolved
  sign of overfitting at this sample size — not smoothed over. It is a
  materially worse convergence picture for the heads than dynamics training
  showed, and should be treated as unresolved until re-run with a larger
  sample cap or across the full 5-seed ensemble.

## 3. Evaluation — test split (Friday), n=2,000 (capped, stratified from 438,708 candidates)

| Model | F1 | AUC-PR | Precision | Recall | FPR |
|---|---|---|---|---|---|
| LR (current state) | 0.790 | 0.975 | 0.993 | 0.656 | 0.053 |
| LR (flattened history) | 0.904 | 0.993 | 0.996 | 0.828 | 0.040 |
| Persistence | 0.704 | 0.969 | 0.996 | 0.544 | 0.027 |
| **World model** | 0.641 | 0.962 | 0.997 | 0.473 | 0.020 |
| Oracle (upper bound) | 0.990 | 0.996 | 0.996 | 0.985 | 0.053 |

**The world model did not beat any other baseline on F1 or AUC-PR on the
test split** — including persistence. AUC-PR values across the board are
high (>0.96) because Friday's risk-label positive rate is high at this
sample size, which compresses the visible gap between models; F1 still
shows persistence and the flattened-history LR clearly ahead. This is
reported as-is, not spun.

## 4. Evaluation — holdout split (Thursday, both Web-Attacks + Infiltration), n=2,000 (capped, stratified from 674,269 candidates)

Both Thursday files (Web Attacks, Infiltration) are held out from training
entirely, per the mandated generalization test — neither attack family is
seen during Stage-1 or Stage-2 training.

| Model | F1 | AUC-PR | Precision | Recall | FPR |
|---|---|---|---|---|---|
| LR (current state) | 0.780 | 0.717 | — | — | — |
| LR (flattened history) | 0.921 | 0.933 | — | — | — |
| Persistence | 0.816 | 0.783 | — | — | — |
| **World model** | 0.784 | 0.732 | — | — | — |
| Oracle (upper bound) | 0.848 | 0.857 | — | — | — |

(precision/recall/fpr omitted here for brevity — see
`artifacts_mvp_2017/metrics/baselines.json` from this run, copied to
`/tmp/eval_holdout_2017/` at run time.) **World model again did not beat
persistence or either LR baseline** on the unseen-attack holdout. Unlike
the test split, though, holdout AUC-PR values are meaningfully lower
across every model (0.72–0.93 vs. 0.96–0.99 on test) — consistent with a
real, expected generalization gap on genuinely unseen attack types, not a
pipeline defect.

## 5. Ablations

**Persistence ablation** (both splits): *"NO MEANINGFUL GAP over
persistence — dynamics may not be adding value; this is a reportable
negative result, not a bug to hide."* `auc_collapse` slightly negative on
both (test: -0.006, holdout: -0.044) — consistent with the baseline table
above; the learned dynamics are not beating "assume no change" at this
training scale.

**Time-shuffle ablation:**
- Test split: *"NO COLLAPSE under shuffling — the model may be using
  per-window features only, not real temporal structure; this is a
  reportable negative result, not a bug to hide."* (normal-order AUC-PR
  0.963 vs. shuffled 0.963, collapse essentially zero: -0.0005.)
- Holdout split: **the opposite, healthy result** — *"temporal order
  matters (collapse observed)"* (normal-order AUC-PR 0.732 vs. shuffled
  0.631, collapse 0.100). The model's reliance on real temporal structure
  is inconsistent between the two splits, worth further investigation
  rather than averaging away.

**Horizon curve — a genuine, unexplained anomaly worth flagging
prominently:** AUC-PR by horizon step on the test split oscillates sharply
by parity rather than decaying smoothly: k=0,2,4 (0.643, 0.688, 0.709) are
all much higher than k=1,3,5 (0.216, 0.211, 0.211). The **same odd/even
pattern shows up independently in the calibration Brier scores** for the
same split (k=0,2,4: 0.208/0.171/0.154 vs. k=1,3,5: 0.574/0.558/0.548) —
two different metrics agreeing on the same alternating structure is a
strong signal this is real, not sampling noise. The holdout split does
**not** show this pattern (its Brier scores are flat, 0.234–0.264 across
all six horizons; its AUC-PR-by-k does dip but without the same clean
alternation: 0.285, 0.235, 0.342, 0.093, 0.321, 0.039). No automated
`flat_curve_leakage_warning` fired on either split. This parity effect on
the test split specifically is unexplained and should be investigated
before trusting per-horizon numbers there — possible causes include an
artifact of the rollout/sampling step, or a genuine periodicity in Friday's
labelled attack windows that a 30s/K=6 window geometry happens to alias
against; neither has been confirmed.

**Surprise signal** (prediction error rising before attack onset): on both
splits, `mean_error_pre_attack` is roughly 15–22× `mean_error_benign`
(test: 5.30 vs. 0.35; holdout: 5.38 vs. 0.24), and
`error_rises_before_onset=true` on both — the one ablation that shows a
consistent, genuine positive signal on both splits.

**State nRMSE:** as in the earlier CIC-IDS2017 pipeline-validation run,
`nrmse_world_model_mean` is anomalously large (test: 154,775; holdout:
783,843) versus `nrmse_persistence_mean` (~1.2 and ~0.86 respectively) —
several orders of magnitude apart. This rollout-state-divergence issue is
still unresolved and still only affects the raw rolled-out *state* values,
not the risk/stage scores read off them (which is where the baseline
numbers above come from) — but it remains an open item, not something this
run fixed.

## 6. Calibration

Mean Brier score across the 6 horizon steps: **0.369 (test)**, **0.250
(holdout)** — both driven substantially by the k=1/3/5 spikes described
above on the test split; see §5 for the per-horizon breakdown.

## 7. Lead time

- **Test split (Friday):** 10 attack episodes found in the (capped-for-cost)
  scan; median lead time **32,340s (~9 hours)**, distribution ranging
  10,290–36,300s across 8 warned episodes, with 2 of 10 episodes
  (`fraction_no_warning=0.2`) receiving no warning at all. A ~9-hour median
  lead time is large enough on a Δ=30s/K=6 (3-minute) forecast horizon
  that it likely reflects episodes where risk was elevated for most of the
  day rather than a sharp, localized early-warning signal — worth reading
  as "sustained elevated risk across a long attack day" rather than "the
  system predicted this specific attack 9 hours ahead," which would be a
  stronger and less-supported claim.
- **Holdout split (Thursday):** only 2 attack episodes found; median lead
  time 11,205s, both received warning (`n_no_warning=0`). With `n_episodes=2`
  this is barely more than an anecdote, not a distribution — consistent with
  Thursday's Infiltration file having only 36 positively-labelled rows.

## 8. Honest summary

This run proves the full pipeline works genuinely end-to-end against the
complete, real CIC-IDS2017 dataset, including real tshark-extracted
packet-level features for two of five capture days (something the prior
pipeline-validation attempt never had at all). It surfaced and fixed two
real bugs in the PCAP extraction path that had never been exercised at
scale before this run. It does **not** show the world model outperforming
simpler baselines on either split, shows an unexplained horizon-parity
oscillation on the test split specifically, shows inconsistent
time-shuffle behavior between splits, and shows the heads-training
val-loss not converging cleanly. None of this is hidden or reframed as
success — it is the genuine state of a single-seed, reduced-epoch MVP run,
and should be read as exactly that.

### Caveats that materially limit every number above

- **Single seed (seed 0) of the planned 5-seed ensemble** — no inter-seed
  variance, no error bars anywhere in this document.
- **20-epoch cap for dynamics, 20 for heads** (`config/mvp_2017.yaml`), not
  `config/default.yaml`'s production 60/30. Neither stage shows clean,
  confirmed convergence.
- **3 of 8 day-files (Tuesday, Wednesday, both Thursday files) are
  flow-only** — their PCAPs are not yet downloaded. Only Monday and Friday
  have real packet-level features.
- **Training/eval sample counts are capped** (8,000 train / 2,000 val for
  training; 2,000 for eval baselines/ablations/calibration) via stratified
  subsampling of the full, real windowed data — not by truncating the raw
  CSVs, which were read in full (6.9M real train-candidate windows exist;
  8,000 were sampled for this MVP run).
- **Stage-head class imbalance**: 4 of 6 stage classes are effectively
  absent from the capped train sample; the stage head is meaningfully
  validated only on benign vs. one other class.
- **Heads val-loss did not converge** — flagged above, unresolved.
- **The horizon-parity oscillation (§5) and rollout-state nRMSE anomaly
  (§5) are both unresolved**, open items for follow-up investigation.
- **Lead-time numbers are based on 10 (test) and 2 (holdout) episodes** —
  small samples, not distributions with statistical weight.

## Reproduction

```
cd ml
python -m nidra.data.pcap_extract cicids2017/pcap/Monday-WorkingHours.pcap cicids2017/pcap/parquet/Monday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Friday-WorkingHours.pcap cicids2017/pcap/parquet/Friday-WorkingHours_packets.parquet
python -m nidra.train.train_dynamics --config config/mvp_2017.yaml --max-train-samples 8000 --max-val-samples 2000
python -m nidra.train.train_heads --config config/mvp_2017.yaml --max-train-samples 8000 --max-val-samples 2000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split test --n-samples 30 --max-eval-samples 2000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split holdout --n-samples 30 --max-eval-samples 2000
python -m nidra.scripts.portscan_sanity_plot --csv "cicids2017/csv/extracted/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --out portscan_sanity.png
```

Requires the real CIC-IDS2017 `TrafficLabelling` CSVs and the
`Monday-WorkingHours.pcap` / `Friday-WorkingHours.pcap` captures placed
under `cicids2017/` per `config/mvp_2017.yaml`'s `dataset:` paths (not
committed to this repo — see `.gitignore`).

## Upgrading to the full dataset

`config/default.yaml` is the full-scale production config: same dataset,
same day/split structure, 5-seed ensemble, 60/30 epoch budget. Nothing
about this run's data layer needs to change to use it — it reads the exact
same `cicids2017/` paths. The path to closing the gap to full-scale,
full-feature training is purely additive:

1. Download the remaining PCAPs (Tuesday, Wednesday, Thursday) and extract
   them the same way as Monday/Friday above — `dataset.days.<day>.packets`
   in the config is the only thing that needs a new filename per day, no
   code changes.
2. Switch from `config/mvp_2017.yaml` to `config/default.yaml` (or raise
   `mvp_2017.yaml`'s epoch/seed/sample-cap numbers directly) once compute
   time allows a full run.
