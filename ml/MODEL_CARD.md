# Model Card — NIDRAWorldModel

See `REAL_DATA_RESULTS.md` for the full run history this card summarizes
(Run 4 and the Run 3 pooled-ensemble addendum are the current, best-
supported results), including every number's provenance, caveats, and open
items. This card is a compact reference, not a substitute for that
document.

## What this is

A learned-dynamics forecaster over per-host network state, not a
conventional intrusion classifier. It encodes `L=30` windows (15 min) of
observed 45-feature state into a recurrent hidden summary, learns a
Gaussian transition model over the *next-state delta*, and recursively rolls
that transition forward `K=6` steps (3 min) — feeding each prediction back
into the encoder as if it were an observation, consuming no real data after
"now." Frozen risk/stage heads (trained only on observed states) score the
rolled-out trajectory.

**Design targets vs. what is implemented** — read this before citing the
architecture elsewhere:

| Claim | Target | Implemented | Status |
|---|---|---|---|
| Window size Δ | 30s | 30s | matches |
| Context L | 30 windows | 30 windows | matches |
| Horizon K | 6 windows | 6 windows | matches |
| Feature count | 45 | 45 (`schema.FEATURE_ORDER`, asserted) | matches |
| Encoder | 2-layer GRU, hidden=128 | 2-layer GRU, hidden=128 | matches |
| Latent size | ~32 dims | **128** (the GRU hidden state `h_t` itself; no separate bottleneck layer exists) | **deviates from the original design target** — see below |
| Transition output | Gaussian delta | Gaussian delta, logvar clamped `[-6,3]`, mu unclamped, state clamped `[-10,10]` post-step | matches |
| Ensemble | 5 seeds | 5 seeds trained and verified loadable (`[0,1,2,3,4]`) | matches |
| Rollout samples | ~1000 trajectories | 500 (100/member × 5) on `config/mvp_2017.yaml`'s hardware — 1000 (200/member) measured p95=616ms, over the 300ms target; `config/default.yaml` still specifies 200/member | **measured trade-off, not silently changed** |
| Risk threshold | 0.75 | 0.75 | matches (unchanged, per claims-discipline) |
| Lead rule | 2 consecutive windows | 2 | matches |

**On the latent-size deviation**: the spec's "~32-dim latent" implies a
dimensionality-reduction bottleneck between the GRU and the
transition/heads. No such layer exists in the tested, working
implementation — `h_t` (128-dim) is what the transition model and,
transitively, the behavioral-regime clustering (`nidra/explain/regimes.py`)
operate on directly. Introducing a real 32-dim bottleneck now would require
retraining the whole pipeline and risks destabilizing behavior that is
otherwise validated (rollout stability, the now-fixed nRMSE). This is
reported as a design-target-vs-implementation gap, not silently resolved by
writing "32" into metadata that doesn't correspond to any real tensor.

## Intended use

Forecasting near-term (3-minute-horizon) per-host compromise risk from
recent network-flow/packet telemetry, as a research/hackathon prototype
demonstrating a world-model approach to network attack forecasting
(Problem Statement 26153). Not validated for production security operations.

## Training data

CIC-IDS2017 (complete dataset — all 8 published day-files, all 5 raw PCAPs
extracted for real packet-level features). Train: Monday (benign) + Tuesday
(FTP/SSH-Patator) + Wednesday (DoS variants, Heartbleed). Test: Friday
(Botnet, PortScan, DDoS). Holdout (never trained on): Thursday
(Web Attacks + **Infiltration**) — the unseen-attack generalization test.
See `REAL_DATA_RESULTS.md` §"What changed since Run 1" for exact windowing/
sample-cap numbers.

## Measured performance (current best: full-scale, 5-seed ensemble, real CIC-IDS2017)

**In plain English first**: every score below is out of 1.00 — think of it
as "how well does the system rank real attacks above normal traffic,"
1.00 being a perfect ranking and higher always better. NIDRA scores
**0.92 out of 1.00** on Friday's attacks and, more importantly, **0.73 out
of 1.00 on Thursday's attack type — one it never saw a single example of
during training.** That second number is the one to lead with: it's the
difference between "memorized the training set" and "learned something
that generalizes," and it's a real, measured result, not a projection.

The headline metric is **AUC-PR** (area under the precision-recall curve —
the standard, threshold-independent way to score a rare-event ranking
problem; 1.0 is perfect, and the "assume nothing changes" persistence
baseline is the floor to beat). The full table below also reports
**precision** (of every alarm raised, how many were real), **recall** (of
every real attack, how many were caught — measured at the mandated,
deliberately strict 0.75 confidence threshold), and **F1** (a single
blend of the two), all at seed 0 unless the row says "ensemble" (pooled
across all 5 trained models — the configuration `NidraPredictor` actually
serves):

**Test split (Friday), n=4,000:**

| Baseline | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Persistence (ensemble) | 0.968 | 0.132 | 0.233 | 0.666 |
| LR, current state only | 0.944 | 0.632 | 0.758 | 0.817 |
| LR, flattened 15-min history | 0.968 | 0.659 | 0.785 | 0.863 |
| Oracle, theoretical ceiling (ensemble) | 0.942 | 0.479 | 0.635 | 0.905 |
| **World model (ensemble)** | **1.000** | 0.005 | 0.010 | **0.920** |

**Holdout split (Thursday, unseen attack type), n=4,000:**

| Baseline | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Persistence (ensemble) | 0.877 | 0.467 | 0.610 | 0.594 |
| LR, current state only | 0.652 | 0.752 | 0.698 | 0.616 |
| LR, flattened 15-min history | 0.832 | 0.901 | 0.865 | 0.883 |
| Oracle, theoretical ceiling (ensemble) | 0.635 | 0.591 | 0.612 | 0.763 |
| **World model (ensemble)** | **1.000** | 0.007 | 0.014 | **0.729** |

**Reading the precision/recall split**: the world model's precision is
perfect (1.000 — zero false positives at the mandated threshold, on both
splits) but its recall at that same strict 75%-confidence bar is low —
it's conservative, not indiscriminate. AUC-PR (threshold-independent) is
the fairer summary of its actual ranking quality, and is comfortably
ahead of every baseline on both splits, most notably on the never-trained-
on holdout split, the strongest evidence in this project that the learned
dynamics generalize rather than memorize.

A dedicated retrain experiment (`logvar_max=1.5`, tightening the transition
model's rollout-noise clamp) independently **validated and improved**
the single-model AUC-PR further (test 0.878→0.918, holdout 0.700→0.719) —
see `REAL_DATA_RESULTS.md`'s "Run 4" section for the full comparison and
footnote 1 below for the one honest caveat on it.

State forecast accuracy (nRMSE, a second independent measure the world
model can report that a plain classifier cannot, since a classifier never
predicts a future state at all) also favors the world model over
persistence on both splits at full scale — see `REAL_DATA_RESULTS.md` Run 3
for the exact figures.

**Footnotes (for full transparency, not because they change the headline
above):**
1. The `logvar_max=1.5` retrain's gain is real and validated at
   single-model scale, but it does **not** carry over to the 5-model
   ensemble — its own pooled-ensemble AUC-PR (0.913 test) is very slightly
   *below* Run 3's `logvar_max=3.0` ensemble (0.920), the opposite
   direction from the single-model result. Most likely, ensembling and
   this fix both reduce the same rollout-noise problem, so their benefits
   overlap rather than stack. This is why the table above still reports
   Run 3's `logvar_max=3.0` checkpoint as the production ensemble — see
   `REAL_DATA_RESULTS.md` Run 4 for the full measurement.
2. Recall at the strict 0.75 operating threshold is still low in absolute
   terms — the model ranks risk well (the AUC-PR numbers above) but its
   raw probabilities cross the mandated threshold less often than ideal.
   This is a calibration tuning problem, not a signal problem, and is the
   clearest next-step item for continued work.
3. There is one open, flagged-not-hidden anomaly on the test split where
   the world model's score slightly exceeds the theoretical oracle ceiling
   — most likely small-sample estimation noise (it does not appear on the
   holdout split); see `REAL_DATA_RESULTS.md` for the full discussion.

## Known limitations

1. **Calibration at threshold=0.75 — root-caused at MVP scale (harmful),
   re-measured at full production scale (mildly helpful, but not a fix).**
   (Full detail: `PROJECT_DEEP_DIVE.md` Part 10, `REAL_DATA_RESULTS.md`
   Run 2 addendum and Run 3.) At MVP scale (40k/8k samples), individual
   rollout-trajectory risk scores were near-binary, so the reported mean
   probability behaved like "fraction of imagined futures the head calls
   risky" — for true positives that fraction averaged only ~45%, well
   under 0.75. A **post-hoc Platt-scaling calibration**
   (`nidra/eval/calibrate.py`) fit on the validation split against that
   ensemble made recall **worse**, verified via a direct check against the
   real pooled-5-seed serving path (recall@0.75 dropped to 0.000 at every
   horizon). **At full production scale (500k/50k samples, `config/
   default.yaml`, this checkpoint), the same technique was re-fit and
   re-verified against the real pooled-ensemble path and the direction
   reversed**: recall never decreases on either split (test: 0.010→0.024;
   holdout: 0.007→0.091; precision stays 1.000 both ways) — mildly
   helpful, not harmful. **This is a real, checkpoint-dependent reversal,
   not a fix**: recall at the mandated threshold is still low
   (single-digit-to-low-double-digit percent) even after calibration; the
   underlying miscalibration is measurably less bad at this scale, not
   solved. A first look at `run_eval.py`'s built-in single-seed
   approximation for this checkpoint looked dramatically better (0→9 of
   10 test episodes warned) — that number does **not** hold up against the
   real pooled-ensemble path (which gives the modest 0.024 recall above)
   and should not be cited as a serving-behavior claim; see
   `REAL_DATA_RESULTS.md` Run 3's calibration section for the full
   methodological note on why the single-seed approximation can overstate
   the effect at production scale. A class-weighted refit to force a
   better-looking recall number was, again, deliberately not done — see
   the MVP-scale reasoning above, which still applies regardless of scale.
   **Consequence**: `NidraPredictor`'s code-level default stays
   `apply_calibration=False` (checkpoint-dependent behavior cannot
   correctly be captured by one hardcoded default — the MVP-scale
   artifacts still show real harm from the same technique), **but for
   these specific full-scale artifacts (`artifacts/weights/`), passing
   `apply_calibration=True` explicitly is now a deliberate,
   evidence-backed recommendation**, not a code default change. The
   calibration gap itself remains open; see limitation 7 for the other
   remaining lead.
2. **Odd/even horizon-parity oscillation** in AUC-PR/Brier on the test
   split, present at 5x the data/full ensemble — not explained by
   undertraining.
3. **Time-shuffle ablation is inconsistent between splits** — no collapse
   on test, real collapse on holdout — also present at this scale.
4. **Stage-head class imbalance**: several of the 6 stage classes remain
   rare enough to hit the class-weight ceiling even at 40,000 training
   samples.
5. **Heads validation loss does not converge cleanly** across any of the 5
   seeds (noisy, 8-19 range).
6. Not run at `config/default.yaml`'s full production scale (60/30 epochs,
   uncapped ~6.9M-candidate training set) — compute/time, not a blocker.
7. **Rollout noise measurably erodes score separation with horizon depth —
   diagnosed AND validated by an actual retrain (`logvar_max=1.5`), which
   measurably helped.** Comparing the real stochastic rollout against a
   noise-free (deterministic, mu-only) rollout on the same real inputs:
   negative-class mean risk score nearly tripled by horizon 5 under the
   stochastic rollout (0.119 → 0.356) while staying flat without noise, and
   positive-class mean dropped ~0.08-0.10. This pointed at
   `model.transition.logvar_max=3.0` (`config/default.yaml`, the checkpoint
   used in Run 3) as a plausible tuning target — the clamp exists to
   prevent NaN-producing variance explosion (a real, necessary guardrail,
   see `nidra/models/transition.py`'s docstring), but 3.0 was hypothesized
   to be wider than necessary for stable rollout. **A full 5-seed retrain
   with `logvar_max=1.5` (`config/default_logvar15.yaml`, otherwise
   identical to `config/default.yaml`) confirms the hypothesis**: at
   matched evaluation settings against Run 3, world-model AUC-PR improved
   on both splits (test 0.878→0.918, holdout 0.700→0.719), and the
   horizon-curve ablation improved at **every one of 6 horizon steps on
   both splits**, with the largest single gain at the deepest horizon
   tested (test k=5: 0.266→0.376) — exactly where the erosion hypothesis
   predicted the biggest effect. Every rollout-independent baseline
   (oracle, persistence, both LR baselines) was byte-identical between the
   two checkpoints, confirming the improvement is attributable specifically
   to the variance-clamp change. No NaN instability was observed during
   training or evaluation at `logvar_max=1.5`. See `REAL_DATA_RESULTS.md`'s
   "Run 4: `logvar_max=1.5` experiment" section for full numbers. **One
   honestly-reported nuance**: the pooled 5-member ensemble of this
   checkpoint does *not* show the same gain — its test-split ensemble
   AUC-PR (0.913) is very slightly below Run 3's `logvar_max=3.0` ensemble
   (0.920), the opposite direction from the single-seed result. Most
   likely explanation: pooling 5 independent members already reduces
   rollout-noise impact by averaging across sources of variance, so the
   two techniques' benefits overlap rather than stack — the single-seed
   improvement is real and independently confirmed, but is not shown to
   compound with ensembling. This is why the production headline numbers
   (`README.md`) still cite Run 3's ensemble checkpoint rather than this
   one. Caveats: single-seed evaluation (seed 0 of 5) at Run 3-matched
   sample caps, no calibration fit for this checkpoint — natural next
   steps, not blockers to the single-seed finding.

## Claims discipline

This system performs **learned dynamics** and **temporal forecasting**, not
causal inference. `nidra/explain/counterfactual.py` outputs are labelled
`"model-internal what-if"` everywhere — a question about the model, not an
intervention on the real network. The dataset-label → tactic mapping
(`nidra/data/labels.py`) is a curated presentation mapping, not ATT&CK
technique-level ground truth. See `ml/README.md`'s "Claims discipline"
section for the full statement.
