# Model Card — NIDRAWorldModel

See `REAL_DATA_RESULTS.md` §Run 2 for the full run this card summarizes,
including every number's provenance, caveats, and open items. This card is
a compact reference, not a substitute for that document.

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

## Measured performance (Run 2, seed 0 unless noted; see full doc for all 5 seeds' training convergence)

- Test split: world model AUC-PR 0.803, beats persistence (0.694) by
  +0.103 — the project's central structural claim, supported at this scale.
- Test split F1/recall at threshold=0.75: **0.082 F1, 0.043 recall** —
  the model ranks risk well but is badly miscalibrated at the mandated
  operating point. **This is the top open problem**, not hidden here.
- Holdout (Infiltration) split: world model AUC-PR 0.635 ≈ persistence
  (0.635) — no measurable generalization edge on a genuinely unseen attack
  type. Honest negative result.
- State forecast nRMSE (scaled units, now numerically sane after the metric
  fix — see `REAL_DATA_RESULTS.md`): test 6.52 (world model) vs. 5.71
  (persistence); holdout 2.53 vs. 2.39.

## Known limitations (all unresolved as of this run)

1. **Calibration at threshold=0.75** produces near-zero recall on the test
   split despite good AUC-PR ranking (§ above). Most consequential open item.
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

## Claims discipline

This system performs **learned dynamics** and **temporal forecasting**, not
causal inference. `nidra/explain/counterfactual.py` outputs are labelled
`"model-internal what-if"` everywhere — a question about the model, not an
intervention on the real network. The dataset-label → tactic mapping
(`nidra/data/labels.py`) is a curated presentation mapping, not ATT&CK
technique-level ground truth. See `ml/README.md`'s "Claims discipline"
section for the full statement.
