# NIDRA — ML subsystem

> Root copy of [`ml/README.md`](ml/README.md); the file references below (`REAL_DATA_RESULTS.md`, `MODEL_CARD.md`, `TRAINING.md`, `EVALUATION.md`, `ARCHITECTURE.md`) live under `ml/`.

Predictive network world model. Problem Statement 26153 (NTRO). NIDRA:
Network Infiltration & Dynamics Recurrent Analyzer for Attack Forecasting.
This is the ML half of NIDRA: data pipeline, world model, training,
evaluation, explainability, and the `NidraPredictor` serving interface. See
`docs/IMPLEMENTATION-ML.md` for the full build spec this implements, and
`docs/HORIZON_PRD.pdf` (original problem-statement PRD, kept under its
original filename) for product framing. This file is the top-level entry
point; `ARCHITECTURE.md` is the two-page architecture document (with
`ARCHITECTURE_DETAIL.md` at engineering depth), and `MODEL_CARD.md` / `TRAINING.md` / `EVALUATION.md` cover
architecture-vs-spec deviations, the training procedure, and the eval
harness in more depth, and `REAL_DATA_RESULTS.md` is the single source of truth for actual
measured numbers. **For the plain-English version of the scores — no
jargon, just what they mean — see the "Results" section of the root
[`README.md`](README.md).**

## What the evidence supports — read first

Three results are solid; the rest of this README is how they were produced and
what limits them. The two-page version is [`ARCHITECTURE.md`](ARCHITECTURE.md).

1. **The world model learns state-transition dynamics.** Six-minute state
   forecasts reduce squared error vs persistence by 0.672 / 0.587 / 0.616
   (val / test / holdout) against a two-lag linear model's 0.653 / 0.565 / 0.595 —
   a 5–6 % margin, on every split, at every horizon k = 1…6, and on active hosts
   alone. The 90 % band covers the true state 98 % of the time on validation.
2. **Against the PS's logistic-regression baseline it raises fewer, more
   precise alerts**, each system at its own validation-chosen threshold
   (`reports/PS_BASELINE_BENCHMARK.md`): false alarms/hour 0.48 vs 4.80 (test)
   and 0.86 vs 4.89 (holdout), precision 0.89 vs 0.48 and 0.85 vs 0.46, F1 0.46 vs
   0.35 on holdout and 0.06 vs 0.07 on test. AP margin +0.025 [+0.005, +0.058] on
   test; +0.197 [−0.000, +0.366] on holdout.
3. **Every number reproduces and every forecast is explained** — provenance
   records on every run, 28 cells re-run bit-identically, SHAP / saliency /
   integrated-gradient attributions with a faithfulness check on every forecast.

What it does not do: no episode is warned before onset on any split; the same
risk head on the *current* state scores about as well as on the rollout, so
advantage 2 is not demonstrably the simulation's; history-reading classifiers
(LR on 30 min of history, a GRU) match or beat it on AP; projected stages are
0.00 accurate where training had none; and the model learns correlation, not
causation. The numbers behind each are in `REAL_DATA_RESULTS.md` (Run 8) and
the Run 9 reports below.

## Run 9 — research phase, concluded (2026-09-24)

Everything in this README, in `MODEL_CARD.md` and in `REAL_DATA_RESULTS.md`
describes **Run 8**, which is the shipped state and is unchanged. Run 9 was a
research phase: CTU-13 integration, a cross-dataset matrix, a history-aware
risk head, and an unseen-attack-family test. It concluded that the Run 8
per-state head should stay (DECISIONS.md D146) and produced **no** new shipped
configuration.

| | |
|---|---|
| `MODEL_CARD_RUN9.md` | what Run 9 has and has not established |
| `reports/RUN9_FINAL_REPORT_2026-09-23.md` | the conclusion, separating demonstrated from unproven |
| `reports/RUN9_QUESTIONS.md` | the fourteen questions the phase was set, and their answers |
| `reports/RUN9_NEGATIVE_RESULTS.md` | 19 entries: what was tried and did not work, including four conclusions this log withdrew |
| `reports/run9/` | the generated tables — corrected scorecard, D145 correction, reproduction check, paired margin intervals |
| `reports/CTU13_MODEL_IMPROVEMENT_2026-09-23.md` | the full experiment log |

Two things from it that bear on how to read the Run 8 numbers here. First, a
defect (D145) was found in the rollout, and **Run 8 is not affected by it** — its
feature regime drops no features, which is the precondition for the defect. Every
CTU-13 and cross-dataset number in Run 9 was re-scored under the fix (21 up, 7 down,
no direction flipped); no Run 8 number needed it. Second, on the slices where Run 9
could ask the question most sharply, **the forecast does not beat a persistence
baseline** — across 28 cells exactly one margin over the strongest baseline in its
cell survives a paired episode-cluster interval with room to spare. That is a finding
about the approach, not about the Run 8 artifacts, and it is stated plainly in
`MODEL_CARD_RUN9.md`.

## Claims discipline (read this before reading any metric below)

This system learns **temporal dynamics** from **observational** data. It
does not perform causal inference, and it does not model attacker intent.

- Use: *learned dynamics*, *forecast*, *simulated trajectory*, *projected
  state*, *model-internal counterfactual*.
- Avoid: *causal simulator*, *"the model understands why"*, *"predicts
  attacker intent"*.
- The stage → ATT&CK tactic and CIC-IDS2017 label → technique tables
  (`nidra/data/attack_mapping.py`, rendered to `docs/ATTACK_MAPPING.md`)
  are a **curated presentation mapping** validated against the published
  attack descriptions, not technique-level ground truth observed in the
  traffic — CIC-IDS2017 does not support that resolution. A projected stage
  sequence is labelled "projected stage sequence (model-internal)".
- Counterfactual rollouts (`nidra/explain/counterfactual.py`) answer
  *"what would this model predict if this feature were held at this
  value"* — a question about the model, not the network. Every result is
  labelled `"model-internal what-if"`, in code and in any UI/API surface,
  never "intervention" or "causal effect".
- `Infiltration` (Thursday) is held out of training entirely — it is used
  only to test generalization to an unseen attack type.

## Architecture

```
raw telemetry (PCAP via tshark + CICFlowMeter CSV)
    -> per-host, per-60s-window state vector (45 features, schema.py)
    -> GRU encoder over L=30 windows of history -> h_t
    -> Gaussian transition model: P(S_t+1 | h_t), predicts a DELTA
    -> recursive K=6 rollout, feeding predictions back as if observed
    -> frozen risk head + frozen stage head applied to predicted states
    -> ensemble of 5 seeds, sampled trajectories -> pooling statistic,
       per-horizon calibration and threshold frozen on validation
    -> forecast: p_compromise per horizon with a trajectory band, stage
       distribution -> ATT&CK tactics, lead time, signed top signals,
       driving window, forecast attributions, model-internal counterfactual
```

The recursion — the model's own prediction fed back into the encoder as if
it were an observation — is the entire "world model" claim. A per-window
classifier cannot do this; it never emits a state to feed back.

### Repository layout

```
nidra/
  config/default.yaml       production config (Δ=60 s, L=30, K=6, 5-seed ensemble)
  config/mvp_2017.yaml      MVP-scale config (fewer epochs/samples) — same real dataset/paths
  nidra/
    data/                   pcap_extract, flow_load, flow_assemble, join, windowize,
                             graph_features, labels, splits, normalize, dataset, onset,
                             attack_mapping, audit, schema (FEATURE_ORDER)
    models/                 encoder, transition, heads (risk, stage, onset), world_model,
                             risk_pooling
    train/                  losses, pipeline, head_data, train_dynamics, train_heads,
                             train_onset
    eval/                   benchmark (primary), eval_set, metrics_natural, operating_point,
                             systems, state_metrics, episode_metrics, gru_classifier;
                             legacy: metrics, baselines, ablations, calibration, run_eval
    explain/                shap_runner, saliency, forecast_attribution, counterfactual,
                             flow_bridge, regimes
    serve/                  predictor.py (NidraPredictor — the ONLY backend import),
                             benchmark.py
    cli/                    forecast (offline PCAP/CSV pipeline), report_html
    scripts/                run_experiment, report_tables, plot_benchmark, demo_forecast, ...
    utils/                  config loading, seeding, provenance
  artifacts/                the shipped model, committed: weights/ (5 checkpoints, onset
                             heads, operating_point.json) scaler/ processed/ metrics/ metadata/
  experiments/              run records (config + provenance + metrics per label),
                             the Δ=30 baseline manifest
  tests/                    unit + integration tests, synthetic fixtures
```

## The canonical 45-feature schema

`nidra/data/schema.py::FEATURE_ORDER` is the single source of truth for
feature ordering. Every module — windowing, normalization, training,
serving, explainability — imports it. Never construct feature order from
dict iteration. `validate_feature_dict()` / `validate_state_array_width()`
fail loudly on any drift.

Groups: flow aggregates (15) · packet aggregates (11) · graph scalars (8) ·
backward-looking dynamics — deltas and 3-window OLS slopes (10) · activity
flag (1).

## Data pipeline

1. **`pcap_extract.py`** — `tshark -T fields`, streamed to parquet in
   chunks (never loads a full capture into memory). Deliberately not
   PyShark/Scapy (~2 orders of magnitude too slow for bulk extraction).
2. **`flow_load.py`** — robust CICFlowMeter CSV loading: strips leading
   column-name spaces, drops mid-file duplicate header rows (caught via
   numeric coercion), logs input/accepted/dropped counts and reasons.
3. **`join.py`** — packet and flow events are each assigned to
   `(host, window)` independently, then the two aggregate tables are
   joined on `(host_id, window_ts)`. No five-tuple packet↔flow matching.
   Host attribution is **source-only** in v1.
4. **`windowize.py`** — absolute epoch-aligned 60s windows (never relative
   to the first observed packet). **Empty windows are real data**: a
   silent host gets a zero-valued row with `is_active=0`, never a dropped
   window (dropping would corrupt every downstream lead-time measurement).
   `new_peer_count` / `neighbour_risk_fraction` are computed in a single
   chronological forward pass (`graph_features.PeerTracker`) — the one
   genuinely stateful feature in the pipeline.
5. **`graph_features.py`** — per-window transient directed graph, discarded
   immediately after 8 scalars are extracted. `local_clustering_coeff`
   samples up to 200 neighbours on dense nodes (a signal, not an exact
   quantity).
6. **`labels.py`** — `stage_label` (curated tactic mapping) and
   `risk_label = 1 if any attack window in (t, t+K]`. Future data is used
   only to build these two targets, never the input.
7. **`splits.py`** — day-based train/test/holdout (train=Mon+Tue+Wed,
   test=Fri, holdout=Thu/Infiltration, never trained on), plus a
   **contiguous trailing time block** for validation within train (never a
   random split), with the cutoff nudged earlier if it would otherwise
   split a live attack episode.
8. **`normalize.py`** — `RobustScaler`, fit on the **training split only**,
   log1p on heavy-tailed count features, clipped to `[-10, 10]`, serialized
   to `artifacts/scaler/`. `FeatureScaler.transform()` refuses to run
   before `fit()`/`load()` — the scaler is never refit at serving time.
   Also computes `reference_std_` at fit time (per-feature std of the
   training population in scaled units) — the stable normalizer
   `eval/metrics.state_nrmse` uses instead of a small eval batch's own std
   (see "Known bugs, fixed" below) — and provides `inverse_transform()` for
   turning a rolled-out/predicted state back into raw units for serving.
9. **`dataset.py`** — builds `[N, L, F]` / `[N, K, F]` windowed arrays with
   traceable metadata (`host_id`, `origin_ts`, `episode_id`, `stage_label`,
   `risk_label`, and per-horizon `future_stage_idx`/`future_is_attack` for
   horizon-curve metrics), kept separate from the numeric tensors.

### Target dataset: CIC-IDS2017 (complete dataset, not CSE-CIC-IDS2018)

NIDRA's real training/evaluation dataset is the complete **CIC-IDS2017**
dataset — all 8 published day-files from the "TrafficLabelling" release
(the sibling "MachineLearningCVE" release strips Source IP/Destination
IP/Timestamp columns entirely and cannot be windowed by host/time).
Switching to CSE-CIC-IDS2018 was considered and explicitly abandoned (it
is far larger than needed for this project's scope, at ~250GB); do not
reintroduce it. `config/default.yaml` is the full-scale production config
(5-seed ensemble, 60/30 epochs) against this dataset; `config/mvp_2017.yaml`
is the same dataset and paths at MVP scale (single seed, fewer epochs,
stratified sample capping) for a faster hackathon-timeline run.

Real, tshark-extracted packet-level features exist for **all 5 raw PCAPs**
(Monday, Tuesday, Wednesday, Thursday, Friday — all downloaded and
extracted), so all 8 day-files carry real packet-level features; none run
in flow-only mode. `windowize.build_state_rows`'s flow-only warning still
exists and still fires loudly (never silently) if a day's `packets:` config
entry is ever unset or its parquet file goes missing. `nidra/data/labels.py`
also carries dormant, unused label-matching rules for CSE-CIC-IDS2018's raw
`Label` strings, kept only as harmless compatibility in case that dataset is
ever added later — it is not part of this project's actual data path.

See `REAL_DATA_RESULTS.md` for the actual end-to-end run against this
dataset: dataset stats, PCAP extraction stats, training convergence,
baseline comparisons, ablations, calibration, and lead time — reported
honestly, including where the world model does not yet beat the
baselines.

## World model (`nidra/models/`)

- **`encoder.py`** — 2-layer GRU, hidden=128, dropout=0.1. Returns both the
  last-step hidden summary and the full recurrent state, so rollout can
  re-enter the GRU one step at a time.
- **`transition.py`** — `H -> 256 -> GELU -> 256 -> GELU -> {mu, logvar}`,
  `logvar` clamped to `[-6, 3]` (non-negotiable: without this, rollout
  reliably produces NaN by k≈3). Predicts a **delta**:
  `S_hat[t+1] = S[t] + mu` — a strong autocorrelation prior, and it makes
  the persistence baseline exactly the `mu=0` case.
- **`heads.py`** — `RiskHead` (F→64→1) and `StageHead` (F→64→6), operating
  on the 45-dim feature vector (not the hidden state) so they apply
  identically to an observed or a rolled-out predicted state.
- **`world_model.py`** — assembles the above; `rollout()` recursively feeds
  each prediction back through the encoder for K steps, consuming no real
  data after the input's last window. Supports deterministic (`stochastic=
  False`) and stochastic sampling (`n_samples` trajectories via
  `repeat_interleave`, vectorized rather than looped).

## Training (`nidra/train/`)

**Stage 1 — dynamics** (`train_dynamics.py`): encoder + transition only,
multi-step unrolled **β-NLL** (`losses.dynamics_loss`, β=0.5 — each element's
Gaussian NLL weighted by `stop_grad(σ²)^β`; measured against the plain NLL,
an MSE auxiliary, non-silent sample weighting and a two-lag linear skip in
`DECISIONS.md` D113) with horizon-discounted weighting (`0.85**k`) and
scheduled sampling (teacher-forcing probability 1.0 → 0.3, linear over the
first 60% of epochs). AdamW, cosine schedule, grad clip 1.0. **Checkpoint
selection on the free-running validation NLL** — the rollout the model
performs at serving time, own predictions fed back — not the teacher-forced
loss (D107).

**Stage 2 — heads** (`train_heads.py`, `head_data.py`): encoder + transition
**frozen** (`model.freeze_dynamics()`), fresh risk/stage heads trained on
**observed states only** — every observed state of the split (2.27M training
rows, 179 positives), `S_t` never a rollout prediction. Imbalanced BCE with
`pos_weight = n_neg / n_pos`, Gaussian input noise σ=0.3, independent
selection: risk head on natural-prevalence validation AP, stage head on
macro-F1 (D108, D114). After this stage `model.freeze_heads()` runs — nothing
is trained after that point. There is no code path that fine-tunes a head on
rollout output, even implicitly.

**Stage 3 — onset head** (`train_onset.py`, D109): an explicit supervised
baseline on `S_t` for P(episode begins within h ∈ {1,3,5,10,15,30} min),
defined only outside episodes (`nidra/data/onset.py`). Scored in the
benchmark as `onset_head_direct`; not part of the served risk curve.

**Ensemble**: 5 seeds (`[0,1,2,3,4]`), identical config, trained
independently; `serve/predictor.py` pools all members' sampled
trajectories before scoring, with the pooling statistic frozen on validation.

Every hyperparameter lives in `config/default.yaml` — nothing important is
hardcoded in a training script — and every recorded run goes through
`nidra/scripts/run_experiment.py`, which writes
`experiments/runs/<label>/{config.yaml,record.json}` with git commit, config
hash, dataset digests and checkpoint hashes. See `TRAINING.md`.

## Evaluation (`nidra/eval/`)

**Primary benchmark** (`benchmark.py`, Run 8 onward — see `EVALUATION.md`):
every positive and pre-onset origin of a split plus capped, re-weighted
benign origins, so every metric is at the split's **natural prevalence**
(test 0.0033, holdout 0.00034). Pooling statistic, per-horizon Platt
calibration and threshold are selected on validation only and frozen in
`artifacts/weights/operating_point.json` before test or holdout are scored.
It reports, with episode-cluster bootstrap intervals:

- **Task A** — the published label (attack on this host within K windows),
  the primary AP; **Task A′** detection (is `S_t` itself attack-labelled),
  kept separate so a classifier's number is never presented as a forecast;
- **Task B** — onset within h minutes, origin outside any episode: the
  forecasting question proper, with very few positives in this dataset;
- **Task C** — per-horizon k: attack at `t+k`, stage top-1, state skill;
- **Attribution** — paired differences `world_model − persistence`,
  `− persistence + learned noise`, `− isotropic noise`, `− deterministic
  rollout`, `− ridge two-lag`, `− LR / GBDT / GRU classifier`. The sign and
  interval of the first is the test of whether the transition model
  contributes anything beyond the head on `S_t`;
- **State-forecast skill** `1 − MSE/MSE_persistence` on the deterministic
  rollout, 90%-band coverage, per-episode lead time / latency, false alarms
  per active-benign hour, reliability.

Baselines scored in the same run: persistence (frozen risk head on `S_t`),
ridge two-lag dynamics, oracle on the true future, LR on `S_t`, LR on the
flattened L-window history, GBDT on `S_t`, a GRU sequence classifier, the
onset head. If the world model cannot beat one of them, that is reported,
not hidden.

**Legacy harness** (`run_eval.py`, `baselines.py`, `ablations.py`,
`metrics.py`): the balanced-subsample evaluation Runs 1–7 were measured
with — four mandated baselines, persistence and time-shuffle ablations,
horizon curve, surprise signal, nRMSE, lead time. Still runs; its numbers
are not comparable with the benchmark's and are never mixed with them.

## Explainability (`nidra/explain/`)

Four deliberately distinct mechanisms (conflating them is called out in
the spec as the most common weakness in submissions of this type):

1. **Why is the current state risky?** KernelSHAP on the frozen risk head
   over the observed state, background = 100 k-means centroids of benign
   training states (never a random background — slow and noisy).
2. **Why this future?** Captum input-gradient saliency from a predicted
   future feature back to the historical input windows — identifies which
   past window(s) drove the forecast.
3. **Why this stage?** SHAP on the frozen stage head over a *predicted*
   state — only expressible because the transition model decodes to named
   features rather than an opaque latent.
4. **What drove this forecast?** (`forecast_attribution.py`) signed
   integrated-gradient attributions of the pooled K-step risk with respect
   to every (window, feature) cell of the input, through the rollout, with a
   **deletion faithfulness check**: removing the top-m attributed cells must
   move the forecast more than removing m random cells, and the result says
   whether it did (`faithful: true/false`).

Plus **counterfactual rollout** (`counterfactual.py`): clamp one named
feature at every rollout step, re-simulate. Labelled `"model-internal
what-if"` everywhere — never "intervention". And the **flow bridge**
(`flow_bridge.py`): a registry (`FEATURE_TO_FLOW_PREDICATE`) resolving top
SHAP features back to the flows that produced them — a lookup, not a
second model. Honest empty result (not a fabricated guess) when
packet-level data isn't available for a predicate that needs it.

## Serving (`nidra/serve/predictor.py`)

`NidraPredictor` is the **only** class the backend imports. Loads
ensemble weights, the scaler, the SHAP background and the validation-frozen
operating point once at construction. `forecast(states, host_id, origin_ts)`
takes raw, unscaled `[L, 45]` history, validates the schema (fails loudly on
width/length/NaN mismatch), and returns the backend's `Forecast` schema
(minus `tenant_id`, which the backend attaches) plus a `risk_curve` (per
horizon: calibrated P(attack at t+k), raw score, trajectory band, cumulative
P(attack within the horizon), the threshold in force), a `progression`
(projected stage sequence, labelled model-internal) and per-horizon ATT&CK
tactics/techniques from `nidra/data/attack_mapping.py`. `forecast_batch`
scores many origins at once for the offline CLI. Holds **no per-host
sequence state** — sequence buffering is the backend/Redis layer's job,
which is what lets any stateless inference worker serve any host.
`benchmark.py` measures latency against the <300ms target (K=6, 5 members,
200 samples/member on CPU); if over target, cut samples toward 100 before
cutting ensemble size.

**Offline pipeline** (`nidra/cli/forecast.py`): a PCAP (tshark → flows
assembled from packets) or a CICFlowMeter CSV → state table → every host's
contexts scored → `forecasts.csv`, `alerts.json`, `summary.json`;
`nidra/cli/report_html.py` renders a static report from that directory.

## Reproducing this

### Without the raw dataset (what a fresh clone can do)

The repo ships the trained ensemble, the fitted scaler, the operating point
and the windowed per-day tables under `artifacts/`, so everything below runs
offline on a CPU with no CIC-IDS2017 download. `build_all_splits` resolves
each day from its committed table first and only falls back to the raw CSV
on a miss (see `nidra/train/pipeline.py`), so a machine without the dataset
is a cache hit, not a failure.

```bash
cd ml
pip install -e .
pytest tests/ -q                                  # synthetic fixtures, a few minutes

# Forecast one real window with the shipped ensemble
python -m nidra.scripts.demo_forecast                 # precedes a real attack
python -m nidra.scripts.demo_forecast --want-risk 0   # benign, for contrast
python -m nidra.scripts.demo_forecast --json          # the full Forecast dict

# Reproduce the published numbers (Run 8). Validation selects and freezes the
# operating point; test and holdout only load it. Every record carries its own
# parameters, so a cheaper run is visibly a cheaper run, never a silent mismatch.
python -m nidra.eval.benchmark --split val --select-operating-point --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split test    --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split holdout --n-samples 60 --n-resamples 300
python -m nidra.scripts.report_tables --run . --splits val,test,holdout
python -m nidra.scripts.plot_benchmark --run . --splits test,holdout --out reports/run8

# Offline forecasting on a capture or CICFlowMeter CSV the model has never seen
python -m nidra.cli.forecast --csv flows.csv --out out/      # or --pcap capture.pcap
python -m nidra.cli.report_html --run out/

# Serving latency on your own hardware
python -m nidra.serve.benchmark --weights-dir artifacts/weights \
    --scaler-path artifacts/scaler/feature_scaler.json --config config/default.yaml
```

`benchmark` rewrites `artifacts/metrics/<split>/benchmark.json` in place. The
committed contents are the published run, so `git diff` after your own run
is the comparison.

### Retraining from the raw dataset

Everything below needs the ~50GB CIC-IDS2017 release in place — see
`PRODUCTION_RUN_GUIDE.md` for acquiring and pointing at it, and `TRAINING.md`
for the stages.

```bash
cd ml
python -m nidra.scripts.run_experiment --label production --seeds 0,1,2,3,4 \
    --stages dynamics,heads,onset,gru_baseline --epochs 24     # ≈4.8 h on an M1, writes artifacts/
# a variant, recorded under experiments/runs/<label>/ and never touching artifacts/:
python -m nidra.scripts.run_experiment --label my_variant --seeds 0 --stages dynamics,heads \
    --epochs 20 --set train_dynamics.beta_nll=0.0
```

The shipped artifacts are the `production` run of 2026-09-20 (Run 8 in
`REAL_DATA_RESULTS.md`: Δ=60 s, 5 seeds, β-NLL dynamics, heads on every split
row). Runs 1–7 were at Δ=30 s on a balanced evaluation subsample and are
superseded; their artifacts are preserved under the `baseline-delta30-run7`
git tag and `experiments/BASELINE_MANIFEST_delta30_run7.json`.
