# NIDRA — ML subsystem

Predictive network world model. Problem Statement 26153 (NTRO). NIDRA:
Network Infiltration & Dynamics Recurrent Analyzer for Attack Forecasting.
This is the ML half of NIDRA: data pipeline, world model, training,
evaluation, explainability, and the `NidraPredictor` serving interface. See
`../docs/IMPLEMENTATION-ML.md` for the full build spec this implements, and
`../docs/HORIZON_PRD.pdf` (original problem-statement PRD, kept under its
original filename) for product framing.

## Claims discipline (read this before reading any metric below)

This system learns **temporal dynamics** from **observational** data. It
does not perform causal inference, and it does not model attacker intent.

- Use: *learned dynamics*, *forecast*, *simulated trajectory*, *projected
  state*, *model-internal counterfactual*.
- Avoid: *causal simulator*, *"the model understands why"*, *"predicts
  attacker intent"*.
- The dataset-label → tactic mapping (`nidra/data/labels.py`) is a
  **curated presentation mapping**, not ATT&CK technique-level ground
  truth — CIC-IDS2017 does not support that resolution.
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
    -> per-host, per-30s-window state vector (45 features, schema.py)
    -> GRU encoder over L=30 windows of history -> h_t
    -> Gaussian transition model: P(S_t+1 | h_t), predicts a DELTA
    -> recursive K=6 rollout, feeding predictions back as if observed
    -> frozen risk head + frozen stage head applied to predicted states
    -> ensemble of 5 seeds, ~1000 sampled trajectories -> quantiles
    -> forecast: p_compromise per horizon, confidence band, stage
       distribution, lead time, signed top signals, driving window,
       model-internal counterfactual
```

The recursion — the model's own prediction fed back into the encoder as if
it were an observation — is the entire "world model" claim. A per-window
classifier cannot do this; it never emits a state to feed back.

### Repository layout

```
nidra/
  config/default.yaml       single source of truth for all hyperparameters/paths
  config/real_smoke.yaml    reduced-scale config for the real-data smoke run (see below)
  nidra/
    data/                   pcap_extract, flow_load, join, windowize, graph_features,
                             labels, splits, normalize, dataset, schema (FEATURE_ORDER)
    models/                 encoder, transition, heads, world_model (+ rollout)
    train/                  losses, pipeline, train_dynamics, train_heads
    eval/                   metrics, baselines, ablations, calibration,
                             lead_time_runner, run_eval (CLI)
    explain/                shap_runner, saliency, counterfactual, flow_bridge
    serve/                  predictor.py (NidraPredictor — the ONLY backend import),
                             benchmark.py
    utils/                  config loading, seeding
  artifacts/                weights/ scaler/ metrics/  (gitignored except .gitkeep)
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
4. **`windowize.py`** — absolute epoch-aligned 30s windows (never relative
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
9. **`dataset.py`** — builds `[N, L, F]` / `[N, K, F]` windowed arrays with
   traceable metadata (`host_id`, `origin_ts`, `episode_id`, `stage_label`,
   `risk_label`, and per-horizon `future_stage_idx`/`future_is_attack` for
   horizon-curve metrics), kept separate from the numeric tensors.

### Target dataset: CSE-CIC-IDS2018 (not CIC-IDS2017)

NIDRA's real training/evaluation dataset is **CSE-CIC-IDS2018**
(`aws s3 sync --no-sign-request ... s3://cse-cic-ids2018/`, ~250GB), not
CIC-IDS2017. `config/mvp_2018.yaml` is the config for a bounded MVP slice of
it once downloaded; `config/default.yaml` is still CIC-IDS2017-shaped and
needs the same dataset-section rework once 2018 lands and its column layout
is confirmed (see the warning block at the top of `config/mvp_2018.yaml`
about CSE-CIC-IDS2018 CSV releases that reportedly drop Source/Destination
IP and Timestamp — which would break `host_id` derivation entirely if true
of this specific download, and has not yet been verified locally).
`nidra/data/labels.py` already has CSE-CIC-IDS2018's known raw label
strings (`FTP-BruteForce`, `Infilteration`'s dataset-own misspelling, etc.)
mapped to the same six-stage taxonomy, cross-referenced only against the
CIC's own published documentation, not yet against real files.

See `REAL_DATA_RESULTS.md` for a CIC-IDS2017 smoke run that predates this
decision — it validates that the pipeline mechanics work end-to-end against
real CICFlowMeter-format data, but its numbers are **not** a claim about
NIDRA's performance on its actual target dataset and should not be cited
as such.

### Known real-data constraint (from the CIC-IDS2017 pipeline-validation run)

CIC-IDS2017's packet-level fields (TTL, TCP window, fragmentation,
retransmissions) require the **raw PCAPs**. This machine has the real
CICFlowMeter flow CSVs (`TrafficLabelling` release — the sibling
`MachineLearningCVE` release strips IP/timestamp columns entirely and
cannot be windowed by host/time) but not the raw CIC-IDS2017 captures.
The pipeline runs correctly in **flow-only mode** against real flow data
(packet-aggregate features zero-filled, `is_active` still derived from
flow activity, and this is logged loudly, not silently absorbed) — see
`windowize.build_state_rows`'s flow-only warning. The tshark extraction
path itself is implemented and unit-tested against synthetic pcap-style
input, and has also been smoke-tested against a real (non-CIC) local pcap
capture; it has not been exercised against a real CIC-IDS2017 or
CSE-CIC-IDS2018 PCAP because neither exists on this machine.

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
multi-step unrolled Gaussian NLL (`losses.dynamics_loss`) with
horizon-discounted weighting (`0.85**k`) and scheduled sampling
(teacher-forcing probability 1.0 → 0.3, linear over the first 60% of
epochs). AdamW, cosine schedule, grad clip 1.0, early stopping on
validation multi-step NLL.

**Stage 2 — heads** (`train_heads.py`): encoder + transition **frozen**
(`model.freeze_dynamics()`), risk/stage heads trained on **observed
states only** — `S_t`, never a rollout prediction. `pos_weight = n_neg /
n_pos` for the risk head; inverse-frequency class weights for the stage
head. After this stage, `model.freeze_heads()` runs — nothing is trained
after that point. There is no code path that fine-tunes a head on rollout
output, even implicitly.

**Ensemble**: 5 seeds (`[0,1,2,3,4]`), identical config, trained
independently; `serve/predictor.py` pools all members' sampled
trajectories before scoring.

Every hyperparameter lives in `config/default.yaml` — nothing important is
hardcoded in a training script.

## Evaluation (`nidra/eval/`)

Four mandated baselines (`baselines.py`): LR on `S_t`; LR on flattened
`S[t-L..t]` (1350 features — *the one that matters*: beating this is the
real test of whether the transition model earns its place); persistence
(`S_hat[t+k]=S_t`, scored with the **same frozen risk head** the world
model uses, isolating the transition model's contribution); oracle (frozen
heads on the true future state — upper bound).

Metrics (`metrics.py`): F1/precision/recall/AUC-PR/FPR (+ alerts/hour/host);
**lead time**, defined exactly once — earliest sustained (`m=2` consecutive
windows) threshold (0.75) crossing before true onset, reporting the full
distribution and no-warning fraction, never only the maximum; per-feature
per-horizon **state nRMSE** (the metric a classifier cannot report at all,
since it never emits a state); Brier score + reliability diagrams.

Falsification suite (`ablations.py`) — genuine experiments, not guaranteed
outcomes:
- **Persistence ablation** — expects a collapse in AUC/nRMSE relative to
  the world model. If it doesn't collapse, that's reported as-is: no
  useful dynamics were learned.
- **Time-shuffle ablation** — shuffles window order within each input
  sequence. Expects collapse; if not, the model is using per-window
  features only and the temporal claim is false.
- **Horizon curve** — AUC/nRMSE vs k. A **flat** curve is flagged as a
  leakage warning, not treated as a good result.
- **Surprise signal** — one-step forecast error, benign vs pre-attack
  windows.

`run_eval.py` is the CLI entry point that runs all of the above against a
trained seed and writes `artifacts/metrics/{baselines,ablations,
calibration,lead_time}.json`.

## Explainability (`nidra/explain/`)

Three deliberately distinct mechanisms (conflating them is called out in
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

Plus **counterfactual rollout** (`counterfactual.py`): clamp one named
feature at every rollout step, re-simulate. Labelled `"model-internal
what-if"` everywhere — never "intervention". And the **flow bridge**
(`flow_bridge.py`): a registry (`FEATURE_TO_FLOW_PREDICATE`) resolving top
SHAP features back to the flows that produced them — a lookup, not a
second model. Honest empty result (not a fabricated guess) when
packet-level data isn't available for a predicate that needs it.

## Serving (`nidra/serve/predictor.py`)

`NidraPredictor` is the **only** class the backend imports. Loads
ensemble weights, the scaler, and the SHAP background once at
construction. `forecast(states, host_id, origin_ts)` takes raw, unscaled
`[L, 45]` history, validates the schema (fails loudly on width/length/NaN
mismatch), and returns the backend's `Forecast` schema (minus `tenant_id`,
which the backend attaches). Holds **no per-host sequence state** —
sequence buffering is the backend/Redis layer's job, which is what lets any
stateless inference worker serve any host. `benchmark.py` measures latency
against the <300ms target (K=6, 5 members, 200 samples/member on CPU); if
over target, cut samples toward 100 before cutting ensemble size.

## Reproducing this

```bash
cd horizon
pip install -e .
pytest tests/ -q                                    # 100+ tests, synthetic fixtures, seconds

python -m nidra.train.train_dynamics --config config/default.yaml       # Stage 1, full ensemble
python -m nidra.train.train_heads    --config config/default.yaml       # Stage 2
python -m nidra.eval.run_eval        --config config/default.yaml --seed 0 --split test
python -m nidra.eval.run_eval        --config config/default.yaml --seed 0 --split holdout   # Infiltration
python -m nidra.serve.benchmark --weights-dir artifacts/weights --scaler-path artifacts/scaler/robust_scaler.joblib
```

Full training (5 seeds × 60 epochs over the complete Monday–Wednesday
corpus) was not run to completion inside this session — see
`REAL_DATA_RESULTS.md` for what was actually measured, at what scale, and
why.
