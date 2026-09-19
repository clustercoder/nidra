# NIDRA ML — Architecture

How the ML subsystem is put together and why. For measured results see
`REAL_DATA_RESULTS.md`; for claims and limitations see `MODEL_CARD.md`; for
the evaluation harness see `EVALUATION.md`.

## The one architectural claim

NIDRA is a **world model**, not a classifier. A classifier scores the
present: given what this host looks like now, how suspicious is it? NIDRA
learns the *dynamics* of a host's behaviour and then simulates forward —
generating states that have not happened yet and scoring those.

The distinction is load-bearing and lives in exactly one place,
`WorldModel.rollout`: the model's own prediction is fed back into the
encoder as if it were an observation. No real data after the input's last
window is consumed anywhere in that loop. This is what produces a lead time
at all; a per-window classifier structurally cannot produce one, because it
has nothing to say about a window that has not arrived.

## The state vector

One host, one 30-second window, 45 floats — `nidra/data/schema.py`'s
`FEATURE_ORDER` is the single source of truth, and every module imports it
rather than reconstructing order from a dict.

| Family | Count | Examples |
|---|---|---|
| Flow aggregates | 15 | `syn_ratio`, `bytes_total`, `iat_var`, `active_flow_count` |
| Packet aggregates | 11 | `ttl_var`, `tcp_window_entropy`, `retrans_rate`, `payload_size_p95` |
| Graph scalars | 8 | `out_degree`, `dst_port_entropy`, `new_peer_count`, `reciprocity` |
| Dynamics (deltas/slopes) | 10 | `d_syn_ratio`, `slope3_out_degree`, `d_iat_var` |
| Activity | 1 | `is_active` |

Geometry, fixed across the pipeline, model and serving: **Δ = 30s** windows,
**L = 30** windows of history (15 minutes), **K = 6** windows of forecast
(3 minutes). So one training example is `[30, 45] → [6, 45]`.

## Data pipeline

```
raw CIC-IDS2017 CSVs  ──┐
                        ├─> windowize ─> state table ─> splits ─> windowed tensors
tshark packet parquet ──┘   [host,ts,45]   + labels      temporal    [N,L,F],[N,K,F]

an uploaded pcap ─> tshark ─> flow_assemble ─┘   (no CSV, no labels)
```

The two inputs are joined on `(host_id, window_ts)` and nothing else — no
five-tuple matching. That join is only as good as the clocks agree, and for a
long time they did not: CIC-IDS2017's CSVs print the capture site's local time
(UTC-3) on a 12-hour dial with no AM/PM marker, while the PCAPs carry true UTC.
`parse_cic_timestamp` undoes both, and the correction is stamped into the
windowed-table cache key and into the saved scaler's metadata, because getting
it wrong changes every packet-derived column while raising nothing. See Run 7
in `REAL_DATA_RESULTS.md`.

An uploaded capture has no CICFlowMeter CSV, so `nidra/data/flow_assemble.py`
reconstructs the fifteen flow features from the packets. That reconstruction is
measured, not assumed: `scripts/validate_flow_assembly.py` compares the same
`(host, window)` computed both ways on the one capture where both views exist.
The volume and timing features track (Spearman 0.85-0.92); the TCP flag ratios
do not, and `urg_ratio` cannot be reconstructed at all. Those residuals travel
with every uploaded analysis in `source.fidelity`, so a page built from
reconstructed flows cannot be read as if it were the labelled replay.

`nidra/data/windowize.py` enforces the invariants that make the rest sound:
window boundaries align to absolute epoch multiples (never to the first
observed event), **empty windows are real data** (a silent host gets a
zero vector with `is_active=0`, never a dropped row — otherwise the model
would learn a sequence that skips time), peer-tracking runs in one
chronological forward pass, and deltas/slopes are strictly backward-looking
with zero padding at sequence starts.

Labels are built in `nidra/data/labels.py`. `risk_label = 1` if any
attack-stage window falls in `(t, t+K]` — that is, the target is *about the
future*, which is what makes this forecasting rather than detection. Future
windows are used **only** to construct target columns, never model input.

Splits are temporal by day, never random: train = Mon/Tue/Wed, test =
Friday ×3, holdout = Thursday ×2 (an attack type held out of training
entirely). Validation is the contiguous trailing 15% time-block of the
train days, not a random sample — a random split would leak a host's future
into its own training history. `assert_no_temporal_overlap` and
`assert_no_episode_leakage` run on every build.

Windowed tensors are built by `build_windowed_arrays`, which applies its
sample cap **during** construction rather than after: the uncapped
production train split is ~6.9M `[30,45]` float32 windows (~35GB), which
OOM-kills the process before the first epoch. The cap is stratified — every
attack-positive window is kept, only benign context is thinned.

## Model

```
x [B,L,45] ──> GRU encoder ──> h_t [B,128] ──> Transition ──> (mu, logvar) [B,45]
               2 layers, 128                   MLP 256, GELU x2
               dropout 0.1                     logvar clamped [-6, 3]
                                                      │
                            next = clamp(cur + mu + ε·exp(½·logvar), ±10)
                                                      │
                              └──── fed back into the encoder ────┘  (×K)

predicted state [.,45] ──> RiskHead  45→64→1  ──> sigmoid  p_compromise
                      └──> StageHead 45→64→6  ──> softmax  stage distribution
```

Three deliberate choices:

**The transition predicts a delta, not an absolute state.** This hands the
model a strong autocorrelation prior so capacity goes into modelling
*change*, and it makes the persistence baseline exactly the `mu = 0` case —
a clean ablation rather than a separate implementation.

**`logvar` is clamped.** Unclamped, the model discovers that predicting
infinite variance on hard features minimises NLL; variance explodes and
rollout returns NaN by about k=3. The clamp is a correctness guardrail, not
a hyperparameter. (Tightening it to 1.5 was tested and rejected — see
`MODEL_CARD.md`.)

**The heads read the 45-dim feature vector, not the GRU hidden state.**
This is what lets one head score an observed state and a simulated state
without distinguishing them, and it keeps KernelSHAP tractable over named
features.

## Training: two stages, strictly ordered

**Stage 1 — dynamics.** Encoder + transition trained on multi-step unrolled
Gaussian NLL with horizon discounting (`0.85^k`, so distant and inherently
more uncertain steps do not dominate the gradient) and scheduled sampling
(teacher forcing annealed 1.0 → 0.3 over the first 60% of epochs). Without
the annealing you get a model that looks fine at k=1 and diverges by k=6.
AdamW, lr 3e-4, 60 epochs, cosine schedule, early stopping on
`val_multistep_nll`.

**Stage 2 — heads.** Dynamics are **frozen**, then the risk and stage heads
are trained on *observed states only*. Training heads on simulated states
would let head error and dynamics error fit each other, and the resulting
risk score would no longer be a statement about a state. AdamW, lr 1e-3, 30
epochs, `pos_weight ≈ 1741` for the risk head against a ~0.06% positive
rate. Selection is on weighted validation loss; selecting on validation
AUC-PR was implemented, measured, and rejected — it improves the validation
metric and makes holdout F1 worse (0.727 → 0.486), because the validation
block carries a 6× different positive rate.

Nothing is trained after stage 2. `freeze_all()` is the end state.

## Ensemble and risk pooling

Five seeds (`ensemble.seeds: [0,1,2,3,4]`), each rolling out 200 sampled
trajectories → **1,000 trajectories** per forecast.

Reducing those 1,000 trajectories to one number is the single most
consequential decision in the system, and it is isolated in
`nidra/models/risk_pooling.py` so evaluation and serving cannot drift apart.

Mean pooling answers *"how risky is the average imagined future"*. For a
rare event where only a minority of sampled futures reach compromise, the
mean sits far below any individual risky trajectory — which is why
mean-pooled scores ranked attacks well (good AUC-PR) yet almost never
crossed the mandated 0.75 alert threshold: **F1 0.010**. Pooling the **85th
percentile** instead asks *"how risky is the riskier tail"* — a different
statistic of the same distribution, not a different model and not a lowered
bar, and monotonic in the underlying per-trajectory risk, so it invents no
separability the ranking did not already have. Same checkpoints, F1 **0.84**.

Those two figures are the Run 6 pooling sweep, measured as a matched pair on
the pre-Run-7 checkpoints; they are quoted together because they isolate the
pooling change and nothing else. The shipped ensemble now scores test F1
**0.906** after the Run 7 data correction and retrain — see
`REAL_DATA_RESULTS.md`.

`q=0.85` was selected by a sweep on both splits. It is scale-specific: at
reduced scale `q=0.5` won, and at full scale `q=0.5` is the *worst* setting
tested (test AUC-PR collapses 0.925 → 0.499). That sweep also predates the
Run 7 retrain; it has not been repeated on the new checkpoints, so `q=0.85`
is carried forward on the earlier evidence rather than re-derived.

## Serving

`nidra/serve/predictor.py::NidraPredictor` is the only surface a backend
imports — never `models.*`, `data.*` or `explain.*` directly. It loads the 5
checkpoints and the scaler once at construction, and is **stateless with
respect to per-host history**: `forecast()` takes the caller's `[L, F]`
buffer as an argument rather than holding one, so any inference worker can
serve any host with no coordination. It returns the risk curve for t+1..t+6
with confidence intervals, lead time, stage distribution, predicted features
in raw units (inverse-transformed out of scaled space), SHAP top signals,
and temporal saliency. Latency target is 300ms; measured ~188ms median on an
M1 CPU.

## Module map

| Path | Responsibility |
|---|---|
| `nidra/data/` | schema, flow/pcap loading, windowing, graph features, labels, splits, normalization, tensor construction |
| `nidra/models/` | encoder, transition, heads, world model + rollout, risk pooling |
| `nidra/train/` | shared pipeline, stage-1 and stage-2 trainers, losses |
| `nidra/eval/` | baselines, ablations, metrics, calibration, lead time, `run_eval` CLI |
| `nidra/explain/` | SHAP, temporal saliency, counterfactual rollout, flow bridge |
| `nidra/serve/` | `NidraPredictor`, latency benchmark |
| `nidra/scripts/` | demo forecast, report generation, calibration fitting, sanity plots |

## Invariants that must not be broken

1. **Rollout consumes no future data.** Anything that reaches into `Y`
   during a forecast invalidates every lead-time number in the repo.
2. **Heads train on observed states only, then freeze.**
3. **`FEATURE_ORDER` is the contract.** Never infer feature position from
   dict order.
4. **Eval and serving pool identically** — both go through
   `risk_pooling.py`. A calibration artifact fitted under one pooling
   statistic is refused under another rather than silently applied.
5. **Splits are temporal.** No random shuffling anywhere near split
   construction.
6. **An artifact states what data it was built from, and is refused when
   that does not match.** The windowed-table cache key carries the flow
   timebase; the scaler's metadata carries it too and
   `prepare_training_data` refits rather than reuse a mismatch. Both exist
   because a stale artifact here does not fail — it trains.
7. **A reconstructed input says so.** Anything derived from assembled rather
   than published flows carries its measured fidelity, and an analysis with
   no ground truth never reports precision or recall.
