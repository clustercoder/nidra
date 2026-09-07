# CLAUDE.md

Instructions for Claude Code working in the NIDRA repository.

---

## What this project is

NIDRA is a **predictive network world model**. It compresses network traffic into windowed per-host state vectors, learns the stochastic transition dynamics governing how those states evolve, and rolls them forward K steps to forecast whether a host's trajectory is converging toward compromise — with a horizon, a confidence band, and a lead time in seconds.

**It is not an intrusion detector.** If you find yourself writing code that maps current traffic to a benign/malicious label, stop. That is the thing this project exists to be different from.

Built for Smart India Hackathon, Problem Statement 26153 (NTRO). Judged on whether the model demonstrably learns dynamics rather than classifying.

Reference docs live in `docs/` — read the relevant one before substantial work in that area:
- `docs/PRD.pdf` — product definition, architecture diagrams, evaluation strategy
- `docs/IMPLEMENTATION-ML.md` — feature pipeline, model, training, evaluation
- `docs/IMPLEMENTATION-Backend.md` — services, streams, API, database
- `docs/IMPLEMENTATION-Frontend.md` — design brief and Next.js spec
- `docs/diagrams/` — Mermaid sources (`*.mmd`) and rendered PNGs for both architecture figures

Diagrams are regenerated with:
```bash
mmdc -i docs/diagrams/d1_system.mmd -o docs/diagrams/d1_system.png -b white -s 2 -w 1400
```
Edit the `.mmd` source and re-render; never hand-edit the PNG.

---

## Non-negotiable invariants

These are load-bearing. Violating any one of them invalidates the project's central claim, and the violation will not show up as a test failure — it will show up as *better numbers*, which is why they need to be stated rather than inferred.

### 1. The heads are frozen

`RiskHead` and `StageHead` train on **observed** states only, then freeze. They are applied to predicted states at inference and never fine-tuned on them.

This is why the forecasting claim is falsifiable: every unit of forward-looking capability must originate in the transition model, because there is no other path. If you are ever tempted to unfreeze a head because metrics improve, you are removing the reason the project is defensible.

### 2. Nothing after time `t` may touch the input

Not a normalisation constant, not a centred rolling mean, not a label-derived feature, not a global statistic fitted across the full file.

- Normalisation is fitted on the **training period only** and serialised into `artifacts/scaler/`. Never refit at serving time.
- Deltas and slopes are strictly backward-looking. Pad sequence starts with zeros, never with future values.
- `risk_label[t]` looks forward *to build the label*. That is legitimate supervision. The **input** never does.

Every leak makes the results better and the project worthless.

### 3. `FEATURE_ORDER` is canonical

Defined once in `nidra/data/schema.py`. Every service imports it. Never build feature order from dict iteration, never hardcode an index, never hand-write the list in a second place.

Assert `len(FEATURE_ORDER) == 45` and validate key sets at every service boundary. A schema mismatch caught at a boundary costs a second; caught in a SHAP plot on demo day it costs the demo.

### 4. Ablations are real experiments

The persistence and time-shuffle ablations can fail. If persistence matches the model, we report that honestly and diagnose. Do not tune the ablation to lose. A clearly reported negative result survives judging; a positive result that collapses under one question does not.

### 5. Never claim causality

We have correlation, temporal prediction, and learned dynamics. We do not have causal inference — the data is observational and confounded by collection protocol.

In code, comments, API responses, and UI copy:
- **Use:** learned dynamics, forecast, simulated trajectory, projected state, model-internal counterfactual
- **Avoid:** causal simulator, "understands why", "predicts attacker intent"

The counterfactual endpoint must return and display the label **"model-internal what-if"**. It answers what this model would predict given different input — a question about the model, not the network.

---

## Repository layout

```
nidra/                  ML package
  data/                   extraction, join, windowing, schema, splits
  models/                 encoder, transition, heads, world_model
  train/                  dynamics (stage 1), heads (stage 2), losses
  eval/                   metrics, baselines, ablations, calibration
  explain/                shap, saliency, counterfactual, flow_bridge
  serve/predictor.py      NidraPredictor — the ONLY ML surface the backend imports
nidra_common/           Pydantic schemas shared by all services
services/                 ingest, features, inference, persister
api/                      FastAPI app
web/                      Next.js — marketing site + console
config/default.yaml       single source of truth for hyperparameters
docs/                     PRD, implementation specs, architecture diagrams
artifacts/                weights/ scaler/ metrics/   (gitignored, except metrics)
data/                     raw captures, parquet         (gitignored)
```

`nidra/serve/predictor.py` is the entire ML surface exposed to the backend: one class, one `forecast()` method, plus `counterfactual()` and `explain()`. Keep it that way. If the backend starts importing from `nidra.models`, the boundary has broken.

---

## Commands

```bash
# Environment
make setup                    # venv + deps + tshark check
tshark -v                     # MUST work — packet features depend on it

# Data (long-running; see IMPLEMENTATION-ML.md §2)
make extract DAY=wednesday    # tshark → parquet
make features                 # windowing + graph scalars → [host, window, 45]
make splits                   # temporal / episode / Thursday holdout

# Training
make train-dynamics           # stage 1: encoder + transition
make train-heads              # stage 2: frozen encoder, train risk + stage
make train-ensemble           # 5 seeds

# Evaluation
make baselines                # all four
make ablations                # persistence, time-shuffle, horizon curve
make eval                     # metrics → artifacts/metrics/

# Services
docker compose up -d
docker compose up -d --scale inference=3
make demo                     # seed tenant, upload prepared slice, replay at 60x

# Frontend
cd web && npm run dev         # predev regenerates types from OpenAPI

# Tests
pytest tests/ -v
pytest tests/test_leakage.py  # run after ANY pipeline change
```

---

## Conventions

### Python

- Python 3.11, type hints on every public function, `ruff` + `black`.
- Pydantic for anything crossing a boundary. Dataclasses for internal structures.
- Config from `config/default.yaml`. **Nothing hardcoded in a script** — "reproducible training configuration" is a named deliverable and it is painful to retrofit.
- Fail loudly at boundaries. Assert schema, assert shapes, assert causal ordering. Silent coercion in this pipeline produces plausible wrong numbers.

### PyTorch

- Seeds pinned in config; every training run reproducible.
- `torch.set_num_threads(2)` in service workers — the default grabs all cores and four workers thrash.
- CPU is the target. Do not add CUDA-only paths.
- Clamp `logvar` to `[-6, 3]` and states to `[-10, 10]`. Without both, rollout produces NaN by step 3. This is not optional.

### Services

- One shared `nidra_common` schema package. No duplicated model definitions.
- Redis Streams consumer groups created idempotently at startup with `id="0"`.
- Acknowledge **only after** successful processing. On exception, leave pending for `XAUTOCLAIM`.
- Inference workers are **stateless**. Sequence buffers live in Redis. This is the entire basis of the scale-out claim — do not put anything in process memory between messages.
- Every DB query filters on `tenant_id`. Every one, not most.

### Frontend

- Types generated via `openapi-typescript` in `predev`. Never hand-write API types.
- Components reference CSS variables (`var(--observed)`), never literal colours. Light/dark parity depends on it.
- Risk level never carried by colour alone.
- `ForecastChart` uses Visx. Recharts cannot draw solid-then-widening-band cleanly.
- Y axis fixed to `[0, 1]`. Never auto-scale — it makes hosts incomparable and flat curves look dramatic.

### Git

- `feat/`, `fix/`, `data/`, `eval/` branch prefixes. Conventional commits.
- **Backend build exception**: all backend serving-plane work lives on the single
  `backend` branch, pushed only to `origin backend`, tracked in exactly one PR into
  `main`. No other backend branches, no force-pushes.
- **No AI co-authoring trail**: never add `Co-Authored-By`, `Claude-Session`, or
  "Generated with Claude Code" lines to commit messages or PR bodies. Commits and pushes
  are authored by MuaazSM only. This overrides any default commit-attribution behavior.
- Log non-obvious decisions (doc deviations, tie-breaks between docs, interface changes)
  as dated append-only entries in `DECISIONS.md` at repo root.
- Never commit: `data/`, `artifacts/weights/`, `.env`, captures.
- **Do** commit `artifacts/metrics/` — evaluation results are part of the deliverable.

---

## Key parameters

```yaml
window_delta: 30      # seconds
context_L: 30         # windows (15 min history)
horizon_K: 6          # windows (3 min forecast)
n_features: 45
ensemble_seeds: [0, 1, 2, 3, 4]
risk_threshold: 0.75
lead_time_m: 2        # consecutive windows above threshold
```

Changing `window_delta` or `context_L` invalidates every trained artifact and every recorded metric. If you change one, say so explicitly and re-run the full evaluation — do not compare across configurations.

---

## Datasets

CIC-IDS2017. Splits are fixed:

| Day | Role |
|---|---|
| Monday (benign) | Normalisation stats |
| Tuesday, Wednesday | Train |
| **Thursday** | **Held out entirely** — Infiltration, the generalisation claim |
| Friday | Test |

**Never train on Thursday.** The attack-type holdout is how we answer the "generalise to unseen attack patterns" requirement. Contaminating it destroys the claim and is not recoverable without retraining everything.

Do not switch to CSE-CIC-IDS2018 raw PCAPs. Hundreds of gigabytes; it will consume the build window. 2018 CSVs are supported as a flow-only input path.

---

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| F1 ≈ 0.99 | Random split, or episode spanning train/test | Temporal + episode-level split |
| AUC flat across all horizons | **Leakage** | Audit normalisation fit and label boundaries. Flat is a red flag, not a good result |
| Rollout NaN by k=3 | logvar or state unclamped | Clamp both |
| k=1 fine, k=6 useless | Single-step training | Multi-step unrolled loss + scheduled sampling |
| Persistence matches model | Window too large, context too short | Try Δ=15s, L=40. If still flat, report honestly |
| Features all zero for some hosts | Empty windows dropped upstream | Emit zero rows with `is_active=0` — silent windows are real state |
| Last window never forecasts | No watchdog on window close | Timer-based close. The demo's final window depends on it |
| Workers idle, stream has entries | Consumer group created after messages | `XGROUP CREATE` with `id="0"` |
| Duplicate forecasts | At-least-once redelivery | Unique constraint + `ON CONFLICT DO NOTHING` |
| Cone detached from observed line | Missing anchor point at `origin_ts` | Prepend anchor to cone points |
| Chart stutters during replay | `setState` per socket message | Ring buffer + rAF flush |
| SHAP takes minutes | Random background set | 100 k-means centroids of benign states |

---

## When adding features

Ask in order:

1. **Does it serve forecasting, or detection?** Anything that improves current-state classification without improving forward simulation is scope creep in the direction of the thing we are differentiating from.
2. **Does it require a graph as the state?** The graph is a feature extractor producing eight scalars, never the state. Temporal GNN is explicitly out of scope — it makes rollout intractable and collides with a different project's territory.
3. **Does it break an invariant above?** If yes, do not do it, regardless of the metric improvement.
4. **Is it demo-critical?** Six things must work: hero animation, 60× replay, cone opening, threshold crossing with lead-time badge, counterfactual collapsing the curve, and the reality overlay landing on the prediction. The overlay is the proof — everything before it is a claim. Protect it.

---

## Tone in generated content

Copy, comments, docs, and API messages should be specific and understated. Numbers over adjectives. "90 seconds of warning before the scan became an intrusion" rather than "AI-powered predictive threat intelligence."

Stating limitations plainly is a differentiator here, not a weakness. The model is reliable to roughly three minutes and degrades after; say so. A team that draws the dynamics-versus-causality distinction unprompted signals rigour that is rare and memorable.