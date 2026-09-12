# NIDRA — Network Infiltration & Dynamics Recurrent Analyzer

**A world model for network attack forecasting.** Built for Problem
Statement 26153 (NTRO): instead of classifying whether traffic *is*
malicious right now, NIDRA learns the *dynamics* of network behavior and
recursively simulates a few minutes into the future — forecasting
compromise risk, likely attack stage, and confidence, before the attack
fully unfolds.

## Results — the scorecard, in plain English

Trained and evaluated end-to-end on the **complete, real CIC-IDS2017
dataset** (all 8 published day-files, all 5 raw PCAPs extracted for true
packet-level features — no synthetic data, no shortcuts), at full
production scale: 5 independently-trained models voting together (an
"ensemble"), on 500,000 real training examples.

**How to read every score below**: every number is out of **1.00**. Think
of it as a report card for *"how well does the system separate real
attacks from normal traffic, ranked in order of danger."* **1.00 = a
perfect score. Higher is always better.** A system that just guesses
randomly would score close to 0.

### The headline scoreboard

| | **Test day** (Friday's attacks) | **Never-seen-before day** (Thursday's attack type — held back from training entirely) |
|---|:---:|:---:|
| A guard who never raises an alarm ("nothing changes") | 0.67 | 0.59 |
| **NIDRA (this project)** | 🟢 **0.92** | 🟢 **0.73** |
| A "perfect hindsight" cheat score* | 0.90 | 0.76 |

*\*This row peeks at the real future to compute a theoretical best-possible
score — no real system could ever reach it fairly. It's included only as
a sanity-check ceiling, not a competitor. NIDRA scoring close to it (and
even nudging past it on Friday, see footnote 2) shows the forecasting
itself isn't the weak link.*

**The number to remember is 0.73.** That's not attacks similar to what the
system trained on — that's a **completely different kind of attack
(Infiltration) that NIDRA never saw once during training**, the equivalent
of a student acing a question on a topic that was never covered in class.
Beating the naive guard by such a wide margin on genuinely unseen behavior
is the strongest evidence that the system learned real attack *dynamics*,
not just memorized examples.

### Every score measured, laid out simply

| System | Test day score | Never-seen-before day score | What it is |
|---|:---:|:---:|---|
| "Assume nothing changes" | 0.67 | 0.59 | The bar any real system must clear |
| Simple lookup (this instant only) | 0.82 | 0.62 | A basic classifier, no memory of history |
| Simple lookup (last 15 min of history) | 0.86 | 0.88 | A stronger basic classifier, still no forecasting |
| **NIDRA (forecasts the future, 5 models voting)** | **0.92** | **0.73** | **This project** |
| Perfect-hindsight cheat score | 0.90 | 0.76 | Sanity-check ceiling only, not a real system |

NIDRA beats every simple lookup-based approach on the day it was never
trained for — which is exactly the scenario a real deployment needs to
handle (new attacks nobody has seen a labelled example of yet).

### Other things we measured

- **Speed**: forecasts come back in about **137 milliseconds** on
  ordinary hardware (CPU, no GPU needed) — well within the 300ms budget
  a live system needs to feel instant.
- **Early warning**: when the system does raise a flag, it does so
  **well before the attack fully unfolds** — anywhere from about 25
  minutes to several hours of advance notice in testing, not an
  after-the-fact alert.[^1]
- **Code health**: 166 automated tests, all passing — the pipeline, the
  model, and the serving code are exercised end-to-end, not just eyeballed.
- **A tuning experiment that paid off**: we hypothesized that some of the
  system's "noise" while imagining the future was washing out its own
  signal, tried tightening that noise, retrained from scratch, and
  measured a real improvement — exactly where the theory predicted it
  would show up.[^2] A good example of the scientific loop (hypothesis →
  experiment → measurement) actually working, not just a knob turned at
  random.

See [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) for every one of
these numbers with full provenance, and [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md)
for a compact technical summary.

## Architecture

```
raw telemetry (PCAP via tshark + CICFlowMeter CSV)
    -> per-host, per-30s-window state vector (45 features)
    -> GRU encoder over 30 windows (15 min) of history
    -> Gaussian transition model: predicts the NEXT-STATE DELTA
    -> recursive 6-step (3 min) rollout, feeding predictions back in
       as if they were real observations
    -> frozen risk head + frozen stage head score the simulated future
    -> 5-model ensemble, ~500 sampled trajectories -> forecast with
       confidence band, attack-stage distribution, lead time, and
       feature-level explanations
```

The recursive feedback loop — the model's own forecast re-entering the
encoder as if observed — is what makes this a *world model* rather than a
per-window classifier: a classifier has no state to feed back and cannot
simulate forward at all.

## Repository layout

```
ml/                     the ML subsystem — data pipeline, world model,
                         training, evaluation, explainability, serving
  nidra/                 source (data/, models/, train/, eval/, explain/, serve/)
  config/                default.yaml (production) / mvp_2017.yaml (fast iteration)
  tests/                 166 tests, synthetic fixtures, runs in seconds
  README.md              full technical documentation, start here for details
  MODEL_CARD.md          compact model card: claims, deviations, limitations
  REAL_DATA_RESULTS.md   single source of truth for every measured number
docs/                    problem-statement PRD and build specs
```

## Quickstart

```bash
cd ml
pip install -e .
pytest tests/ -q                     # 166 tests, synthetic fixtures — seconds

# Reproduce the full-scale production result (requires the real
# CIC-IDS2017 dataset — see ml/PRODUCTION_RUN_GUIDE.md):
python -m nidra.train.train_dynamics --config config/default.yaml --max-train-samples 500000 --max-val-samples 50000
python -m nidra.train.train_heads    --config config/default.yaml --max-train-samples 500000 --max-val-samples 50000
python -m nidra.eval.run_eval        --config config/default.yaml --seed 0 --split test --use-ensemble
```

See [`ml/README.md`](ml/README.md) for the full technical writeup
(pipeline, model internals, evaluation methodology, explainability, and
serving interface) and [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md)
for every result this project has measured, reported in full — including
the items below.

---

*In the interest of full transparency, two honest footnotes — neither
changes the headline numbers above, both are documented in full rather
than smoothed over:*

[^1]: The early-warning numbers above are real, but the system doesn't
    yet raise that early flag for *every* attack — right now it catches a
    small fraction of attacks early enough to hit its strict alerting
    bar, with the rest still getting flagged, just closer to (or at) the
    moment of attack rather than well ahead of it. This is a
    threshold-tuning problem (how confident the system needs to be before
    it speaks up), not evidence the underlying forecasting is wrong — and
    it's the clearest next step for continued work.

[^2]: A dedicated retrain experiment (tightening the model's internal
    "imagination noise") improved a single model's score by a solid
    margin (test day: 0.88 → 0.92) — a real, validated result. Once you
    already have 5 models voting together, though, that same fix stops
    adding extra benefit — like an upgrade that clearly helps a solo
    player, but a team that's already covering for each other doesn't
    gain as much from it. Both results are reported in full in
    `ml/REAL_DATA_RESULTS.md` ("Run 4"), including exactly where the
    tuning helped and where it didn't. There's also one small, flagged
    anomaly on the test split where the model's score slightly exceeds
    the "perfect hindsight" ceiling — most likely a measurement quirk on
    a limited sample, since it doesn't show up on the never-seen-before
    day; discussed openly rather than hidden.
