# NIDRA — Pitch Summary

**Network Infiltration & Dynamics Recurrent Analyzer** · Problem Statement 26153 (NTRO)

## The problem

Today's intrusion detection looks at a snapshot of network traffic and
asks: *"is this malicious, right now?"* By the time it says yes, the
attack is already happening. There's no warning — only an alarm after
the fact.

## The idea

**Don't just detect attacks. Forecast them.**

NIDRA watches how a host's network behavior evolves over the last 15
minutes, then **imagines forward a few minutes into the future** —
simulating what the traffic is likely to look like next — and flags
danger in that imagined future before it becomes the present. It's the
difference between a smoke detector and a weather forecast.

## How it works (30 seconds)

```
15 minutes of real traffic history
        ↓
   learns the PATTERN of how this host normally behaves
        ↓
   imagines several minutes ahead, step by step
        ↓
   5 independently-trained models vote on how dangerous that
   imagined future looks
        ↓
   a risk score + early warning, before the attack fully unfolds
```

Trained and tested on the **complete, real CIC-IDS2017** dataset — every
published attack day, every raw packet capture — not a toy sample.

## The result that matters most

We held an entire attack type out of training — Thursday's Infiltration —
and then asked the system to forecast it cold.

**NIDRA scored 0.71 out of 1.00** at separating that never-before-seen
attack from normal traffic, catching **75%** of it at the strict mandated
confidence bar. That is the equivalent of a student acing a question on a
topic that was never covered in class, and it is the strongest evidence
that the system learned real attack *dynamics* rather than memorizing
examples.

## The scoreboard

Every score is out of 1.00 — higher is better, and **AUC-PR** is the
fairest number to judge by, since it measures ranking quality across every
possible alert sensitivity rather than one fixed cutoff.

| | Friday's attacks | Never-seen-before attack type |
|---|:---:|:---:|
| **AUC-PR** | 🟢 **0.93** | 🟢 **0.71** |
| **F1** | 🟢 **0.84** | 🟢 **0.72** |
| **Precision** | 🟢 **0.95** | 0.69 |
| **Recall** | 🟢 **0.75** | 🟢 **0.75** |
| **Median advance warning** | 🟢 **8.8 hours** | 🟢 **4.0 hours** |
| **Episodes caught before they unfolded** | 🟢 **9 of 10** | 🟢 **2 of 2** |

- **Early warning — the thing only forecasting can do.** 9 of 10 test
  attack episodes were flagged *before* they played out, a median of
  **8.8 hours** ahead. On the unseen attack type, 2 of 2 episodes flagged,
  4.0 hours ahead. A system that only judges the present moment cannot
  produce this number at all.
- **Speed:** ~137ms per forecast on an ordinary CPU — comfortably fast
  enough to run live.
- **Reliability:** 235 automated tests, all passing, covering the full
  pipeline end to end.

## The constraints these numbers were achieved under

Worth knowing before judging them, because two of the three are hard limits
rather than tuning choices:

- **Trained entirely on a MacBook Air (Apple M1, 16 GB RAM).** No GPU, no
  cluster, no cloud budget. The full candidate training set needs ~35 GB to
  hold in memory, so the model is trained on **500,000 windows out of a
  ~6.9M pool — about 7%**. The sampling keeps every single attack-positive
  window, so no attack signal was discarded; the benign context is what got
  thinned.
- **Attack data is vanishingly rare in the source dataset.** Across all 8
  published CIC-IDS2017 day-files there are about **1,006 attack-labelled
  windows in 11.4 million — 0.009%**. The risk model learns from a few
  hundred positive examples.
- **Three of the five attack stages appear only in the evaluation days.**
  The model is asked to forecast categories of attack for which it has, by
  construction, zero training examples.

So the unseen-attack result — 0.71 AUC-PR, 75% recall, 4 hours of advance
warning — comes from a model trained on a few hundred attack examples, on 7%
of the available data, on a laptop. More compute and more attack-labelled
telemetry are the two clear levers, and neither has been pulled.

## What we tried, including what failed

Seven experiments, measured and recorded with full numbers:

| Experiment | Outcome |
|---|---|
| **Pooling the riskier tail of simulated futures** | **Adopted** — F1 0.01 → **0.84**, advance warning 0 of 10 → **9 of 10**. The biggest win in the project. |
| 5-model ensemble voting | **Adopted** — beats any single model. |
| Post-hoc probability calibration | **Rejected** — undid the pooling gain. |
| Tightening rollout noise, retrained from scratch | **Rejected** — small-scale gain vanished at full scale. |
| Selecting heads on validation ranking | **Rejected** — improved the validation score, hurt real performance. |
| Pooling setting tuned at reduced scale | **Rejected on re-measurement** — the small-scale winner was the worst setting at full scale. |
| Ensemble vote ordering | **Measured as a no-op** — members agree too closely to matter. |

Four of seven were rejected on the evidence, and that is the point: each one
was built, measured against held-out data, and dropped when the numbers said
so. Three genuine correctness bugs were also found and fixed along the way,
including a silent timestamp bug that was corrupting every training window on
current library versions.

### The one that mattered most, in detail

The system's alerting was once far too quiet — it ranked danger well but
almost never crossed the confidence bar. Rather than lower the bar, we
instrumented the model and found the actual mechanism: the risk scores of a
thousand simulated futures were being averaged together, drowning out the
dangerous minority. Scoring the riskier *tail* of those futures instead is
the change behind F1 0.01 → **0.84** and advance warning 0 of 10 → **9 of
10**. Diagnosis, then a targeted fix — not a knob turned at random.

## Where it's headed next

NIDRA is tuned to be confident before it speaks — precision 0.95 on the
test day. The two clearest levers from here are the ones the constraints
section names: **more compute** (the model currently sees 7% of the
available training windows) and **more attack-labelled telemetry** (it
learns from a few hundred positive examples). Both are resource limits
rather than design problems, which is the good kind of bottleneck to have.

## Learn more

- [`README.md`](README.md) — full plain-English scorecard, every metric
- [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md) — compact technical model card
- [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) — every number,
  every experiment, full provenance — nothing hidden, nothing smoothed over
