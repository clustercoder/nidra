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

- **Early warning — the thing only forecasting can do.** 9 of 10 test
  attack episodes were flagged *before* they played out, a median of
  **8.8 hours** ahead. On the unseen attack type, 2 of 2 episodes flagged,
  4.0 hours ahead. A system that only judges the present moment cannot
  produce this number at all.
- **Speed:** ~137ms per forecast on an ordinary CPU — comfortably fast
  enough to run live.
- **Reliability:** 221 automated tests, all passing, covering the full
  pipeline end to end.

## The science actually worked

The system's own alerting was once far too quiet — it ranked danger well
but almost never crossed the confidence bar. Rather than lower the bar, we
instrumented the model, found the actual mechanism (the risk of many
simulated futures was being averaged together, drowning out the minority
that looked dangerous), and fixed it at the source. Scoring the riskier
tail of those futures instead took F1 from 0.01 to **0.84** and advance
warning from 0 of 10 episodes to **9 of 10**.

We test our hypotheses honestly, which means some of them lose: a
rollout-noise retrain and a post-hoc calibration step were both built,
measured, and rejected because the data did not support them.
Hypothesis → experiment → measurement, not a knob turned at random.

## Where it's headed next

NIDRA is tuned to be confident before it speaks, which is why its
precision on the test day is 0.95. The remaining work is catching
*every* attack early. That's a tuning dial (how confident is confident
enough), not a redesign — and it's the clearest next step.

## Learn more

- [`README.md`](README.md) — full plain-English scorecard, every metric
- [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md) — compact technical model card
- [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) — every number,
  every experiment, full provenance — nothing hidden, nothing smoothed over
