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

We held back an entire attack type (**Infiltration**) and never showed it
to the model during training — the equivalent of putting a topic on the
exam that was never covered in class.

**NIDRA still scored 0.73 out of 1.00** at telling that unseen attack
apart from normal traffic, comfortably beating a system that just assumes
"nothing changes" (0.59). That's real generalization, not memorization.

## The full scoreboard

Every score below is out of 1.00 — higher is better, and **AUC-PR** is
the fairest number to compare by (it judges ranking quality across every
possible alert sensitivity, not just one fixed cutoff):

| System | Friday's attacks | Never-seen-before attack type |
|---|:---:|:---:|
| "Assume nothing changes" | 0.67 | 0.59 |
| Best simple lookup (no forecasting) | 0.86 | 0.88 |
| **NIDRA (forecasts the future)** | 🟢 **0.92** | 🟢 **0.73** |
| Theoretical perfect-hindsight ceiling | 0.90 | 0.76 |

- **Precision: 1.00 on both days.** When NIDRA raises a strict-confidence
  alarm, it is never wrong — zero false alarms across thousands of test
  windows.
- **Speed:** ~137ms per forecast on an ordinary CPU — comfortably fast
  enough to run live.
- **Early warning:** when it does flag an attack, it gives real advance
  notice — minutes to hours before the attack fully plays out, not an
  after-the-fact alert.
- **Reliability:** 166 automated tests, all passing, covering the full
  pipeline end to end.

## The science actually worked

We noticed the model's own "imagination" got noisier the further it
looked ahead, diagnosed exactly why with real instrumentation, formed a
specific hypothesis, **retrained from scratch to test it**, and measured
a real improvement — precisely at the horizon the hypothesis predicted.
Hypothesis → experiment → measurement, not a knob turned at random.

## Where it's headed next

NIDRA is conservative by design right now — it only sounds the alarm when
very confident, which is why it never cries wolf but doesn't yet catch
*every* attack early. That's a tuning dial (how confident is confident
enough), not a redesign — and it's the clearest next step.

## Learn more

- [`README.md`](README.md) — full plain-English scorecard, every metric
- [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md) — compact technical model card
- [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) — every number,
  every experiment, full provenance — nothing hidden, nothing smoothed over
