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

**NIDRA scored 0.69 out of 1.00** at telling that unseen attack apart from
normal traffic — well ahead of a system that just assumes "nothing changes"
(0.59), but, stated plainly, **behind the best simple lookup method (0.88)
on that same unseen attack type.** So the learned
dynamics do transfer to an attack never trained on, but on pure ranking
quality a plain statistical method over the last 15 minutes of history
still does that particular job better. What that method cannot do at all
is forecast forward and give hours of advance warning, which is the thing
NIDRA is actually for.

## The full scoreboard

Every score below is out of 1.00 — higher is better, and **AUC-PR** is
the fairest number to compare by (it judges ranking quality across every
possible alert sensitivity, not just one fixed cutoff):

| System | Friday's attacks | Never-seen-before attack type |
|---|:---:|:---:|
| "Assume nothing changes" | 0.65 | 0.59 |
| Best simple lookup (no forecasting) | 0.86 | 🟢 **0.88** |
| **NIDRA (forecasts the future)** | 🟢 **0.93** | 0.69 |
| Theoretical perfect-hindsight ceiling | 0.90 | 0.76 |

NIDRA wins the left column and loses the right one — the green marks say
which, rather than implying it wins both. Only NIDRA produces a forecast
with lead time, so the right column is a ranking-quality loss, not a
head-to-head loss at the job NIDRA does.

- **Precision 0.95 on the test day at 0.74 recall.** An earlier version of
  this page advertised precision 1.00 — true, but at ~1% recall, which meant
  it almost never raised the alarm at all. Fixing how the system summarizes
  its imagined futures traded a little of that precision for catching most of
  the attacks: F1 went from 0.01 to 0.84.
- **Speed:** ~137ms per forecast on an ordinary CPU — comfortably fast
  enough to run live.
- **Early warning:** 9 of 10 test attack episodes flagged before they
  unfolded, a median of about 8.8 hours ahead (previously 0 of 10).
- **Reliability:** 221 automated tests, all passing, covering the full
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
