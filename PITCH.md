# NIDRA — Pitch Summary

**Network Infiltration & Dynamics Recurrent Analyzer** · Problem Statement 26153 (NTRO)

## The problem

Intrusion detection today looks at a snapshot of traffic and asks *"is this
malicious, right now?"* It treats each flow in isolation and throws away the
thing that makes an attack an attack: it unfolds over time.

## The idea

**Model how the network behaves, then simulate forward.**

NIDRA reads the last 30 minutes of each host's traffic, learns how that host's
behaviour tends to change from one minute to the next, and projects the next six
minutes step by step. It then scores the projected future for attack risk, maps it
to MITRE ATT&CK stages, and shows which traffic features drove the call.

```
30 minutes of real traffic per host (flow + packet features, every 60 s)
        ↓
   a learned model of how that traffic changes minute to minute
        ↓
   six minutes simulated forward, 1,000 sampled futures, 5 models
        ↓
   a risk curve with an uncertainty band, projected attack stages,
   and the features behind the forecast — offline, on a laptop CPU
```

Trained and tested on the **complete, real CIC-IDS2017** dataset — every attack
day, every raw packet capture — and, in the latest research phase, on **CTU-13**
as a second corpus.

## What it demonstrably does

**1. It learns how network traffic evolves.** Its six-minute forecasts of a
host's next state have **59–67 % lower error than assuming nothing changes**,
and **5–6 % lower than a linear model** fit to the same task — on every data
split, at every horizon from one to six minutes. The linear model gets most of
the way; NIDRA's edge over it is modest and it never goes away. This is the
"world model" claim, measured directly.

**2. It raises fewer, more precise alarms than the logistic-regression baseline
the problem statement asks for.** Both at their own best threshold, chosen before
the test data was seen:

| | NIDRA | Logistic regression |
|---|:---:|:---:|
| false alarms per hour, Friday's attacks | **0.48** | 4.80 |
| false alarms per hour, never-seen attack type | **0.86** | 4.89 |
| precision, never-seen attack type | **0.85** | 0.46 |
| F1, never-seen attack type | **0.46** | 0.35 |
| average precision, Friday / never-seen | **0.058 / 0.438** | 0.033 / 0.241 |

**10× fewer false alarms** on the test day and **5.7× fewer** on an attack type
held out of training entirely. A security team that stops trusting its alerts
stops reading them; this is the property that keeps them read. (On Friday's F1
alone, logistic regression is slightly ahead, 0.07 to 0.06.)

**3. Nothing is hidden and everything reproduces.** Every forecast comes with
the traffic features driving it. Every result is re-runnable from a recorded
configuration — 28 evaluation runs were repeated and matched to the last digit.
It runs fully offline in about 128 ms per forecast on a MacBook Air.

## What it does not do — yet

We say this before anyone asks, because a result that survives the hard
questions is worth more than a bigger number that doesn't.

- **It does not give advance warning on this data.** No attack in the test or
  held-out days was flagged before it began. CIC-IDS2017's attacks start from an
  outside machine with almost no warning signs on the victim beforehand — the
  training data holds 15 examples of "an attack begins within five minutes" in
  2.27 million host-minutes. Earlier drafts of this document claimed hours of
  warning, then 8 of 10 attacks caught early; both are withdrawn.
- **The simulation does not yet beat a simpler reading.** Scoring the current
  state with the same risk model does about as well as scoring the simulated
  future. The learned dynamics are real (point 1); turning them into better
  attack forecasts is the open problem.
- **Sequence classifiers are competitive.** A classifier reading the full
  30-minute history matches NIDRA on average precision, and ranks Friday's
  botnet traffic better.
- **It learns correlation, not cause.** The data is observational. NIDRA
  projects what traffic will probably look like, not why; its what-if tool is
  labelled "model-internal what-if" everywhere.

## Constraints

- **One MacBook Air** (Apple M1, 16 GB), no GPU, no cloud. A five-model training
  run takes about five hours.
- **Attacks are rare in the data.** The risk model learns from roughly 150
  attack-labelled training states; attacks are 0.03–0.4 % of host-minutes on the
  evaluation days, and every score here is measured at that real rate rather than
  on a rebalanced sample.
- **Three attack stages never appear in training.** Reconnaissance, command and
  control and lateral movement occur only on the evaluation days.

## How we worked

Every design choice was measured on validation data and frozen before the test
data was scored. Much of the work was finding where our own numbers were wrong:

| What we found | What changed |
|---|---|
| The flow and packet records were joined on clocks 3 and 12 hours apart, so 11 of 45 features were always zero | Fixed, retrained |
| Our old headline (F1 0.91) used a scoring rule picked by looking at the test day, on a rebalanced sample | Withdrawn; every setting is now chosen on validation, at the real attack rate |
| The simulation wrote untrained values into feature slots a dataset doesn't have, and they compounded | Fixed across every code path (D145) |
| A more complex risk model won on its training objective | Tested before adopting; it failed a pass/fail bar set in advance, so it was not adopted (D146) |
| The benchmark scored logistic regression at *our* model's threshold, overstating its false alarms 9–14× | Corrected: each system gets its own threshold (D148) |

Nineteen failed or corrected experiments are written up in
[`ml/reports/RUN9_NEGATIVE_RESULTS.md`](ml/reports/RUN9_NEGATIVE_RESULTS.md).

## Where it's headed next

The model learns network dynamics; the risk layer reading those dynamics is the
bottleneck. The next step is a risk model that makes better use of the simulated
trajectory without being trained on it, which keeps the forecast honest. Better
data for advance warning — attacks with a visible run-up on the same host —
matters more than a bigger model.

## Learn more

- [`README.md`](README.md) — results, limits and quickstart
- [`ml/ARCHITECTURE.md`](ml/ARCHITECTURE.md) — the two-page architecture document
- [`ml/reports/PS_BASELINE_BENCHMARK.md`](ml/reports/PS_BASELINE_BENCHMARK.md) — the
  logistic-regression comparison, with each system at its own threshold
- [`ml/REAL_DATA_RESULTS.md`](ml/REAL_DATA_RESULTS.md) — every number and every
  experiment, with provenance
