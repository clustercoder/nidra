# Run 9 — what did not work, and what each failure taught

§33: *never hide failed experiments, never delete poor results*. This is the
list. Every entry names the intervention, the number that killed it, and what
the phase did differently afterwards — because an experiment that changed
nothing about the next one was not worth running.

Nine entries. Six are interventions that did not beat their baseline; three are
mistakes in the measurement apparatus that produced confident wrong numbers
before being caught, which belong here for the same reason.

---

## Interventions that did not work

### 1. Score fusion of the two frozen heads

**Idea.** The stage head ranks CTU's c2 at ROC 0.854 and recon at 0.638 where
the risk head sits at 0.434 and 0.320 — below chance. Both heads are frozen and
already trained, so combining their scores adds no parameter and fits nothing on
the rows it scores. The signal is inside the model; take it.

**Result.** All five parameter-free rules lose against the published head:
noisy-or −0.0026 AP, max −0.0097, mean −0.0233, geometric −0.0379. Per stage a
fused score lands *between* the two heads (c2: 0.603–0.682 against 0.434 and
0.828), never above either.

**What it taught.** One pooled ranking cannot hold two orderings at once: the
risk head's confident scores on 172 exfil windows outrank the stage head's
correct ordering of 24 c2 ones. The fix therefore cannot be post-hoc arithmetic
on outputs — it has to be in the objective. That is what motivated the
stage-balanced experiment (§3.13), and it is motivated by this failure rather
than by a guess. *(§3.9, D128.)*

### 2. `context_noise` as a fix for the exposure mismatch

**Idea.** The history-aware head is trained on observed hidden states and
applied to rollout hidden states, which are off-distribution. Adding noise to
the context during head training should close the gap.

**Result.** It closed the gap and *erased* the transition model's measured
contribution — the `persistence_rollout` margin, the only interval in the
project excluding zero. σ=0.1 bought +0.007 AP, inside the noise of 283
positives; σ=0.3 was worse on every column.

**What it taught.** A head made insensitive enough to its context to tolerate
rollout error is also insensitive to what the rollout got *right*. The knob
stays, defaulting to 0.0, because the diagnosis is correct and a mechanism that
closes the gap without flattening the head is still the obvious next thing to
try. *(§3.6, D126.)*

### 3. Uncertainty as a head input

**Idea.** The transition model predicts a log-variance; a head that reads it
should know when to distrust its own state prediction.

**Result.** `logvar` did not earn its place in the head-component ablation.
*(§3.4.)*

**What it taught.** Ordered against the other components, predicted uncertainty
is not what the risk head is missing — history is. It also answers §31 Q5
directly, which is why the ablation was run over components rather than a single
proposed head.

### 4. A finer Δ to recover CTU's precursors

**Idea.** CTU's pre-onset windows are empty at Δ=60. Perhaps a 60-second window
is averaging a short precursor away, and Δ=15 would see it.

**Result.** Killed before it was built, by one query. The infected host has
emitted nothing for a median of 4.5 minutes (validation) or 47 minutes (train)
before an attack begins, at worst 21 hours. Splitting a silent minute yields
silent quarter-minutes.

**What it taught.** Measure the data property an intervention assumes before
paying for the intervention — changing `window_delta` invalidates every artifact
and every recorded metric in the project. It also redirected the
advance-warning work to CIC, where the host *is* active right up to onset in 83%
of validation cases. *(§3.15, D134.)*

### 5. Reading the aggregate head-ablation gain as the result

**Idea.** `state` → `state+hidden` raises validation AP in all three training
regimes (+0.136 CTU, +0.104 CIC, +0.128 combined). Three replications.

**Result.** On CTU the gain is largely the head getting better at picking out
the single infected host: both heads score *below* a host-level constant, and
within that host they reach ROC 0.656 and 0.662. On CIC the gain survives
(within-host ROC 0.9245 against 0.8256).

**What it taught.** An aggregate replication shows an effect is not a fluke of
one split; it does not show the effect is the one being claimed. Every per-group
and per-split number in this project now carries the host/timing decomposition,
in the benchmark, the scorecard and the report tables. *(§3.11, §3.12, D130,
D131.)*

### 6. The 58× lift on the silence floor

**Idea.** On rows where the state is provably uninformative, the history-aware
head reaches 58× the state-only ceiling. The strongest single number in the
phase.

**Result.** All 58 positives sit on one host. The host-mean collapse reproduces
the ranking at ROC 0.9993, and within that host the head scores 0.98× lift, ROC
0.3836 — worse than a constant.

**What it taught.** The tell was visible in the number itself: a ranking cannot
be 58× lift and ROC 0.4242 at the same time unless the gain is one block of rows
lifted wholesale. Lift and ROC disagreeing is now something to check rather than
something to pick between. *(§3.10, D129.)*

---

## Measurement mistakes that produced confident wrong numbers

These are not experiments that failed. They are cases where the apparatus lied
and was caught, which is worth recording because the next apparatus can lie the
same way.

### 7. The separability probe scored unscored rows as zero

Every CTU validation attack group has its positives on exactly one host, so
host-grouped cross-validation leaves those positives in no scoring fold. They
kept a default score of 0.0, which put every positive at the bottom and produced
a confident **ROC of 0.100** for all four groups — a number that reads as a
finding. Rows no fold could score are now dropped, single-positive-host groups
fall back to row-stratified folds flagged `host_leaky`, and `n_positive_hosts`
is reported. *(D127.)*

### 8. A failed probe read as proof of unlearnability

With that bug fixed, `ctu_4:c2` came back at ROC 0.443 and was written up as
"not separable at all — no head architecture fixes that". NIDRA's own frozen
stage head then reached **ROC 0.864** on the same 23 windows under the same
cross-host construction. The probe was the weaker learner, not the ceiling.

The verdict is one-sided and now says so in the docstring, the property and the
table: a `yes` proves the signal exists and transfers, a `no` proves only that
this probe missed it. The retracted paragraph is struck through in §3.8 rather
than deleted. *(§3.8, §3.9, D128.)*

### 9. A provenance record that misstated which split a capture was in

`dataset.days[].role` is a hand-written annotation and had drifted from
`cfg["splits"]`, which is what actually assigns days: `ctu_4` and `ctu_6` are
annotated `role: test` and are the validation captures. Nothing downstream reads
the annotation, so no result is affected — but a record that misstates a split
is worse than no record. `provenance.py` now derives the role from
`cfg["splits"]`. *(§3.16, D135.)*

Two shell-level mistakes belong in the same spirit and are recorded in
`DECISIONS.md` rather than here: a zsh glob silently turned the entire onset
2×2 into four 0-second no-ops that "completed" successfully, and editing a
running shell script corrupted its read position so a benchmark re-ran a variant
it had been told to skip. Both are visible in the experiment matrix as 0-minute
rows.

---

## What this list is for

Three of the six interventions were killed by a measurement that cost minutes
(4), by a decomposition of a result already in hand (5, 6), or by an ablation
that was going to be run anyway (3). One was killed by building the thing and
finding it lost (1), one by building it and finding it won on the wrong axis
(2). That ratio is the argument for measuring the assumption before building the
intervention, and it is why §3.13 was pre-registered before it was run.
