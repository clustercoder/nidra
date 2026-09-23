# Run 9 — what did not work, and what each failure taught

§33: *never hide failed experiments, never delete poor results*. This is the
list. Every entry names the intervention, the number that killed it, and what
the phase did differently afterwards — because an experiment that changed
nothing about the next one was not worth running.

Twelve entries. Six are interventions that did not beat their baseline, five are
mistakes in the measurement apparatus that produced confident wrong numbers or
no numbers at all, and one is a pre-registered explanation whose own test
refused it.

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

### 11. Scoring a model with a scaler from a different feature regime

**What happened.** A probe loaded `config/combined_eval_ctu.yaml` directly. The
queue runs that config with `--set artifacts.scaler_dir=<source run>/artifacts/scaler`;
without the override it resolves to the shared `ml/artifacts/scaler`, which a
later training run had rewritten at the full-45 feature regime — against a model
trained at `cross_core`'s 32.

**Why nothing caught it.** A dropped feature is **zeroed, not removed**. Shapes
match, `load_state_dict` succeeds, the rollout runs, the AP is finite and
plausible. The only symptom was a number 2.9× away from the same cell's
independently recorded value, and it was noticed only because that recorded
value existed to check against.

**Scope.** All 20 recorded cross-evaluation cells were audited; every one used a
scaler from its own model's family. No published number is affected. The probe
was the only thing wrong.

**What it changed.** `load_models` now compares the loaded scaler's drop set
against each checkpoint's recorded `dropped_features` and refuses on mismatch
(D142, §3.28). The drop set rather than a file hash: a hash would false-alarm on
an equivalent refit, is not recorded anywhere, and would not say what differs.
The transfer configs are still not self-contained, which the guard converts from
a wrong answer into a refusal but does not solve.

### 12. A queue that could not report a failure, and configs that could not express a head

**What happened.** Three `state+hidden` transfer arms — `CIC → CTU`,
`CTU → CIC`, `CIC+CTU → CTU` — crashed on every split and produced no metrics.
The transfer configs do not declare `model.risk_head.components`, so a 45-wide
head was built against a 173-wide checkpoint.

**Why nothing caught it.** `load_state_dict` raised a clear shape error. The
queue then ran `echo "END $label/$split rc=$?"` **after a pipeline**, so `$?`
was `tail`'s status and all three were recorded `rc=0` — three seconds after
their START, beside fourteen-minute neighbours. Six cells were missing from the
cross-dataset matrix and the only symptom was three dashes in a scorecard dry
run done before the report queue reached it.

**What it changed.** The cells are re-queued with the override, verified to load
first. The recovery queue reads `${pipestatus[1]}` and prints `FAILED`.
`load_models` refuses the mismatch by name with the fix quoted (D143). And
§3.25's "all 42 exited zero" is withdrawn — that column was measuring `tail`.

**The part worth keeping.** Two independent slips were required: a config that
could not express the variant, and a log that could not report a failure. The
thing that saved it was a convention rather than a test — the scorecard renders
a missing cell as a gap instead of borrowing a neighbour's number, so the
absence was visible the moment anyone looked. Three dashes were recoverable;
three plausible numbers would not have been.

---

## Explanations that their own pre-registered test refused

### 10. The attractor account of §3.26's rising-with-horizon AP

**Idea.** Two of fifteen cross-evaluation cells have a deterministic forecast
whose AP *rises* with horizon (1.29× and 2.04× from k=1 to k=6, against 0.01–0.68×
everywhere else) and beats the true-future oracle at all six horizons. §3.26
proposed that the recursive rollout drifts toward class-dependent attractors of
its own dynamics, and that those attractors separate infected hosts from benign
ones better than real states do.

**Pre-registered criterion,** written into §3.26 before the measurement: the
combined model's predicted terminal states separate the classes better than the
true future states do, *while the CIC-only model's do not*.

**The number that refused it.** Fisher-style separation (distance between class
means over mean within-class spread) on 2,400 sampled eval rows, 400 positive:

| cell | anomalous? | true `S[t+6]` | predicted `Ŝ[t+6]` | amplification |
|---|---|---|---|---|
| comb → CIC, test | **yes** | 1.177 | 2.435 | 2.07× |
| comb → CTU, holdout | **yes** | 0.911 | 1.813 | 1.99× |
| CIC → CIC, test | no | 1.257 | 2.173 | **1.73×** |
| CTU → CTU, test | no | 0.636 | 0.627 | 0.99× |

The CIC-only model amplifies separation too, at 1.73×, and it shows no anomaly.
The criterion required it not to. The two anomalous cells do have the largest
amplification, and the ordering is consistent — but 1.73× against 2.07× is not a
distinction that separates an anomaly from a non-anomaly, and treating it as one
after the fact is what pre-registration exists to prevent.

The drift numbers point the other way as well. The rollout does not *collapse*;
it **over-moves**. Predicted benign hosts travel 7.1, 7.4, 8.3 and 7.8 units from
their origin where the real ones travel 3.3, 2.9, 5.0 and 3.5 — roughly double, in
every cell. An attractor account predicts shrinking spread and shrinking drift.

**What it taught us.** Two things. The instrument was weakly coupled to the
question: L2 separation between class means in raw feature space is not what a
nonlinear frozen head reads, so even a clean positive would have been indirect
evidence. A test of "does the head score predicted states better" has to be run
through the head. And the phenomenon is now better specified than the
explanation: the rollout systematically inflates both class separation *and*
benign drift, in three of four cells, which is a property of the transition
model worth its own investigation and is not what §3.26 guessed.

§3.24's two cells stay **unexplained**, and the log says so rather than keeping
the hypothesis alive on a partial match.

---

### 13. "The decomposable head is a dead end" — closed on CTU, wrong in general

§3.22 read the CTU component table and concluded that Q13's explainability cost
was probably intrinsic: `state+logvar` carried a seventh of what the hidden state
carried, so there was no decomposable head to retreat to. The same section
pre-committed to running the four missing CIC arms, on the reasoning that a
negative should close a direction properly rather than leave it standing as an
untested suggestion.

The CIC arms ran. `state+logvar` reaches **0.7754 against `state+hidden`'s
0.7825** — 99.1%, inside the ±0.047 seed spread — where CTU gave 14%.

The negative result here is the *generalisation*, not the direction: "the AP gain
and the explainability loss are the same thing" was true of the dataset it was
measured on and false as a statement about the model. It had been written into
Q13 and Q14 as a general limit. Both are revised.

What made the difference was running the control on the other dataset. The
conclusion was already written and would have survived unchallenged. *(§3.38.)*

### 14. A defect, not an intervention: the rollout's phantom features

Listed here because the rule is that nothing gets hidden, and because it
invalidates numbers this log has already published rather than an idea it was
considering.

The transition loss masks dropped features, so the network is untrained on those
outputs; the rollout fed them back and they compounded to rms 2.03 by k=6 — more
magnitude than the real features, in slots that are exactly zero in both the
input and the truth. Every CTU and cross-dataset number in this phase was
produced under it.

Two findings in this log looked, briefly, as though the defect explained them.
§3.24's "the oracle is not an upper bound" and the deterministic arm's
below-chance ROC both have an obvious candidate mechanism in the phantom drift. A
test was pre-registered in §3.37 before the re-scores ran: masking should remove
the oracle-beating.

**It did not, and both candidate explanations are dead.** On `comb2cic/test` the
raw arm went from beating its oracle by +0.095 to beating it by **+0.150**, and
the deterministic arm's ROC moved 0.254 → 0.266. §3.24's cells stay unexplained
and entry 10 stands exactly as written.

So the defect is real, it invalidates numbers, and it explains neither of the two
things it looked like it would explain. Worth separating those: a defect large
enough to be obviously important is not thereby the cause of whatever else is
unexplained nearby.

And a claim of my own, written off the first re-scored cell and withdrawn two
cells later. Raw AP: comb → CIC test **+0.0548**, CTU → CTU test +0.0050, comb →
CTU test **−0.0147**. "The phantom was costing accuracy, not manufacturing it"
was true of the cell it was drafted from and false of the third. An untrained
signal projected through untrained weights has no reason to point the same way
twice, and one cell was never enough to say which way it pointed. *(§3.37,
D145.)*

---

### 15. A fix that was a no-op, and reported itself as a result

Not an experiment — a near-miss, listed because it would have produced a
published number.

D145's mask was wired at each call site: `benchmark.load_models` and
`NidraPredictor`. The *training* entry point was missed. Re-running §3.38's
decomposable-head comparison "under the fix" returned `state+logvar` =
**0.7754417090891179**, bit-identical to the unfixed run, because
`build_head_context` was calling `mask_logvar` on a model whose mask was still
`None`.

The run exited zero. It wrote a full provenance record. Its conclusion —
"§3.38's result is unaffected by D145" — is a perfectly reasonable finding, and
would have gone into the report as one.

What caught it was that bit-identical is *too good* for a retraining run. Two
head trainings that differ in their inputs do not agree to sixteen significant
figures; two that agree to sixteen significant figures had identical inputs. The
same property that made §3.36's reproducibility check strong made this one
impossible to miss once looked at.

Two changes followed. The mask now comes from `world_model_from_config`, the
single construction point, so a path cannot be added without it. And the
re-queued comparison carries `state` and `state+hidden` as controls: they do not
read `logvar`, so they must come back unchanged — if *everything* comes back
unchanged again, the harness is lying rather than the result being negative.
*(D145.)*

---

## What this list is for

Three of the six interventions were killed by a measurement that cost minutes
(4), by a decomposition of a result already in hand (5, 6), or by an ablation
that was going to be run anyway (3). One was killed by building the thing and
finding it lost (1), one by building it and finding it won on the wrong axis
(2). That ratio is the argument for measuring the assumption before building the
intervention, and it is why §3.13 was pre-registered before it was run.

Entry 10 is the first case of pre-registration refusing one of this log's own
explanations rather than one of its interventions. The criterion was written
down while the hypothesis still looked obvious, and the control that killed it —
a CIC-only model showing the same effect — is one that would have been very easy
not to run.
