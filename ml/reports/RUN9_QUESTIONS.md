# Run 9 — the fourteen questions, answered as the evidence arrives

Living document. Every answer cites the section of
`CTU13_MODEL_IMPROVEMENT_2026-09-23.md` and the decision record it rests on, and
carries a status:

- **answered** — measured, with the artifact committed.
- **partial** — measured on validation only, or on one seed, or under a caveat
  that the running experiments will lift.
- **pending** — no evidence yet; what will settle it is named.

Status as of 2026-09-23, with the cross-dataset matrix still running. Nothing
here reads test or holdout except where explicitly stated; no such number
exists yet.

---

## Q1 — Does CTU-13 contain enough temporal precursor information to make genuine advance warning feasible?

**Answered: no.** Two independent measurements say so. The infected host has
emitted nothing for a median of 4.5 minutes (validation) or 47 minutes (train)
before an attack begins, so no finer window recovers a precursor — the Δ=15
idea is closed by measurement. And a probe fit on that host's own training rows
and scored on its validation rows returns AP equal to the base rate and ROC of
exactly **0.5000**: a constant, which is all any function can produce when every
row it scores is the same vector. *(§3.15.)*

CIC is not the same and must not be answered with it: in 83% of CIC validation
onsets the host is active in the window immediately before the attack, and the
same probe reaches **ROC 0.7500** there against a model at the floor. Most of
that is `is_active` alone (0.6783), so the genuine precursor content is the gap
between the two — real, and modest.

The detail behind the CTU answer: Every pre-onset positive on CTU is a
silent window — 71 of 71 on train, 84 of 84 on validation at `is_active == 0` —
and most are at the exact silence floor, bit-identical to 1,026,383 and 47,650
negatives respectively. Inside that stratum the best AP any function of the
state can reach is the stratum's own prevalence, 0.001216 on validation, and
the published state-only head emits exactly one distinct score across all
47,708 rows and lands on it. Twenty percent of CTU validation's positives are
in that stratum. History is the only thing that can distinguish them, and §3.10
shows what history actually does with them: all 58 floor positives sit on one
host, the host-mean collapse reproduces the ranking at ROC 0.9993, and within
that host the head scores 0.98× lift — worse than a constant.
*(§3.10, D129.)*

CIC has the same structure more weakly: its pre-onset windows are silent 24 of
45 times on train and 9 of 22 on validation, so this is not purely a CTU
artifact, but CTU's advance-warning signal specifically is a silence
phenomenon.

## Q2 — Does adding CTU-13 improve CIC-IDS2017 generalization?

**Pending.** The combined dynamics model and both its head variants are
trained; the `comb2cic` test and holdout benchmarks are in the running matrix.
The comparison is `comb2cic` against `cic2cic` at the same feature mask and the
same operating-point discipline.

## Q3 — Does training on CIC + CTU improve CTU performance?

**Pending.** Same matrix, `comb2ctu` against the CTU-only arm.

Head-training validation AP is available and is *not* an answer: the combined
regime's 0.3850 / 0.5130 and CTU-only's 0.3531 / 0.4894 are computed on
different validation sets with different prevalences, so they are not
comparable. *(§3.12, D131.)*

## Q4 — Does the history-aware risk head fix the Run 8 failure?

**Partial, and the honest answer so far is "on CIC yes, on CTU mostly no".**

The aggregate says yes three times over: `state` → `state+hidden` moves
validation AP 0.3531 → 0.4894 (CTU), 0.6781 → 0.7825 (CIC), 0.3850 → 0.5130
(combined), each with its own frozen encoder and transition so the head is the
only thing that changes. *(§3.12, D131.)*

The decomposition disagrees across datasets. On CTU both heads score *below* a
host-level constant (host-mean AP 0.7579 against 0.3531 and 0.4894) and reach
within-host ROC 0.6559 and 0.6621. On CIC both beat it and the history-aware
head reaches within-host ROC 0.9245 against 0.8256 — the gain is reproduced
inside the host, where identity is constant and only timing is left.
*(§3.11, D130.)*

Two limits on that. Both figures are validation, where each corpus has exactly
one infected host, so neither can speak to the ten-host test splits. And on CIC
that one host is `172.16.0.1`, the external attack machine, not a compromised
internal workstation. *(§3.14, D133.)*

## Q5 — Does uncertainty improve attack-risk forecasting?

**Answered: no, not as a head input.** The `logvar` component was included in
the head-component ablation and did not earn its place. *(§3.4.)* The related
`context_noise` intervention — built to fix the diagnosed exposure mismatch
between observed and rollout hidden states — closed the gap and *erased* the
transition model's measured contribution, and was recorded as a negative result
rather than adopted. *(§3.6, D126.)*

## Q6 — Does trajectory-aware risk outperform state-only risk?

**Partial — the same answer as Q4, since this is the same experiment read a
different way.** Aggregate: yes, in all three regimes. Within-host: yes clearly
on CIC, marginally on CTU. On the rows where the state is provably
uninformative, the history-aware head reaches 58× the state-only ceiling and
every unit of that is host recognition. *(§3.10, §3.11, §3.12.)*

## Q7 — Does the world model outperform a strong GRU sequence classifier?

**Pending on the matrix.** The GRU classifier is trained (`ctu_gru`) and is one
of the systems every benchmark scores, so the comparison lands with the matrix.

## Q8 — Does the world model provide additional value over a history classifier?

**Partial.** The strict form of this question is the `persistence_rollout`
ablation — the same forward simulation with only the next state swapped, which
isolates the transition model's own contribution. On CTU validation it gave
+0.0331 AP [+0.000203, +0.0551], the first interval in the project excluding
zero. *(§3.5.)* The first cross-dataset benchmark to land (`cic2cic_state`, val)
puts the world model at AP 0.7911 against persistence at 0.7715, and reproduces
that ordering within-host (0.8499 against 0.8162), which is the first evidence
the margin is not host identity. One seed, one split.

## Q9 — Does the model generalize to completely unseen attack families/scenarios?

**Pending.** Three things bear on it and none has landed: the CTU test and
holdout splits (families absent from training), the leave-one-family-out run
without Neris (queued), and the two transfer arms `cic2ctu` and `ctu2cic`.

## Q10 — Does it produce genuine advance warning?

**Answered for CTU: no, and the reason is in the data. On CIC: not yet, and
there is a measured ceiling on how much it could.** CIC's pre-onset windows are
non-empty and weakly separable (probe ROC 0.75, of which `is_active` alone
gives 0.68) while the model sits at the prevalence floor — a quantified,
actionable gap, and a small one. *(§3.15.)*

For CTU, see Q1. The pre-onset
rows carry no state information and the history-aware head answers them with
the infected host's identity rather than the moment. *(§3.10, D129.)* Run 8
reported Task B near the prevalence floor on CIC and attributed it to a scarcity
of same-host precursors; CTU *has* precursors and they are empty. Two datasets,
two mechanisms, the same conclusion — and this corpus family cannot support a
strong advance-warning claim.

## Q11 — What is the best configuration under natural prevalence?

**Pending, deliberately.** Selection happens on validation only (§19), the
screening is one seed per cell, and Stage B — three seeds for the two surviving
head variants — is queued. No configuration will be called best before it has
been confirmed across seeds and scored on a split that was not used to pick it.

## Q12 — Which improvements are statistically supported rather than noise?

**Partial.** Exactly one interval in this phase excludes zero: the world model's
margin over `persistence_rollout` on CTU validation, +0.0331 [+0.000203,
+0.0551], by episode-cluster bootstrap. *(§3.5.)* Everything else is either a
point estimate from one seed, or an interval that includes zero. The head
ablation's three-regime replication (§3.12) is consistency evidence, not a
confidence interval, and the three rows are not comparable to each other.

## Q13 — What is still limiting NIDRA?

**Answered, and the answer changed during the phase.** Three limits, in the
order they bind:

1. **The single-infected-host structure of both corpora.** Training and
   validation have exactly one infected host each — not one per attack group,
   one per split. No architecture, objective or curriculum touches that.
   *(§3.14, D133.)*
2. **The risk head's pooled objective.** It ranks recon and c2 *below chance*
   (0.320 and 0.434) where the frozen stage head, trained on the same states
   with class weights, ranks them at 0.638 and 0.854 — so the signal is inside
   the model and the published composite does not use it. No scalar fusion of
   the two recovers it, because one pooled ranking cannot hold both orderings.
   *(§3.9, D128.)* The stage-balanced objective is pre-registered and queued.
   *(§3.13, D132.)*
3. **The label's pre-onset windows are empty on CTU.** See Q1 and Q10.

What is *not* the limit, contrary to how §3.8 originally read it: attack-group
separability. `ctu_4:c2` was written up as unreachable on a probe's ROC of
0.443 and NIDRA's own stage head then reached 0.864 on the same windows under
the same cross-host construction. That conclusion is withdrawn and the probe's
verdict is now documented as one-sided. *(§3.8 correction, D128.)*

## Q14 — What should the next research phase be?

**Partial — three directions are already evidence-backed, and the matrix may add
a fourth.**

1. **A corpus with two infected hosts in the same attack stage on the same
   split.** This is the binding limitation and no modelling change addresses it.
   Until then, every per-group number in this project needs the within-host
   decomposition attached, which is now what the benchmark and the scorecard do.
2. **The stage-aware risk objective**, pre-registered at §3.13. If criterion 1
   fails, the pooled objective is not the explanation and this direction closes
   — which is itself worth knowing.
3. **Advance warning, pursued on CIC rather than CTU.** This replaces the Δ=15
   idea, which was tested and withdrawn the same day it was written: CTU's host
   is silent for minutes before onset, so no window size recovers a precursor.
   CIC's pre-onset windows *are* non-empty and carry ROC 0.75 of separable
   signal against a model at the floor. The ceiling is modest and most of it is
   bare activity, so the honest framing of the work is closing a small measured
   gap, not unlocking advance warning. *(§3.15.)*

Ruled out by measurement rather than opinion: score fusion of the two frozen
heads (§3.9), `context_noise` as a fix for the exposure mismatch (§3.6, D126),
uncertainty as a head input (§3.4), and a finer Δ for CTU precursors (§3.15).
