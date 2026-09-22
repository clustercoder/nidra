### Frozen-head fusion screen — observed states, **val**

Both heads are already trained and frozen; these rules add no parameter and fit nothing on the rows they score. Screening only — one seed, no bootstrap, observed states. A rule that wins here earns a rollout-level implementation and a confirmation run; it is not itself a result.

**composite risk label** — 288 positives

| rule | AP | ΔAP vs published | ROC |
|---|---|---|---|
| risk head alone (published) | 0.4894 | — | 0.7441 |
| noisy-or | 0.4868 | -0.0026 | 0.7380 |
| max | 0.4797 | -0.0097 | 0.7354 |
| mean | 0.4662 | -0.0233 | 0.7380 |
| geometric mean | 0.4516 | -0.0379 | 0.7327 |
| stage 1-P(benign) alone | 0.2946 | -0.1948 | 0.7411 |

Verdict: **no fusion rule beats the published head**.

**Per stage, one-vs-rest.** Where the two heads disagree, a scalar rule lands between them rather than above either — the risk head's confident scores on the majority stage outrank the stage head's correct ordering of the minority one.

| stage | positives | risk head alone (published) | stage 1-P(benign) alone | max | mean | noisy-or | geometric mean |
|---|---|---|---|---|---|---|---|
| c2 | 24 | 0.434 | 0.828 | 0.682 | 0.681 | 0.681 | 0.603 |
| exfil | 172 | 0.915 | 0.917 | 0.916 | 0.916 | 0.916 | 0.917 |
| recon | 17 | 0.320 | 0.585 | 0.520 | 0.520 | 0.520 | 0.425 |

ROC, not AP: at these base rates AP is dominated by a handful of rows.
