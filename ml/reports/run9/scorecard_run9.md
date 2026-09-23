| Training | Evaluated on | split | rows | prevalence | AP [95% CI] | ROC | P | R | F1 | FA/h | alerts/h | FA rate on active benign | best baseline (AP) | oracle AP / deterministic AP | state skill vs persistence / ridge | onset AP 5/15 | episodes warned | positive hosts | within-host ROC |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---:|---:|
| CIC | CIC | test | 21,173 | 0.00417 | 0.117 [0.039, 0.299] | 0.760 | 0.107 | 0.026 | 0.042 | 25.80 | 28.90 | 0.00714 | noised_persistence (0.098) | 0.051 / 0.025 | — / 0.693 | 0.004 / 0.008 | 0 / 15 | 9 | 0.6129 |
| CIC | CIC | holdout | 20,171 | 0.00034 | 0.387 [0.002, 0.912] | 0.839 | 0.725 | 0.352 | 0.474 | 2.01 | 7.32 | 0.00047 | lr_flattened_history (0.381) | 0.375 / 0.319 | — / 0.671 | 0.008 / 0.092 | 1 / 5 | 2 | 0.7052 |
| CTU | CTU | test | 25,099 | 0.00749 | 0.319 [0.130, 0.586] | 0.691 | 0.718 | 0.306 | 0.429 | 15.04 | 53.35 | 0.00376 | noised_persistence (0.353) | 0.352 / 0.330 | — / 0.089 | 0.003 / 0.008 | 0 / 131 | 10 | 0.6143 |
| CTU | CTU | holdout | 21,045 | 0.00409 | 0.227 [0.003, 0.290] | 0.720 | 0.982 | 0.183 | 0.308 | 0.20 | 10.97 | 0.00007 | noised_persistence (0.245) | 0.197 / 0.196 | — / 0.138 | 0.004 / 0.005 | 0 / 6 | 3 | 0.2777 |
| CIC | CTU | test | 25,099 | 0.00749 | 0.010 [0.005, 0.022] | 0.580 | 0.011 | 0.037 | 0.017 | 398.21 | 402.79 | 0.09648 | noised_persistence (0.013) | 0.012 / 0.009 | — / 0.867 | 0.002 / 0.004 | 4 / 131 | 10 | 0.6238 |
| CIC | CTU | holdout | 21,045 | 0.00409 | 0.016 [0.000, 0.054] | 0.758 | 0.000 | 0.000 | 0.000 | 349.01 | 349.01 | 0.11052 | noised_persistence (0.017) | 0.013 / 0.010 | — / 0.868 | 0.000 / 0.000 | 0 / 6 | 3 | 0.7245 |
| CTU | CIC | test | 21,173 | 0.00417 | 0.004 [0.001, 0.007] | 0.222 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00000 | gbdt_current_state (0.024) | 0.002 / 0.002 | — / -0.005 | 0.000 / 0.001 | 0 / 15 | 9 | 0.4792 |
| CTU | CIC | holdout | 20,171 | 0.00034 | 0.000 [0.000, 0.001] | 0.133 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00000 | gbdt_current_state (0.001) | 0.000 / 0.000 | — / -0.014 | 0.000 / 0.000 | 0 / 5 | 2 | 0.3609 |
| CIC+CTU | CIC | test | 21,173 | 0.00417 | 0.199 [0.068, 0.355] | 0.950 | 0.784 | 0.022 | 0.043 | 0.72 | 3.32 | 0.00020 | noised_persistence (0.107) | 0.052 / 0.041 | — / 0.235 | 0.006 / 0.015 | 0 / 15 | 9 | 0.5971 |
| CIC+CTU | CIC | holdout | 20,171 | 0.00034 | 0.339 [0.004, 0.789] | 0.923 | 0.823 | 0.393 | 0.532 | 1.27 | 7.20 | 0.00030 | persistence_rollout (0.356) | 0.363 / 0.344 | — / 0.204 | 0.069 / 0.293 | 1 / 5 | 2 | 0.6090 |
| CIC+CTU | CTU | test | 25,099 | 0.00749 | 0.354 [0.184, 0.583] | 0.789 | 0.658 | 0.360 | 0.466 | 23.43 | 68.53 | 0.00586 | persistence (0.362) | 0.400 / 0.286 | — / 0.279 | 0.003 / 0.008 | 7 / 131 | 10 | 0.7563 |
| CIC+CTU | CTU | holdout | 21,045 | 0.00409 | 0.240 [0.001, 0.318] | 0.836 | 0.709 | 0.199 | 0.311 | 4.82 | 16.56 | 0.00087 | gbdt_current_state (0.299) | 0.206 / 0.197 | — / 0.256 | 0.000 / 0.000 | 0 / 6 | 3 | 0.5514 |

**oracle AP / deterministic AP.** The oracle is the frozen head on the TRUE future state under a single-trajectory readout and no Platt layer. The published system pools ~200 stochastic trajectories and then calibrates, so the ratio of the two is NOT a fraction of an upper bound — the published score exceeds the oracle in half the cells measured, because its readout is a better estimator and the oracle is given neither half of it. The matched comparison is the deterministic column, which the oracle does bound in 12 of 14 (§3.24).

### Calibration

| Training | Evaluated on | split | Brier raw | ECE raw | Brier calibrated | ECE calibrated | base rate | Brier of that constant | beats it? |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| CIC | CIC | test | 0.00561 | 0.0503 | 0.00517 | 0.0473 | 0.00417 | 0.00415 | **no** |
| CIC | CIC | holdout | 0.00095 | 0.0514 | 0.00028 | 0.0499 | 0.00034 | 0.00034 | yes |
| CTU | CTU | test | 0.00602 | 0.0470 | 0.00604 | 0.0456 | 0.00749 | 0.00743 | yes |
| CTU | CTU | holdout | 0.00350 | 0.0474 | 0.00352 | 0.0469 | 0.00409 | 0.00408 | yes |
| CIC | CTU | test | 0.08012 | 0.1521 | 0.03600 | 0.0744 | 0.00749 | 0.00743 | **no** |
| CIC | CTU | holdout | 0.07365 | 0.1487 | 0.03358 | 0.0787 | 0.00409 | 0.00408 | **no** |
| CTU | CIC | test | 0.00449 | 0.0468 | 0.00417 | 0.0458 | 0.00417 | 0.00415 | **no** |
| CTU | CIC | holdout | 0.00063 | 0.0504 | 0.00035 | 0.0497 | 0.00034 | 0.00034 | **no** |
| CIC+CTU | CIC | test | 0.00369 | 0.0468 | 0.00403 | 0.0459 | 0.00417 | 0.00415 | yes |
| CIC+CTU | CIC | holdout | 0.00033 | 0.0502 | 0.00026 | 0.0499 | 0.00034 | 0.00034 | yes |
| CIC+CTU | CTU | test | 0.00602 | 0.0478 | 0.00566 | 0.0456 | 0.00749 | 0.00743 | yes |
| CIC+CTU | CTU | holdout | 0.00359 | 0.0481 | 0.00332 | 0.0468 | 0.00409 | 0.00408 | yes |

ECE is weighted by each bin's **population** weight, not its sampled row count — the evaluation subsamples negatives, so those are different numbers. At these prevalences Brier is dominated by the negatives, so the last two columns are the ones that make it readable: predicting the base rate for every row and never moving scores p(1−p), and a **no** in the final column means the calibrated score does not beat that. It is not a verdict on the ranking, which is what AP measures — but a probability that loses to a constant should not be displayed to an operator as a probability.
