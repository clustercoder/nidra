| Training | Evaluated on | split | rows | prevalence | AP [95% CI] | ROC | P | R | F1 | FA/h | alerts/h | FA rate on active benign | best baseline (AP) | oracle AP / deterministic AP | state skill vs persistence / ridge | onset AP 5/15 | episodes warned | positive hosts | within-host ROC |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---:|---:|
| CIC | CIC (state) | val | 20,113 | 0.00012 | 0.781 | 0.904 | 0.955 | 0.753 | 0.842 | 0.38 | 8.40 | 0.00007 | world_model_deterministic (0.793) | 0.797 / 0.793 | — / 0.694 | 0.000 / 0.000 | 0 / 2 | 1¹ | 0.8571 |
| CIC | CIC (state) | test | 21,173 | 0.00417 | 0.111 | 0.870 | 0.745 | 0.033 | 0.063 | 1.32 | 5.16 | 0.00037 | noised_persistence (0.127) | 0.155 / 0.039 | — / 0.711 | 0.006 / 0.011 | 0 / 15 | 9 | 0.5933 |
| CIC | CIC (state) | holdout | 20,171 | 0.00034 | 0.332 | 0.830 | 0.511 | 0.279 | 0.361 | 4.02 | 8.22 | 0.00093 | lr_flattened_history (0.381) | 0.489 / 0.312 | — / 0.689 | 0.002 / 0.003 | 0 / 5 | 2 | 0.6730 |
| CTU | CTU (state) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CTU | CTU (state) | test | 25,099 | 0.00749 | 0.196 | 0.817 | 0.097 | 0.009 | 0.017 | 11.06 | 12.24 | 0.00277 | gru_classifier (0.291) | 0.214 / 0.154 | — / 0.101 | 0.001 / 0.003 | 0 / 131 | 10 | 0.8415 |
| CTU | CTU (state) | holdout | 21,045 | 0.00409 | 0.188 | 0.780 | 1.000 | 0.058 | 0.109 | 0.00 | 3.40 | 0.00000 | noised_persistence (0.222) | 0.188 / 0.188 | — / 0.149 | 0.000 / 0.001 | 0 / 6 | 3 | 0.4963 |
| CIC | CTU (state) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CIC | CTU (state) | test | 25,099 | 0.00749 | 0.010 | 0.604 | 0.011 | 0.019 | 0.014 | 207.97 | 210.32 | 0.05204 | noised_persistence (0.014) | 0.011 / 0.005 | — / 0.873 | 0.001 / 0.003 | 0 / 131 | 10 | 0.6487 |
| CIC | CTU (state) | holdout | 21,045 | 0.00409 | 0.035 | 0.863 | 0.002 | 0.005 | 0.002 | 186.18 | 186.46 | 0.06353 | noised_persistence (0.033) | 0.034 / 0.026 | — / 0.874 | 0.000 / 0.001 | 0 / 6 | 3 | 0.7154 |
| CTU | CIC (state) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CTU | CIC (state) | test | 21,173 | 0.00417 | 0.045 | 0.767 | 1.000 | 0.001 | 0.002 | 0.00 | 0.12 | 0.00000 | noised_persistence (0.088) | 0.071 / 0.013 | — / 0.006 | 0.003 / 0.006 | 0 / 15 | 9 | 0.4585 |
| CTU | CIC (state) | holdout | 20,171 | 0.00034 | 0.001 | 0.485 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00000 | noised_persistence (0.002) | 0.005 / 0.000 | — / -0.005 | 0.001 / 0.003 | 0 / 5 | 2 | 0.2298 |
| CIC+CTU | CIC (state) | val | 20,451 | 0.00046 | 0.391 | 0.902 | 0.569 | 0.444 | 0.499 | 8.48 | 19.69 | 0.00160 | gbdt_current_state (0.419) | 0.442 / 0.334 | — / 0.247 | 0.000 / 0.000 | 0 / 10 | 2 | 0.8204 |
| CIC+CTU | CIC (state) | test | 21,173 | 0.00417 | 0.164 | 0.972 | 0.912 | 0.021 | 0.041 | 0.24 | 2.72 | 0.00007 | world_model_deterministic (0.157) | 0.089 / 0.157 | — / 0.267 | 0.005 / 0.017 | 0 / 15 | 9 | 0.5046 |
| CIC+CTU | CIC (state) | holdout | 20,171 | 0.00034 | 0.224 | 0.979 | 0.751 | 0.115 | 0.199 | 0.57 | 2.30 | 0.00013 | noised_persistence (0.311) | 0.354 / 0.182 | — / 0.240 | 0.004 / 0.007 | 0 / 5 | 2 | 0.7565 |
| CIC+CTU | CTU (state) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CIC+CTU | CTU (state) | test | 25,099 | 0.00749 | 0.157 | 0.795 | 0.253 | 0.046 | 0.078 | 17.11 | 22.91 | 0.00428 | gbdt_current_state (0.244) | 0.200 / 0.166 | — / 0.289 | 0.001 / 0.003 | 0 / 131 | 10 | 0.8433 |
| CIC+CTU | CTU (state) | holdout | 21,045 | 0.00409 | 0.307 | 0.978 | 1.000 | 0.129 | 0.228 | 0.00 | 7.60 | 0.00000 | world_model_deterministic (0.339) | 0.232 / 0.339 | — / 0.270 | 0.000 / 0.000 | 0 / 6 | 3 | 0.8977 |
| CIC | CIC (state+hidden) | val | 20,113 | 0.00012 | 0.832 | 0.999 | 0.957 | 0.787 | 0.863 | 0.38 | 8.76 | 0.00007 | persistence (0.890) | 0.943 / 0.770 | — / 0.694 | 0.068 / 0.062 | 0 / 2 | 1¹ | 0.8672 |
| CIC | CIC (state+hidden) | test | 21,173 | 0.00417 | 0.064 | 0.560 | 0.105 | 0.040 | 0.058 | 40.26 | 44.97 | 0.01115 | noised_persistence (0.072) | 0.051 / 0.014 | — / 0.711 | 0.004 / 0.006 | 1 / 15 | 9 | 0.6035 |
| CIC | CIC (state+hidden) | holdout | 20,171 | 0.00034 | 0.380 | 0.734 | 0.521 | 0.352 | 0.420 | 4.88 | 10.19 | 0.00113 | lr_flattened_history (0.381) | 0.375 / 0.236 | — / 0.689 | 0.004 / 0.049 | 1 / 5 | 2 | 0.7503 |
| CTU | CTU (state+hidden) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CTU | CTU (state+hidden) | test | 25,099 | 0.00749 | 0.322 | 0.680 | 0.810 | 0.262 | 0.396 | 7.67 | 40.42 | 0.00192 | noised_persistence (0.339) | 0.352 / 0.305 | — / 0.101 | 0.003 / 0.010 | 0 / 131 | 10 | 0.5956 |
| CTU | CTU (state+hidden) | holdout | 21,045 | 0.00409 | 0.221 | 0.769 | 1.000 | 0.155 | 0.268 | 0.00 | 9.13 | 0.00000 | noised_persistence (0.247) | 0.197 / 0.195 | — / 0.149 | 0.005 / 0.006 | 0 / 6 | 3 | 0.2418 |
| CIC | CTU (state+hidden) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CIC | CTU (state+hidden) | test | 25,099 | 0.00749 | 0.009 [0.005, 0.018] | 0.468 | 0.013 | 0.050 | 0.021 | 477.96 | 484.24 | 0.11581 | noised_persistence (0.010) | 0.012 / 0.007 | — / 0.873 | 0.002 / 0.004 | 5 / 131 | 10 | 0.4828 |
| CIC | CTU (state+hidden) | holdout | 21,045 | 0.00409 | 0.014 [0.000, 0.052] | 0.703 | 0.001 | 0.007 | 0.002 | 425.53 | 425.93 | 0.13585 | noised_persistence (0.015) | 0.013 / 0.008 | — / 0.874 | 0.000 / 0.000 | 0 / 6 | 3 | 0.7523 |
| CTU | CIC (state+hidden) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CTU | CIC (state+hidden) | test | 21,173 | 0.00417 | 0.004 [0.001, 0.006] | 0.161 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00000 | gbdt_current_state (0.024) | 0.002 / 0.002 | — / 0.006 | 0.000 / 0.001 | 0 / 15 | 9 | 0.4721 |
| CTU | CIC (state+hidden) | holdout | 20,171 | 0.00034 | 0.000 [0.000, 0.001] | 0.086 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00000 | gbdt_current_state (0.001) | 0.000 / 0.000 | — / -0.005 | 0.000 / 0.000 | 0 / 5 | 2 | 0.3572 |
| CIC+CTU | CIC (state+hidden) | val | 20,451 | 0.00046 | 0.538 | 0.904 | 0.702 | 0.559 | 0.622 | 6.01 | 20.14 | 0.00113 | persistence (0.535) | 0.545 / 0.449 | — / 0.247 | 0.021 / 0.020 | 1 / 10 | 2 | 0.7855 |
| CIC+CTU | CIC (state+hidden) | test | 21,173 | 0.00417 | 0.158 | 0.889 | 0.675 | 0.021 | 0.041 | 1.19 | 3.67 | 0.00033 | noised_persistence (0.079) | 0.052 / 0.041 | — / 0.267 | 0.005 / 0.009 | 0 / 15 | 9 | 0.5931 |
| CIC+CTU | CIC (state+hidden) | holdout | 20,171 | 0.00034 | 0.328 | 0.900 | 0.860 | 0.402 | 0.547 | 0.99 | 7.04 | 0.00023 | persistence_rollout (0.356) | 0.363 / 0.346 | — / 0.240 | 0.071 / 0.285 | 1 / 5 | 2 | 0.6200 |
| CIC+CTU | CTU (state+hidden) | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CIC+CTU | CTU (state+hidden) | test | 25,099 | 0.00749 | 0.373 [0.204, 0.586] | 0.795 | 0.687 | 0.352 | 0.465 | 20.04 | 64.06 | 0.00501 | persistence (0.362) | 0.400 / 0.321 | — / 0.289 | 0.003 / 0.009 | 7 / 131 | 10 | 0.7786 |
| CIC+CTU | CTU (state+hidden) | holdout | 21,045 | 0.00409 | 0.247 [0.000, 0.344] | 0.929 | 0.677 | 0.199 | 0.308 | 5.60 | 17.34 | 0.00113 | gbdt_current_state (0.299) | 0.206 / 0.225 | — / 0.270 | 0.000 / 0.000 | 0 / 6 | 3 | 0.7262 |

**oracle AP / deterministic AP.** The oracle is the frozen head on the TRUE future state under a single-trajectory readout and no Platt layer. The published system pools ~200 stochastic trajectories and then calibrates, so the ratio of the two is NOT a fraction of an upper bound — the published score exceeds the oracle in half the cells measured, because its readout is a better estimator and the oracle is given neither half of it. The matched comparison is the deterministic column, which the oracle does bound in 12 of 14 (§3.24).

¹ Every positive in that evaluation sits on a single host, so its AP does not separate the attack's behaviour from that host's identity. **Within-host ROC** is the column that does: on the infected host alone, does the system order the attack windows above that host's own benign ones? It is prevalence-independent and is the only column here that compares fairly across datasets.

### Gaps

**8 of 36 cells have no benchmark.** A gap is not a zero and not a failure: this table cannot distinguish a cell that was never scheduled from one that crashed. Each path below is where it looked.

- `CTU → CTU (state)`, val — `experiments/runs/xeval_ctu2ctu_state/artifacts/metrics/val/benchmark.json`
- `CIC → CTU (state)`, val — `experiments/runs/xeval_cic2ctu_state/artifacts/metrics/val/benchmark.json`
- `CTU → CIC (state)`, val — `experiments/runs/xeval_ctu2cic_state/artifacts/metrics/val/benchmark.json`
- `CIC+CTU → CTU (state)`, val — `experiments/runs/xeval_comb2ctu_state/artifacts/metrics/val/benchmark.json`
- `CTU → CTU (state+hidden)`, val — `experiments/runs/xeval_ctu2ctu_state+hidden/artifacts/metrics/val/benchmark.json`
- `CIC → CTU (state+hidden)`, val — `experiments/runs/xeval_cic2ctu_state+hidden/artifacts/metrics/val/benchmark.json`
- `CTU → CIC (state+hidden)`, val — `experiments/runs/xeval_ctu2cic_state+hidden/artifacts/metrics/val/benchmark.json`
- `CIC+CTU → CTU (state+hidden)`, val — `experiments/runs/xeval_comb2ctu_state+hidden/artifacts/metrics/val/benchmark.json`

### Calibration

| Training | Evaluated on | split | Brier raw | ECE raw | Brier calibrated | ECE calibrated | base rate | Brier of that constant | beats it? |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| CIC | CIC (state) | val | 0.08979 | 0.2807 | 0.00005 | 0.0500 | 0.00012 | 0.00012 | yes |
| CIC | CIC (state) | test | 0.08719 | 0.2661 | 0.00402 | 0.0462 | 0.00417 | 0.00415 | yes |
| CIC | CIC (state) | holdout | 0.08860 | 0.2773 | 0.00030 | 0.0499 | 0.00034 | 0.00034 | yes |
| CTU | CTU (state) | test | 0.00674 | 0.0468 | 0.00706 | 0.0440 | 0.00749 | 0.00743 | yes |
| CTU | CTU (state) | holdout | 0.00357 | 0.0470 | 0.00386 | 0.0469 | 0.00409 | 0.00408 | yes |
| CIC | CTU (state) | test | 0.13315 | 0.3032 | 0.01901 | 0.0555 | 0.00749 | 0.00743 | **no** |
| CIC | CTU (state) | holdout | 0.11526 | 0.2880 | 0.01571 | 0.0591 | 0.00409 | 0.00408 | **no** |
| CTU | CIC (state) | test | 0.00405 | 0.0460 | 0.00410 | 0.0459 | 0.00417 | 0.00415 | yes |
| CTU | CIC (state) | holdout | 0.00039 | 0.0498 | 0.00035 | 0.0497 | 0.00034 | 0.00034 | **no** |
| CIC+CTU | CIC (state) | val | 0.00140 | 0.0538 | 0.00037 | 0.0497 | 0.00046 | 0.00046 | yes |
| CIC+CTU | CIC (state) | test | 0.00380 | 0.0503 | 0.00407 | 0.0458 | 0.00417 | 0.00415 | yes |
| CIC+CTU | CIC (state) | holdout | 0.00100 | 0.0524 | 0.00033 | 0.0497 | 0.00034 | 0.00034 | yes |
| CIC+CTU | CTU (state) | test | 0.00929 | 0.0577 | 0.00735 | 0.0427 | 0.00749 | 0.00743 | yes |
| CIC+CTU | CTU (state) | holdout | 0.00399 | 0.0537 | 0.00391 | 0.0461 | 0.00409 | 0.00408 | yes |
| CIC | CIC (state+hidden) | val | 0.06725 | 0.2334 | 0.00007 | 0.0501 | 0.00012 | 0.00012 | yes |
| CIC | CIC (state+hidden) | test | 0.06332 | 0.2109 | 0.00565 | 0.0479 | 0.00417 | 0.00415 | **no** |
| CIC | CIC (state+hidden) | holdout | 0.06534 | 0.2273 | 0.00039 | 0.0501 | 0.00034 | 0.00034 | **no** |
| CTU | CTU (state+hidden) | test | 0.00614 | 0.0479 | 0.00643 | 0.0457 | 0.00749 | 0.00743 | yes |
| CTU | CTU (state+hidden) | holdout | 0.00359 | 0.0477 | 0.00375 | 0.0470 | 0.00409 | 0.00408 | yes |
| CIC | CTU (state+hidden) | test | 0.17761 | 0.3452 | 0.04297 | 0.0833 | 0.00749 | 0.00743 | **no** |
| CIC | CTU (state+hidden) | holdout | 0.17505 | 0.3480 | 0.04051 | 0.0879 | 0.00409 | 0.00408 | **no** |
| CTU | CIC (state+hidden) | test | 0.00486 | 0.0476 | 0.00418 | 0.0458 | 0.00417 | 0.00415 | **no** |
| CTU | CIC (state+hidden) | holdout | 0.00090 | 0.0510 | 0.00035 | 0.0497 | 0.00034 | 0.00034 | **no** |
| CIC+CTU | CIC (state+hidden) | val | 0.00048 | 0.0504 | 0.00029 | 0.0498 | 0.00046 | 0.00046 | yes |
| CIC+CTU | CIC (state+hidden) | test | 0.00384 | 0.0464 | 0.00404 | 0.0459 | 0.00417 | 0.00415 | yes |
| CIC+CTU | CIC (state+hidden) | holdout | 0.00032 | 0.0501 | 0.00026 | 0.0499 | 0.00034 | 0.00034 | yes |
| CIC+CTU | CTU (state+hidden) | test | 0.00605 | 0.0482 | 0.00573 | 0.0457 | 0.00749 | 0.00743 | yes |
| CIC+CTU | CTU (state+hidden) | holdout | 0.00363 | 0.0486 | 0.00332 | 0.0468 | 0.00409 | 0.00408 | yes |

ECE is weighted by each bin's **population** weight, not its sampled row count — the evaluation subsamples negatives, so those are different numbers. At these prevalences Brier is dominated by the negatives, so the last two columns are the ones that make it readable: predicting the base rate for every row and never moving scores p(1−p), and a **no** in the final column means the calibrated score does not beat that. It is not a verdict on the ranking, which is what AP measures — but a probability that loses to a constant should not be displayed to an operator as a probability.
