### Systems — val
**val** — 20451 scored rows, natural prevalence 0.00046, operating point `mean|q=-|max`, threshold 0.088 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.538 | — | 0.904 | 0.70 / 0.56 / 0.62 | 6.01 | 0.666 | 0.021 / 0.020 |
| NIDRA (stochastic rollout) | 0.524 | [0.007, 0.791] | 0.925 | 0.04 / 0.65 / 0.08 | 386.07 | 0.665 | 0.021 / 0.018 |
| NIDRA (deterministic rollout) | 0.449 | [0.007, 0.753] | 0.793 | 0.37 / 0.60 / 0.46 | 26.15 | 0.567 | 0.010 / 0.009 |
| persistence + learned noise (mean disabled) | 0.525 | [0.002, 0.798] | 0.843 | 0.02 / 0.67 / 0.04 | 762.45 | 0.664 | 0.013 / 0.009 |
| persistence + isotropic noise | 0.495 | — | 0.765 | 0.00 / 0.79 / 0.00 | 37502.79 | 0.631 | 0.007 / 0.005 |
| persistence (risk head on S_t) | 0.535 | [0.016, 0.811] | 0.723 | 0.69 / 0.59 / 0.64 | 6.78 | 0.659 | 0.027 / 0.025 |
| oracle: risk head on the true future | 0.545 | — | 0.773 | 0.49 / 0.64 / 0.56 | 16.81 | 0.633 | 0.038 / 0.029 |
| logistic regression on S_t | 0.201 | [0.000, 0.528] | 0.746 | 0.01 / 0.62 / 0.02 | 1689.32 | 0.256 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.147 | [0.000, 0.529] | 0.627 | 0.03 / 0.54 / 0.06 | 390.64 | 0.191 | 0.001 / 0.002 |
| gradient-boosted trees on S_t | 0.419 | [0.003, 0.684] | 0.766 | 0.02 / 0.66 / 0.03 | 1082.36 | 0.538 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.92 / R 0.30 / F1 0.45, 122 alerts.

### Attribution — val
**val** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.010 | [-0.081, 0.031] |
| world_model - persistence_rollout | 0.122 | [0.003, 0.216] |
| world_model - noised_persistence | -0.001 | [-0.038, 0.041] |
| world_model - isotropic_noise_persistence | 0.029 | [-0.029, 0.086] |
| world_model - world_model_deterministic | 0.076 | [-0.006, 0.168] |
| world_model_deterministic - persistence | -0.086 | [-0.170, -0.003] |
| noised_persistence - persistence | -0.010 | [-0.086, 0.055] |
| world_model - lr_current_state | 0.323 | [0.003, 0.496] |
| world_model - lr_flattened_history | 0.377 | [0.007, 0.537] |
| world_model - gbdt_current_state | 0.105 | [-0.024, 0.197] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.575, period2_persistence 0.053, ridge_two_lag 0.436; NIDRA vs ridge 0.247.

### Horizon — val
**val** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00035 | 0.652 | 0.595 | 0.666 | 0.00643 | 0.42 (281) |
| 2 | 2 | 0.00035 | 0.644 | 0.557 | 0.639 | 0.00719 | 0.29 (281) |
| 3 | 3 | 0.00035 | 0.628 | 0.517 | 0.648 | 0.00791 | 0.18 (281) |
| 4 | 4 | 0.00035 | 0.602 | 0.477 | 0.686 | 0.00855 | 0.12 (281) |
| 5 | 5 | 0.00035 | 0.594 | 0.443 | 0.679 | 0.00911 | 0.06 (281) |
| 6 | 6 | 0.00035 | 0.588 | 0.415 | 0.636 | 0.00905 | 0.03 (281) |

### Onset forecasting — val
**val** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 0.006 | 0.007 | — | 0.000 | 0.001 | — |
| 3 | 28 | 0.015 | 0.018 | — | 0.000 | 0.001 | — |
| 5 | 46 | 0.021 | 0.027 | — | 0.000 | 0.001 | — |
| 10 | 86 | 0.020 | 0.026 | — | 0.000 | 0.002 | — |
| 15 | 106 | 0.020 | 0.025 | — | 0.000 | 0.002 | — |
| 30 | 134 | 0.017 | 0.022 | — | 0.002 | 0.003 | — |

### Episodes — val
**val** — per episode at the selected threshold (0.088), 2 consecutive windows: 10 episodes, 1 warned before onset, 3 alerted inside the episode; median lead 360.0 s, median latency 1.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313409420 | 5 | 28 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313410320 | 4 | 9 | 0.002 | no (—) | no (—) |
| 147.32.84.165@1313411220 | 25 | 11 | 0.002 | no (—) | no (—) |
| 147.32.84.165@1313413500 | 0 | 12 | 0.064 | no (—) | no (—) |
| 147.32.84.165@1313414280 | 46 | 12 | 0.038 | no (—) | no (—) |
| 147.32.84.165@1313417880 | 10 | 13 | 0.053 | no (—) | no (—) |
| 147.32.84.165@1313419200 | 32 | 12 | 0.000 | no (—) | yes (4.0) |
| 147.32.84.165@1313489340 | 116 | 6 | 0.685 | yes (360) | yes (1.0) |
| 172.16.0.1@1499188140 | 62 | 1 | 0.000 | no (—) | yes (0.0) |
| 172.16.0.1@1499278320 | 21 | 30 | 0.035 | no (—) | no (—) |

### Per attack group — val
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_6:exfil` | menti | 116 | 1 | 1¹ | 0.899 | 0.817 | 0.876 | 0.489 |
| `tuesday:initial_access` | — | 63 | 1 | 1¹ | 0.347 | 0.462 | 0.460 | 0.033 |
| `ctu_4:exfil` | rbot | 61 | 5 | 1¹ | 0.285 | 0.314 | 0.265 | 0.420 |
| `ctu_4:c2` | rbot | 56 | 6 | 1¹ | 0.001 | 0.008 | 0.004 | 0.169 |
| `ctu_4:recon` | rbot | 44 | 6 | 1¹ | 0.002 | 0.001 | 0.006 | 0.500 |
| `wednesday:initial_access` | — | 26 | 1 | 1¹ | 0.000 | 0.000 | 0.000 | 0.323 |
| `ctu_6:c2` | menti | 6 | 1 | 1¹ | 0.122 | 0.089 | 0.075 | 0.139 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — val
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.634 | 0.880 | 0.9980 | 2 | 0.8140 | 0.947 | 1.16× | 0.7855 | carries timing signal |
| world_model | 0.626 | 0.886 | 0.9980 | 2 | 0.8140 | 0.947 | 1.16× | 0.7947 | carries timing signal |
| world_model_deterministic | 0.607 | 0.888 | 0.9977 | 2 | 0.8140 | 0.935 | 1.15× | 0.7337 | carries timing signal |
| noised_persistence | 0.596 | 0.815 | 0.9980 | 2 | 0.8140 | 0.947 | 1.16× | 0.7809 | carries timing signal |
| isotropic_noise_persistence | 0.572 | 0.777 | 0.9980 | 2 | 0.8140 | 0.945 | 1.16× | 0.7608 | carries timing signal |
| persistence | 0.622 | 0.837 | 0.9977 | 2 | 0.8140 | 0.926 | 1.14× | 0.6907 | carries timing signal |
| oracle_true_future | 0.657 | 0.851 | 0.9977 | 2 | 0.8140 | 0.938 | 1.15× | 0.7254 | carries timing signal |
| lr_current_state | 0.386 | 0.797 | 0.9718 | 2 | 0.8140 | 0.918 | 1.13× | 0.6947 | carries timing signal |
| lr_flattened_history | 0.368 | 0.752 | 0.9903 | 2 | 0.8140 | 0.912 | 1.12× | 0.6670 | carries timing signal |
| gbdt_current_state | 0.552 | 0.855 | 0.9973 | 2 | 0.8140 | 0.924 | 1.14× | 0.7218 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — val
**val** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00048 | 0.05042 | 0.00046 | 0.00046 | 10/10 |
| calibrated | 0.00029 | 0.04985 | 0.00046 | 0.00046 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `mean|q=-|max`, threshold 0.088 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.158 | — | 0.889 | 0.67 / 0.02 / 0.04 | 1.19 | 0.135 | 0.005 / 0.009 |
| NIDRA (stochastic rollout) | 0.147 | [0.045, 0.273] | 0.913 | 0.18 / 0.18 / 0.18 | 98.67 | 0.124 | 0.004 / 0.009 |
| NIDRA (deterministic rollout) | 0.041 | [0.005, 0.188] | 0.254 | 0.65 / 0.04 / 0.07 | 2.27 | 0.052 | 0.002 / 0.002 |
| persistence + learned noise (mean disabled) | 0.079 | [0.016, 0.173] | 0.864 | 0.10 / 0.28 / 0.14 | 307.84 | 0.077 | 0.002 / 0.004 |
| persistence + isotropic noise | 0.026 | — | 0.330 | 0.00 / 0.38 / 0.00 | 17700.32 | 0.035 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.034 | [0.004, 0.145] | 0.194 | 0.80 / 0.02 / 0.05 | 0.72 | 0.042 | 0.002 / 0.003 |
| oracle: risk head on the true future | 0.052 | — | 0.367 | 0.64 / 0.04 / 0.08 | 2.63 | 0.059 | 0.022 / 0.009 |
| logistic regression on S_t | 0.046 | [0.010, 0.139] | 0.390 | 0.04 / 0.27 / 0.08 | 669.00 | 0.049 | 0.001 / 0.004 |
| logistic regression on the L-window history | 0.013 | [0.003, 0.087] | 0.088 | 0.04 / 0.03 / 0.03 | 101.31 | 0.005 | 0.055 / 0.030 |
| gradient-boosted trees on S_t | 0.065 | [0.021, 0.213] | 0.360 | 0.05 / 0.20 / 0.07 | 484.85 | 0.081 | 0.001 / 0.002 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.02 / F1 0.04, 17 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.113 | [0.037, 0.220] |
| world_model - persistence_rollout | 0.112 | [0.042, 0.216] |
| world_model - noised_persistence | 0.068 | [0.027, 0.147] |
| world_model - isotropic_noise_persistence | 0.121 | [0.044, 0.226] |
| world_model - world_model_deterministic | 0.106 | [0.017, 0.218] |
| world_model_deterministic - persistence | 0.006 | [-0.002, 0.037] |
| noised_persistence - persistence | 0.045 | [0.004, 0.092] |
| world_model - lr_current_state | 0.101 | [0.030, 0.198] |
| world_model - lr_flattened_history | 0.134 | [0.025, 0.239] |
| world_model - gbdt_current_state | 0.082 | [-0.006, 0.187] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.565, period2_persistence 0.045, ridge_two_lag 0.406; NIDRA vs ridge 0.267.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.072 | 0.044 | 0.042 | 0.03042 | 0.02 (664) |
| 2 | 2 | 0.00292 | 0.084 | 0.037 | 0.038 | 0.03039 | 0.00 (662) |
| 3 | 3 | 0.00291 | 0.109 | 0.035 | 0.039 | 0.03045 | 0.00 (661) |
| 4 | 4 | 0.00291 | 0.097 | 0.033 | 0.037 | 0.03059 | 0.00 (660) |
| 5 | 5 | 0.00290 | 0.083 | 0.025 | 0.036 | 0.03063 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.085 | 0.024 | 0.037 | 0.03052 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.001 | 0.001 | — | 0.002 | 0.014 | — |
| 3 | 36 | 0.002 | 0.003 | — | 0.001 | 0.043 | — |
| 5 | 60 | 0.005 | 0.002 | — | 0.001 | 0.055 | — |
| 10 | 113 | 0.007 | 0.001 | — | 0.001 | 0.038 | — |
| 15 | 161 | 0.009 | 0.003 | — | 0.002 | 0.030 | — |
| 30 | 299 | 0.019 | 0.003 | — | 0.004 | 0.023 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.088), 2 consecutive windows: 15 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.034 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.003 | no (—) | no (—) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.001 | no (—) | no (—) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.002 | no (—) | no (—) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.012 | no (—) | no (—) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.007 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | yes (0.0) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.014 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.014 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.006 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 0.023 | no (—) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.009 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.017 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.017 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `friday_morning:c2` | — | 862 | 7 | 7 | 0.101 | 0.010 | 0.005 | 0.508 |
| `friday_portscan:recon` | — | 58 | 6 | 1¹ | 0.046 | 0.118 | 0.023 | 0.408 |
| `friday_ddos:exfil` | — | 26 | 2 | 2 | 0.776 | 0.769 | 0.731 | -0.786 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.272 | 0.855 | 0.9283 | 9 | 0.3927 | 0.478 | 1.22× | 0.5931 | carries timing signal |
| world_model | 0.264 | 0.863 | 0.9286 | 9 | 0.3927 | 0.471 | 1.20× | 0.5880 | carries timing signal |
| world_model_deterministic | 0.079 | 0.381 | 0.7569 | 9 | 0.3927 | 0.483 | 1.23× | 0.5765 | carries timing signal |
| noised_persistence | 0.154 | 0.792 | 0.8905 | 9 | 0.3927 | 0.461 | 1.17× | 0.5748 | carries timing signal |
| isotropic_noise_persistence | 0.061 | 0.398 | 0.1999 | 9 | 0.3927 | 0.458 | 1.17× | 0.5543 | carries timing signal |
| persistence | 0.071 | 0.345 | 0.7817 | 9 | 0.3927 | 0.471 | 1.20× | 0.5599 | carries timing signal |
| oracle_true_future | 0.116 | 0.507 | 0.8592 | 9 | 0.3927 | 0.505 | 1.29× | 0.6001 | carries timing signal |
| lr_current_state | 0.101 | 0.509 | 0.8797 | 9 | 0.3927 | 0.408 | 1.04× | 0.4956 | carries timing signal |
| lr_flattened_history | 0.041 | 0.200 | 0.2787 | 9 | 0.3927 | 0.378 | 0.96× | 0.4845 | carries timing signal |
| gbdt_current_state | 0.131 | 0.498 | 0.8901 | 9 | 0.3927 | 0.463 | 1.18× | 0.5634 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00384 | 0.04637 | 0.00417 | 0.00415 | 10/10 |
| calibrated | 0.00404 | 0.04592 | 0.00417 | 0.00415 | 6/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `mean|q=-|max`, threshold 0.088 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.328 | — | 0.900 | 0.86 / 0.40 / 0.55 | 0.99 | 0.356 | 0.071 / 0.285 |
| NIDRA (stochastic rollout) | 0.329 | [0.003, 0.772] | 0.909 | 0.06 / 0.49 / 0.10 | 123.27 | 0.355 | 0.069 / 0.289 |
| NIDRA (deterministic rollout) | 0.346 | [0.000, 0.844] | 0.425 | 0.36 / 0.40 / 0.38 | 10.76 | 0.501 | 0.050 / 0.250 |
| persistence + learned noise (mean disabled) | 0.311 | [0.001, 0.762] | 0.835 | 0.02 / 0.49 / 0.04 | 387.43 | 0.354 | 0.063 / 0.277 |
| persistence + isotropic noise | 0.335 | — | 0.445 | 0.00 / 0.44 / 0.00 | 29627.25 | 0.428 | 0.039 / 0.246 |
| persistence (risk head on S_t) | 0.341 | [0.000, 0.843] | 0.414 | 0.84 / 0.39 / 0.54 | 1.11 | 0.489 | 0.060 / 0.270 |
| oracle: risk head on the true future | 0.363 | — | 0.438 | 0.71 / 0.40 / 0.51 | 2.42 | 0.536 | 0.066 / 0.272 |
| logistic regression on S_t | 0.065 | [0.001, 0.231] | 0.563 | 0.01 / 0.46 / 0.02 | 779.13 | 0.115 | 0.000 / 0.002 |
| logistic regression on the L-window history | 0.021 | [0.000, 0.054] | 0.225 | 0.01 / 0.14 / 0.03 | 141.10 | 0.008 | 0.001 / 0.005 |
| gradient-boosted trees on S_t | 0.285 | [0.001, 0.734] | 0.504 | 0.01 / 0.43 / 0.02 | 528.91 | 0.507 | 0.001 / 0.003 |

At the mandated 0.75 threshold on the calibrated score: P 0.69 / R 0.07 / F1 0.13, 13 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.013 | [-0.071, 0.030] |
| world_model - persistence_rollout | -0.027 | [-0.114, 0.028] |
| world_model - noised_persistence | 0.018 | [0.001, 0.031] |
| world_model - isotropic_noise_persistence | -0.006 | [-0.067, 0.030] |
| world_model - world_model_deterministic | -0.017 | [-0.086, 0.028] |
| world_model_deterministic - persistence | 0.004 | [-0.001, 0.032] |
| noised_persistence - persistence | -0.030 | [-0.085, 0.009] |
| world_model - lr_current_state | 0.264 | [0.002, 0.616] |
| world_model - lr_flattened_history | 0.308 | [-0.012, 0.750] |
| world_model - gbdt_current_state | 0.044 | [-0.005, 0.106] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.585, period2_persistence 0.051, ridge_two_lag 0.455; NIDRA vs ridge 0.240.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.333 | 0.457 | 0.485 | 0.00255 | 0.00 (68) |
| 2 | 2 | 0.00019 | 0.339 | 0.472 | 0.468 | 0.00264 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.319 | 0.400 | 0.440 | 0.00284 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.307 | 0.317 | 0.430 | 0.00290 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.297 | 0.219 | 0.422 | 0.00294 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.273 | 0.198 | 0.412 | 0.00288 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.018 | 0.018 | — | 0.000 | 0.000 | — |
| 3 | 12 | 0.043 | 0.039 | — | 0.001 | 0.001 | — |
| 5 | 20 | 0.071 | 0.060 | — | 0.001 | 0.001 | — |
| 10 | 38 | 0.178 | 0.172 | — | 0.002 | 0.006 | — |
| 15 | 52 | 0.285 | 0.270 | — | 0.003 | 0.005 | — |
| 30 | 73 | 0.215 | 0.209 | — | 0.002 | 0.004 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.088), 2 consecutive windows: 5 episodes, 1 warned before onset, 2 alerted inside the episode; median lead 840.0 s, median latency 14.5 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | yes (29.0) |
| 172.16.0.1@1499346900 | 27 | 14 | 0.981 | yes (840) | yes (0.0) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.028 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.006 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.035 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `thursday_infiltration:lateral` | — | 73 | 3 | 1¹ | 0.006 | 0.001 | 0.000 | 0.434 |
| `thursday_web:initial_access` | — | 49 | 2 | 1¹ | 0.768 | 0.889 | 0.850 | 0.197 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.356 | 0.883 | 0.9808 | 2 | 0.3664 | 0.563 | 1.54× | 0.6200 | carries timing signal |
| world_model | 0.359 | 0.881 | 0.9808 | 2 | 0.3664 | 0.562 | 1.53× | 0.6142 | carries timing signal |
| world_model_deterministic | 0.351 | 0.458 | 0.9144 | 2 | 0.3664 | 0.557 | 1.52× | 0.5204 | carries timing signal |
| noised_persistence | 0.324 | 0.796 | 0.9395 | 2 | 0.3664 | 0.551 | 1.50× | 0.5984 | carries timing signal |
| isotropic_noise_persistence | 0.341 | 0.468 | 0.4157 | 2 | 0.3664 | 0.556 | 1.52× | 0.5183 | carries timing signal |
| persistence | 0.347 | 0.442 | 0.9961 | 2 | 0.3664 | 0.551 | 1.50× | 0.5143 | carries timing signal |
| oracle_true_future | 0.373 | 0.454 | 0.9845 | 2 | 0.3664 | 0.550 | 1.50× | 0.4550 | carries timing signal |
| lr_current_state | 0.123 | 0.613 | 0.9199 | 2 | 0.3664 | 0.571 | 1.56× | 0.5433 | carries timing signal |
| lr_flattened_history | 0.037 | 0.347 | 0.7223 | 2 | 0.3664 | 0.513 | 1.40× | 0.5442 | carries timing signal |
| gbdt_current_state | 0.323 | 0.584 | 0.9750 | 2 | 0.3664 | 0.619 | 1.69× | 0.5985 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00032 | 0.05006 | 0.00034 | 0.00034 | 9/10 |
| calibrated | 0.00026 | 0.04986 | 0.00034 | 0.00034 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
