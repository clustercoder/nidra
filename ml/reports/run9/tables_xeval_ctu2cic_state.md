_val: no benchmark.json at experiments/runs/xeval_ctu2cic_state/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `mean|q=-|max`, threshold 0.182 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.045 | — | 0.767 | 1.00 / 0.00 / 0.00 | 0.00 | 0.033 | 0.003 / 0.006 |
| NIDRA (stochastic rollout) | 0.064 | [0.012, 0.131] | 0.787 | 0.18 / 0.02 / 0.04 | 11.37 | 0.046 | 0.004 / 0.009 |
| NIDRA (deterministic rollout) | 0.013 | [0.002, 0.031] | 0.182 | 0.17 / 0.02 / 0.04 | 12.08 | 0.010 | 0.002 / 0.003 |
| persistence + learned noise (mean disabled) | 0.088 | [0.016, 0.186] | 0.880 | 0.14 / 0.06 / 0.09 | 44.89 | 0.066 | 0.005 / 0.014 |
| persistence + isotropic noise | 0.066 | — | 0.829 | 0.14 / 0.09 / 0.11 | 64.39 | 0.052 | 0.005 / 0.012 |
| persistence (risk head on S_t) | 0.024 | [0.003, 0.060] | 0.220 | 0.13 / 0.05 / 0.07 | 35.07 | 0.019 | 0.005 / 0.006 |
| ridge two-lag dynamics + risk head | 0.074 | [0.011, 0.164] | 0.668 | 0.13 / 0.21 / 0.16 | 165.06 | 0.053 | 0.003 / 0.013 |
| oracle: risk head on the true future | 0.071 | — | 0.520 | 0.13 / 0.17 / 0.15 | 137.65 | 0.043 | 0.023 / 0.031 |
| logistic regression on S_t | 0.016 | [0.001, 0.034] | 0.238 | 0.03 / 0.11 / 0.05 | 386.36 | 0.010 | 0.002 / 0.003 |
| logistic regression on the L-window history | 0.003 | [0.001, 0.008] | 0.049 | 0.02 / 0.01 / 0.01 | 43.03 | 0.002 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.024 | [0.002, 0.053] | 0.268 | 0.08 / 0.14 / 0.10 | 193.76 | 0.015 | 0.020 / 0.022 |
| GRU sequence classifier | 0.017 | [0.001, 0.043] | 0.131 | 0.00 / 0.00 / 0.00 | 0.00 | 0.010 | 0.012 / 0.024 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.041 | [0.008, 0.076] |
| world_model - persistence_rollout | 0.041 | [0.008, 0.076] |
| world_model - noised_persistence | -0.024 | [-0.054, -0.002] |
| world_model - isotropic_noise_persistence | -0.002 | [-0.013, 0.008] |
| world_model - world_model_deterministic | 0.052 | [0.010, 0.105] |
| world_model - ridge_two_lag | -0.009 | [-0.038, 0.005] |
| world_model_deterministic - persistence | -0.011 | [-0.030, -0.001] |
| world_model_deterministic - ridge_two_lag | -0.061 | [-0.137, -0.009] |
| noised_persistence - persistence | 0.064 | [0.012, 0.126] |
| world_model - lr_current_state | 0.049 | [0.009, 0.105] |
| world_model - lr_flattened_history | 0.062 | [0.011, 0.125] |
| world_model - gbdt_current_state | 0.040 | [0.002, 0.086] |
| world_model - gru_classifier | 0.048 | [0.008, 0.098] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.430, period2_persistence 0.041, ridge_two_lag 0.426; NIDRA vs ridge 0.006.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.055 | 0.011 | 0.019 | 0.03100 | 0.00 (664) |
| 2 | 2 | 0.00292 | 0.041 | 0.009 | 0.019 | 0.03099 | 0.00 (662) |
| 3 | 3 | 0.00291 | 0.020 | 0.002 | 0.018 | 0.03103 | 0.00 (661) |
| 4 | 4 | 0.00291 | 0.008 | 0.001 | 0.019 | 0.03105 | 0.00 (660) |
| 5 | 5 | 0.00290 | 0.004 | 0.001 | 0.017 | 0.03099 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.004 | 0.001 | 0.018 | 0.03089 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.002 | 0.017 | 0.002 | 0.084 | 0.000 | — |
| 3 | 36 | 0.002 | 0.006 | 0.003 | 0.030 | 0.000 | — |
| 5 | 60 | 0.003 | 0.005 | 0.003 | 0.020 | 0.000 | — |
| 10 | 113 | 0.004 | 0.006 | 0.008 | 0.025 | 0.000 | — |
| 15 | 161 | 0.006 | 0.006 | 0.013 | 0.022 | 0.000 | — |
| 30 | 299 | 0.016 | 0.011 | 0.039 | 0.024 | 0.001 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.182), 2 consecutive windows: 15 episodes, 0 warned before onset, 0 alerted inside the episode; median lead None s, median latency None min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.025 | no (—) | no (—) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.003 | no (—) | no (—) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.026 | no (—) | no (—) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.024 | no (—) | no (—) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.024 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | no (—) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.038 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.070 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.060 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 0.023 | no (—) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.057 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.040 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.068 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `friday_morning:c2` | — | 862 | 7 | 7 | 0.068 | 0.077 | 0.026 | 0.367 |
| `friday_portscan:recon` | — | 58 | 6 | 1¹ | 0.001 | 0.000 | 0.000 | 0.405 |
| `friday_ddos:exfil` | — | 26 | 2 | 2 | 0.000 | 0.000 | 0.000 | -1.300 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.128 | 0.702 | 0.9201 | 9 | 0.3927 | 0.374 | 0.95× | 0.4585 | **host identity** |
| world_model | 0.150 | 0.727 | 0.9258 | 9 | 0.3927 | 0.370 | 0.94× | 0.4571 | **host identity** |
| world_model_deterministic | 0.045 | 0.226 | 0.8964 | 9 | 0.3927 | 0.357 | 0.91× | 0.4371 | carries timing signal |
| noised_persistence | 0.162 | 0.794 | 0.9199 | 9 | 0.3927 | 0.366 | 0.93× | 0.4549 | **host identity** |
| isotropic_noise_persistence | 0.144 | 0.752 | 0.9006 | 9 | 0.3927 | 0.359 | 0.91× | 0.4454 | **host identity** |
| persistence | 0.064 | 0.369 | 0.8834 | 9 | 0.3927 | 0.379 | 0.96× | 0.4815 | carries timing signal |
| ridge_two_lag | 0.148 | 0.686 | 0.9092 | 9 | 0.3927 | 0.350 | 0.89× | 0.4245 | **host identity** |
| oracle_true_future | 0.137 | 0.529 | 0.8705 | 9 | 0.3927 | 0.377 | 0.96× | 0.4503 | carries timing signal |
| lr_current_state | 0.055 | 0.452 | 0.8770 | 9 | 0.3927 | 0.364 | 0.93× | 0.4745 | carries timing signal |
| lr_flattened_history | 0.026 | 0.134 | 0.3637 | 9 | 0.3927 | 0.336 | 0.86× | 0.3964 | carries timing signal |
| gbdt_current_state | 0.065 | 0.329 | 0.8685 | 9 | 0.3927 | 0.369 | 0.94× | 0.4496 | carries timing signal |
| gru_classifier | 0.046 | 0.220 | 0.4657 | 9 | 0.3927 | 0.383 | 0.98× | 0.4294 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00405 | 0.04602 | 0.00417 | 0.00415 | 7/10 |
| calibrated | 0.00410 | 0.04586 | 0.00417 | 0.00415 | 3/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `mean|q=-|max`, threshold 0.182 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.001 | — | 0.485 | 0.00 / 0.00 / 0.00 | 0.00 | 0.000 | 0.001 / 0.003 |
| NIDRA (stochastic rollout) | 0.002 | [0.000, 0.005] | 0.487 | 0.00 / 0.00 / 0.00 | 12.92 | 0.000 | 0.001 / 0.005 |
| NIDRA (deterministic rollout) | 0.000 | [0.000, 0.001] | 0.089 | 0.00 / 0.00 / 0.00 | 15.06 | 0.000 | 0.000 / 0.002 |
| persistence + learned noise (mean disabled) | 0.002 | [0.001, 0.006] | 0.516 | 0.01 / 0.03 / 0.01 | 56.69 | 0.001 | 0.002 / 0.005 |
| persistence + isotropic noise | 0.001 | — | 0.456 | 0.01 / 0.05 / 0.02 | 76.92 | 0.000 | 0.001 / 0.004 |
| persistence (risk head on S_t) | 0.001 | [0.000, 0.002] | 0.099 | 0.01 / 0.03 / 0.02 | 43.89 | 0.000 | 0.001 / 0.005 |
| ridge two-lag dynamics + risk head | 0.002 | [0.000, 0.011] | 0.237 | 0.01 / 0.09 / 0.01 | 180.50 | 0.000 | 0.003 / 0.006 |
| oracle: risk head on the true future | 0.005 | — | 0.246 | 0.01 / 0.12 / 0.02 | 166.11 | 0.000 | 0.018 / 0.030 |
| logistic regression on S_t | 0.000 | [0.000, 0.001] | 0.141 | 0.00 / 0.03 / 0.00 | 470.74 | 0.000 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.000 | [0.000, 0.000] | 0.007 | 0.00 / 0.00 / 0.00 | 95.69 | 0.000 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.001 | [0.000, 0.005] | 0.311 | 0.01 / 0.09 / 0.01 | 215.67 | 0.000 | 0.000 / 0.003 |
| GRU sequence classifier | 0.000 | [0.000, 0.000] | 0.070 | 0.00 / 0.00 / 0.00 | 0.00 | 0.000 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.001 | [-0.000, 0.003] |
| world_model - persistence_rollout | 0.001 | [-0.000, 0.003] |
| world_model - noised_persistence | -0.001 | [-0.002, 0.000] |
| world_model - isotropic_noise_persistence | 0.000 | [-0.001, 0.001] |
| world_model - world_model_deterministic | 0.001 | [0.000, 0.005] |
| world_model - ridge_two_lag | -0.000 | [-0.006, 0.001] |
| world_model_deterministic - persistence | -0.000 | [-0.002, 0.000] |
| world_model_deterministic - ridge_two_lag | -0.001 | [-0.010, 0.000] |
| noised_persistence - persistence | 0.002 | [0.000, 0.004] |
| world_model - lr_current_state | 0.001 | [0.000, 0.004] |
| world_model - lr_flattened_history | 0.001 | [0.000, 0.005] |
| world_model - gbdt_current_state | 0.000 | [-0.004, 0.004] |
| world_model - gru_classifier | 0.001 | [0.000, 0.005] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.457, period2_persistence 0.045, ridge_two_lag 0.459; NIDRA vs ridge -0.005.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.000 | 0.000 | 0.002 | 0.00339 | 0.00 (68) |
| 2 | 2 | 0.00019 | 0.000 | 0.000 | 0.001 | 0.00334 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.000 | 0.000 | 0.002 | 0.00328 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.000 | 0.000 | 0.001 | 0.00323 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.000 | 0.000 | 0.001 | 0.00318 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.000 | 0.000 | 0.001 | 0.00313 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | — |
| 3 | 12 | 0.001 | 0.001 | 0.002 | 0.000 | 0.000 | — |
| 5 | 20 | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 | — |
| 10 | 38 | 0.002 | 0.002 | 0.005 | 0.002 | 0.000 | — |
| 15 | 52 | 0.003 | 0.005 | 0.006 | 0.003 | 0.000 | — |
| 30 | 73 | 0.004 | 0.004 | 0.007 | 0.002 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.182), 2 consecutive windows: 5 episodes, 0 warned before onset, 0 alerted inside the episode; median lead None s, median latency None min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | no (—) |
| 172.16.0.1@1499346900 | 27 | 14 | 0.019 | no (—) | no (—) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.035 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.044 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.053 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `thursday_infiltration:lateral` | — | 73 | 3 | 1¹ | 0.001 | 0.007 | 0.001 | 0.230 |
| `thursday_web:initial_access` | — | 49 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.157 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.008 | 0.435 | 0.8612 | 2 | 0.3664 | 0.257 | 0.70× | 0.2298 | carries timing signal |
| world_model | 0.008 | 0.440 | 0.8554 | 2 | 0.3664 | 0.245 | 0.67× | 0.2208 | carries timing signal |
| world_model_deterministic | 0.004 | 0.105 | 0.7659 | 2 | 0.3664 | 0.241 | 0.66× | 0.1902 | carries timing signal |
| noised_persistence | 0.008 | 0.458 | 0.8012 | 2 | 0.3664 | 0.249 | 0.68× | 0.2261 | carries timing signal |
| isotropic_noise_persistence | 0.007 | 0.393 | 0.7470 | 2 | 0.3664 | 0.261 | 0.71× | 0.2147 | carries timing signal |
| persistence | 0.004 | 0.149 | 0.6432 | 2 | 0.3664 | 0.249 | 0.68× | 0.1852 | carries timing signal |
| ridge_two_lag | 0.007 | 0.258 | 0.5926 | 2 | 0.3664 | 0.249 | 0.68× | 0.1674 | carries timing signal |
| oracle_true_future | 0.012 | 0.239 | 0.5684 | 2 | 0.3664 | 0.274 | 0.75× | 0.1977 | carries timing signal |
| lr_current_state | 0.004 | 0.319 | 0.6251 | 2 | 0.3664 | 0.261 | 0.71× | 0.2709 | carries timing signal |
| lr_flattened_history | 0.003 | 0.043 | 0.1923 | 2 | 0.3664 | 0.247 | 0.67× | 0.2151 | carries timing signal |
| gbdt_current_state | 0.007 | 0.353 | 0.5670 | 2 | 0.3664 | 0.339 | 0.92× | 0.4532 | carries timing signal |
| gru_classifier | 0.003 | 0.188 | 0.1539 | 2 | 0.3664 | 0.369 | 1.01× | 0.4909 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00039 | 0.04978 | 0.00034 | 0.00034 | 6/10 |
| calibrated | 0.00035 | 0.04967 | 0.00034 | 0.00034 | 2/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
