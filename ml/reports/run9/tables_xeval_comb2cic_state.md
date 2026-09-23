### Systems — val
**val** — 20451 scored rows, natural prevalence 0.00046, operating point `p_above_half|q=-|max`, threshold 0.041 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.391 | — | 0.902 | 0.57 / 0.44 / 0.50 | 8.48 | 0.498 | 0.000 / 0.000 |
| NIDRA (stochastic rollout) | 0.389 | [0.002, 0.666] | 0.902 | 0.01 / 0.72 / 0.02 | 2271.29 | 0.495 | 0.000 / 0.000 |
| NIDRA (deterministic rollout) | 0.334 | [0.002, 0.653] | 0.968 | 0.04 / 0.58 / 0.08 | 349.57 | 0.425 | 0.000 / 0.001 |
| persistence + learned noise (mean disabled) | 0.398 | [0.002, 0.678] | 0.893 | 0.01 / 0.73 / 0.02 | 2340.49 | 0.506 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.372 | — | 0.808 | 0.00 / 0.88 / 0.00 | 38520.11 | 0.474 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.385 | [0.002, 0.670] | 0.774 | 0.17 / 0.56 / 0.27 | 67.83 | 0.488 | 0.000 / 0.000 |
| ridge two-lag dynamics + risk head | 0.137 | [0.001, 0.459] | 0.903 | 0.02 / 0.60 / 0.04 | 642.60 | 0.177 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.442 | — | 0.948 | 0.05 / 0.64 / 0.10 | 282.37 | 0.513 | 0.021 / 0.009 |
| logistic regression on S_t | 0.201 | [0.000, 0.528] | 0.746 | 0.00 / 0.84 / 0.00 | 51295.86 | 0.256 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.147 | [0.000, 0.529] | 0.627 | 0.00 / 0.59 / 0.00 | 30777.02 | 0.191 | 0.001 / 0.002 |
| gradient-boosted trees on S_t | 0.419 | [0.003, 0.684] | 0.766 | 0.00 / 0.89 / 0.00 | 50481.95 | 0.538 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.86 / R 0.17 / F1 0.28, 73 alerts.

### Attribution — val
**val** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.004 | [-0.036, 0.045] |
| world_model - persistence_rollout | 0.004 | [-0.036, 0.045] |
| world_model - noised_persistence | -0.009 | [-0.062, 0.027] |
| world_model - isotropic_noise_persistence | 0.017 | [-0.023, 0.050] |
| world_model - world_model_deterministic | 0.054 | [-0.017, 0.124] |
| world_model - ridge_two_lag | 0.252 | [0.000, 0.396] |
| world_model_deterministic - persistence | -0.050 | [-0.105, 0.016] |
| world_model_deterministic - ridge_two_lag | 0.198 | [0.001, 0.352] |
| noised_persistence - persistence | 0.013 | [-0.013, 0.040] |
| world_model - lr_current_state | 0.188 | [-0.006, 0.299] |
| world_model - lr_flattened_history | 0.242 | [0.000, 0.392] |
| world_model - gbdt_current_state | -0.030 | [-0.089, 0.050] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.575, period2_persistence 0.053, ridge_two_lag 0.436; NIDRA vs ridge 0.247.

### Horizon — val
**val** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00035 | 0.526 | 0.466 | 0.530 | 0.00974 | 0.44 (281) |
| 2 | 2 | 0.00035 | 0.440 | 0.438 | 0.536 | 0.01167 | 0.32 (281) |
| 3 | 3 | 0.00035 | 0.392 | 0.396 | 0.564 | 0.01269 | 0.19 (281) |
| 4 | 4 | 0.00035 | 0.298 | 0.399 | 0.563 | 0.01318 | 0.11 (281) |
| 5 | 5 | 0.00035 | 0.227 | 0.378 | 0.559 | 0.01337 | 0.04 (281) |
| 6 | 6 | 0.00035 | 0.157 | 0.338 | 0.519 | 0.01351 | 0.01 (281) |

### Onset forecasting — val
**val** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | — |
| 3 | 28 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | — |
| 5 | 46 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | — |
| 10 | 86 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | — |
| 15 | 106 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | — |
| 30 | 134 | 0.000 | 0.001 | 0.000 | 0.002 | 0.003 | — |

### Episodes — val
**val** — per episode at the selected threshold (0.041), 2 consecutive windows: 10 episodes, 0 warned before onset, 3 alerted inside the episode; median lead None s, median latency 3.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313409420 | 5 | 28 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313410320 | 4 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313411220 | 25 | 11 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313413500 | 0 | 12 | 0.006 | no (—) | no (—) |
| 147.32.84.165@1313414280 | 46 | 12 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313417880 | 10 | 13 | 0.006 | no (—) | no (—) |
| 147.32.84.165@1313419200 | 32 | 12 | 0.004 | no (—) | yes (3.0) |
| 147.32.84.165@1313489340 | 116 | 6 | 0.006 | no (—) | yes (3.0) |
| 172.16.0.1@1499188140 | 62 | 1 | 0.000 | no (—) | yes (21.0) |
| 172.16.0.1@1499278320 | 21 | 30 | 0.021 | no (—) | no (—) |

### Per attack group — val
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_6:exfil` | menti | 116 | 1 | 1¹ | 0.733 | 0.744 | 0.647 | 0.489 |
| `tuesday:initial_access` | — | 63 | 1 | 1¹ | 0.147 | 0.190 | 0.201 | 0.033 |
| `ctu_4:exfil` | rbot | 61 | 5 | 1¹ | 0.167 | 0.237 | 0.168 | 0.420 |
| `ctu_4:c2` | rbot | 56 | 6 | 1¹ | 0.000 | 0.004 | 0.000 | 0.169 |
| `ctu_4:recon` | rbot | 44 | 6 | 1¹ | 0.000 | 0.000 | 0.000 | 0.500 |
| `wednesday:initial_access` | — | 26 | 1 | 1¹ | 0.000 | 0.001 | 0.001 | 0.323 |
| `ctu_6:c2` | menti | 6 | 1 | 1¹ | 0.000 | 0.048 | 0.000 | 0.139 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — val
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.531 | 0.826 | 0.9966 | 2 | 0.8140 | 0.952 | 1.17× | 0.8204 | carries timing signal |
| world_model | 0.522 | 0.825 | 0.9973 | 2 | 0.8140 | 0.949 | 1.17× | 0.8205 | carries timing signal |
| world_model_deterministic | 0.514 | 0.884 | 0.9924 | 2 | 0.8140 | 0.956 | 1.17× | 0.8209 | carries timing signal |
| noised_persistence | 0.539 | 0.810 | 0.9980 | 2 | 0.8140 | 0.948 | 1.16× | 0.8196 | carries timing signal |
| isotropic_noise_persistence | 0.522 | 0.758 | 0.9980 | 2 | 0.8140 | 0.939 | 1.15× | 0.7640 | carries timing signal |
| persistence | 0.536 | 0.834 | 0.9977 | 2 | 0.8140 | 0.931 | 1.14× | 0.7400 | carries timing signal |
| ridge_two_lag | 0.334 | 0.811 | 0.9795 | 2 | 0.8140 | 0.943 | 1.16× | 0.7990 | carries timing signal |
| oracle_true_future | 0.594 | 0.870 | 0.9950 | 2 | 0.8140 | 0.970 | 1.19× | 0.8897 | carries timing signal |
| lr_current_state | 0.386 | 0.797 | 0.9718 | 2 | 0.8140 | 0.918 | 1.13× | 0.6947 | carries timing signal |
| lr_flattened_history | 0.368 | 0.752 | 0.9903 | 2 | 0.8140 | 0.912 | 1.12× | 0.6670 | carries timing signal |
| gbdt_current_state | 0.552 | 0.855 | 0.9973 | 2 | 0.8140 | 0.924 | 1.14× | 0.7218 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — val
**val** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00140 | 0.05379 | 0.00046 | 0.00046 | 10/10 |
| calibrated | 0.00037 | 0.04972 | 0.00046 | 0.00046 | 4/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `p_above_half|q=-|max`, threshold 0.041 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.164 | — | 0.972 | 0.91 / 0.02 / 0.04 | 0.24 | 0.141 | 0.005 / 0.017 |
| NIDRA (stochastic rollout) | 0.167 | [0.057, 0.347] | 0.972 | 0.14 / 0.96 / 0.24 | 715.70 | 0.142 | 0.006 / 0.018 |
| NIDRA (deterministic rollout) | 0.157 | [0.046, 0.281] | 0.932 | 0.14 / 0.38 / 0.21 | 265.38 | 0.133 | 0.006 / 0.013 |
| persistence + learned noise (mean disabled) | 0.127 | [0.053, 0.242] | 0.960 | 0.10 / 0.84 / 0.17 | 937.30 | 0.117 | 0.003 / 0.008 |
| persistence + isotropic noise | 0.066 | — | 0.768 | 0.01 / 0.88 / 0.01 | 19349.90 | 0.073 | 0.000 / 0.002 |
| persistence (risk head on S_t) | 0.045 | [0.008, 0.151] | 0.244 | 0.22 / 0.04 / 0.07 | 17.94 | 0.053 | 0.000 / 0.001 |
| ridge two-lag dynamics + risk head | 0.080 | [0.029, 0.170] | 0.734 | 0.12 / 0.27 / 0.17 | 229.70 | 0.064 | 0.002 / 0.007 |
| oracle: risk head on the true future | 0.089 | — | 0.604 | 0.14 / 0.14 / 0.14 | 94.62 | 0.084 | 0.049 / 0.019 |
| logistic regression on S_t | 0.046 | [0.010, 0.139] | 0.390 | 0.00 / 0.48 / 0.00 | 25678.21 | 0.049 | 0.001 / 0.004 |
| logistic regression on the L-window history | 0.013 | [0.003, 0.087] | 0.088 | 0.00 / 0.04 / 0.00 | 12179.03 | 0.005 | 0.055 / 0.030 |
| gradient-boosted trees on S_t | 0.065 | [0.021, 0.213] | 0.360 | 0.00 / 0.47 / 0.00 | 25221.93 | 0.081 | 0.001 / 0.002 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.122 | [0.029, 0.285] |
| world_model - persistence_rollout | 0.122 | [0.029, 0.285] |
| world_model - noised_persistence | 0.040 | [-0.004, 0.133] |
| world_model - isotropic_noise_persistence | 0.101 | [0.021, 0.256] |
| world_model - world_model_deterministic | 0.010 | [-0.034, 0.071] |
| world_model - ridge_two_lag | 0.087 | [0.020, 0.203] |
| world_model_deterministic - persistence | 0.112 | [0.024, 0.230] |
| world_model_deterministic - ridge_two_lag | 0.077 | [0.014, 0.176] |
| noised_persistence - persistence | 0.082 | [0.025, 0.171] |
| world_model - lr_current_state | 0.121 | [0.032, 0.276] |
| world_model - lr_flattened_history | 0.154 | [0.028, 0.333] |
| world_model - gbdt_current_state | 0.102 | [-0.017, 0.257] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.565, period2_persistence 0.045, ridge_two_lag 0.406; NIDRA vs ridge 0.267.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.122 | 0.081 | 0.053 | 0.03085 | 0.03 (664) |
| 2 | 2 | 0.00292 | 0.122 | 0.093 | 0.050 | 0.03064 | 0.01 (662) |
| 3 | 3 | 0.00291 | 0.121 | 0.113 | 0.051 | 0.03058 | 0.00 (661) |
| 4 | 4 | 0.00291 | 0.109 | 0.119 | 0.051 | 0.03060 | 0.00 (660) |
| 5 | 5 | 0.00290 | 0.104 | 0.110 | 0.051 | 0.03051 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.105 | 0.105 | 0.051 | 0.03053 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.001 | 0.000 | 0.000 | 0.002 | 0.014 | — |
| 3 | 36 | 0.003 | 0.000 | 0.001 | 0.001 | 0.043 | — |
| 5 | 60 | 0.005 | 0.000 | 0.002 | 0.001 | 0.055 | — |
| 10 | 113 | 0.011 | 0.000 | 0.006 | 0.001 | 0.038 | — |
| 15 | 161 | 0.017 | 0.001 | 0.007 | 0.002 | 0.030 | — |
| 30 | 299 | 0.034 | 0.002 | 0.011 | 0.004 | 0.023 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.041), 2 consecutive windows: 15 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 1.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.003 | no (—) | no (—) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.004 | no (—) | no (—) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.005 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | yes (1.0) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.020 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.018 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.020 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 0.027 | no (—) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.020 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.021 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.018 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `friday_morning:c2` | — | 862 | 7 | 7 | 0.140 | 0.042 | 0.012 | 0.508 |
| `friday_portscan:recon` | — | 58 | 6 | 1¹ | 0.023 | 0.121 | 0.012 | 0.408 |
| `friday_ddos:exfil` | — | 26 | 2 | 2 | 0.755 | 0.736 | 0.769 | -0.786 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.260 | 0.916 | 0.9172 | 9 | 0.3927 | 0.407 | 1.04× | 0.5046 | **host identity** |
| world_model | 0.265 | 0.917 | 0.9373 | 9 | 0.3927 | 0.423 | 1.08× | 0.5223 | **host identity** |
| world_model_deterministic | 0.250 | 0.874 | 0.9239 | 9 | 0.3927 | 0.414 | 1.05× | 0.5196 | **host identity** |
| noised_persistence | 0.208 | 0.873 | 0.9127 | 9 | 0.3927 | 0.433 | 1.10× | 0.5428 | carries timing signal |
| isotropic_noise_persistence | 0.136 | 0.700 | 0.8831 | 9 | 0.3927 | 0.437 | 1.11× | 0.5421 | carries timing signal |
| persistence | 0.084 | 0.399 | 0.8825 | 9 | 0.3927 | 0.443 | 1.13× | 0.5559 | carries timing signal |
| ridge_two_lag | 0.171 | 0.684 | 0.9240 | 9 | 0.3927 | 0.429 | 1.09× | 0.5419 | **host identity** |
| oracle_true_future | 0.162 | 0.568 | 0.8973 | 9 | 0.3927 | 0.435 | 1.11× | 0.5318 | carries timing signal |
| lr_current_state | 0.101 | 0.509 | 0.8797 | 9 | 0.3927 | 0.408 | 1.04× | 0.4956 | carries timing signal |
| lr_flattened_history | 0.041 | 0.200 | 0.2787 | 9 | 0.3927 | 0.378 | 0.96× | 0.4845 | carries timing signal |
| gbdt_current_state | 0.131 | 0.498 | 0.8901 | 9 | 0.3927 | 0.463 | 1.18× | 0.5634 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00380 | 0.05033 | 0.00417 | 0.00415 | 10/10 |
| calibrated | 0.00407 | 0.04584 | 0.00417 | 0.00415 | 2/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `p_above_half|q=-|max`, threshold 0.041 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.224 | — | 0.979 | 0.75 / 0.11 / 0.20 | 0.57 | 0.344 | 0.004 / 0.007 |
| NIDRA (stochastic rollout) | 0.194 | [0.006, 0.617] | 0.978 | 0.02 / 0.94 / 0.03 | 825.29 | 0.295 | 0.005 / 0.007 |
| NIDRA (deterministic rollout) | 0.182 | [0.007, 0.595] | 0.863 | 0.03 / 0.55 / 0.05 | 317.52 | 0.300 | 0.004 / 0.010 |
| persistence + learned noise (mean disabled) | 0.311 | [0.006, 0.764] | 0.942 | 0.01 / 0.83 / 0.02 | 1115.63 | 0.549 | 0.002 / 0.004 |
| persistence + isotropic noise | 0.280 | — | 0.780 | 0.00 / 0.82 / 0.00 | 29440.99 | 0.509 | 0.000 / 0.001 |
| persistence (risk head on S_t) | 0.307 | [0.000, 0.772] | 0.411 | 0.23 / 0.31 / 0.27 | 15.47 | 0.581 | 0.000 / 0.001 |
| ridge two-lag dynamics + risk head | 0.085 | [0.007, 0.241] | 0.848 | 0.04 / 0.64 / 0.07 | 241.35 | 0.061 | 0.002 / 0.005 |
| oracle: risk head on the true future | 0.354 | — | 0.566 | 0.08 / 0.46 / 0.13 | 81.76 | 0.525 | 0.251 / 0.099 |
| logistic regression on S_t | 0.065 | [0.001, 0.231] | 0.563 | 0.00 / 0.63 / 0.00 | 41782.61 | 0.115 | 0.000 / 0.002 |
| logistic regression on the L-window history | 0.021 | [0.000, 0.054] | 0.225 | 0.00 / 0.16 / 0.00 | 24292.66 | 0.008 | 0.001 / 0.005 |
| gradient-boosted trees on S_t | 0.285 | [0.001, 0.734] | 0.504 | 0.00 / 0.55 / 0.00 | 41192.77 | 0.507 | 0.001 / 0.003 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.114 | [-0.391, 0.131] |
| world_model - persistence_rollout | -0.114 | [-0.391, 0.131] |
| world_model - noised_persistence | -0.118 | [-0.315, 0.075] |
| world_model - isotropic_noise_persistence | -0.086 | [-0.276, 0.119] |
| world_model - world_model_deterministic | 0.012 | [-0.122, 0.157] |
| world_model - ridge_two_lag | 0.109 | [-0.085, 0.566] |
| world_model_deterministic - persistence | -0.125 | [-0.394, 0.036] |
| world_model_deterministic - ridge_two_lag | 0.097 | [-0.146, 0.511] |
| noised_persistence - persistence | 0.004 | [-0.070, 0.065] |
| world_model - lr_current_state | 0.128 | [-0.036, 0.475] |
| world_model - lr_flattened_history | 0.172 | [0.005, 0.606] |
| world_model - gbdt_current_state | -0.091 | [-0.325, 0.136] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.585, period2_persistence 0.051, ridge_two_lag 0.455; NIDRA vs ridge 0.240.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.405 | 0.457 | 0.547 | 0.00322 | 0.00 (68) |
| 2 | 2 | 0.00019 | 0.214 | 0.335 | 0.544 | 0.00320 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.086 | 0.169 | 0.536 | 0.00318 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.085 | 0.114 | 0.525 | 0.00315 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.039 | 0.056 | 0.518 | 0.00311 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.036 | 0.038 | 0.507 | 0.00308 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 3 | 12 | 0.003 | 0.000 | 0.000 | 0.001 | 0.001 | — |
| 5 | 20 | 0.004 | 0.000 | 0.002 | 0.001 | 0.001 | — |
| 10 | 38 | 0.006 | 0.000 | 0.002 | 0.002 | 0.006 | — |
| 15 | 52 | 0.007 | 0.001 | 0.005 | 0.003 | 0.005 | — |
| 30 | 73 | 0.010 | 0.001 | 0.006 | 0.002 | 0.004 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.041), 2 consecutive windows: 5 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 32.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | yes (32.0) |
| 172.16.0.1@1499346900 | 27 | 14 | 0.007 | no (—) | no (—) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.019 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.021 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.020 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `thursday_infiltration:lateral` | — | 73 | 3 | 1¹ | 0.026 | 0.002 | 0.000 | 0.434 |
| `thursday_web:initial_access` | — | 49 | 2 | 1¹ | 0.369 | 0.860 | 0.759 | 0.197 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.306 | 0.946 | 0.9841 | 2 | 0.3664 | 0.738 | 2.01× | 0.7565 | carries timing signal |
| world_model | 0.269 | 0.945 | 0.9745 | 2 | 0.3664 | 0.721 | 1.97× | 0.7495 | carries timing signal |
| world_model_deterministic | 0.251 | 0.825 | 0.9649 | 2 | 0.3664 | 0.608 | 1.66× | 0.5947 | carries timing signal |
| noised_persistence | 0.356 | 0.888 | 0.9900 | 2 | 0.3664 | 0.682 | 1.86× | 0.6994 | carries timing signal |
| isotropic_noise_persistence | 0.320 | 0.742 | 0.9504 | 2 | 0.3664 | 0.663 | 1.81× | 0.6680 | carries timing signal |
| persistence | 0.317 | 0.477 | 0.9678 | 2 | 0.3664 | 0.601 | 1.64× | 0.5838 | carries timing signal |
| ridge_two_lag | 0.190 | 0.816 | 0.9863 | 2 | 0.3664 | 0.699 | 1.91× | 0.7456 | carries timing signal |
| oracle_true_future | 0.381 | 0.567 | 0.9632 | 2 | 0.3664 | 0.635 | 1.73× | 0.5474 | carries timing signal |
| lr_current_state | 0.123 | 0.613 | 0.9199 | 2 | 0.3664 | 0.571 | 1.56× | 0.5433 | carries timing signal |
| lr_flattened_history | 0.037 | 0.347 | 0.7223 | 2 | 0.3664 | 0.513 | 1.40× | 0.5442 | carries timing signal |
| gbdt_current_state | 0.323 | 0.584 | 0.9750 | 2 | 0.3664 | 0.619 | 1.69× | 0.5985 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00100 | 0.05242 | 0.00034 | 0.00034 | 9/10 |
| calibrated | 0.00033 | 0.04966 | 0.00034 | 0.00034 | 1/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
