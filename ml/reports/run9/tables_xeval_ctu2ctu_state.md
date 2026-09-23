_val: no benchmark.json at experiments/runs/xeval_ctu2ctu_state/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 25099 scored rows, natural prevalence 0.00749, operating point `mean|q=-|max`, threshold 0.182 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.196 | — | 0.817 | 0.10 / 0.01 / 0.02 | 11.06 | 0.261 | 0.001 / 0.003 |
| NIDRA (stochastic rollout) | 0.215 | [0.090, 0.514] | 0.819 | 0.56 / 0.25 / 0.34 | 24.75 | 0.287 | 0.001 / 0.003 |
| NIDRA (deterministic rollout) | 0.154 | [0.056, 0.444] | 0.691 | 0.52 / 0.25 / 0.33 | 28.17 | 0.202 | 0.002 / 0.005 |
| persistence + learned noise (mean disabled) | 0.190 | [0.079, 0.444] | 0.773 | 0.52 / 0.29 / 0.37 | 33.96 | 0.256 | 0.001 / 0.003 |
| persistence + isotropic noise | 0.188 | — | 0.751 | 0.52 / 0.32 / 0.40 | 36.59 | 0.254 | 0.001 / 0.002 |
| persistence (risk head on S_t) | 0.149 | [0.057, 0.411] | 0.588 | 0.53 / 0.23 / 0.32 | 26.06 | 0.202 | 0.002 / 0.004 |
| ridge two-lag dynamics + risk head | 0.184 | [0.073, 0.450] | 0.600 | 0.23 / 0.40 / 0.29 | 165.66 | 0.247 | 0.002 / 0.003 |
| oracle: risk head on the true future | 0.214 | — | 0.670 | 0.50 / 0.43 / 0.46 | 54.26 | 0.263 | 0.003 / 0.004 |
| logistic regression on S_t | 0.197 | [0.094, 0.407] | 0.808 | 0.11 / 0.43 / 0.18 | 425.24 | 0.265 | 0.002 / 0.004 |
| logistic regression on the L-window history | 0.042 | [0.013, 0.163] | 0.421 | 0.22 / 0.16 / 0.19 | 74.17 | 0.053 | 0.001 / 0.003 |
| gradient-boosted trees on S_t | 0.242 | [0.111, 0.470] | 0.632 | 0.27 / 0.41 / 0.33 | 135.80 | 0.325 | 0.002 / 0.004 |
| GRU sequence classifier | 0.291 | [0.121, 0.504] | 0.576 | 0.70 / 0.31 / 0.43 | 16.74 | 0.388 | 0.002 / 0.005 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.00 / F1 0.00, 3 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.066 | [0.029, 0.132] |
| world_model - persistence_rollout | 0.066 | [0.029, 0.132] |
| world_model - noised_persistence | 0.025 | [0.010, 0.070] |
| world_model - isotropic_noise_persistence | 0.027 | [0.009, 0.082] |
| world_model - world_model_deterministic | 0.061 | [0.019, 0.123] |
| world_model - ridge_two_lag | 0.031 | [-0.004, 0.094] |
| world_model_deterministic - persistence | 0.005 | [-0.011, 0.053] |
| world_model_deterministic - ridge_two_lag | -0.029 | [-0.053, 0.030] |
| noised_persistence - persistence | 0.041 | [0.016, 0.073] |
| world_model - lr_current_state | 0.018 | [-0.033, 0.190] |
| world_model - lr_flattened_history | 0.173 | [0.074, 0.361] |
| world_model - gbdt_current_state | -0.027 | [-0.065, 0.067] |
| world_model - gru_classifier | -0.076 | [-0.184, 0.087] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.545, period2_persistence 0.028, ridge_two_lag 0.494; NIDRA vs ridge 0.101.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00568 | 0.261 | 0.202 | 0.197 | 0.10258 | 0.25 (2768) |
| 2 | 2 | 0.00565 | 0.248 | 0.173 | 0.199 | 0.10291 | 0.11 (2774) |
| 3 | 3 | 0.00566 | 0.209 | 0.096 | 0.207 | 0.10396 | 0.02 (2781) |
| 4 | 4 | 0.00566 | 0.138 | 0.014 | 0.196 | 0.10607 | 0.00 (2787) |
| 5 | 5 | 0.00568 | 0.091 | 0.007 | 0.186 | 0.10743 | 0.00 (2795) |
| 6 | 6 | 0.00571 | 0.061 | 0.006 | 0.200 | 0.10910 | 0.00 (2804) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 126 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 3 | 382 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | — |
| 5 | 635 | 0.001 | 0.002 | 0.002 | 0.002 | 0.001 | — |
| 10 | 1188 | 0.002 | 0.003 | 0.003 | 0.003 | 0.002 | — |
| 15 | 1523 | 0.003 | 0.004 | 0.003 | 0.004 | 0.003 | — |
| 30 | 2171 | 0.004 | 0.006 | 0.005 | 0.006 | 0.005 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.182), 2 consecutive windows: 131 episodes, 0 warned before onset, 2 alerted inside the episode; median lead None s, median latency 115.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313504340 | 216 | 0 | — | no (—) | yes (69.0) |
| 147.32.84.165@1313519940 | 12 | 14 | 0.052 | no (—) | no (—) |
| 147.32.84.165@1313522640 | 863 | 30 | 0.027 | no (—) | yes (161.0) |
| 147.32.84.165@1313586240 | 0 | 30 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313591040 | 164 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313664420 | 0 | 8 | 0.022 | no (—) | no (—) |
| 147.32.84.165@1313665440 | 2 | 16 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313668380 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313669160 | 16 | 12 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313670780 | 2 | 10 | 0.035 | no (—) | no (—) |
| 147.32.84.165@1313675220 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313675820 | 0 | 9 | 0.002 | no (—) | no (—) |
| 147.32.84.165@1313676360 | 9 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313677800 | 1 | 14 | 0.025 | no (—) | no (—) |
| 147.32.84.165@1313678460 | 5 | 10 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313679420 | 2 | 10 | 0.029 | no (—) | no (—) |
| 147.32.84.191@1313586240 | 0 | 5 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313592180 | 149 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313665560 | 0 | 17 | 0.043 | no (—) | no (—) |
| 147.32.84.191@1313666580 | 0 | 16 | 0.005 | no (—) | no (—) |
| 147.32.84.191@1313668380 | 0 | 29 | 0.002 | no (—) | no (—) |
| 147.32.84.191@1313669160 | 16 | 12 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313670720 | 3 | 9 | 0.025 | no (—) | no (—) |
| 147.32.84.191@1313675280 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313675820 | 0 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313676360 | 10 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313677800 | 0 | 14 | 0.007 | no (—) | no (—) |
| 147.32.84.191@1313678460 | 6 | 10 | 0.006 | no (—) | no (—) |
| 147.32.84.191@1313679420 | 2 | 9 | 0.031 | no (—) | no (—) |
| 147.32.84.192@1313586240 | 1 | 4 | 0.003 | no (—) | no (—) |
| 147.32.84.192@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313592480 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313665500 | 0 | 9 | 0.032 | no (—) | no (—) |
| 147.32.84.192@1313666580 | 0 | 17 | 0.019 | no (—) | no (—) |
| 147.32.84.192@1313668380 | 0 | 29 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313669280 | 14 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313670720 | 3 | 9 | 0.025 | no (—) | no (—) |
| 147.32.84.192@1313675280 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313675820 | 0 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313676360 | 9 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313677800 | 0 | 14 | 0.009 | no (—) | no (—) |
| 147.32.84.192@1313678460 | 7 | 10 | 0.010 | no (—) | no (—) |
| 147.32.84.192@1313679480 | 1 | 10 | 0.025 | no (—) | no (—) |
| 147.32.84.193@1313586240 | 0 | 3 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313592600 | 140 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313665620 | 0 | 10 | 0.038 | no (—) | no (—) |
| 147.32.84.193@1313666580 | 0 | 15 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313668380 | 0 | 29 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313669100 | 18 | 11 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313670720 | 3 | 8 | 0.022 | no (—) | no (—) |
| 147.32.84.193@1313675340 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313675820 | 0 | 7 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313676360 | 9 | 8 | 0.006 | no (—) | no (—) |
| 147.32.84.193@1313677800 | 0 | 14 | 0.006 | no (—) | no (—) |
| 147.32.84.193@1313678460 | 7 | 10 | 0.021 | no (—) | no (—) |
| 147.32.84.193@1313679480 | 0 | 9 | 0.020 | no (—) | no (—) |
| 147.32.84.204@1313586240 | 0 | 1 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313588580 | 0 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313592600 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313665500 | 0 | 7 | 0.037 | no (—) | no (—) |
| 147.32.84.204@1313666520 | 0 | 16 | 0.027 | no (—) | no (—) |
| 147.32.84.204@1313668380 | 0 | 30 | 0.006 | no (—) | no (—) |
| 147.32.84.204@1313669280 | 14 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313670720 | 3 | 9 | 0.024 | no (—) | no (—) |
| 147.32.84.204@1313675340 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.204@1313675820 | 0 | 7 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313676360 | 9 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313677800 | 0 | 14 | 0.016 | no (—) | no (—) |
| 147.32.84.204@1313678460 | 8 | 10 | 0.007 | no (—) | no (—) |
| 147.32.84.204@1313679420 | 2 | 7 | 0.022 | no (—) | no (—) |
| 147.32.84.205@1313586240 | 1 | 0 | — | no (—) | no (—) |
| 147.32.84.205@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313592600 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313665620 | 1 | 9 | 0.038 | no (—) | no (—) |
| 147.32.84.205@1313666640 | 0 | 16 | 0.022 | no (—) | no (—) |
| 147.32.84.205@1313668380 | 1 | 28 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313669160 | 15 | 12 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313670780 | 2 | 11 | 0.022 | no (—) | no (—) |
| 147.32.84.205@1313675460 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313676360 | 10 | 8 | 0.005 | no (—) | no (—) |
| 147.32.84.205@1313677800 | 0 | 14 | 0.002 | no (—) | no (—) |
| 147.32.84.205@1313678460 | 9 | 10 | 0.020 | no (—) | no (—) |
| 147.32.84.205@1313679480 | 1 | 7 | 0.028 | no (—) | no (—) |
| 147.32.84.206@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.206@1313592600 | 146 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313666460 | 1 | 15 | 0.026 | no (—) | no (—) |
| 147.32.84.206@1313668380 | 0 | 30 | 0.017 | no (—) | no (—) |
| 147.32.84.206@1313669100 | 16 | 11 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313670720 | 3 | 10 | 0.024 | no (—) | no (—) |
| 147.32.84.206@1313675460 | 6 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.206@1313676360 | 9 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.206@1313677800 | 1 | 14 | 0.015 | no (—) | no (—) |
| 147.32.84.206@1313678460 | 9 | 10 | 0.012 | no (—) | no (—) |
| 147.32.84.206@1313679480 | 0 | 6 | 0.013 | no (—) | no (—) |
| 147.32.84.207@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313592600 | 143 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313666460 | 0 | 16 | 0.027 | no (—) | no (—) |
| 147.32.84.207@1313668380 | 5 | 30 | 0.026 | no (—) | no (—) |
| 147.32.84.207@1313669220 | 15 | 9 | 0.002 | no (—) | no (—) |
| 147.32.84.207@1313670720 | 3 | 9 | 0.032 | no (—) | no (—) |
| 147.32.84.207@1313675520 | 5 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313676360 | 9 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313677800 | 0 | 14 | 0.012 | no (—) | no (—) |
| 147.32.84.207@1313678460 | 9 | 10 | 0.006 | no (—) | no (—) |
| 147.32.84.207@1313679540 | 0 | 7 | 0.040 | no (—) | no (—) |
| 147.32.84.208@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.208@1313590380 | 0 | 29 | 0.005 | no (—) | no (—) |
| 147.32.84.208@1313592600 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313593380 | 131 | 6 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313666400 | 0 | 15 | 0.023 | no (—) | no (—) |
| 147.32.84.208@1313668380 | 4 | 30 | 0.010 | no (—) | no (—) |
| 147.32.84.208@1313669100 | 17 | 7 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313670720 | 2 | 9 | 0.024 | no (—) | no (—) |
| 147.32.84.208@1313675520 | 5 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313676360 | 9 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.208@1313677800 | 20 | 14 | 0.013 | no (—) | no (—) |
| 147.32.84.208@1313679540 | 0 | 8 | 0.038 | no (—) | no (—) |
| 147.32.84.209@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313592600 | 7 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313593500 | 127 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313666340 | 0 | 15 | 0.072 | no (—) | no (—) |
| 147.32.84.209@1313668380 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.209@1313669280 | 25 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313675520 | 5 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313676360 | 9 | 8 | 0.006 | no (—) | no (—) |
| 147.32.84.209@1313677800 | 0 | 14 | 0.024 | no (—) | no (—) |
| 147.32.84.209@1313678460 | 9 | 10 | 0.005 | no (—) | no (—) |
| 147.32.84.209@1313679540 | 0 | 7 | 0.017 | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_9:exfil` | neris | 1373 | 11 | 10 | 0.405 | 0.387 | 0.306 | 0.444 |
| `ctu_8:c2` | murlo | 1069 | 3 | 1¹ | 0.036 | 0.038 | 0.013 | 0.300 |
| `ctu_10:recon` | rbot | 488 | 59 | 10 | 0.001 | 0.001 | 0.001 | 0.213 |
| `ctu_10:exfil` | rbot | 307 | 38 | 10 | 0.001 | 0.001 | 0.000 | 0.520 |
| `ctu_10:c2` | rbot | 232 | 45 | 10 | 0.001 | 0.001 | 0.001 | 0.082 |
| `ctu_9:recon` | neris | 151 | 22 | 10 | 0.000 | 0.005 | 0.001 | 0.083 |
| `ctu_9:c2` | neris | 35 | 8 | 8 | 0.002 | 0.014 | 0.003 | 0.120 |
| `ctu_8:recon` | murlo | 31 | 3 | 1¹ | 0.037 | 0.009 | 0.011 | 0.402 |
| `ctu_8:exfil` | murlo | 2 | 2 | 1¹ | 0.000 | 0.008 | 0.000 | 0.093 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.529 | 0.754 | 0.9661 | 10 | 0.7192 | 0.941 | 1.31× | 0.8415 | carries timing signal |
| world_model | 0.533 | 0.754 | 0.9563 | 10 | 0.7192 | 0.942 | 1.31× | 0.8439 | carries timing signal |
| world_model_deterministic | 0.458 | 0.732 | 0.9527 | 10 | 0.7192 | 0.856 | 1.19× | 0.6241 | carries timing signal |
| noised_persistence | 0.474 | 0.689 | 0.9648 | 10 | 0.7192 | 0.934 | 1.30× | 0.8137 | carries timing signal |
| isotropic_noise_persistence | 0.467 | 0.666 | 0.9630 | 10 | 0.7192 | 0.926 | 1.29× | 0.7958 | carries timing signal |
| persistence | 0.451 | 0.760 | 0.9563 | 10 | 0.7192 | 0.779 | 1.08× | 0.4951 | **host identity** |
| ridge_two_lag | 0.453 | 0.663 | 0.9503 | 10 | 0.7192 | 0.825 | 1.15× | 0.5772 | carries timing signal |
| oracle_true_future | 0.507 | 0.726 | 0.9545 | 10 | 0.7192 | 0.843 | 1.17× | 0.6366 | carries timing signal |
| lr_current_state | 0.567 | 0.841 | 0.9268 | 10 | 0.7192 | 0.905 | 1.26× | 0.7924 | carries timing signal |
| lr_flattened_history | 0.266 | 0.559 | 0.9407 | 10 | 0.7192 | 0.705 | 0.98× | 0.3565 | **host identity** |
| gbdt_current_state | 0.518 | 0.752 | 0.9525 | 10 | 0.7192 | 0.817 | 1.14× | 0.5630 | carries timing signal |
| gru_classifier | 0.491 | 0.680 | 0.9526 | 10 | 0.7192 | 0.809 | 1.13× | 0.5344 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00674 | 0.04683 | 0.00749 | 0.00743 | 10/10 |
| calibrated | 0.00706 | 0.04402 | 0.00749 | 0.00743 | 9/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 21045 scored rows, natural prevalence 0.00409, operating point `mean|q=-|max`, threshold 0.182 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.188 | — | 0.780 | 1.00 / 0.06 / 0.11 | 0.00 | 0.194 | 0.000 / 0.001 |
| NIDRA (stochastic rollout) | 0.203 | [0.001, 0.256] | 0.794 | 0.83 / 0.18 / 0.29 | 2.15 | 0.210 | 0.000 / 0.001 |
| NIDRA (deterministic rollout) | 0.188 | [0.001, 0.207] | 0.294 | 0.75 / 0.18 / 0.29 | 3.52 | 0.198 | 0.001 / 0.001 |
| persistence + learned noise (mean disabled) | 0.222 | [0.001, 0.314] | 0.898 | 0.67 / 0.18 / 0.28 | 5.27 | 0.231 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.207 | — | 0.867 | 0.51 / 0.18 / 0.27 | 10.16 | 0.216 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.186 | [0.000, 0.209] | 0.299 | 0.64 / 0.18 / 0.28 | 5.86 | 0.195 | 0.000 / 0.000 |
| ridge two-lag dynamics + risk head | 0.180 | [0.002, 0.209] | 0.315 | 0.17 / 0.19 / 0.18 | 54.90 | 0.186 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.188 | — | 0.308 | 0.31 / 0.19 / 0.23 | 24.10 | 0.194 | 0.000 / 0.000 |
| logistic regression on S_t | 0.045 | [0.001, 0.135] | 0.354 | 0.05 / 0.21 / 0.08 | 231.99 | 0.046 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.074 | [0.000, 0.119] | 0.251 | 0.13 / 0.12 / 0.13 | 47.11 | 0.077 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.198 | [0.001, 0.237] | 0.533 | 0.22 / 0.20 / 0.21 | 41.76 | 0.206 | 0.000 / 0.000 |
| GRU sequence classifier | 0.189 | [0.000, 0.205] | 0.277 | 0.97 / 0.18 / 0.31 | 0.39 | 0.198 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.018 | [0.000, 0.050] |
| world_model - persistence_rollout | 0.018 | [0.000, 0.050] |
| world_model - noised_persistence | -0.019 | [-0.061, 0.001] |
| world_model - isotropic_noise_persistence | -0.004 | [-0.037, 0.003] |
| world_model - world_model_deterministic | 0.015 | [-0.001, 0.050] |
| world_model - ridge_two_lag | 0.023 | [-0.007, 0.049] |
| world_model_deterministic - persistence | 0.002 | [-0.002, 0.005] |
| world_model_deterministic - ridge_two_lag | 0.008 | [-0.007, 0.015] |
| noised_persistence - persistence | 0.037 | [0.000, 0.105] |
| world_model - lr_current_state | 0.159 | [-0.005, 0.176] |
| world_model - lr_flattened_history | 0.129 | [0.001, 0.149] |
| world_model - gbdt_current_state | 0.005 | [-0.001, 0.022] |
| world_model - gru_classifier | 0.014 | [0.001, 0.054] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.625, period2_persistence 0.060, ridge_two_lag 0.559; NIDRA vs ridge 0.149.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00391 | 0.208 | 0.196 | 0.190 | 0.04442 | 0.18 (992) |
| 2 | 2 | 0.00391 | 0.199 | 0.175 | 0.183 | 0.04508 | 0.14 (992) |
| 3 | 3 | 0.00391 | 0.139 | 0.065 | 0.189 | 0.04581 | 0.06 (994) |
| 4 | 4 | 0.00391 | 0.054 | 0.003 | 0.192 | 0.04641 | 0.01 (994) |
| 5 | 5 | 0.00391 | 0.023 | 0.002 | 0.191 | 0.04670 | 0.00 (994) |
| 6 | 6 | 0.00392 | 0.009 | 0.002 | 0.189 | 0.04700 | 0.00 (996) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 3 | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 5 | 18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 10 | 27 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 15 | 27 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 30 | 27 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.182), 2 consecutive windows: 6 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 62.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313428560 | 977 | 0 | — | no (—) | yes (62.0) |
| 147.32.84.165@1313752680 | 5 | 8 | 0.026 | no (—) | no (—) |
| 147.32.84.165@1313753400 | 9 | 7 | 0.027 | no (—) | no (—) |
| 147.32.84.191@1313753100 | 14 | 8 | 0.017 | no (—) | no (—) |
| 147.32.84.192@1313752500 | 15 | 0 | — | no (—) | no (—) |
| 147.32.84.192@1313754000 | 0 | 4 | 0.025 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_13:exfil` | virut | 752 | 1 | 1¹ | 0.264 | 0.251 | 0.252 | 0.558 |
| `ctu_13:recon` | virut | 150 | 1 | 1¹ | 0.002 | 0.000 | 0.000 | 0.596 |
| `ctu_12:c2` | nsis_ay | 63 | 5 | 3 | 0.004 | 0.001 | 0.000 | 0.505 |
| `ctu_13:c2` | virut | 47 | 1 | 1¹ | 0.001 | 0.000 | 0.000 | 0.531 |
| `ctu_12:recon` | nsis_ay | 18 | 4 | 2 | 0.001 | 0.004 | 0.000 | 0.414 |
| `ctu_12:exfil` | nsis_ay | 10 | 2 | 1¹ | 0.001 | 0.003 | 0.002 | 0.356 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.267 | 0.696 | 0.9930 | 3 | 0.9933 | 0.995 | 1.00× | 0.4963 | **host identity** |
| world_model | 0.281 | 0.711 | 0.9953 | 3 | 0.9933 | 0.995 | 1.00× | 0.4788 | **host identity** |
| world_model_deterministic | 0.225 | 0.359 | 0.9953 | 3 | 0.9933 | 0.989 | 1.00× | 0.2026 | **host identity** |
| noised_persistence | 0.314 | 0.789 | 0.9927 | 3 | 0.9933 | 0.997 | 1.00× | 0.6625 | **host identity** |
| isotropic_noise_persistence | 0.291 | 0.748 | 0.9669 | 3 | 0.9933 | 0.998 | 1.01× | 0.8030 | **host identity** |
| persistence | 0.230 | 0.504 | 0.9900 | 3 | 0.9933 | 0.991 | 1.00× | 0.2606 | **host identity** |
| ridge_two_lag | 0.229 | 0.389 | 0.9934 | 3 | 0.9933 | 0.991 | 1.00× | 0.2422 | **host identity** |
| oracle_true_future | 0.238 | 0.389 | 0.9815 | 3 | 0.9933 | 0.991 | 1.00× | 0.2735 | **host identity** |
| lr_current_state | 0.144 | 0.567 | 0.9535 | 3 | 0.9933 | 0.991 | 1.00× | 0.2525 | **host identity** |
| lr_flattened_history | 0.141 | 0.438 | 0.9795 | 3 | 0.9933 | 0.995 | 1.00× | 0.5328 | **host identity** |
| gbdt_current_state | 0.309 | 0.786 | 0.9995 | 3 | 0.9933 | 0.992 | 1.00× | 0.3234 | **host identity** |
| gru_classifier | 0.218 | 0.383 | 0.9776 | 3 | 0.9933 | 0.998 | 1.01× | 0.8015 | **host identity** |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00357 | 0.04695 | 0.00409 | 0.00408 | 10/10 |
| calibrated | 0.00386 | 0.04691 | 0.00409 | 0.00408 | 6/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
