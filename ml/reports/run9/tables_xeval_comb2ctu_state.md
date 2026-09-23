_val: no benchmark.json at experiments/runs/xeval_comb2ctu_state/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 25099 scored rows, natural prevalence 0.00749, operating point `p_above_half|q=-|max`, threshold 0.041 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.157 | — | 0.795 | 0.25 / 0.05 / 0.08 | 17.11 | 0.211 | 0.001 / 0.003 |
| NIDRA (stochastic rollout) | 0.134 | [0.060, 0.356] | 0.795 | 0.03 / 0.63 / 0.05 | 2897.18 | 0.179 | 0.001 / 0.003 |
| NIDRA (deterministic rollout) | 0.166 | [0.070, 0.438] | 0.672 | 0.20 / 0.42 / 0.27 | 207.71 | 0.222 | 0.001 / 0.003 |
| persistence + learned noise (mean disabled) | 0.116 | [0.053, 0.270] | 0.782 | 0.03 / 0.60 / 0.05 | 2738.70 | 0.154 | 0.001 / 0.003 |
| persistence + isotropic noise | 0.113 | — | 0.768 | 0.01 / 0.92 / 0.02 | 13219.60 | 0.151 | 0.001 / 0.003 |
| persistence (risk head on S_t) | 0.150 | [0.063, 0.382] | 0.623 | 0.36 / 0.33 / 0.34 | 73.71 | 0.201 | 0.002 / 0.004 |
| ridge two-lag dynamics + risk head | 0.114 | [0.052, 0.297] | 0.743 | 0.07 / 0.44 / 0.12 | 718.46 | 0.155 | 0.001 / 0.003 |
| oracle: risk head on the true future | 0.200 | — | 0.723 | 0.20 / 0.50 / 0.29 | 244.92 | 0.233 | 0.003 / 0.003 |
| logistic regression on S_t | 0.114 | [0.044, 0.231] | 0.810 | 0.01 / 0.95 / 0.02 | 14342.83 | 0.151 | 0.002 / 0.003 |
| logistic regression on the L-window history | 0.022 | [0.009, 0.083] | 0.423 | 0.01 / 0.25 / 0.01 | 5490.58 | 0.026 | 0.001 / 0.003 |
| gradient-boosted trees on S_t | 0.244 | [0.123, 0.486] | 0.792 | 0.01 / 0.91 / 0.02 | 13677.15 | 0.328 | 0.002 / 0.004 |

At the mandated 0.75 threshold on the calibrated score: P 0.33 / R 0.01 / F1 0.01, 57 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.015 | [-0.058, 0.013] |
| world_model - persistence_rollout | -0.015 | [-0.058, 0.013] |
| world_model - noised_persistence | 0.018 | [-0.001, 0.086] |
| world_model - isotropic_noise_persistence | 0.021 | [0.001, 0.084] |
| world_model - world_model_deterministic | -0.032 | [-0.118, 0.004] |
| world_model - ridge_two_lag | 0.020 | [-0.028, 0.120] |
| world_model_deterministic - persistence | 0.016 | [-0.000, 0.084] |
| world_model_deterministic - ridge_two_lag | 0.052 | [0.006, 0.198] |
| noised_persistence - persistence | -0.034 | [-0.128, 0.007] |
| world_model - lr_current_state | 0.020 | [-0.058, 0.179] |
| world_model - lr_flattened_history | 0.112 | [0.051, 0.277] |
| world_model - gbdt_current_state | -0.110 | [-0.172, -0.049] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.549, period2_persistence 0.025, ridge_two_lag 0.366; NIDRA vs ridge 0.289.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00568 | 0.210 | 0.225 | 0.187 | 0.10648 | 0.20 (2768) |
| 2 | 2 | 0.00565 | 0.164 | 0.215 | 0.189 | 0.10765 | 0.03 (2774) |
| 3 | 3 | 0.00566 | 0.111 | 0.202 | 0.196 | 0.10864 | 0.00 (2781) |
| 4 | 4 | 0.00566 | 0.072 | 0.173 | 0.201 | 0.10941 | 0.00 (2787) |
| 5 | 5 | 0.00568 | 0.048 | 0.131 | 0.187 | 0.10992 | 0.00 (2795) |
| 6 | 6 | 0.00571 | 0.038 | 0.091 | 0.198 | 0.11059 | 0.00 (2804) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 126 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 3 | 382 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | — |
| 5 | 635 | 0.001 | 0.002 | 0.001 | 0.002 | 0.001 | — |
| 10 | 1188 | 0.002 | 0.003 | 0.002 | 0.003 | 0.002 | — |
| 15 | 1523 | 0.003 | 0.004 | 0.003 | 0.004 | 0.003 | — |
| 30 | 2171 | 0.004 | 0.005 | 0.004 | 0.006 | 0.005 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.041), 2 consecutive windows: 131 episodes, 0 warned before onset, 6 alerted inside the episode; median lead None s, median latency 71.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313504340 | 216 | 0 | — | no (—) | yes (68.0) |
| 147.32.84.165@1313519940 | 12 | 14 | 0.006 | no (—) | no (—) |
| 147.32.84.165@1313522640 | 863 | 30 | 0.003 | no (—) | yes (0.0) |
| 147.32.84.165@1313586240 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313591040 | 164 | 30 | 0.003 | no (—) | yes (151.0) |
| 147.32.84.165@1313664420 | 0 | 8 | 0.005 | no (—) | no (—) |
| 147.32.84.165@1313665440 | 2 | 16 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313668380 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313669160 | 16 | 12 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313670780 | 2 | 10 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313675220 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313675820 | 0 | 9 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313676360 | 9 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313677800 | 1 | 14 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313678460 | 5 | 10 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313679420 | 2 | 10 | 0.006 | no (—) | no (—) |
| 147.32.84.191@1313586240 | 0 | 5 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313592180 | 149 | 30 | 0.000 | no (—) | yes (74.0) |
| 147.32.84.191@1313665560 | 0 | 17 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313666580 | 0 | 16 | 0.003 | no (—) | no (—) |
| 147.32.84.191@1313668380 | 0 | 29 | 0.003 | no (—) | no (—) |
| 147.32.84.191@1313669160 | 16 | 12 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313670720 | 3 | 9 | 0.006 | no (—) | no (—) |
| 147.32.84.191@1313675280 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313675820 | 0 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313676360 | 10 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.191@1313677800 | 0 | 14 | 0.005 | no (—) | no (—) |
| 147.32.84.191@1313678460 | 6 | 10 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313679420 | 2 | 9 | 0.008 | no (—) | no (—) |
| 147.32.84.192@1313586240 | 1 | 4 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313592480 | 139 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313665500 | 0 | 9 | 0.008 | no (—) | no (—) |
| 147.32.84.192@1313666580 | 0 | 17 | 0.003 | no (—) | no (—) |
| 147.32.84.192@1313668380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313669280 | 14 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313670720 | 3 | 9 | 0.006 | no (—) | no (—) |
| 147.32.84.192@1313675280 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313675820 | 0 | 8 | 0.006 | no (—) | no (—) |
| 147.32.84.192@1313676360 | 9 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.192@1313677800 | 0 | 14 | 0.005 | no (—) | no (—) |
| 147.32.84.192@1313678460 | 7 | 10 | 0.003 | no (—) | no (—) |
| 147.32.84.192@1313679480 | 1 | 10 | 0.009 | no (—) | no (—) |
| 147.32.84.193@1313586240 | 0 | 3 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313592600 | 140 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313665620 | 0 | 10 | 0.007 | no (—) | no (—) |
| 147.32.84.193@1313666580 | 0 | 15 | 0.004 | no (—) | no (—) |
| 147.32.84.193@1313668380 | 0 | 29 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313669100 | 18 | 11 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313670720 | 3 | 8 | 0.007 | no (—) | no (—) |
| 147.32.84.193@1313675340 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313675820 | 0 | 7 | 0.004 | no (—) | no (—) |
| 147.32.84.193@1313676360 | 9 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313677800 | 0 | 14 | 0.007 | no (—) | no (—) |
| 147.32.84.193@1313678460 | 7 | 10 | 0.004 | no (—) | no (—) |
| 147.32.84.193@1313679480 | 0 | 9 | 0.007 | no (—) | no (—) |
| 147.32.84.204@1313586240 | 0 | 1 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313592600 | 139 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313665500 | 0 | 7 | 0.006 | no (—) | no (—) |
| 147.32.84.204@1313666520 | 0 | 16 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313668380 | 0 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313669280 | 14 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313670720 | 3 | 9 | 0.006 | no (—) | no (—) |
| 147.32.84.204@1313675340 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313675820 | 0 | 7 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313676360 | 9 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313677800 | 0 | 14 | 0.007 | no (—) | no (—) |
| 147.32.84.204@1313678460 | 8 | 10 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313679420 | 2 | 7 | 0.007 | no (—) | no (—) |
| 147.32.84.205@1313586240 | 1 | 0 | — | no (—) | no (—) |
| 147.32.84.205@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313592600 | 139 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313665620 | 1 | 9 | 0.005 | no (—) | no (—) |
| 147.32.84.205@1313666640 | 0 | 16 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313668380 | 1 | 28 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313669160 | 15 | 12 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313670780 | 2 | 11 | 0.007 | no (—) | no (—) |
| 147.32.84.205@1313675460 | 6 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313676360 | 10 | 8 | 0.005 | no (—) | no (—) |
| 147.32.84.205@1313677800 | 0 | 14 | 0.007 | no (—) | no (—) |
| 147.32.84.205@1313678460 | 9 | 10 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313679480 | 1 | 7 | 0.006 | no (—) | no (—) |
| 147.32.84.206@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313592600 | 146 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313666460 | 1 | 15 | 0.004 | no (—) | no (—) |
| 147.32.84.206@1313668380 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313669100 | 16 | 11 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313670720 | 3 | 10 | 0.007 | no (—) | no (—) |
| 147.32.84.206@1313675460 | 6 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313676360 | 9 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.206@1313677800 | 1 | 14 | 0.006 | no (—) | no (—) |
| 147.32.84.206@1313678460 | 9 | 10 | 0.005 | no (—) | no (—) |
| 147.32.84.206@1313679480 | 0 | 6 | 0.007 | no (—) | no (—) |
| 147.32.84.207@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.207@1313592600 | 143 | 30 | 0.000 | no (—) | yes (56.0) |
| 147.32.84.207@1313666460 | 0 | 16 | 0.007 | no (—) | no (—) |
| 147.32.84.207@1313668380 | 5 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.207@1313669220 | 15 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313670720 | 3 | 9 | 0.007 | no (—) | no (—) |
| 147.32.84.207@1313675520 | 5 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313676360 | 9 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.207@1313677800 | 0 | 14 | 0.005 | no (—) | no (—) |
| 147.32.84.207@1313678460 | 9 | 10 | 0.004 | no (—) | no (—) |
| 147.32.84.207@1313679540 | 0 | 7 | 0.007 | no (—) | no (—) |
| 147.32.84.208@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313590380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313592600 | 6 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313593380 | 131 | 6 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313666400 | 0 | 15 | 0.005 | no (—) | no (—) |
| 147.32.84.208@1313668380 | 4 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313669100 | 17 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313670720 | 2 | 9 | 0.008 | no (—) | no (—) |
| 147.32.84.208@1313675520 | 5 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313676360 | 9 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313677800 | 20 | 14 | 0.006 | no (—) | no (—) |
| 147.32.84.208@1313679540 | 0 | 8 | 0.006 | no (—) | no (—) |
| 147.32.84.209@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313592600 | 7 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313593500 | 127 | 8 | 0.000 | no (—) | yes (89.0) |
| 147.32.84.209@1313666340 | 0 | 15 | 0.007 | no (—) | no (—) |
| 147.32.84.209@1313668380 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313669280 | 25 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313675520 | 5 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313676360 | 9 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.209@1313677800 | 0 | 14 | 0.006 | no (—) | no (—) |
| 147.32.84.209@1313678460 | 9 | 10 | 0.006 | no (—) | no (—) |
| 147.32.84.209@1313679540 | 0 | 7 | 0.007 | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_9:exfil` | neris | 1373 | 11 | 10 | 0.235 | 0.323 | 0.291 | 0.465 |
| `ctu_8:c2` | murlo | 1069 | 3 | 1¹ | 0.023 | 0.032 | 0.012 | 0.319 |
| `ctu_10:recon` | rbot | 488 | 59 | 10 | 0.001 | 0.002 | 0.001 | 0.222 |
| `ctu_10:exfil` | rbot | 307 | 38 | 10 | 0.006 | 0.012 | 0.003 | 0.539 |
| `ctu_10:c2` | rbot | 232 | 45 | 10 | 0.001 | 0.001 | 0.001 | 0.076 |
| `ctu_9:recon` | neris | 151 | 22 | 10 | 0.000 | 0.002 | 0.000 | 0.097 |
| `ctu_9:c2` | neris | 35 | 8 | 8 | 0.000 | 0.006 | 0.002 | 0.161 |
| `ctu_8:recon` | murlo | 31 | 3 | 1¹ | 0.015 | 0.008 | 0.007 | 0.359 |
| `ctu_8:exfil` | murlo | 2 | 2 | 1¹ | 0.000 | 0.002 | 0.000 | 0.148 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.443 | 0.688 | 0.9149 | 10 | 0.7192 | 0.921 | 1.28× | 0.8433 | carries timing signal |
| world_model | 0.418 | 0.688 | 0.8636 | 10 | 0.7192 | 0.918 | 1.28× | 0.8430 | carries timing signal |
| world_model_deterministic | 0.443 | 0.637 | 0.9482 | 10 | 0.7192 | 0.883 | 1.23× | 0.6784 | carries timing signal |
| noised_persistence | 0.389 | 0.662 | 0.9275 | 10 | 0.7192 | 0.916 | 1.27× | 0.8412 | carries timing signal |
| isotropic_noise_persistence | 0.395 | 0.675 | 0.9054 | 10 | 0.7192 | 0.920 | 1.28× | 0.8158 | carries timing signal |
| persistence | 0.457 | 0.749 | 0.9545 | 10 | 0.7192 | 0.814 | 1.13× | 0.5587 | carries timing signal |
| ridge_two_lag | 0.398 | 0.619 | 0.8978 | 10 | 0.7192 | 0.912 | 1.27× | 0.8230 | carries timing signal |
| oracle_true_future | 0.504 | 0.685 | 0.9524 | 10 | 0.7192 | 0.921 | 1.28× | 0.7797 | carries timing signal |
| lr_current_state | 0.473 | 0.809 | 0.9100 | 10 | 0.7192 | 0.912 | 1.27× | 0.8094 | carries timing signal |
| lr_flattened_history | 0.224 | 0.541 | 0.8931 | 10 | 0.7192 | 0.708 | 0.98× | 0.3537 | carries timing signal |
| gbdt_current_state | 0.601 | 0.834 | 0.9544 | 10 | 0.7192 | 0.894 | 1.24× | 0.7644 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00929 | 0.05766 | 0.00749 | 0.00743 | 10/10 |
| calibrated | 0.00735 | 0.04265 | 0.00749 | 0.00743 | 3/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 21045 scored rows, natural prevalence 0.00409, operating point `p_above_half|q=-|max`, threshold 0.041 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.307 | — | 0.978 | 1.00 / 0.13 / 0.23 | 0.00 | 0.318 | 0.000 / 0.000 |
| NIDRA (stochastic rollout) | 0.324 | [0.001, 0.522] | 0.979 | 0.03 / 0.98 / 0.06 | 1703.78 | 0.336 | 0.000 / 0.000 |
| NIDRA (deterministic rollout) | 0.339 | [0.003, 0.540] | 0.961 | 0.25 / 0.42 / 0.31 | 76.38 | 0.349 | 0.001 / 0.002 |
| persistence + learned noise (mean disabled) | 0.210 | [0.000, 0.330] | 0.948 | 0.03 / 0.92 / 0.06 | 1652.03 | 0.218 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.222 | — | 0.904 | 0.01 / 0.96 / 0.01 | 10622.25 | 0.231 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.206 | [0.000, 0.293] | 0.561 | 0.31 / 0.19 / 0.23 | 24.62 | 0.216 | 0.000 / 0.000 |
| ridge two-lag dynamics + risk head | 0.304 | [0.001, 0.638] | 0.962 | 0.13 / 0.82 / 0.22 | 328.29 | 0.315 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.232 | — | 0.918 | 0.10 / 0.20 / 0.13 | 104.61 | 0.241 | 0.000 / 0.000 |
| logistic regression on S_t | 0.045 | [0.000, 0.164] | 0.835 | 0.00 / 0.94 / 0.01 | 12416.22 | 0.046 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.016 | [0.000, 0.112] | 0.247 | 0.00 / 0.15 / 0.00 | 4601.87 | 0.016 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.299 | [0.001, 0.533] | 0.957 | 0.00 / 0.98 / 0.01 | 11985.12 | 0.312 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.01 / F1 0.01, 7 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.118 | [0.001, 0.231] |
| world_model - persistence_rollout | 0.118 | [0.001, 0.231] |
| world_model - noised_persistence | 0.114 | [0.000, 0.200] |
| world_model - isotropic_noise_persistence | 0.103 | [0.001, 0.172] |
| world_model - world_model_deterministic | -0.014 | [-0.044, 0.008] |
| world_model - ridge_two_lag | 0.021 | [-0.136, 0.075] |
| world_model_deterministic - persistence | 0.133 | [0.003, 0.251] |
| world_model_deterministic - ridge_two_lag | 0.035 | [-0.116, 0.078] |
| noised_persistence - persistence | 0.004 | [-0.007, 0.048] |
| world_model - lr_current_state | 0.279 | [0.000, 0.384] |
| world_model - lr_flattened_history | 0.309 | [0.001, 0.472] |
| world_model - gbdt_current_state | 0.025 | [-0.024, 0.056] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.612, period2_persistence 0.052, ridge_two_lag 0.468; NIDRA vs ridge 0.270.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00391 | 0.232 | 0.212 | 0.202 | 0.04546 | 0.19 (992) |
| 2 | 2 | 0.00391 | 0.268 | 0.275 | 0.217 | 0.04573 | 0.12 (992) |
| 3 | 3 | 0.00391 | 0.294 | 0.368 | 0.218 | 0.04604 | 0.03 (994) |
| 4 | 4 | 0.00391 | 0.278 | 0.418 | 0.220 | 0.04626 | 0.00 (994) |
| 5 | 5 | 0.00391 | 0.240 | 0.432 | 0.220 | 0.04634 | 0.00 (994) |
| 6 | 6 | 0.00392 | 0.211 | 0.431 | 0.217 | 0.04663 | 0.00 (996) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 3 | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 5 | 18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 10 | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 15 | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 30 | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.041), 2 consecutive windows: 6 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 46.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313428560 | 977 | 0 | — | no (—) | yes (46.0) |
| 147.32.84.165@1313752680 | 5 | 8 | 0.005 | no (—) | no (—) |
| 147.32.84.165@1313753400 | 9 | 7 | 0.009 | no (—) | no (—) |
| 147.32.84.191@1313753100 | 14 | 8 | 0.008 | no (—) | no (—) |
| 147.32.84.192@1313752500 | 15 | 0 | — | no (—) | no (—) |
| 147.32.84.192@1313754000 | 0 | 4 | 0.009 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_13:exfil` | virut | 752 | 1 | 1¹ | 0.381 | 0.287 | 0.272 | 0.566 |
| `ctu_13:recon` | virut | 150 | 1 | 1¹ | 0.018 | 0.009 | 0.002 | 0.579 |
| `ctu_12:c2` | nsis_ay | 63 | 5 | 3 | 0.003 | 0.001 | 0.000 | 0.516 |
| `ctu_13:c2` | virut | 47 | 1 | 1¹ | 0.005 | 0.003 | 0.000 | 0.532 |
| `ctu_12:recon` | nsis_ay | 18 | 4 | 2 | 0.000 | 0.001 | 0.000 | 0.386 |
| `ctu_12:exfil` | nsis_ay | 10 | 2 | 1¹ | 0.000 | 0.001 | 0.001 | 0.372 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.488 | 0.933 | 0.9959 | 3 | 0.9933 | 0.999 | 1.01× | 0.8977 | **host identity** |
| world_model | 0.508 | 0.938 | 0.9959 | 3 | 0.9933 | 0.999 | 1.01× | 0.9133 | **host identity** |
| world_model_deterministic | 0.531 | 0.920 | 0.9978 | 3 | 0.9933 | 0.997 | 1.00× | 0.6353 | **host identity** |
| noised_persistence | 0.331 | 0.856 | 0.9909 | 3 | 0.9933 | 0.999 | 1.01× | 0.8666 | **host identity** |
| isotropic_noise_persistence | 0.358 | 0.826 | 0.9650 | 3 | 0.9933 | 0.999 | 1.01× | 0.9217 | **host identity** |
| persistence | 0.314 | 0.704 | 0.9772 | 3 | 0.9933 | 0.994 | 1.00× | 0.4626 | **host identity** |
| ridge_two_lag | 0.554 | 0.922 | 0.9961 | 3 | 0.9933 | 0.999 | 1.01× | 0.9129 | **host identity** |
| oracle_true_future | 0.388 | 0.876 | 0.9723 | 3 | 0.9933 | 0.999 | 1.01× | 0.9045 | **host identity** |
| lr_current_state | 0.162 | 0.827 | 0.9238 | 3 | 0.9933 | 0.998 | 1.00× | 0.7556 | **host identity** |
| lr_flattened_history | 0.072 | 0.389 | 0.9765 | 3 | 0.9933 | 0.990 | 1.00× | 0.2411 | **host identity** |
| gbdt_current_state | 0.581 | 0.958 | 0.9988 | 3 | 0.9933 | 0.998 | 1.00× | 0.8283 | **host identity** |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00399 | 0.05365 | 0.00409 | 0.00408 | 10/10 |
| calibrated | 0.00391 | 0.04614 | 0.00409 | 0.00408 | 4/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
