_val: no benchmark.json at experiments/runs/xeval_cic2ctu_state/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 25099 scored rows, natural prevalence 0.00749, operating point `mean|q=-|integrated`, threshold 0.502 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.010 | — | 0.604 | 0.01 / 0.02 / 0.01 | 207.97 | 0.009 | 0.001 / 0.003 |
| NIDRA (stochastic rollout) | 0.009 | [0.005, 0.017] | 0.568 | 0.01 / 0.18 / 0.02 | 2211.98 | 0.008 | 0.001 / 0.003 |
| NIDRA (deterministic rollout) | 0.005 | [0.004, 0.009] | 0.297 | 0.01 / 0.04 / 0.02 | 415.68 | 0.004 | 0.001 / 0.003 |
| persistence + learned noise (mean disabled) | 0.014 | [0.007, 0.024] | 0.712 | 0.01 / 0.22 / 0.02 | 2269.05 | 0.015 | 0.001 / 0.003 |
| persistence + isotropic noise | 0.008 | — | 0.472 | 0.01 / 0.24 / 0.02 | 2851.06 | 0.009 | 0.001 / 0.002 |
| persistence (risk head on S_t) | 0.007 | [0.003, 0.018] | 0.391 | 0.01 / 0.03 / 0.01 | 382.51 | 0.008 | 0.001 / 0.003 |
| ridge two-lag dynamics + risk head | 0.011 | [0.005, 0.022] | 0.523 | 0.01 / 0.14 / 0.02 | 1303.26 | 0.011 | 0.001 / 0.003 |
| oracle: risk head on the true future | 0.011 | — | 0.506 | 0.02 / 0.16 / 0.04 | 841.37 | 0.005 | 0.005 / 0.004 |
| logistic regression on S_t | 0.005 | [0.004, 0.008] | 0.226 | 0.01 / 0.03 / 0.01 | 824.96 | 0.004 | 0.002 / 0.004 |
| logistic regression on the L-window history | 0.006 | [0.005, 0.013] | 0.238 | 0.03 / 0.08 / 0.04 | 342.29 | 0.004 | 0.005 / 0.007 |
| gradient-boosted trees on S_t | 0.007 | [0.005, 0.010] | 0.282 | 0.01 / 0.02 / 0.01 | 333.51 | 0.006 | 0.002 / 0.004 |

At the mandated 0.75 threshold on the calibrated score: P 0.01 / R 0.01 / F1 0.01, 5126 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.002 | [-0.003, 0.005] |
| world_model - persistence_rollout | 0.002 | [-0.002, 0.007] |
| world_model - noised_persistence | -0.005 | [-0.009, -0.001] |
| world_model - isotropic_noise_persistence | 0.001 | [-0.002, 0.004] |
| world_model - world_model_deterministic | 0.004 | [0.002, 0.008] |
| world_model - ridge_two_lag | -0.001 | [-0.007, 0.003] |
| world_model_deterministic - persistence | -0.002 | [-0.010, 0.001] |
| world_model_deterministic - ridge_two_lag | -0.005 | [-0.014, -0.001] |
| noised_persistence - persistence | 0.007 | [0.002, 0.011] |
| world_model - lr_current_state | 0.004 | [0.001, 0.009] |
| world_model - lr_flattened_history | 0.003 | [-0.003, 0.008] |
| world_model - gbdt_current_state | 0.003 | [-0.001, 0.008] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.395, period2_persistence 0.025, ridge_two_lag -3.759; NIDRA vs ridge 0.873.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00568 | 0.008 | 0.005 | 0.008 | 0.13403 | 0.00 (2768) |
| 2 | 2 | 0.00565 | 0.008 | 0.005 | 0.008 | 0.12806 | 0.00 (2774) |
| 3 | 3 | 0.00566 | 0.008 | 0.004 | 0.008 | 0.12057 | 0.00 (2781) |
| 4 | 4 | 0.00566 | 0.007 | 0.003 | 0.009 | 0.11578 | 0.00 (2787) |
| 5 | 5 | 0.00568 | 0.006 | 0.003 | 0.009 | 0.11274 | 0.00 (2795) |
| 6 | 6 | 0.00571 | 0.005 | 0.003 | 0.012 | 0.11189 | 0.00 (2804) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 126 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | — |
| 3 | 382 | 0.001 | 0.001 | 0.001 | 0.001 | 0.003 | — |
| 5 | 635 | 0.001 | 0.001 | 0.001 | 0.002 | 0.005 | — |
| 10 | 1188 | 0.002 | 0.003 | 0.002 | 0.003 | 0.007 | — |
| 15 | 1523 | 0.003 | 0.003 | 0.003 | 0.004 | 0.007 | — |
| 30 | 2171 | 0.004 | 0.005 | 0.005 | 0.006 | 0.008 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.502), 2 consecutive windows: 131 episodes, 0 warned before onset, 16 alerted inside the episode; median lead None s, median latency 10.5 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313504340 | 216 | 0 | — | no (—) | no (—) |
| 147.32.84.165@1313519940 | 12 | 14 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313522640 | 863 | 30 | 0.009 | no (—) | no (—) |
| 147.32.84.165@1313586240 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313591040 | 164 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313664420 | 0 | 8 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313665440 | 2 | 16 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313668380 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.165@1313669160 | 16 | 12 | 0.001 | no (—) | yes (12.0) |
| 147.32.84.165@1313670780 | 2 | 10 | 0.071 | no (—) | no (—) |
| 147.32.84.165@1313675220 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313675820 | 0 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313677800 | 1 | 14 | 0.002 | no (—) | no (—) |
| 147.32.84.165@1313678460 | 5 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313679420 | 2 | 10 | 0.101 | no (—) | no (—) |
| 147.32.84.191@1313586240 | 0 | 5 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313592180 | 149 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313665560 | 0 | 17 | 0.070 | no (—) | no (—) |
| 147.32.84.191@1313666580 | 0 | 16 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313668380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313669160 | 16 | 12 | 0.000 | no (—) | yes (12.0) |
| 147.32.84.191@1313670720 | 3 | 9 | 0.015 | no (—) | no (—) |
| 147.32.84.191@1313675280 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.191@1313675820 | 0 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313676360 | 10 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313677800 | 0 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313678460 | 6 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313679420 | 2 | 9 | 0.035 | no (—) | no (—) |
| 147.32.84.192@1313586240 | 1 | 4 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313592480 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313665500 | 0 | 9 | 0.032 | no (—) | no (—) |
| 147.32.84.192@1313666580 | 0 | 17 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313668380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313669280 | 14 | 14 | 0.000 | no (—) | yes (10.0) |
| 147.32.84.192@1313670720 | 3 | 9 | 0.037 | no (—) | no (—) |
| 147.32.84.192@1313675280 | 0 | 30 | 0.008 | no (—) | no (—) |
| 147.32.84.192@1313675820 | 0 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313677800 | 0 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313678460 | 7 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313679480 | 1 | 10 | 0.084 | no (—) | no (—) |
| 147.32.84.193@1313586240 | 0 | 3 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313592600 | 140 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313665620 | 0 | 10 | 0.040 | no (—) | no (—) |
| 147.32.84.193@1313666580 | 0 | 15 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313668380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313669100 | 18 | 11 | 0.000 | no (—) | yes (13.0) |
| 147.32.84.193@1313670720 | 3 | 8 | 0.030 | no (—) | no (—) |
| 147.32.84.193@1313675340 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313675820 | 0 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313677800 | 0 | 14 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313678460 | 7 | 10 | 0.000 | no (—) | yes (2.0) |
| 147.32.84.193@1313679480 | 0 | 9 | 0.044 | no (—) | no (—) |
| 147.32.84.204@1313586240 | 0 | 1 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313592600 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313665500 | 0 | 7 | 0.042 | no (—) | no (—) |
| 147.32.84.204@1313666520 | 0 | 16 | 0.012 | no (—) | no (—) |
| 147.32.84.204@1313668380 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313669280 | 14 | 14 | 0.000 | no (—) | yes (10.0) |
| 147.32.84.204@1313670720 | 3 | 9 | 0.015 | no (—) | no (—) |
| 147.32.84.204@1313675340 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313675820 | 0 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313677800 | 0 | 14 | 0.002 | no (—) | no (—) |
| 147.32.84.204@1313678460 | 8 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313679420 | 2 | 7 | 0.031 | no (—) | no (—) |
| 147.32.84.205@1313586240 | 1 | 0 | — | no (—) | no (—) |
| 147.32.84.205@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313592600 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313665620 | 1 | 9 | 0.043 | no (—) | no (—) |
| 147.32.84.205@1313666640 | 0 | 16 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313668380 | 1 | 28 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313669160 | 15 | 12 | 0.001 | no (—) | yes (12.0) |
| 147.32.84.205@1313670780 | 2 | 11 | 0.042 | no (—) | no (—) |
| 147.32.84.205@1313675460 | 6 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.205@1313676360 | 10 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313677800 | 0 | 14 | 0.002 | no (—) | no (—) |
| 147.32.84.205@1313678460 | 9 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313679480 | 1 | 7 | 0.027 | no (—) | no (—) |
| 147.32.84.206@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313592600 | 146 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.206@1313666460 | 1 | 15 | 0.041 | no (—) | no (—) |
| 147.32.84.206@1313668380 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313669100 | 16 | 11 | 0.002 | no (—) | yes (13.0) |
| 147.32.84.206@1313670720 | 3 | 10 | 0.027 | no (—) | no (—) |
| 147.32.84.206@1313675460 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313677800 | 1 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313678460 | 9 | 10 | 0.001 | no (—) | yes (7.0) |
| 147.32.84.206@1313679480 | 0 | 6 | 0.096 | no (—) | no (—) |
| 147.32.84.207@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313592600 | 143 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313666460 | 0 | 16 | 0.013 | no (—) | no (—) |
| 147.32.84.207@1313668380 | 5 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313669220 | 15 | 9 | 0.000 | no (—) | yes (11.0) |
| 147.32.84.207@1313670720 | 3 | 9 | 0.096 | no (—) | yes (1.0) |
| 147.32.84.207@1313675520 | 5 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.207@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313677800 | 0 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313678460 | 9 | 10 | 0.001 | no (—) | yes (2.0) |
| 147.32.84.207@1313679540 | 0 | 7 | 0.028 | no (—) | no (—) |
| 147.32.84.208@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313590380 | 0 | 29 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313592600 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313593380 | 131 | 6 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313666400 | 0 | 15 | 0.028 | no (—) | no (—) |
| 147.32.84.208@1313668380 | 4 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313669100 | 17 | 7 | 0.000 | no (—) | yes (13.0) |
| 147.32.84.208@1313670720 | 2 | 9 | 0.020 | no (—) | no (—) |
| 147.32.84.208@1313675520 | 5 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313677800 | 20 | 14 | 0.001 | no (—) | yes (13.0) |
| 147.32.84.208@1313679540 | 0 | 8 | 0.042 | no (—) | no (—) |
| 147.32.84.209@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313592600 | 7 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.209@1313593500 | 127 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313666340 | 0 | 15 | 0.042 | no (—) | no (—) |
| 147.32.84.209@1313668380 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313669280 | 25 | 14 | 0.000 | no (—) | yes (10.0) |
| 147.32.84.209@1313675520 | 5 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.209@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313677800 | 0 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313678460 | 9 | 10 | 0.001 | no (—) | yes (2.0) |
| 147.32.84.209@1313679540 | 0 | 7 | 0.040 | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_9:exfil` | neris | 1373 | 11 | 10 | 0.003 | 0.001 | 0.001 | 0.208 |
| `ctu_8:c2` | murlo | 1069 | 3 | 1¹ | 0.004 | 0.005 | 0.007 | 0.001 |
| `ctu_10:recon` | rbot | 488 | 59 | 10 | 0.001 | 0.009 | 0.001 | 0.171 |
| `ctu_10:exfil` | rbot | 307 | 38 | 10 | 0.005 | 0.012 | 0.004 | 0.458 |
| `ctu_10:c2` | rbot | 232 | 45 | 10 | 0.001 | 0.001 | 0.000 | 0.003 |
| `ctu_9:recon` | neris | 151 | 22 | 10 | 0.000 | 0.000 | 0.000 | 0.093 |
| `ctu_9:c2` | neris | 35 | 8 | 8 | 0.000 | 0.000 | 0.000 | 0.108 |
| `ctu_8:recon` | murlo | 31 | 3 | 1¹ | 0.000 | 0.000 | 0.000 | 0.329 |
| `ctu_8:exfil` | murlo | 2 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.123 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.119 | 0.429 | 0.5872 | 10 | 0.7192 | 0.824 | 1.15× | 0.6487 | carries timing signal |
| world_model | 0.121 | 0.434 | 0.4024 | 10 | 0.7192 | 0.796 | 1.11× | 0.5895 | carries timing signal |
| world_model_deterministic | 0.102 | 0.312 | 0.5182 | 10 | 0.7192 | 0.622 | 0.87× | 0.2548 | carries timing signal |
| noised_persistence | 0.121 | 0.444 | 0.3005 | 10 | 0.7192 | 0.924 | 1.28× | 0.8265 | carries timing signal |
| isotropic_noise_persistence | 0.103 | 0.323 | 0.1828 | 10 | 0.7192 | 0.804 | 1.12× | 0.5353 | carries timing signal |
| persistence | 0.101 | 0.288 | 0.5064 | 10 | 0.7192 | 0.750 | 1.04× | 0.4397 | carries timing signal |
| ridge_two_lag | 0.127 | 0.419 | 0.4045 | 10 | 0.7192 | 0.827 | 1.15× | 0.5764 | carries timing signal |
| oracle_true_future | 0.135 | 0.422 | 0.5788 | 10 | 0.7192 | 0.875 | 1.22× | 0.6195 | carries timing signal |
| lr_current_state | 0.118 | 0.359 | 0.5680 | 10 | 0.7192 | 0.575 | 0.80× | 0.1674 | carries timing signal |
| lr_flattened_history | 0.118 | 0.308 | 0.8045 | 10 | 0.7192 | 0.623 | 0.87× | 0.2116 | carries timing signal |
| gbdt_current_state | 0.159 | 0.525 | 0.5059 | 10 | 0.7192 | 0.614 | 0.85× | 0.1644 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.13315 | 0.30318 | 0.00749 | 0.00743 | 10/10 |
| calibrated | 0.01901 | 0.05553 | 0.00749 | 0.00743 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 21045 scored rows, natural prevalence 0.00409, operating point `mean|q=-|integrated`, threshold 0.502 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.035 | — | 0.863 | 0.00 / 0.00 / 0.00 | 186.18 | 0.035 | 0.000 / 0.001 |
| NIDRA (stochastic rollout) | 0.030 | [0.000, 0.110] | 0.838 | 0.03 / 0.69 / 0.05 | 1390.19 | 0.029 | 0.000 / 0.001 |
| NIDRA (deterministic rollout) | 0.026 | [0.000, 0.095] | 0.677 | 0.03 / 0.17 / 0.05 | 329.57 | 0.026 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.033 | [0.000, 0.124] | 0.903 | 0.03 / 0.71 / 0.05 | 1609.58 | 0.033 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.030 | — | 0.757 | 0.02 / 0.71 / 0.04 | 1977.79 | 0.031 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.028 | [0.000, 0.108] | 0.711 | 0.04 / 0.21 / 0.07 | 305.34 | 0.029 | 0.000 / 0.000 |
| ridge two-lag dynamics + risk head | 0.016 | [0.000, 0.059] | 0.492 | 0.02 / 0.31 / 0.04 | 879.44 | 0.017 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.034 | — | 0.736 | 0.06 / 0.69 / 0.11 | 614.97 | 0.035 | 0.000 / 0.000 |
| logistic regression on S_t | 0.005 | [0.000, 0.016] | 0.300 | 0.02 / 0.18 / 0.03 | 600.07 | 0.005 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.002 | [0.000, 0.007] | 0.214 | 0.00 / 0.00 / 0.00 | 285.04 | 0.002 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.004 | [0.000, 0.011] | 0.184 | 0.00 / 0.00 / 0.00 | 176.89 | 0.004 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 2776 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.002 | [-0.003, 0.005] |
| world_model - persistence_rollout | 0.002 | [-0.003, 0.005] |
| world_model - noised_persistence | -0.003 | [-0.013, 0.001] |
| world_model - isotropic_noise_persistence | -0.000 | [-0.008, 0.002] |
| world_model - world_model_deterministic | 0.003 | [0.000, 0.012] |
| world_model - ridge_two_lag | 0.013 | [-0.000, 0.048] |
| world_model_deterministic - persistence | -0.002 | [-0.009, 0.000] |
| world_model_deterministic - ridge_two_lag | 0.010 | [-0.001, 0.045] |
| noised_persistence - persistence | 0.005 | [0.000, 0.015] |
| world_model - lr_current_state | 0.025 | [0.000, 0.091] |
| world_model - lr_flattened_history | 0.027 | [0.000, 0.101] |
| world_model - gbdt_current_state | 0.026 | [-0.000, 0.097] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.454, period2_persistence 0.051, ridge_two_lag -3.334; NIDRA vs ridge 0.874.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00391 | 0.036 | 0.033 | 0.028 | 0.08026 | 0.00 (992) |
| 2 | 2 | 0.00391 | 0.030 | 0.022 | 0.028 | 0.07286 | 0.00 (992) |
| 3 | 3 | 0.00391 | 0.025 | 0.008 | 0.029 | 0.06028 | 0.00 (994) |
| 4 | 4 | 0.00391 | 0.019 | 0.002 | 0.028 | 0.05256 | 0.00 (994) |
| 5 | 5 | 0.00391 | 0.012 | 0.002 | 0.029 | 0.04812 | 0.00 (994) |
| 6 | 6 | 0.00392 | 0.007 | 0.002 | 0.029 | 0.04744 | 0.00 (996) |

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
**holdout** — per episode at the selected threshold (0.502), 2 consecutive windows: 6 episodes, 0 warned before onset, 0 alerted inside the episode; median lead None s, median latency None min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313428560 | 977 | 0 | — | no (—) | no (—) |
| 147.32.84.165@1313752680 | 5 | 8 | 0.018 | no (—) | no (—) |
| 147.32.84.165@1313753400 | 9 | 7 | 0.062 | no (—) | no (—) |
| 147.32.84.191@1313753100 | 14 | 8 | 0.025 | no (—) | no (—) |
| 147.32.84.192@1313752500 | 15 | 0 | — | no (—) | no (—) |
| 147.32.84.192@1313754000 | 0 | 4 | 0.086 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_13:exfil` | virut | 752 | 1 | 1¹ | 0.022 | 0.025 | 0.022 | 0.426 |
| `ctu_13:recon` | virut | 150 | 1 | 1¹ | 0.006 | 0.010 | 0.006 | 0.454 |
| `ctu_12:c2` | nsis_ay | 63 | 5 | 3 | 0.001 | 0.000 | 0.000 | 0.432 |
| `ctu_13:c2` | virut | 47 | 1 | 1¹ | 0.002 | 0.003 | 0.002 | 0.427 |
| `ctu_12:recon` | nsis_ay | 18 | 4 | 2 | 0.000 | 0.000 | 0.000 | 0.284 |
| `ctu_12:exfil` | nsis_ay | 10 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.261 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.121 | 0.748 | 0.8980 | 3 | 0.9933 | 0.998 | 1.00× | 0.7154 | carries timing signal |
| world_model | 0.109 | 0.733 | 0.9096 | 3 | 0.9933 | 0.997 | 1.00× | 0.6696 | **host identity** |
| world_model_deterministic | 0.097 | 0.636 | 0.9038 | 3 | 0.9933 | 0.997 | 1.00× | 0.6830 | **host identity** |
| noised_persistence | 0.105 | 0.728 | 0.8700 | 3 | 0.9933 | 0.999 | 1.01× | 0.8424 | carries timing signal |
| isotropic_noise_persistence | 0.102 | 0.654 | 0.8449 | 3 | 0.9933 | 0.999 | 1.01× | 0.8169 | carries timing signal |
| persistence | 0.096 | 0.620 | 0.8845 | 3 | 0.9933 | 0.998 | 1.00× | 0.7446 | carries timing signal |
| ridge_two_lag | 0.070 | 0.439 | 0.7026 | 3 | 0.9933 | 0.995 | 1.00× | 0.4758 | carries timing signal |
| oracle_true_future | 0.127 | 0.679 | 0.8935 | 3 | 0.9933 | 0.998 | 1.01× | 0.7729 | carries timing signal |
| lr_current_state | 0.053 | 0.481 | 0.8955 | 3 | 0.9933 | 0.993 | 1.00× | 0.4412 | carries timing signal |
| lr_flattened_history | 0.040 | 0.427 | 0.5450 | 3 | 0.9933 | 0.998 | 1.01× | 0.7714 | carries timing signal |
| gbdt_current_state | 0.074 | 0.597 | 0.2301 | 3 | 0.9933 | 0.988 | 0.99× | 0.3007 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.11526 | 0.28801 | 0.00409 | 0.00408 | 10/10 |
| calibrated | 0.01571 | 0.05907 | 0.00409 | 0.00408 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
