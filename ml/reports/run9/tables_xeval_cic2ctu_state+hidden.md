_val: no benchmark.json at experiments/runs/xeval_cic2ctu_state+hidden/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 25099 scored rows, natural prevalence 0.00749, operating point `mean|q=-|integrated`, threshold 0.882 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.009 | [0.005, 0.018] | 0.468 | 0.01 / 0.05 / 0.02 | 477.96 | 0.007 | 0.002 / 0.004 |
| NIDRA (stochastic rollout) | 0.008 | [0.005, 0.013] | 0.423 | 0.01 / 0.08 / 0.02 | 900.98 | 0.006 | 0.002 / 0.004 |
| NIDRA (deterministic rollout) | 0.007 | [0.005, 0.010] | 0.330 | 0.01 / 0.07 / 0.02 | 813.34 | 0.005 | 0.002 / 0.004 |
| persistence + learned noise (mean disabled) | 0.010 | [0.005, 0.020] | 0.622 | 0.01 / 0.07 / 0.02 | 951.48 | 0.009 | 0.001 / 0.003 |
| persistence + isotropic noise | 0.007 | — | 0.396 | 0.01 / 0.08 / 0.02 | 1100.68 | 0.005 | 0.001 / 0.003 |
| persistence (risk head on S_t) | 0.008 | [0.004, 0.018] | 0.443 | 0.00 / 0.02 / 0.01 | 400.75 | 0.006 | 0.002 / 0.004 |
| oracle: risk head on the true future | 0.012 | — | 0.511 | 0.02 / 0.08 / 0.03 | 649.01 | 0.006 | 0.006 / 0.006 |
| logistic regression on S_t | 0.005 | [0.004, 0.008] | 0.226 | 0.01 / 0.03 / 0.01 | 621.11 | 0.004 | 0.002 / 0.004 |
| logistic regression on the L-window history | 0.006 | [0.005, 0.013] | 0.238 | 0.03 / 0.06 / 0.04 | 274.40 | 0.004 | 0.005 / 0.007 |
| gradient-boosted trees on S_t | 0.007 | [0.005, 0.010] | 0.282 | 0.02 / 0.02 / 0.02 | 124.49 | 0.006 | 0.002 / 0.004 |

At the mandated 0.75 threshold on the calibrated score: P 0.01 / R 0.06 / F1 0.02, 16209 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.001 | [-0.004, 0.001] |
| world_model - persistence_rollout | -0.000 | [-0.003, 0.002] |
| world_model - noised_persistence | -0.003 | [-0.006, -0.000] |
| world_model - isotropic_noise_persistence | 0.001 | [-0.001, 0.001] |
| world_model - world_model_deterministic | 0.001 | [-0.000, 0.003] |
| world_model_deterministic - persistence | -0.002 | [-0.007, 0.001] |
| noised_persistence - persistence | 0.002 | [-0.000, 0.003] |
| world_model - lr_current_state | 0.003 | [0.000, 0.006] |
| world_model - lr_flattened_history | 0.001 | [-0.005, 0.005] |
| world_model - gbdt_current_state | 0.001 | [-0.001, 0.004] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.395, period2_persistence 0.025, ridge_two_lag -3.759; NIDRA vs ridge 0.873.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00568 | 0.006 | 0.006 | 0.006 | 0.17117 | 0.00 (2768) |
| 2 | 2 | 0.00565 | 0.006 | 0.004 | 0.006 | 0.16816 | 0.00 (2774) |
| 3 | 3 | 0.00566 | 0.005 | 0.004 | 0.006 | 0.16234 | 0.00 (2781) |
| 4 | 4 | 0.00566 | 0.005 | 0.004 | 0.006 | 0.15580 | 0.00 (2787) |
| 5 | 5 | 0.00568 | 0.004 | 0.003 | 0.006 | 0.14904 | 0.00 (2795) |
| 6 | 6 | 0.00571 | 0.004 | 0.003 | 0.006 | 0.14222 | 0.00 (2804) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 126 | 0.000 | 0.000 | — | 0.000 | 0.001 | — |
| 3 | 382 | 0.001 | 0.001 | — | 0.001 | 0.003 | — |
| 5 | 635 | 0.002 | 0.002 | — | 0.002 | 0.005 | — |
| 10 | 1188 | 0.003 | 0.004 | — | 0.003 | 0.007 | — |
| 15 | 1523 | 0.004 | 0.004 | — | 0.004 | 0.007 | — |
| 30 | 2171 | 0.004 | 0.005 | — | 0.006 | 0.008 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.882), 2 consecutive windows: 131 episodes, 5 warned before onset, 37 alerted inside the episode; median lead 540.0 s, median latency 2.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313504340 | 216 | 0 | — | no (—) | no (—) |
| 147.32.84.165@1313519940 | 12 | 14 | 0.012 | no (—) | no (—) |
| 147.32.84.165@1313522640 | 863 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313586240 | 0 | 30 | 0.006 | no (—) | no (—) |
| 147.32.84.165@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313591040 | 164 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313664420 | 0 | 8 | 0.035 | no (—) | no (—) |
| 147.32.84.165@1313665440 | 2 | 16 | 0.019 | no (—) | no (—) |
| 147.32.84.165@1313668380 | 0 | 30 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313669160 | 16 | 12 | 0.036 | no (—) | yes (12.0) |
| 147.32.84.165@1313670780 | 2 | 10 | 0.658 | no (—) | yes (0.0) |
| 147.32.84.165@1313675220 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313675820 | 0 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313676360 | 9 | 8 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313677800 | 1 | 14 | 0.015 | no (—) | no (—) |
| 147.32.84.165@1313678460 | 5 | 10 | 0.041 | no (—) | yes (2.0) |
| 147.32.84.165@1313679420 | 2 | 10 | 0.825 | no (—) | no (—) |
| 147.32.84.191@1313586240 | 0 | 5 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313592180 | 149 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313665560 | 0 | 17 | 0.023 | no (—) | no (—) |
| 147.32.84.191@1313666580 | 0 | 16 | 0.016 | no (—) | no (—) |
| 147.32.84.191@1313668380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313669160 | 16 | 12 | 0.055 | no (—) | yes (12.0) |
| 147.32.84.191@1313670720 | 3 | 9 | 0.999 | yes (540) | yes (1.0) |
| 147.32.84.191@1313675280 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313675820 | 0 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313676360 | 10 | 8 | 0.010 | no (—) | yes (3.0) |
| 147.32.84.191@1313677800 | 0 | 14 | 0.020 | no (—) | no (—) |
| 147.32.84.191@1313678460 | 6 | 10 | 0.036 | no (—) | yes (2.0) |
| 147.32.84.191@1313679420 | 2 | 9 | 0.837 | no (—) | no (—) |
| 147.32.84.192@1313586240 | 1 | 4 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313592480 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313665500 | 0 | 9 | 0.055 | no (—) | no (—) |
| 147.32.84.192@1313666580 | 0 | 17 | 0.200 | no (—) | no (—) |
| 147.32.84.192@1313668380 | 0 | 29 | 0.003 | no (—) | no (—) |
| 147.32.84.192@1313669280 | 14 | 14 | 0.016 | no (—) | yes (10.0) |
| 147.32.84.192@1313670720 | 3 | 9 | 0.576 | no (—) | yes (1.0) |
| 147.32.84.192@1313675280 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313675820 | 0 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313676360 | 9 | 8 | 0.044 | no (—) | yes (3.0) |
| 147.32.84.192@1313677800 | 0 | 14 | 0.037 | no (—) | no (—) |
| 147.32.84.192@1313678460 | 7 | 10 | 0.080 | no (—) | yes (2.0) |
| 147.32.84.192@1313679480 | 1 | 10 | 0.947 | no (—) | no (—) |
| 147.32.84.193@1313586240 | 0 | 3 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313592600 | 140 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313665620 | 0 | 10 | 0.030 | no (—) | no (—) |
| 147.32.84.193@1313666580 | 0 | 15 | 0.009 | no (—) | no (—) |
| 147.32.84.193@1313668380 | 0 | 29 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313669100 | 18 | 11 | 0.029 | no (—) | yes (13.0) |
| 147.32.84.193@1313670720 | 3 | 8 | 0.414 | no (—) | yes (1.0) |
| 147.32.84.193@1313675340 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313675820 | 0 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313676360 | 9 | 8 | 0.022 | no (—) | yes (3.0) |
| 147.32.84.193@1313677800 | 0 | 14 | 0.017 | no (—) | no (—) |
| 147.32.84.193@1313678460 | 7 | 10 | 0.060 | no (—) | yes (2.0) |
| 147.32.84.193@1313679480 | 0 | 9 | 1.000 | yes (540) | no (—) |
| 147.32.84.204@1313586240 | 0 | 1 | 0.002 | no (—) | no (—) |
| 147.32.84.204@1313588580 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313592600 | 139 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.204@1313665500 | 0 | 7 | 0.026 | no (—) | no (—) |
| 147.32.84.204@1313666520 | 0 | 16 | 0.006 | no (—) | no (—) |
| 147.32.84.204@1313668380 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313669280 | 14 | 14 | 0.126 | no (—) | yes (10.0) |
| 147.32.84.204@1313670720 | 3 | 9 | 0.952 | no (—) | yes (1.0) |
| 147.32.84.204@1313675340 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313675820 | 0 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313676360 | 9 | 8 | 0.006 | no (—) | no (—) |
| 147.32.84.204@1313677800 | 0 | 14 | 0.013 | no (—) | no (—) |
| 147.32.84.204@1313678460 | 8 | 10 | 0.058 | no (—) | yes (2.0) |
| 147.32.84.204@1313679420 | 2 | 7 | 0.893 | no (—) | no (—) |
| 147.32.84.205@1313586240 | 1 | 0 | — | no (—) | no (—) |
| 147.32.84.205@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313592600 | 139 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313665620 | 1 | 9 | 0.058 | no (—) | no (—) |
| 147.32.84.205@1313666640 | 0 | 16 | 0.043 | no (—) | no (—) |
| 147.32.84.205@1313668380 | 1 | 28 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313669160 | 15 | 12 | 0.123 | no (—) | yes (1.0) |
| 147.32.84.205@1313670780 | 2 | 11 | 0.782 | no (—) | yes (0.0) |
| 147.32.84.205@1313675460 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313676360 | 10 | 8 | 0.004 | no (—) | yes (3.0) |
| 147.32.84.205@1313677800 | 0 | 14 | 0.026 | no (—) | no (—) |
| 147.32.84.205@1313678460 | 9 | 10 | 0.015 | no (—) | yes (2.0) |
| 147.32.84.205@1313679480 | 1 | 7 | 0.427 | no (—) | no (—) |
| 147.32.84.206@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.206@1313592600 | 146 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313666460 | 1 | 15 | 0.015 | no (—) | no (—) |
| 147.32.84.206@1313668380 | 0 | 30 | 0.016 | no (—) | no (—) |
| 147.32.84.206@1313669100 | 16 | 11 | 0.059 | no (—) | yes (13.0) |
| 147.32.84.206@1313670720 | 3 | 10 | 0.773 | no (—) | yes (1.0) |
| 147.32.84.206@1313675460 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313676360 | 9 | 8 | 0.076 | no (—) | yes (3.0) |
| 147.32.84.206@1313677800 | 1 | 14 | 0.025 | no (—) | no (—) |
| 147.32.84.206@1313678460 | 9 | 10 | 0.044 | no (—) | yes (2.0) |
| 147.32.84.206@1313679480 | 0 | 6 | 0.974 | no (—) | no (—) |
| 147.32.84.207@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313592600 | 143 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313666460 | 0 | 16 | 0.013 | no (—) | no (—) |
| 147.32.84.207@1313668380 | 5 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.207@1313669220 | 15 | 9 | 0.021 | no (—) | yes (11.0) |
| 147.32.84.207@1313670720 | 3 | 9 | 0.991 | yes (540) | yes (0.0) |
| 147.32.84.207@1313675520 | 5 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.207@1313676360 | 9 | 8 | 0.011 | no (—) | yes (3.0) |
| 147.32.84.207@1313677800 | 0 | 14 | 0.011 | no (—) | no (—) |
| 147.32.84.207@1313678460 | 9 | 10 | 0.036 | no (—) | yes (2.0) |
| 147.32.84.207@1313679540 | 0 | 7 | 0.997 | yes (480) | no (—) |
| 147.32.84.208@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313590380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313592600 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313593380 | 131 | 6 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313666400 | 0 | 15 | 0.008 | no (—) | no (—) |
| 147.32.84.208@1313668380 | 4 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.208@1313669100 | 17 | 7 | 0.013 | no (—) | yes (13.0) |
| 147.32.84.208@1313670720 | 2 | 9 | 0.826 | no (—) | yes (0.0) |
| 147.32.84.208@1313675520 | 5 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313676360 | 9 | 8 | 0.020 | no (—) | yes (3.0) |
| 147.32.84.208@1313677800 | 20 | 14 | 0.019 | no (—) | yes (13.0) |
| 147.32.84.208@1313679540 | 0 | 8 | 0.501 | no (—) | no (—) |
| 147.32.84.209@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313592600 | 7 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.209@1313593500 | 127 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313666340 | 0 | 15 | 0.021 | no (—) | no (—) |
| 147.32.84.209@1313668380 | 0 | 30 | 0.005 | no (—) | no (—) |
| 147.32.84.209@1313669280 | 25 | 14 | 0.029 | no (—) | yes (10.0) |
| 147.32.84.209@1313675520 | 5 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313676360 | 9 | 8 | 0.012 | no (—) | yes (3.0) |
| 147.32.84.209@1313677800 | 0 | 14 | 0.016 | no (—) | no (—) |
| 147.32.84.209@1313678460 | 9 | 10 | 0.112 | no (—) | yes (2.0) |
| 147.32.84.209@1313679540 | 0 | 7 | 0.991 | yes (480) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_9:exfil` | neris | 1373 | 11 | 10 | 0.001 | 0.002 | 0.001 | 0.208 |
| `ctu_8:c2` | murlo | 1069 | 3 | 1¹ | 0.003 | 0.005 | 0.005 | 0.001 |
| `ctu_10:recon` | rbot | 488 | 59 | 10 | 0.002 | 0.008 | 0.002 | 0.171 |
| `ctu_10:exfil` | rbot | 307 | 38 | 10 | 0.007 | 0.010 | 0.005 | 0.458 |
| `ctu_10:c2` | rbot | 232 | 45 | 10 | 0.001 | 0.001 | 0.001 | 0.003 |
| `ctu_9:recon` | neris | 151 | 22 | 10 | 0.000 | 0.000 | 0.000 | 0.093 |
| `ctu_9:c2` | neris | 35 | 8 | 8 | 0.000 | 0.000 | 0.000 | 0.108 |
| `ctu_8:recon` | murlo | 31 | 3 | 1¹ | 0.000 | 0.000 | 0.000 | 0.329 |
| `ctu_8:exfil` | murlo | 2 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.123 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.112 | 0.368 | 0.6107 | 10 | 0.7192 | 0.765 | 1.06× | 0.4828 | carries timing signal |
| world_model | 0.108 | 0.348 | 0.3062 | 10 | 0.7192 | 0.731 | 1.02× | 0.4204 | carries timing signal |
| world_model_deterministic | 0.111 | 0.338 | 0.4874 | 10 | 0.7192 | 0.679 | 0.94× | 0.2986 | carries timing signal |
| noised_persistence | 0.115 | 0.408 | 0.2925 | 10 | 0.7192 | 0.838 | 1.16× | 0.6865 | carries timing signal |
| isotropic_noise_persistence | 0.108 | 0.351 | 0.2676 | 10 | 0.7192 | 0.725 | 1.01× | 0.4016 | carries timing signal |
| persistence | 0.121 | 0.402 | 0.7072 | 10 | 0.7192 | 0.735 | 1.02× | 0.4403 | carries timing signal |
| oracle_true_future | 0.145 | 0.453 | 0.6999 | 10 | 0.7192 | 0.811 | 1.13× | 0.5428 | carries timing signal |
| lr_current_state | 0.118 | 0.359 | 0.5680 | 10 | 0.7192 | 0.575 | 0.80× | 0.1674 | carries timing signal |
| lr_flattened_history | 0.118 | 0.308 | 0.8045 | 10 | 0.7192 | 0.623 | 0.87× | 0.2116 | carries timing signal |
| gbdt_current_state | 0.159 | 0.525 | 0.5059 | 10 | 0.7192 | 0.614 | 0.85× | 0.1644 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.17761 | 0.34520 | 0.00749 | 0.00743 | 10/10 |
| calibrated | 0.04297 | 0.08327 | 0.00749 | 0.00743 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 21045 scored rows, natural prevalence 0.00409, operating point `mean|q=-|integrated`, threshold 0.882 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.014 | [0.000, 0.052] | 0.703 | 0.00 / 0.01 / 0.00 | 425.53 | 0.014 | 0.000 / 0.000 |
| NIDRA (stochastic rollout) | 0.013 | [0.000, 0.047] | 0.682 | 0.02 / 0.23 / 0.03 | 802.31 | 0.013 | 0.000 / 0.000 |
| NIDRA (deterministic rollout) | 0.008 | [0.000, 0.029] | 0.559 | 0.01 / 0.09 / 0.01 | 699.35 | 0.008 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.015 | [0.000, 0.054] | 0.795 | 0.02 / 0.32 / 0.04 | 811.08 | 0.016 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.011 | — | 0.618 | 0.02 / 0.31 / 0.04 | 947.69 | 0.011 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.012 | [0.000, 0.042] | 0.661 | 0.00 / 0.00 / 0.00 | 356.98 | 0.012 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.013 | — | 0.686 | 0.01 / 0.07 / 0.01 | 583.57 | 0.014 | 0.000 / 0.000 |
| logistic regression on S_t | 0.005 | [0.000, 0.016] | 0.300 | 0.02 / 0.13 / 0.03 | 443.27 | 0.005 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.002 | [0.000, 0.007] | 0.214 | 0.00 / 0.00 / 0.00 | 245.27 | 0.002 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.004 | [0.000, 0.011] | 0.184 | 0.00 / 0.00 / 0.00 | 68.66 | 0.004 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.03 / F1 0.01, 8634 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.001 | [-0.001, 0.003] |
| world_model - persistence_rollout | 0.000 | [-0.001, 0.004] |
| world_model - noised_persistence | -0.003 | [-0.009, -0.000] |
| world_model - isotropic_noise_persistence | 0.002 | [0.000, 0.007] |
| world_model - world_model_deterministic | 0.005 | [0.000, 0.017] |
| world_model_deterministic - persistence | -0.004 | [-0.016, -0.000] |
| noised_persistence - persistence | 0.003 | [0.000, 0.011] |
| world_model - lr_current_state | 0.008 | [-0.000, 0.028] |
| world_model - lr_flattened_history | 0.010 | [0.000, 0.039] |
| world_model - gbdt_current_state | 0.009 | [-0.001, 0.034] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.454, period2_persistence 0.051, ridge_two_lag -3.334; NIDRA vs ridge 0.874.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00391 | 0.015 | 0.013 | 0.012 | 0.12744 | 0.00 (992) |
| 2 | 2 | 0.00391 | 0.015 | 0.013 | 0.012 | 0.12385 | 0.00 (992) |
| 3 | 3 | 0.00391 | 0.014 | 0.011 | 0.012 | 0.11554 | 0.00 (994) |
| 4 | 4 | 0.00391 | 0.012 | 0.008 | 0.012 | 0.10722 | 0.00 (994) |
| 5 | 5 | 0.00391 | 0.010 | 0.005 | 0.012 | 0.09789 | 0.00 (994) |
| 6 | 6 | 0.00392 | 0.008 | 0.004 | 0.012 | 0.09105 | 0.00 (996) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 3 | 10 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 5 | 18 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 10 | 27 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 15 | 27 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 30 | 27 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.882), 2 consecutive windows: 6 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 951.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313428560 | 977 | 0 | — | no (—) | yes (951.0) |
| 147.32.84.165@1313752680 | 5 | 8 | 0.008 | no (—) | no (—) |
| 147.32.84.165@1313753400 | 9 | 7 | 0.030 | no (—) | no (—) |
| 147.32.84.191@1313753100 | 14 | 8 | 0.032 | no (—) | no (—) |
| 147.32.84.192@1313752500 | 15 | 0 | — | no (—) | no (—) |
| 147.32.84.192@1313754000 | 0 | 4 | 0.003 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_13:exfil` | virut | 752 | 1 | 1¹ | 0.009 | 0.010 | 0.010 | 0.426 |
| `ctu_13:recon` | virut | 150 | 1 | 1¹ | 0.003 | 0.004 | 0.002 | 0.454 |
| `ctu_12:c2` | nsis_ay | 63 | 5 | 3 | 0.000 | 0.000 | 0.000 | 0.432 |
| `ctu_13:c2` | virut | 47 | 1 | 1¹ | 0.001 | 0.001 | 0.001 | 0.427 |
| `ctu_12:recon` | nsis_ay | 18 | 4 | 2 | 0.000 | 0.000 | 0.000 | 0.284 |
| `ctu_12:exfil` | nsis_ay | 10 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.261 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.061 | 0.572 | 0.7427 | 3 | 0.9933 | 0.998 | 1.00× | 0.7523 | carries timing signal |
| world_model | 0.057 | 0.553 | 0.6979 | 3 | 0.9933 | 0.998 | 1.00× | 0.7391 | carries timing signal |
| world_model_deterministic | 0.046 | 0.470 | 0.6012 | 3 | 0.9933 | 0.998 | 1.00× | 0.7505 | carries timing signal |
| noised_persistence | 0.062 | 0.599 | 0.7026 | 3 | 0.9933 | 0.998 | 1.00× | 0.7518 | carries timing signal |
| isotropic_noise_persistence | 0.052 | 0.510 | 0.4563 | 3 | 0.9933 | 0.998 | 1.00× | 0.7140 | carries timing signal |
| persistence | 0.063 | 0.568 | 0.7480 | 3 | 0.9933 | 0.998 | 1.00× | 0.7202 | carries timing signal |
| oracle_true_future | 0.068 | 0.592 | 0.7672 | 3 | 0.9933 | 0.998 | 1.00× | 0.6935 | carries timing signal |
| lr_current_state | 0.053 | 0.481 | 0.8955 | 3 | 0.9933 | 0.993 | 1.00× | 0.4412 | carries timing signal |
| lr_flattened_history | 0.040 | 0.427 | 0.5450 | 3 | 0.9933 | 0.998 | 1.01× | 0.7714 | carries timing signal |
| gbdt_current_state | 0.074 | 0.597 | 0.2301 | 3 | 0.9933 | 0.988 | 0.99× | 0.3007 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.17505 | 0.34801 | 0.00409 | 0.00408 | 10/10 |
| calibrated | 0.04051 | 0.08791 | 0.00409 | 0.00408 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.
