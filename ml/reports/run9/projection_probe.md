## §3.41 verdict: **REFUTED**, and the same at every threshold in 0.003, 0.01, 0.03, 0.1.

- **sd > 0.003** — refuted: a cell that beats its oracle has its rollout no closer to the training cloud than the truth: cic2cic_state/test (1.686), cic2cic_state+hidden/test (1.688), cic2cic_state+hidden/holdout (1.943), cic2ctu_state/test (1.424), cic2ctu_state/holdout (1.321), cic2ctu_state+hidden/test (1.434), cic2ctu_state+hidden/holdout (1.336), comb2cic_state/test (1.334), comb2cic_state+hidden/test (1.337), comb2ctu_state/holdout (1.143), comb2ctu_state+hidden/holdout (1.151), ctu2cic_state/test (2.054), ctu2cic_state/holdout (2.211), ctu2cic_state+hidden/test (2.057), ctu2cic_state+hidden/holdout (2.211), ctu2ctu_state/holdout (1.344), ctu2ctu_state+hidden/holdout (1.347)
- **sd > 0.01** — refuted: a cell that beats its oracle has its rollout no closer to the training cloud than the truth: cic2cic_state/test (1.686), cic2cic_state+hidden/test (1.688), cic2cic_state+hidden/holdout (1.943), cic2ctu_state/test (1.424), cic2ctu_state/holdout (1.321), cic2ctu_state+hidden/test (1.434), cic2ctu_state+hidden/holdout (1.336), comb2cic_state/test (1.334), comb2cic_state+hidden/test (1.337), comb2ctu_state/holdout (1.143), comb2ctu_state+hidden/holdout (1.151), ctu2cic_state/test (2.054), ctu2cic_state/holdout (2.211), ctu2cic_state+hidden/test (2.057), ctu2cic_state+hidden/holdout (2.211), ctu2ctu_state/holdout (1.344), ctu2ctu_state+hidden/holdout (1.347)
- **sd > 0.03** — refuted: a cell that beats its oracle has its rollout no closer to the training cloud than the truth: cic2cic_state/test (1.317), cic2cic_state+hidden/test (1.315), cic2cic_state+hidden/holdout (1.496), cic2ctu_state/test (1.235), cic2ctu_state/holdout (1.114), cic2ctu_state+hidden/test (1.242), cic2ctu_state+hidden/holdout (1.121), comb2cic_state/test (1.025), comb2cic_state+hidden/test (1.021), ctu2cic_state/test (1.182), ctu2cic_state/holdout (1.239), ctu2cic_state+hidden/test (1.172), ctu2cic_state+hidden/holdout (1.251)
- **sd > 0.1** — refuted: a cell that beats its oracle has its rollout no closer to the training cloud than the truth: cic2cic_state+hidden/holdout (1.039)

Distances below are at the default threshold 0.01.

| cell | beats its oracle | d(truth) | d(rollout) | ratio | ratio across the sweep |
|---|:--:|---:|---:|---:|---|
| comb2ctu_state+hidden/test | no | 7.561 | 8.627 | 1.141 | 1.14 / 1.14 / 1.06 / 0.86 |
| comb2ctu_state/holdout | yes | 6.347 | 7.256 | 1.143 | 1.14 / 1.14 / 0.95 / 0.80 |
| comb2ctu_state/test | no | 7.561 | 8.651 | 1.144 | 1.14 / 1.14 / 1.06 / 0.85 |
| comb2ctu_state+hidden/holdout | yes | 6.347 | 7.304 | 1.151 | 1.15 / 1.15 / 0.95 / 0.80 |
| ctu2ctu_state+hidden/test | no | 8.540 | 11.026 | 1.291 | 1.29 / 1.29 / 1.00 / 0.83 |
| ctu2ctu_state/test | no | 8.540 | 11.088 | 1.298 | 1.30 / 1.30 / 1.00 / 0.83 |
| cic2ctu_state/holdout | yes | 14.471 | 19.118 | 1.321 | 1.32 / 1.32 / 1.11 / 0.90 |
| comb2cic_state/test | yes | 6.180 | 8.246 | 1.334 | 1.33 / 1.33 / 1.02 / 0.88 |
| cic2ctu_state+hidden/holdout | yes | 14.471 | 19.340 | 1.336 | 1.34 / 1.34 / 1.12 / 0.91 |
| comb2cic_state+hidden/test | yes | 6.180 | 8.261 | 1.337 | 1.34 / 1.34 / 1.02 / 0.88 |
| ctu2ctu_state/holdout | yes | 6.633 | 8.913 | 1.344 | 1.34 / 1.34 / 0.89 / 0.77 |
| ctu2ctu_state+hidden/holdout | yes | 6.633 | 8.937 | 1.347 | 1.35 / 1.35 / 0.89 / 0.77 |
| comb2cic_state+hidden/holdout | no | 5.906 | 8.143 | 1.379 | 1.38 / 1.38 / 1.06 / 0.91 |
| comb2cic_state/holdout | no | 5.906 | 8.220 | 1.392 | 1.39 / 1.39 / 1.06 / 0.91 |
| cic2ctu_state/test | yes | 15.961 | 22.730 | 1.424 | 1.42 / 1.42 / 1.24 / 0.95 |
| cic2ctu_state+hidden/test | yes | 15.961 | 22.888 | 1.434 | 1.43 / 1.43 / 1.24 / 0.96 |
| cic2cic_state/test | yes | 4.898 | 8.256 | 1.686 | 1.69 / 1.69 / 1.32 / 0.91 |
| cic2cic_state+hidden/test | yes | 4.898 | 8.269 | 1.688 | 1.69 / 1.69 / 1.31 / 0.90 |
| cic2cic_state/holdout | no | 4.145 | 8.048 | 1.941 | 1.94 / 1.94 / 1.50 / 1.04 |
| cic2cic_state+hidden/holdout | yes | 4.145 | 8.053 | 1.943 | 1.94 / 1.94 / 1.50 / 1.04 |
| ctu2cic_state/test | yes | 7.039 | 14.455 | 2.054 | 2.05 / 2.05 / 1.18 / 0.94 |
| ctu2cic_state+hidden/test | yes | 7.039 | 14.475 | 2.057 | 2.06 / 2.06 / 1.17 / 0.95 |
| ctu2cic_state/holdout | yes | 6.628 | 14.652 | 2.211 | 2.21 / 2.21 / 1.24 / 0.98 |
| ctu2cic_state+hidden/holdout | yes | 6.628 | 14.653 | 2.211 | 2.21 / 2.21 / 1.25 / 0.97 |
