**`world_model_calibrated` against the strongest baseline in each cell, paired episode-cluster bootstrap: the interval excludes zero in 3 of 28 cells** (3 of 24 once validation is set aside), and is entirely below zero in 3. The baseline is picked per cell as the highest-AP one there, so a cell cannot win by beating a rival that happened to be weak in it.
 **5 of the 6 counted cells clear zero by less than 0.001** and are flagged *knife edge* below: the rule was fixed before the data was seen and is applied as written, but an endpoint at 1e-05 is not a margin.

| cell | split | published AP | strongest baseline | its AP | margin | 95% CI | clusters | |
|---|---|---:|---|---:|---:|---|---:|---|
| cic2cic_state | val | 0.9238 | noised_persistence | 0.9303 | -0.0066 | [-0.0932, +0.2945] | 3305 |  |
| cic2cic_state | test | 0.1404 | noised_persistence | 0.1370 | +0.0035 | [-0.0366, +0.0315] | 1742 |  |
| cic2cic_state | holdout | 0.5688 | persistence | 0.5964 | -0.0275 | [-0.0664, +0.0436] | 2054 |  |
| cic2cic_state+hidden | val | 0.9905 | persistence | 0.9847 | +0.0058 | [+0.0000, +0.3142] | 3305 |  |
| cic2cic_state+hidden | test | 0.1047 | noised_persistence | 0.0945 | +0.0102 | [-0.0038, +0.0634] | 1742 |  |
| cic2cic_state+hidden | holdout | 0.5792 | noised_persistence | 0.5150 | +0.0642 | [-0.1046, +0.3027] | 2054 |  |
| cic2ctu_state | test | 0.0119 | noised_persistence | 0.0154 | -0.0034 | [-0.0095, +0.0001] | 559 |  |
| cic2ctu_state | holdout | 0.0370 | noised_persistence | 0.0340 | **+0.0030** | [+1.4e-05, +0.0152] | 365 | knife edge |
| cic2ctu_state+hidden | test | 0.0087 | noised_persistence | 0.0117 | **-0.0030** | [-0.0053, -0.0005] | 559 | knife edge |
| cic2ctu_state+hidden | holdout | 0.0157 | noised_persistence | 0.0172 | **-0.0015** | [-0.0059, -1.6e-05] | 365 | knife edge |
| comb2cic_state | val | 0.5111 | gbdt_current_state | 0.5375 | -0.0264 | [-0.1087, +0.0772] | 3439 |  |
| comb2cic_state | test | 0.1514 | noised_persistence | 0.1205 | **+0.0309** | [+0.0001, +0.0812] | 1742 | knife edge |
| comb2cic_state | holdout | 0.4792 | persistence_rollout | 0.5806 | -0.1013 | [-0.3781, +0.0685] | 2054 |  |
| comb2cic_state+hidden | val | 0.6392 | persistence | 0.6586 | -0.0193 | [-0.1229, +0.0324] | 3439 |  |
| comb2cic_state+hidden | test | 0.1654 | noised_persistence | 0.0994 | **+0.0660** | [+0.0235, +0.1296] | 1742 |  |
| comb2cic_state+hidden | holdout | 0.3869 | persistence_rollout | 0.5390 | -0.1522 | [-0.3064, +0.0128] | 2054 |  |
| comb2ctu_state | test | 0.2502 | gbdt_current_state | 0.3277 | -0.0775 | [-0.1350, +0.0634] | 559 |  |
| comb2ctu_state | holdout | 0.2926 | ridge_two_lag | 0.3148 | -0.0222 | [-0.2041, +0.0345] | 365 |  |
| comb2ctu_state+hidden | test | 0.4608 | noised_persistence | 0.4686 | -0.0078 | [-0.0458, +0.0749] | 559 |  |
| comb2ctu_state+hidden | holdout | 0.2476 | gbdt_current_state | 0.3121 | -0.0645 | [-0.2151, +0.0001] | 365 |  |
| ctu2cic_state | test | 0.0821 | noised_persistence | 0.0675 | +0.0146 | [-0.0062, +0.0396] | 1742 |  |
| ctu2cic_state | holdout | 0.0021 | noised_persistence | 0.0007 | +0.0014 | [-4.0e-05, +0.0071] | 2054 |  |
| ctu2cic_state+hidden | test | 0.0029 | gbdt_current_state | 0.0146 | **-0.0117** | [-0.0376, -0.0003] | 1742 | knife edge |
| ctu2cic_state+hidden | holdout | 0.0002 | gbdt_current_state | 0.0005 | -0.0003 | [-0.0029, +4.5e-05] | 2054 |  |
| ctu2ctu_state | test | 0.2557 | gru_classifier | 0.3878 | -0.1320 | [-0.2808, +0.0978] | 559 |  |
| ctu2ctu_state | holdout | 0.2483 | noised_persistence | 0.2313 | +0.0170 | [-0.0018, +0.0487] | 365 |  |
| ctu2ctu_state+hidden | test | 0.4173 | noised_persistence | 0.4594 | -0.0421 | [-0.0812, +0.0192] | 559 |  |
| ctu2ctu_state+hidden | holdout | 0.2313 | noised_persistence | 0.2512 | -0.0199 | [-0.0610, +0.0080] | 365 |  |
