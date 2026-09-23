**All 28 re-run cells reproduce** within 0.005 AP, on identical rows.

| cell | split | recorded AP | re-run AP | Δ | reproduces | gained CI | note |
|---|---|---:|---:|---:|:--:|:--:|---|
| cic2cic_state | val | 0.7809 | 0.7809 | 0.0000 | yes | yes |  |
| cic2cic_state | test | 0.1110 | 0.1110 | 0.0000 | yes | yes |  |
| cic2cic_state | holdout | 0.3323 | 0.3323 | 0.0000 | yes | yes |  |
| cic2cic_state+hidden | val | 0.8316 | 0.8316 | 0.0000 | yes | yes |  |
| cic2cic_state+hidden | test | 0.0642 | 0.0642 | 0.0000 | yes | yes |  |
| cic2cic_state+hidden | holdout | 0.3797 | 0.3797 | 0.0000 | yes | yes |  |
| cic2ctu_state | test | 0.0098 | 0.0098 | 0.0000 | yes | yes |  |
| cic2ctu_state | holdout | 0.0354 | 0.0354 | 0.0000 | yes | yes |  |
| cic2ctu_state+hidden | test | 0.0087 | 0.0087 | 0.0000 | yes | yes |  |
| cic2ctu_state+hidden | holdout | 0.0142 | 0.0142 | 0.0000 | yes | yes |  |
| comb2cic_state | val | 0.3914 | 0.3914 | 0.0000 | yes | yes |  |
| comb2cic_state | test | 0.1645 | 0.1645 | 0.0000 | yes | yes |  |
| comb2cic_state | holdout | 0.2243 | 0.2243 | 0.0000 | yes | yes |  |
| comb2cic_state+hidden | val | 0.5381 | 0.5381 | 0.0000 | yes | yes |  |
| comb2cic_state+hidden | test | 0.1584 | 0.1584 | 0.0000 | yes | yes |  |
| comb2cic_state+hidden | holdout | 0.3282 | 0.3282 | 0.0000 | yes | yes |  |
| comb2ctu_state | test | 0.1568 | 0.1568 | 0.0000 | yes | yes |  |
| comb2ctu_state | holdout | 0.3070 | 0.3070 | 0.0000 | yes | yes |  |
| comb2ctu_state+hidden | test | 0.3734 | 0.3734 | 0.0000 | yes | yes |  |
| comb2ctu_state+hidden | holdout | 0.2469 | 0.2469 | 0.0000 | yes | yes |  |
| ctu2cic_state | test | 0.0448 | 0.0448 | 0.0000 | yes | yes |  |
| ctu2cic_state | holdout | 0.0012 | 0.0012 | 0.0000 | yes | yes |  |
| ctu2cic_state+hidden | test | 0.0035 | 0.0035 | 0.0000 | yes | yes |  |
| ctu2cic_state+hidden | holdout | 0.0003 | 0.0003 | 0.0000 | yes | yes |  |
| ctu2ctu_state | test | 0.1964 | 0.1964 | 0.0000 | yes | yes |  |
| ctu2ctu_state | holdout | 0.1884 | 0.1884 | 0.0000 | yes | yes |  |
| ctu2ctu_state+hidden | test | 0.3221 | 0.3221 | 0.0000 | yes | yes |  |
| ctu2ctu_state+hidden | holdout | 0.2206 | 0.2206 | 0.0000 | yes | yes |  |

A re-run is seeded but not bit-identical — the bootstrap resamples and the rollout draws — so agreement is judged at 0.005 AP, an order of magnitude below the smallest difference this phase draws a conclusion from. A row marked **no** for *different rows* or *different prevalence* did not drift: it scored different data, which is the worse finding.
