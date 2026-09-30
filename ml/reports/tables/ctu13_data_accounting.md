### CTU-13 training data accounting (Δ=60 s, L=30, K=6, cross_core)

| split | captures | raw flows | canonical states | active | eligible origins | positives | sampled / epoch | coverage / epoch | mean touches | state exposures |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 3 | 9,343,396 | 1,419,278 | 244,235 | 1,401,778 | 1,384 | 500,000 | 35.7% | 7.1 | 300,000,000 |
| val | 2 | 1,679,995 | 98,876 | 35,014 | 86,941 | 283 | 50,000 | 57.5% | 11.5 | 30,000,000 |
| test | 3 | 6,351,529 | 508,517 | 123,963 | 492,417 | 3,688 | — | — | — | — |
| holdout | 3 | 2,380,452 | 267,388 | 56,866 | 253,983 | 1,040 | — | — | — | — |

Totals: 11 captures, 19,755,372 raw flows, 2,294,059 canonical states (460,078 active), 2,235,119 eligible origins of which 6,395 positive, 330,000,000 state exposures over training. Sampling columns apply to the splits the model is fit on; test and holdout are read once.
