### Attack-group separability probe — **val**

A supervised probe fit directly on each group's one-vs-rest label, host-grouped 5-fold. It reads the labels of the split it scores, so it is an upper bound in the same sense the oracle is — a diagnostic, never a system, and never a selection signal.

| group | positives | base rate | probe AP | lift | probe ROC | separable | forecast AP |
|---|---|---|---|---|---|---|---|
| ctu_6:exfil | 122 | 0.00123 | 0.001 | 1× | 0.100 | **no** | 0.870 |
| ctu_4:exfil | 50 | 0.00051 | 0.001 | 1× | 0.101 | **no** | 0.255 |
| ctu_4:c2 | 23 | 0.00023 | 0.000 | 1× | 0.101 | **no** | 0.001 |
| ctu_4:recon | 17 | 0.00017 | 0.000 | 1× | 0.101 | **no** | 0.001 |
