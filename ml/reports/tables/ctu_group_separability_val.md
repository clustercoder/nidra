### Attack-group separability probe — **val**

A supervised probe fit directly on each group's one-vs-rest label. It reads the labels of the split it scores, so it is an upper bound in the same sense the oracle is — a diagnostic, never a system, and never a selection signal. Folds are host-grouped where the positives sit on two or more hosts; where they sit on one, a host-grouped split cannot be formed and the row-stratified fallback lets the probe recognise the host rather than the behaviour, which is marked. AP is not comparable to the benchmark's per-group AP — the benchmark scores a stratified subsample with capped negatives and this scores every row — but ROC is.

**The last column is one-sided.** A `yes` proves the signal exists and transfers, so a model that misses the group is failing at something achievable. A `no` proves only that this probe missed it: NIDRA's own frozen stage head ranked `ctu_4:c2` at ROC 0.864 after this probe returned 0.443 on the same windows under the same cross-host construction. Read `no` as *unproven*, never as *unlearnable*.

| group | positives | positive hosts | base rate | probe AP | lift | probe ROC | CV | probe found it |
|---|---|---|---|---|---|---|---|---|
| ctu_6:exfil | 122 | 1 | 0.00123 | 0.747 | 605× | 0.996 | **row-stratified (host-leaky)** | yes |
| ctu_4:exfil | 50 | 1 | 0.00051 | 0.145 | 287× | 0.946 | **row-stratified (host-leaky)** | yes |
| ctu_4:c2 | 23 | 1 | 0.00023 | 0.623 | 2680× | 0.947 | **row-stratified (host-leaky)** | yes |
| ctu_4:recon | 17 | 1 | 0.00017 | 0.173 | 1009× | 0.913 | **row-stratified (host-leaky)** | yes |
