### Frozen head diagnostic per stage — observed states, **val**

The stage head's own class probability against that stage one-vs-rest, beside the risk head's score on the same rows. Observed states only: both heads are scored where they were trained, so a gap is about the objective rather than about rollout error.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| recon | 17 | 0.00017 | 0.003 | 20× | 0.638 | 0.000 | 0.320 |
| c2 | 24 | 0.00024 | 0.001 | 6× | 0.854 | 0.000 | 0.434 |
| exfil | 172 | 0.00174 | 0.301 | 173× | 0.910 | 0.799 | 0.915 |
