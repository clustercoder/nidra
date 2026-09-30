### Frozen head diagnostic per stage — observed states, **test**

The stage head's own class probability against that stage one-vs-rest, beside the risk head's score on the same rows. Observed states only: both heads are scored where they were trained, so a gap is about the objective rather than about rollout error.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| recon | 27 | 0.00009 | 0.204 | 2266× | 0.952 | 0.164 | 0.961 |
| c2 | 648 | 0.00216 | 0.021 | 10× | 0.514 | 0.015 | 0.347 |
| exfil | 22 | 0.00007 | 0.192 | 2611× | 1.000 | 0.884 | 1.000 |
