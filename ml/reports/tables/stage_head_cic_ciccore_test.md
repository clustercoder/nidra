### Frozen head diagnostic per stage — observed states, **test**

The stage head's own class probability against that stage one-vs-rest, beside the risk head's score on the same rows. Observed states only: both heads are scored where they were trained, so a gap is about the objective rather than about rollout error.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| recon | 27 | 0.00009 | 0.086 | 951× | 0.743 | 0.164 | 0.959 |
| c2 | 648 | 0.00216 | 0.026 | 12× | 0.596 | 0.028 | 0.629 |
| exfil | 22 | 0.00007 | 0.492 | 6697× | 1.000 | 0.966 | 1.000 |
