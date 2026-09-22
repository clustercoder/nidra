### What a head's scores encode over the whole split — **val**

The same decomposition applied to every row, not only the uninformative ones. Here a state-only head has real features to work with, so its columns are informative too and the comparison between the two heads is the point. **host-mean** replaces each score by its host's mean, keeping only host identity. **within-host** restricts to the infected host(s), where identity is constant and only the timing question remains — which is the question advance warning asks.

| head | rows | positives | AP | lift over ceiling | ROC | host-mean AP | host-mean ROC | within-host prevalence | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| state-only | 98876 | 288 | 0.353066 | 121.21× | 0.7281 | 0.7579 | 0.9995 | 0.7579 | 0.8763 | 1.16× | 0.6559 | carries timing signal |
| state+hidden | 98876 | 288 | 0.489438 | 168.03× | 0.7441 | 0.7579 | 0.9995 | 0.7579 | 0.8951 | 1.18× | 0.6621 | carries timing signal |

A lift that disappears within the host is the head recognising *who*, not *when*.
