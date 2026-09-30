### What a head's scores encode inside the floor stratum — **val**

Every row in the stratum is the same state vector. A state-only head can only emit a constant, so it lands exactly on the ceiling; a history-aware head varies, and these two columns say what the variation is. **host-mean** replaces each score by its host's mean, keeping only host identity. **within-host** restricts to the infected host, where identity is constant and only the timing question remains — which is the question advance warning asks.

| head | rows | positives | AP | lift over ceiling | ROC | host-mean AP | host-mean ROC | within-host prevalence | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| state-only | 47708 | 58 | 0.001216 | 1.00× | 0.5000 | 0.0012 | 0.5000 | 0.4567 | 0.4567 | 1.00× | 0.5000 | constant — at the ceiling, as a state-only head must be |
| state+hidden | 47708 | 58 | 0.070700 | 58.15× | 0.4242 | 0.4567 | 0.9993 | 0.4567 | 0.4460 | 0.98× | 0.3836 | **host identity** |

A lift over the ceiling that disappears within the host is the head recognising *who*, not *when*.
