### Stage-transition lead time — flagged before the campaign's completing phase?

Times are capture-local (UTC-3). A flag during an earlier phase is that phase recognised in time to act before the completing one; it is not a forecast that names the completing phase.

| split | host | phases (onset stage) | completing phase | flagged before | lead (min) | max score before | windows scored / expected | completing phase detected (latency) |
|---|---|---|---|---|---|---|---|---|
| test | 172.16.0.1 | 13:05 recon, 13:52 recon, 14:14 recon, 14:51 recon, 15:03 recon, 15:21 recon, 15:56 exfil | 15:56 exfil | yes, 14:51 | 65 | 1.000 | 121 / 201 | yes (0 min) |
| holdout | 172.16.0.1 | 09:15 initial_access, 10:15 initial_access | 10:15 initial_access | yes, 09:44 | 31 | 0.987 | 30 / 90 | yes (1 min) |
| holdout | 192.168.10.8 | 14:19 lateral, 14:28 lateral, 15:04 lateral | 15:04 lateral | no | — | 0.044 | 73 / 75 | yes (8 min) |

- **val**: 0 of 0 multi-phase campaigns flagged before completion at threshold 0.718; 2 single-episode campaigns not scored.
- **test**: 1 of 1 multi-phase campaigns flagged before completion at threshold 0.718; 9 single-episode campaigns not scored.
- **holdout**: 1 of 2 multi-phase campaigns flagged before completion at threshold 0.718; 0 single-episode campaigns not scored.
