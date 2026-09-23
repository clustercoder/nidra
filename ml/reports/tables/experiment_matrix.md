### Experiment matrix — every run on disk

39 runs with a provenance record, 26 directories without one. Generated from the records, so a run that failed or produced an unwelcome result is here by construction and one that was deleted is visibly missing.

⚠ Rows span 2 geometries (Δ60 L15 K3, Δ60 L30 K6) and 1 feature regimes (cross_core). Those are the two things that make a comparison meaningless without saying so; rows differing in either are not comparable.

| run | stage | corpora | days (train/val/test/holdout) | regime | geometry | stages | seeds | epochs | init from | commit | wall | weights | metrics |
|---|---|---|---|---|---|---|---|---|---|---|---:|---|---|
| `cic_core_dyn` | train | CIC-IDS2017 | 2tr 3te 2ho | cross_core | Δ60 L30 K6 | dynamics | 0 | 20 | — | `8f5b5390` | 75m | ✓ | — |
| `cic_core_heads` | head_ablation | CIC-IDS2017 | 2tr 3te 2ho | cross_core | Δ60 L30 K6 | — | 0,1,2,3,4 | — | cic_core_dyn | `05458199` | — | — | — |
| `comb_dyn` | train | CIC-IDS2017+CTU-13 | 5tr 2va 6te 4ho | cross_core | Δ60 L30 K6 | dynamics | 0 | 20 | — | `59469f18` | 81m | ✓ | — |
| `combined_heads` | head_ablation | CIC-IDS2017+CTU-13 | 5tr 2va 6te 4ho | cross_core | Δ60 L30 K6 | — | 0,1,2,3,4 | — | comb_dyn | `05458199` | — | — | — |
| `ctu_confirm` | head_ablation | CTU-13 | 3tr 2va 3te 3ho | cross_core | Δ60 L30 K6 | — | 0,1,2,3,4 | — | ctu_dyn | `ee794000` | — | — | — |
| `ctu_dyn` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | dynamics | 0 | 20 | — | `5e1d6e18` | 45m | ✓ | — |
| `ctu_dyn_s1` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | dynamics | 1 | 20 | — | `eae98d93` | 79m | ✓ | — |
| `ctu_dyn_s2` | train | CTU-13 | 3tr 2va 3te 3ho | cross_core | Δ60 L30 K6 | dynamics | 2 | 20 | — | `192e8e86` | 86m | ✓ | — |
| `ctu_gru` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | gru_baseline | 0 | — | ctu_dyn | `59469f18` | 4m | ✓ | — |
| `ctu_heads` | head_ablation | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | — | 0,1,2,3,4 | — | ctu_dyn | `8f5b5390` | — | — | — |
| `ctu_onset_hidden_hazard` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | onset | 0 | — | ctu_dyn | `63d3ba4b` | 3m | ✓ | — |
| `ctu_onset_hidden_indep` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | onset | 0 | — | ctu_dyn | `61258de5` | 3m | ✓ | — |
| `ctu_onset_state_hazard` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | onset | 0 | — | ctu_dyn | `59469f18` | 0m | ✓ | — |
| `ctu_onset_state_indep` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | onset | 0 | — | ctu_dyn | `59469f18` | 0m | ✓ | — |
| `ctu_sh_cn01` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | heads | 0 | — | ctu_dyn | `61258de5` | 8m | ✓ | ✓ |
| `ctu_sh_cn03` | train | CTU-13 | 3tr 7te 3ho | cross_core | Δ60 L30 K6 | heads | 0 | — | ctu_dyn | `63d3ba4b` | 7m | ✓ | ✓ |
| `geomA_L15K3` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 30 | — | `4897a042` | 38m | ✓ | ✓ |
| `geomB_L30K6` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L30 K6 | dynamics,heads | 0 | 30 | — | `9aaf3c4c` | 64m | ✓ | — |
| `headsA_bal_nonoise` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads | 0 | — | geomA_L15K3 | `405977a3` | 1m | ✓ | ✓ |
| `headsA_bal_r5_lr3` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads | 0 | — | geomA_L15K3 | `542fc99f` | 1m | ✓ | ✓ |
| `headsA_imb_noise` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads | 0 | — | geomA_L15K3 | `542fc99f` | 2m | ✓ | ✓ |
| `headsA_new` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | geomA_L15K3 | `405977a3` | 1m | ✓ | ✓ |
| `headsB_imb_noise` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L30 K6 | heads | 0 | — | geomB_L30K6 | `88c18ccc` | 1m | ✓ | ✓ |
| `lodo_without_tuesday` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L30 K6 | dynamics,heads | 0 | 20 | — | `02ea6d9c` | 49m | ✓ | ✓ |
| `lodo_without_wednesday` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L30 K6 | dynamics,heads | 0 | 20 | — | `abe2fe59` | 69m | ✓ | ✓ |
| `production` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L30 K6 | dynamics,heads,onset,gru_baseline | 0,1,2,3,4 | 24 | — | `8beccae6` | 4.8h | — | — |
| `var_betanll` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `ca762b48` | 23m | ✓ | ✓ |
| `var_betanll_h2` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | var_betanll | `eb7b3e8f` | 2m | ✓ | ✓ |
| `var_betanll_nonsilent` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `eb7b3e8f` | 27m | ✓ | ✓ |
| `var_linskip` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `ca762b48` | 25m | ✓ | ✓ |
| `var_linskip_betanll` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `eb7b3e8f` | 26m | ✓ | ✓ |
| `var_linskip_betanll_h2` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | var_linskip_betanll | `eb7b3e8f` | 2m | ✓ | ✓ |
| `var_linskip_h2` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | var_linskip | `eb7b3e8f` | 2m | ✓ | ✓ |
| `var_mseaux` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `92eb4dfd` | 20m | ✓ | ✓ |
| `var_mseaux_h2` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | var_mseaux | `eb7b3e8f` | 2m | ✓ | ✓ |
| `var_nonsilent` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `eb7b3e8f` | 26m | ✓ | ✓ |
| `var_nonsilent_h2` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | var_nonsilent | `eb7b3e8f` | 2m | ✓ | ✓ |
| `var_plain` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | dynamics,heads | 0 | 20 | — | `88c18ccc` | 19m | ✓ | ✓ |
| `var_plain_h2` | train | CIC-IDS2017 (TrafficLabelling CSVs + project tshark packet parquet) | 2tr 3te 2ho | — | Δ60 L15 K3 | heads,onset | 0 | — | var_plain | `eb7b3e8f` | 2m | ✓ | ✓ |

Recorded wall clock across all runs: **17.9 hours** (single machine, Apple M1, 8 cores, 16 GB; several runs overlapped, so elapsed time is less than the sum).

Directories without a provenance record — head-ablation variants, which carry their parent run's record: `cic_core_heads__prep`, `cic_core_heads__state`, `cic_core_heads__state+hidden`, `combined_heads__prep`, `combined_heads__state`, `combined_heads__state+hidden`, `ctu_confirm__prep`, `ctu_confirm__state`, `ctu_confirm__state+hidden`, `ctu_heads__hidden`, `ctu_heads__prep`, `ctu_heads__state`, `ctu_heads__state+hidden`, `ctu_heads__state+hidden+delta+logvar`, `ctu_heads__state+hidden+logvar`, `ctu_heads__state+logvar`, `lofo_without_neris_dyn`, `xeval_cic2cic_state`, `xeval_cic2cic_state+hidden`, `xeval_cic2ctu_state`, `xeval_comb2cic_state`, `xeval_comb2cic_state+hidden`, `xeval_comb2ctu_state`, `xeval_ctu2cic_state`, `xeval_ctu2ctu_state`, `xeval_ctu2ctu_state+hidden`.
