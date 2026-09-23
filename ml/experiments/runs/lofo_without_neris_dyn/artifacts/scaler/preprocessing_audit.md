# Preprocessing audit

Fit rows: 500,000 (active: 89,187). Clip: ±10. Dropped: urg_ratio, iat_max, ttl_mean, ttl_var, tcp_window_mean, tcp_window_entropy, frag_flag_rate, payload_size_mean, payload_size_var, payload_size_p95, payload_size_entropy, retrans_count, retrans_rate, reciprocity, d_retrans_rate. Flagged: 2.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 1] | 3.76 | 0.54 | 3.76 | 0.000 | 0.00 | slope3_syn_ratio (0.57) |  |
| ack_ratio | unit | [0, 1] | 3.38 | 0.57 | 3.38 | 0.000 | 0.00 | rst_ratio (0.69) |  |
| rst_ratio | unit | [0, 1] | 7.16 | 0.84 | 7.16 | 0.000 | 0.00 | ack_ratio (0.69) |  |
| fin_ratio | unit | [0, 0.667] | 2.24 | 0.65 | 2.24 | 0.000 | 0.00 | psh_ratio (0.82) |  |
| psh_ratio | unit | [0, 0.667] | 2.19 | 0.62 | 2.19 | 0.000 | 0.00 | fin_ratio (0.82) |  |
| urg_ratio | drop (constant on active training rows (value 0); redundant with is_active) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| bytes_total | log1p | [60, 4.01e+09] | 116.46 | 0.00 | 0.74 | 0.000 | -3.25 | active_flow_count (0.71) |  |
| bytes_up_down_ratio | log1p | [0, 8.47e+03] | 241.42 | 0.26 | 2.90 | 0.000 | -0.87 | iat_mean (0.37) |  |
| pkts_per_flow_mean | log1p | [1, 2.77e+06] | 222.93 | 0.00 | 1.64 | 0.000 | -1.64 | bytes_total (0.64) |  |
| flow_duration_mean | log1p | [0, 3.6e+09] | 2.21 | 0.11 | 0.27 | 0.000 | -1.77 | iat_mean (0.98) |  |
| flow_duration_var | log1p | [0, 3.24e+18] | 3.67 | 0.36 | 0.36 | 0.000 | -1.03 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 3.6e+09] | 6.89 | 0.12 | 0.20 | 0.000 | -1.76 | flow_duration_mean (0.98) |  |
| iat_var | log1p | [0, 3.23e+18] | 25.37 | 0.36 | 0.42 | 0.000 | -1.03 | flow_duration_var (0.99) |  |
| iat_max | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| active_flow_count | log1p | [1, 1.18e+04] | 50.76 | 0.00 | 1.32 | 0.000 | -1.36 | bytes_total (0.71) |  |
| ttl_mean | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| ttl_var | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| tcp_window_mean | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| tcp_window_entropy | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| frag_flag_rate | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| payload_size_mean | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| payload_size_var | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| payload_size_p95 | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| payload_size_entropy | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| retrans_count | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| retrans_rate | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| out_degree | log1p | [1, 738] | 10.96 | 0.00 | 2.76 | 0.000 | -1.52 | dst_ip_entropy (0.97) |  |
| in_degree | log1p | [0, 61] | 29.32 | 0.90 | 3.94 | 0.001 | -0.30 | neighbour_risk_fraction (0.22) |  |
| dst_ip_entropy | zscore | [-0, 6.6] | 2.54 | 0.54 | 2.54 | 0.000 | -0.63 | out_degree (0.97) | log1p would reduce |skew| 2.54 -> 1.13 |
| dst_port_entropy | zscore | [-0, 6.6] | 3.88 | 0.59 | 3.88 | 0.000 | -0.54 | dst_ip_entropy (0.76) | log1p would reduce |skew| 3.88 -> 1.61 |
| new_peer_count | log1p | [0, 738] | 18.26 | 0.87 | 4.19 | 0.000 | -0.30 | out_degree (0.78) |  |
| neighbour_risk_fraction | unit | [0, 1] | 3.42 | 0.88 | 3.42 | 0.000 | 0.00 | in_degree (0.22) |  |
| local_clustering_coeff | unit | [0, 1] | 18.88 | 0.99 | 18.88 | 0.000 | 0.00 | in_degree (0.14) |  |
| reciprocity | drop (constant on active training rows (value 1); redundant with is_active) | [1, 1] | 0.00 | 0.00 | — | 0.000 | 0.00 | — |  |
| d_syn_ratio | unit | [-1, 1] | 4.05 | 0.56 | 4.05 | 0.000 | 0.00 | slope3_syn_ratio (0.71) |  |
| d_dst_port_entropy | zscore | [-4.83, 4.54] | 0.27 | 0.57 | 0.27 | 0.000 | -0.16 | d_out_degree (0.60) |  |
| d_out_degree | asinh | [-359, 490] | 1.19 | 0.32 | 0.59 | 0.000 | -0.32 | d_dst_port_entropy (0.60) |  |
| d_new_peer_count | asinh | [-237, 424] | 4.36 | 0.85 | 0.01 | 0.000 | -0.04 | d_out_degree (0.59) |  |
| d_iat_var | asinh | [-3.23e+18, 3.23e+18] | 3.90 | 0.31 | 0.20 | 0.000 | -0.16 | d_dst_port_entropy (0.51) |  |
| d_retrans_rate | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| slope3_syn_ratio | unit | [-0.5, 0.5] | 3.64 | 0.56 | 3.64 | 0.000 | 0.00 | d_syn_ratio (0.71) |  |
| slope3_dst_port_entropy | zscore | [-2.3, 2.64] | 0.35 | 0.56 | 0.35 | 0.000 | -0.15 | d_dst_port_entropy (0.57) |  |
| slope3_out_degree | asinh | [-972, 177] | 47.15 | 0.31 | 0.43 | 0.000 | -0.27 | slope3_dst_port_entropy (0.57) |  |
| slope3_iat_var | asinh | [-1.57e+18, 1.62e+18] | 9.77 | 0.31 | 0.20 | 0.000 | -0.16 | slope3_dst_port_entropy (0.50) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
