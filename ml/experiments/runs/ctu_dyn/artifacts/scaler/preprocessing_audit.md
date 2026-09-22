# Preprocessing audit

Fit rows: 500,000 (active: 85,841). Clip: ±10. Dropped: urg_ratio, iat_max, ttl_mean, ttl_var, tcp_window_mean, tcp_window_entropy, frag_flag_rate, payload_size_mean, payload_size_var, payload_size_p95, payload_size_entropy, retrans_count, retrans_rate, reciprocity, d_retrans_rate. Flagged: 2.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 1] | 4.18 | 0.55 | 4.18 | 0.000 | 0.00 | slope3_syn_ratio (0.63) |  |
| ack_ratio | unit | [0, 1] | 3.83 | 0.59 | 3.83 | 0.000 | 0.00 | psh_ratio (0.63) |  |
| rst_ratio | unit | [0, 1] | 10.89 | 0.85 | 10.89 | 0.000 | 0.00 | ack_ratio (0.60) |  |
| fin_ratio | unit | [0, 0.667] | 2.12 | 0.64 | 2.12 | 0.000 | 0.00 | psh_ratio (0.82) |  |
| psh_ratio | unit | [0, 0.667] | 2.08 | 0.62 | 2.08 | 0.000 | 0.00 | fin_ratio (0.82) |  |
| urg_ratio | drop (constant on active training rows (value 0); redundant with is_active) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| bytes_total | log1p | [60, 1.54e+09] | 64.72 | 0.00 | 0.71 | 0.000 | -3.25 | active_flow_count (0.71) |  |
| bytes_up_down_ratio | log1p | [0, 3.15e+03] | 128.22 | 0.26 | 2.94 | 0.000 | -0.86 | iat_mean (0.37) |  |
| pkts_per_flow_mean | log1p | [1, 5.05e+05] | 120.92 | 0.00 | 1.58 | 0.000 | -1.65 | bytes_total (0.65) |  |
| flow_duration_mean | log1p | [0, 3.6e+09] | 2.14 | 0.11 | 0.27 | 0.000 | -1.77 | iat_mean (0.97) |  |
| flow_duration_var | log1p | [0, 3.24e+18] | 3.63 | 0.35 | 0.32 | 0.000 | -1.05 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 3.6e+09] | 6.70 | 0.12 | 0.19 | 0.000 | -1.75 | flow_duration_mean (0.97) |  |
| iat_var | log1p | [0, 3.24e+18] | 27.03 | 0.35 | 0.38 | 0.000 | -1.05 | flow_duration_var (0.99) |  |
| iat_max | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| active_flow_count | log1p | [1, 1.07e+04] | 33.27 | 0.00 | 1.35 | 0.000 | -1.37 | bytes_total (0.71) |  |
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
| out_degree | log1p | [1, 2.02e+03] | 22.08 | 0.00 | 2.79 | 0.000 | -1.53 | dst_ip_entropy (0.97) |  |
| in_degree | log1p | [0, 61] | 28.35 | 0.91 | 4.14 | 0.001 | -0.29 | neighbour_risk_fraction (0.22) |  |
| dst_ip_entropy | zscore | [-0, 7.58] | 2.44 | 0.54 | 2.44 | 0.000 | -0.66 | out_degree (0.97) | log1p would reduce |skew| 2.44 -> 1.06 |
| dst_port_entropy | zscore | [-0, 7.58] | 3.75 | 0.58 | 3.75 | 0.000 | -0.57 | dst_ip_entropy (0.75) | log1p would reduce |skew| 3.75 -> 1.45 |
| new_peer_count | log1p | [0, 2e+03] | 47.02 | 0.87 | 4.33 | 0.000 | -0.30 | out_degree (0.78) |  |
| neighbour_risk_fraction | unit | [0, 1] | 3.56 | 0.89 | 3.56 | 0.000 | 0.00 | in_degree (0.22) |  |
| local_clustering_coeff | unit | [0, 1] | 18.62 | 0.99 | 18.62 | 0.000 | 0.00 | in_degree (0.14) |  |
| reciprocity | drop (constant on active training rows (value 1); redundant with is_active) | [1, 1] | 0.00 | 0.00 | — | 0.000 | 0.00 | — |  |
| d_syn_ratio | unit | [-1, 1] | 4.40 | 0.54 | 4.40 | 0.000 | 0.00 | slope3_syn_ratio (0.73) |  |
| d_dst_port_entropy | zscore | [-4.31, 4.64] | 0.22 | 0.56 | 0.22 | 0.000 | -0.16 | d_out_degree (0.60) |  |
| d_out_degree | asinh | [-623, 1.09e+03] | 12.51 | 0.31 | 0.59 | 0.000 | -0.32 | d_dst_port_entropy (0.60) |  |
| d_new_peer_count | asinh | [-547, 1.11e+03] | 30.06 | 0.85 | 0.01 | 0.000 | -0.04 | d_out_degree (0.58) |  |
| d_iat_var | asinh | [-3.24e+18, 3.24e+18] | 7.49 | 0.30 | 0.21 | 0.000 | -0.16 | d_dst_port_entropy (0.51) |  |
| d_retrans_rate | drop (declared) | [0, 0] | 0.00 | 1.00 | — | 0.000 | 0.00 | — |  |
| slope3_syn_ratio | unit | [-0.5, 0.5] | 4.05 | 0.54 | 4.05 | 0.000 | 0.00 | d_syn_ratio (0.73) |  |
| slope3_dst_port_entropy | zscore | [-2.39, 2.76] | 0.14 | 0.56 | 0.14 | 0.000 | -0.15 | slope3_out_degree (0.58) |  |
| slope3_out_degree | asinh | [-381, 550] | 5.97 | 0.30 | 0.47 | 0.000 | -0.27 | slope3_dst_port_entropy (0.58) |  |
| slope3_iat_var | asinh | [-1.62e+18, 1.62e+18] | 5.68 | 0.30 | 0.21 | 0.000 | -0.17 | slope3_dst_port_entropy (0.49) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
