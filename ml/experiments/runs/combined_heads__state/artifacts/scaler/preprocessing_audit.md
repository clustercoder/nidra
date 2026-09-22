# Preprocessing audit

Fit rows: 500,000 (active: 51,187). Clip: ±10. Dropped: iat_max, ttl_mean, ttl_var, tcp_window_mean, tcp_window_entropy, frag_flag_rate, payload_size_mean, payload_size_var, payload_size_p95, payload_size_entropy, retrans_count, retrans_rate, d_retrans_rate. Flagged: 2.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 1] | 3.63 | 0.64 | 3.63 | 0.000 | 0.00 | slope3_syn_ratio (0.65) |  |
| ack_ratio | unit | [0, 1] | 1.99 | 0.55 | 1.99 | 0.000 | 0.00 | neighbour_risk_fraction (0.76) |  |
| rst_ratio | unit | [0, 1] | 13.41 | 0.88 | 13.41 | 0.000 | 0.00 | syn_ratio (0.25) |  |
| fin_ratio | unit | [0, 0.667] | 3.56 | 0.74 | 3.56 | 0.000 | 0.00 | psh_ratio (0.76) |  |
| psh_ratio | unit | [0, 1] | 2.92 | 0.71 | 2.92 | 0.000 | 0.00 | fin_ratio (0.76) |  |
| urg_ratio | unit | [0, 0.5] | 3.75 | 0.92 | 3.75 | 0.000 | 0.00 | ack_ratio (0.66) |  |
| bytes_total | log1p | [0, 1.29e+09] | 62.31 | 0.20 | 0.15 | 0.000 | -1.56 | active_flow_count (0.74) |  |
| bytes_up_down_ratio | log1p | [0, 1.28e+04] | 184.88 | 0.42 | 3.36 | 0.000 | -0.64 | iat_mean (0.37) |  |
| pkts_per_flow_mean | log1p | [0, 1.3e+05] | 83.55 | 0.14 | 1.35 | 0.000 | -1.29 | flow_duration_mean (0.75) |  |
| flow_duration_mean | log1p | [0, 3.6e+09] | 2.81 | 0.22 | 0.14 | 0.000 | -1.27 | iat_mean (0.98) |  |
| flow_duration_var | log1p | [0, 3.24e+18] | 4.50 | 0.46 | 0.67 | 0.000 | -0.86 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 3.54e+09] | 7.85 | 0.22 | 0.18 | 0.000 | -1.28 | flow_duration_mean (0.98) |  |
| iat_var | log1p | [0, 3.24e+18] | 34.05 | 0.46 | 0.71 | 0.000 | -0.86 | flow_duration_var (0.99) |  |
| iat_max | drop (declared) | [0, 1.2e+08] | 9.05 | 0.84 | — | 0.000 | 0.00 | — |  |
| active_flow_count | log1p | [0, 1.78e+04] | 44.63 | 0.14 | 1.35 | 0.000 | -1.10 | bytes_total (0.74) |  |
| ttl_mean | drop (declared) | [0, 255] | 2.40 | 0.70 | — | 0.000 | 0.00 | — |  |
| ttl_var | drop (declared) | [0, 9.22e+03] | 21.42 | 0.97 | — | 0.000 | 0.00 | — |  |
| tcp_window_mean | drop (declared) | [0, 6.55e+04] | 6.83 | 0.70 | — | 0.000 | 0.00 | — |  |
| tcp_window_entropy | drop (declared) | [0, 2.74] | 3.34 | 0.79 | — | 0.000 | 0.00 | — |  |
| frag_flag_rate | drop (declared) | [0, 0.0168] | 70.06 | 1.00 | — | 0.000 | 0.00 | — |  |
| payload_size_mean | drop (declared) | [0, 3.76e+03] | 4.25 | 0.75 | — | 0.000 | 0.00 | — |  |
| payload_size_var | drop (declared) | [0, 2.53e+07] | 40.55 | 0.77 | — | 0.000 | 0.00 | — |  |
| payload_size_p95 | drop (declared) | [0, 7.45e+03] | 3.01 | 0.76 | — | 0.000 | 0.00 | — |  |
| payload_size_entropy | drop (declared) | [0, 2.52] | 1.94 | 0.77 | — | 0.000 | 0.00 | — |  |
| retrans_count | drop (declared) | [0, 4.52e+04] | 114.00 | 0.91 | — | 0.000 | 0.00 | — |  |
| retrans_rate | drop (declared) | [0, 1] | 6.82 | 0.91 | — | 0.000 | 0.00 | — |  |
| out_degree | log1p | [0, 1.12e+03] | 16.02 | 0.14 | 2.08 | 0.000 | -1.17 | dst_ip_entropy (0.94) |  |
| in_degree | log1p | [0, 118] | 17.41 | 0.72 | 2.89 | 0.000 | -0.52 | neighbour_risk_fraction (0.28) |  |
| dst_ip_entropy | zscore | [-0, 7.01] | 2.50 | 0.63 | 2.50 | 0.000 | -0.56 | out_degree (0.94) | log1p would reduce |skew| 2.50 -> 1.39 |
| dst_port_entropy | zscore | [-0, 7.02] | 3.52 | 0.64 | 3.52 | 0.000 | -0.54 | dst_ip_entropy (0.63) | log1p would reduce |skew| 3.52 -> 1.52 |
| new_peer_count | log1p | [0, 1.09e+03] | 27.95 | 0.83 | 3.86 | 0.000 | -0.34 | dst_ip_entropy (0.75) |  |
| neighbour_risk_fraction | unit | [0, 1] | 1.76 | 0.78 | 1.76 | 0.000 | 0.00 | ack_ratio (0.76) |  |
| local_clustering_coeff | unit | [0, 1] | 17.19 | 0.98 | 17.19 | 0.000 | 0.00 | in_degree (0.10) |  |
| reciprocity | unit | [0, 1] | 2.05 | 0.14 | 2.05 | 0.000 | 0.00 | bytes_total (0.64) |  |
| d_syn_ratio | unit | [-1, 1] | 2.52 | 0.62 | 2.52 | 0.000 | 0.00 | slope3_syn_ratio (0.70) |  |
| d_dst_port_entropy | zscore | [-4.12, 5.72] | 0.47 | 0.61 | 0.46 | 0.000 | -0.14 | slope3_dst_port_entropy (0.59) |  |
| d_out_degree | asinh | [-374, 327] | 1.98 | 0.36 | 0.47 | 0.000 | -0.27 | d_new_peer_count (0.62) |  |
| d_new_peer_count | asinh | [-274, 236] | 3.16 | 0.80 | 0.07 | 0.000 | -0.05 | d_out_degree (0.62) |  |
| d_iat_var | asinh | [-3.16e+18, 3.24e+18] | 10.77 | 0.41 | 0.13 | 0.000 | -0.14 | d_dst_port_entropy (0.45) |  |
| d_retrans_rate | drop (declared) | [-0.951, 0.95] | 0.59 | 0.87 | — | 0.000 | 0.00 | — |  |
| slope3_syn_ratio | unit | [-0.5, 0.5] | 3.17 | 0.62 | 3.17 | 0.000 | 0.00 | d_syn_ratio (0.70) |  |
| slope3_dst_port_entropy | zscore | [-2.21, 2.86] | 0.50 | 0.61 | 0.50 | 0.000 | -0.15 | d_dst_port_entropy (0.59) |  |
| slope3_out_degree | asinh | [-381, 206] | 3.61 | 0.36 | 0.35 | 0.000 | -0.23 | d_out_degree (0.52) |  |
| slope3_iat_var | asinh | [-1.62e+18, 1.62e+18] | 1.01 | 0.42 | 0.13 | 0.000 | -0.14 | slope3_dst_port_entropy (0.44) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
