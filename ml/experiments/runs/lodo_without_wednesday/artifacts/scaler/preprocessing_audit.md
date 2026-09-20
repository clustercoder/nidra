# Preprocessing audit

Fit rows: 500,000 (active: 25,707). Clip: ±10. Dropped: none. Flagged: 4.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 0.5] | 3.48 | 0.86 | 3.48 | 0.000 | 0.00 | slope3_syn_ratio (0.83) |  |
| ack_ratio | unit | [0, 0.5] | 0.38 | 0.49 | 0.38 | 0.000 | 0.00 | neighbour_risk_fraction (0.90) |  |
| rst_ratio | unit | [0, 0.0357] | 62.45 | 1.00 | 62.45 | 0.000 | 0.00 | iat_var (0.07) |  |
| fin_ratio | unit | [0, 0.5] | 22.36 | 0.97 | 22.36 | 0.000 | 0.00 | iat_mean (0.10) |  |
| psh_ratio | unit | [0, 0.167] | 6.74 | 0.94 | 6.74 | 0.000 | 0.00 | bytes_total (0.60) |  |
| urg_ratio | unit | [0, 0.5] | 1.45 | 0.71 | 1.45 | 0.000 | 0.00 | ack_ratio (0.66) |  |
| bytes_total | log1p | [0, 5.38e+08] | 89.07 | 0.69 | 2.12 | 0.000 | -0.54 | flow_duration_var (0.83) |  |
| bytes_up_down_ratio | log1p | [0, 1.1e+04] | 139.11 | 0.81 | 4.44 | 0.000 | -0.33 | bytes_total (0.42) |  |
| pkts_per_flow_mean | log1p | [0, 1.06e+05] | 157.38 | 0.47 | 1.51 | 0.000 | -0.86 | flow_duration_mean (0.86) |  |
| flow_duration_mean | log1p | [0, 1.2e+08] | 9.33 | 0.47 | 1.60 | 0.000 | -0.71 | iat_max (0.99) |  |
| flow_duration_var | log1p | [0, 3.56e+15] | 5.88 | 0.73 | 2.21 | 0.000 | -0.46 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 6.14e+07] | 17.79 | 0.47 | 1.45 | 0.000 | -0.74 | flow_duration_mean (0.99) |  |
| iat_var | log1p | [0, 1.45e+15] | 27.47 | 0.73 | 2.15 | 0.000 | -0.47 | flow_duration_var (0.99) |  |
| iat_max | log1p | [0, 1.2e+08] | 5.08 | 0.47 | 1.60 | 0.000 | -0.70 | flow_duration_mean (0.99) |  |
| active_flow_count | log1p | [0, 1.86e+03] | 13.79 | 0.47 | 2.34 | 0.000 | -0.68 | out_degree (0.89) |  |
| ttl_mean | zscore | [0, 255] | 0.93 | 0.00 | 0.93 | 0.000 | -1.33 | tcp_window_entropy (0.14) | log1p would reduce |skew| 0.93 -> 0.25 |
| ttl_var | log1p | [0, 9.22e+03] | 10.85 | 0.90 | 5.75 | 0.000 | -0.19 | tcp_window_entropy (0.19) |  |
| tcp_window_mean | log1p | [0, 6.55e+04] | 3.62 | 0.02 | 0.96 | 0.000 | -3.51 | payload_size_var (0.48) |  |
| tcp_window_entropy | zscore | [0, 2.74] | 1.56 | 0.32 | 1.56 | 0.000 | -0.89 | tcp_window_mean (0.36) | log1p would reduce |skew| 1.56 -> 0.87 |
| frag_flag_rate | unit | [0, 0.0182] | 39.09 | 1.00 | 39.09 | 0.000 | 0.00 | retrans_count (0.10) |  |
| payload_size_mean | log1p | [0, 3.02e+03] | 1.97 | 0.19 | 0.52 | 0.000 | -1.65 | payload_size_p95 (0.98) |  |
| payload_size_var | log1p | [0, 9.63e+06] | 4.84 | 0.24 | 0.56 | 0.000 | -1.53 | payload_size_p95 (0.95) |  |
| payload_size_p95 | log1p | [0, 7.3e+03] | 1.06 | 0.19 | 0.72 | 0.000 | -1.73 | payload_size_mean (0.98) |  |
| payload_size_entropy | zscore | [0, 2.55] | 0.12 | 0.24 | 0.12 | 0.000 | -1.34 | payload_size_var (0.80) |  |
| retrans_count | log1p | [0, 7.79e+04] | 63.48 | 0.71 | 2.63 | 0.000 | -0.48 | retrans_rate (0.69) |  |
| retrans_rate | unit | [0, 1] | 3.33 | 0.71 | 3.33 | 0.000 | 0.00 | retrans_count (0.69) |  |
| out_degree | log1p | [0, 207] | 11.12 | 0.47 | 2.51 | 0.000 | -0.74 | active_flow_count (0.89) |  |
| in_degree | log1p | [0, 121] | 10.45 | 0.29 | 2.16 | 0.000 | -1.05 | dst_ip_entropy (0.66) |  |
| dst_ip_entropy | zscore | [-0, 4.07] | 3.71 | 0.88 | 3.71 | 0.000 | -0.31 | out_degree (0.86) | log1p would reduce |skew| 3.71 -> 3.01 |
| dst_port_entropy | zscore | [-0, 5.72] | 2.47 | 0.76 | 2.47 | 0.000 | -0.47 | active_flow_count (0.67) | log1p would reduce |skew| 2.47 -> 1.74 |
| new_peer_count | log1p | [0, 204] | 21.35 | 0.81 | 4.38 | 0.000 | -0.39 | out_degree (0.65) |  |
| neighbour_risk_fraction | unit | [0, 1] | 0.28 | 0.52 | 0.28 | 0.000 | 0.00 | ack_ratio (0.90) |  |
| local_clustering_coeff | unit | [0, 1] | 14.66 | 0.97 | 14.66 | 0.000 | 0.00 | retrans_count (0.18) |  |
| reciprocity | unit | [0, 1] | 0.14 | 0.47 | 0.14 | 0.000 | 0.00 | neighbour_risk_fraction (0.84) |  |
| d_syn_ratio | unit | [-0.5, 0.5] | 1.11 | 0.82 | 1.11 | 0.000 | 0.00 | syn_ratio (0.76) |  |
| d_dst_port_entropy | zscore | [-3.95, 5.72] | 0.72 | 0.71 | 0.72 | 0.000 | -0.14 | slope3_dst_port_entropy (0.63) |  |
| d_out_degree | asinh | [-151, 206] | 2.06 | 0.47 | 0.53 | 0.000 | -0.19 | d_new_peer_count (0.71) |  |
| d_new_peer_count | asinh | [-170, 203] | 0.52 | 0.74 | 0.26 | 0.000 | -0.09 | d_out_degree (0.71) |  |
| d_iat_var | asinh | [-9.45e+14, 1.45e+15] | 7.96 | 0.68 | 0.06 | 0.000 | -0.07 | d_dst_port_entropy (0.40) |  |
| d_retrans_rate | unit | [-1, 1] | 0.17 | 0.59 | 0.17 | 0.000 | 0.00 | retrans_rate (0.64) |  |
| slope3_syn_ratio | unit | [-0.25, 0.25] | 2.13 | 0.83 | 2.13 | 0.000 | 0.00 | syn_ratio (0.83) |  |
| slope3_dst_port_entropy | zscore | [-1.98, 2.86] | 1.19 | 0.72 | 1.18 | 0.000 | -0.17 | dst_port_entropy (0.66) |  |
| slope3_out_degree | asinh | [-71, 102] | 3.01 | 0.52 | 0.42 | 0.000 | -0.16 | d_out_degree (0.43) |  |
| slope3_iat_var | asinh | [-4.54e+14, 7.26e+14] | 7.60 | 0.69 | 0.01 | 0.000 | -0.09 | d_iat_var (0.40) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
