# Preprocessing audit

Fit rows: 500,000 (active: 26,176). Clip: ±10. Dropped: none. Flagged: 4.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 0.5] | 3.68 | 0.87 | 3.68 | 0.000 | 0.00 | slope3_syn_ratio (0.80) |  |
| ack_ratio | unit | [0, 0.5] | 0.44 | 0.50 | 0.44 | 0.000 | 0.00 | neighbour_risk_fraction (0.88) |  |
| rst_ratio | unit | [0, 0.0333] | 70.72 | 1.00 | 70.72 | 0.000 | 0.00 | iat_var (0.07) |  |
| fin_ratio | unit | [0, 0.5] | 16.83 | 0.97 | 16.83 | 0.000 | 0.00 | dst_port_entropy (0.14) |  |
| psh_ratio | unit | [0, 0.295] | 9.20 | 0.93 | 9.20 | 0.000 | 0.00 | flow_duration_var (0.57) |  |
| urg_ratio | unit | [0, 0.5] | 1.53 | 0.73 | 1.53 | 0.000 | 0.00 | ack_ratio (0.66) |  |
| bytes_total | log1p | [0, 6.15e+08] | 60.19 | 0.66 | 2.08 | 0.000 | -0.56 | active_flow_count (0.84) |  |
| bytes_up_down_ratio | log1p | [0, 1.28e+04] | 115.82 | 0.78 | 4.37 | 0.000 | -0.35 | bytes_total (0.43) |  |
| pkts_per_flow_mean | log1p | [0, 4.67e+04] | 87.63 | 0.48 | 1.58 | 0.000 | -0.85 | flow_duration_mean (0.86) |  |
| flow_duration_mean | log1p | [0, 1.2e+08] | 8.59 | 0.48 | 1.60 | 0.000 | -0.70 | iat_max (1.00) |  |
| flow_duration_var | log1p | [0, 3.59e+15] | 5.56 | 0.74 | 2.22 | 0.000 | -0.46 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 6.14e+07] | 16.05 | 0.48 | 1.46 | 0.000 | -0.73 | flow_duration_mean (0.99) |  |
| iat_var | log1p | [0, 1.78e+15] | 33.26 | 0.74 | 2.16 | 0.000 | -0.46 | flow_duration_var (0.99) |  |
| iat_max | log1p | [0, 1.2e+08] | 4.97 | 0.48 | 1.60 | 0.000 | -0.70 | flow_duration_mean (1.00) |  |
| active_flow_count | log1p | [0, 1.78e+04] | 38.31 | 0.48 | 2.57 | 0.000 | -0.67 | out_degree (0.86) |  |
| ttl_mean | zscore | [0, 255] | 0.98 | 0.00 | 0.98 | 0.000 | -1.32 | tcp_window_mean (0.18) | log1p would reduce |skew| 0.98 -> 0.30 |
| ttl_var | log1p | [0, 9.41e+03] | 11.20 | 0.90 | 5.84 | 0.000 | -0.19 | tcp_window_entropy (0.19) |  |
| tcp_window_mean | log1p | [0, 6.55e+04] | 3.70 | 0.02 | 0.84 | 0.000 | -3.57 | payload_size_var (0.48) |  |
| tcp_window_entropy | zscore | [0, 2.74] | 1.59 | 0.32 | 1.59 | 0.000 | -0.88 | tcp_window_mean (0.34) | log1p would reduce |skew| 1.59 -> 0.88 |
| frag_flag_rate | unit | [0, 0.0154] | 39.30 | 1.00 | 39.30 | 0.000 | 0.00 | retrans_count (0.10) |  |
| payload_size_mean | log1p | [0, 3.34e+03] | 2.06 | 0.19 | 0.49 | 0.000 | -1.64 | payload_size_p95 (0.98) |  |
| payload_size_var | log1p | [0, 1.11e+07] | 5.41 | 0.24 | 0.51 | 0.000 | -1.51 | payload_size_p95 (0.94) |  |
| payload_size_p95 | log1p | [0, 8.76e+03] | 1.16 | 0.19 | 0.68 | 0.000 | -1.71 | payload_size_mean (0.98) |  |
| payload_size_entropy | zscore | [0, 2.5] | 0.13 | 0.24 | 0.13 | 0.000 | -1.32 | payload_size_var (0.81) |  |
| retrans_count | log1p | [0, 6.79e+04] | 46.20 | 0.71 | 2.93 | 0.000 | -0.48 | retrans_rate (0.69) |  |
| retrans_rate | unit | [0, 1] | 3.53 | 0.71 | 3.53 | 0.000 | 0.00 | retrans_count (0.69) |  |
| out_degree | log1p | [0, 223] | 11.64 | 0.48 | 2.47 | 0.000 | -0.74 | active_flow_count (0.86) |  |
| in_degree | log1p | [0, 132] | 10.56 | 0.30 | 2.09 | 0.000 | -1.04 | dst_ip_entropy (0.65) |  |
| dst_ip_entropy | zscore | [-0, 4.09] | 3.69 | 0.88 | 3.69 | 0.000 | -0.31 | out_degree (0.85) | log1p would reduce |skew| 3.69 -> 3.02 |
| dst_port_entropy | zscore | [-0, 6.47] | 2.63 | 0.77 | 2.60 | 0.000 | -0.47 | active_flow_count (0.64) | log1p would reduce |skew| 2.63 -> 1.81 |
| new_peer_count | log1p | [0, 174] | 18.51 | 0.80 | 4.46 | 0.000 | -0.40 | d_new_peer_count (0.65) |  |
| neighbour_risk_fraction | unit | [0, 1] | 0.35 | 0.53 | 0.35 | 0.000 | 0.00 | ack_ratio (0.88) |  |
| local_clustering_coeff | unit | [0, 1] | 15.51 | 0.97 | 15.51 | 0.000 | 0.00 | retrans_count (0.19) |  |
| reciprocity | unit | [0, 1] | 0.10 | 0.48 | 0.10 | 0.000 | 0.00 | neighbour_risk_fraction (0.83) |  |
| d_syn_ratio | unit | [-0.5, 0.5] | 0.90 | 0.82 | 0.90 | 0.000 | 0.00 | syn_ratio (0.74) |  |
| d_dst_port_entropy | zscore | [-3.66, 5.72] | 0.72 | 0.72 | 0.71 | 0.000 | -0.12 | slope3_dst_port_entropy (0.63) |  |
| d_out_degree | asinh | [-156, 223] | 2.54 | 0.48 | 0.44 | 0.000 | -0.18 | d_new_peer_count (0.73) |  |
| d_new_peer_count | asinh | [-104, 173] | 2.47 | 0.74 | 0.17 | 0.000 | -0.09 | d_out_degree (0.73) |  |
| d_iat_var | asinh | [-1.42e+15, 1e+15] | 1.30 | 0.69 | 0.05 | 0.000 | -0.07 | slope3_iat_var (0.43) |  |
| d_retrans_rate | unit | [-1, 1] | 0.38 | 0.59 | 0.38 | 0.000 | 0.00 | retrans_rate (0.63) |  |
| slope3_syn_ratio | unit | [-0.25, 0.25] | 2.02 | 0.84 | 2.02 | 0.000 | 0.00 | syn_ratio (0.80) |  |
| slope3_dst_port_entropy | zscore | [-1.65, 3.23] | 1.10 | 0.74 | 1.05 | 0.000 | -0.15 | d_dst_port_entropy (0.63) |  |
| slope3_out_degree | asinh | [-89, 111] | 2.47 | 0.53 | 0.29 | 0.000 | -0.16 | d_out_degree (0.46) |  |
| slope3_iat_var | asinh | [-8.9e+14, 8.9e+14] | 4.67 | 0.70 | 0.01 | 0.000 | -0.08 | d_iat_var (0.43) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
