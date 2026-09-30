# Preprocessing audit

Fit rows: 500,000 (active: 26,007). Clip: ±10. Dropped: none. Flagged: 4.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 0.5] | 3.70 | 0.87 | 3.70 | 0.000 | 0.00 | slope3_syn_ratio (0.80) |  |
| ack_ratio | unit | [0, 0.5] | 0.43 | 0.50 | 0.43 | 0.000 | 0.00 | neighbour_risk_fraction (0.87) |  |
| rst_ratio | unit | [0, 0.0355] | 66.41 | 1.00 | 66.41 | 0.000 | 0.00 | iat_var (0.08) |  |
| fin_ratio | unit | [0, 0.5] | 18.01 | 0.97 | 18.01 | 0.000 | 0.00 | dst_port_entropy (0.16) |  |
| psh_ratio | unit | [0, 0.333] | 10.30 | 0.93 | 10.30 | 0.000 | 0.00 | flow_duration_var (0.56) |  |
| urg_ratio | unit | [0, 0.5] | 1.52 | 0.73 | 1.52 | 0.000 | 0.00 | ack_ratio (0.65) |  |
| bytes_total | log1p | [0, 5.94e+08] | 59.34 | 0.66 | 2.08 | 0.000 | -0.56 | active_flow_count (0.84) |  |
| bytes_up_down_ratio | log1p | [0, 2.26e+04] | 150.48 | 0.78 | 4.45 | 0.000 | -0.35 | bytes_total (0.42) |  |
| pkts_per_flow_mean | log1p | [0, 6.78e+04] | 127.36 | 0.47 | 1.54 | 0.000 | -0.86 | flow_duration_mean (0.86) |  |
| flow_duration_mean | log1p | [-1, 1.2e+08] | 8.46 | 0.47 | 1.59 | 0.000 | -0.70 | iat_max (1.00) |  |
| flow_duration_var | log1p | [0, 3.6e+15] | 5.52 | 0.74 | 2.20 | 0.000 | -0.46 | iat_var (0.99) |  |
| iat_mean | log1p | [-1, 4.29e+07] | 14.67 | 0.47 | 1.45 | 0.000 | -0.73 | flow_duration_mean (0.99) |  |
| iat_var | log1p | [0, 1.78e+15] | 29.60 | 0.74 | 2.15 | 0.000 | -0.47 | flow_duration_var (0.99) |  |
| iat_max | log1p | [-1, 1.2e+08] | 4.86 | 0.47 | 1.59 | 0.000 | -0.70 | flow_duration_mean (1.00) |  |
| active_flow_count | log1p | [0, 1.78e+04] | 38.19 | 0.47 | 2.57 | 0.000 | -0.67 | out_degree (0.86) |  |
| ttl_mean | zscore | [0, 255] | 1.01 | 0.00 | 1.01 | 0.000 | -1.32 | tcp_window_mean (0.18) | log1p would reduce |skew| 1.01 -> 0.32 |
| ttl_var | log1p | [0, 9.7e+03] | 10.83 | 0.90 | 5.85 | 0.000 | -0.18 | tcp_window_entropy (0.20) |  |
| tcp_window_mean | log1p | [0, 6.55e+04] | 3.79 | 0.02 | 0.83 | 0.000 | -3.59 | payload_size_var (0.50) |  |
| tcp_window_entropy | zscore | [0, 2.74] | 1.51 | 0.32 | 1.51 | 0.000 | -0.89 | tcp_window_mean (0.35) | log1p would reduce |skew| 1.51 -> 0.84 |
| frag_flag_rate | unit | [0, 0.0197] | 35.08 | 1.00 | 35.08 | 0.000 | 0.00 | retrans_count (0.10) |  |
| payload_size_mean | log1p | [0, 3.02e+03] | 2.00 | 0.19 | 0.49 | 0.000 | -1.63 | payload_size_p95 (0.98) |  |
| payload_size_var | log1p | [0, 1.22e+07] | 6.04 | 0.24 | 0.50 | 0.000 | -1.50 | payload_size_p95 (0.94) |  |
| payload_size_p95 | log1p | [0, 7.3e+03] | 1.07 | 0.19 | 0.68 | 0.000 | -1.71 | payload_size_mean (0.98) |  |
| payload_size_entropy | zscore | [0, 2.55] | 0.13 | 0.24 | 0.13 | 0.000 | -1.32 | payload_size_var (0.81) |  |
| retrans_count | log1p | [0, 7.79e+04] | 68.35 | 0.71 | 2.84 | 0.000 | -0.48 | retrans_rate (0.69) |  |
| retrans_rate | unit | [0, 1] | 3.49 | 0.71 | 3.49 | 0.000 | 0.00 | retrans_count (0.69) |  |
| out_degree | log1p | [0, 194] | 10.99 | 0.47 | 2.49 | 0.000 | -0.75 | active_flow_count (0.86) |  |
| in_degree | log1p | [0, 107] | 9.51 | 0.30 | 2.09 | 0.000 | -1.04 | dst_ip_entropy (0.65) |  |
| dst_ip_entropy | zscore | [-0, 3.88] | 3.68 | 0.87 | 3.68 | 0.000 | -0.32 | out_degree (0.86) | log1p would reduce |skew| 3.68 -> 3.00 |
| dst_port_entropy | zscore | [-0, 5.72] | 2.49 | 0.77 | 2.48 | 0.000 | -0.47 | active_flow_count (0.63) | log1p would reduce |skew| 2.49 -> 1.78 |
| new_peer_count | log1p | [0, 119] | 15.97 | 0.80 | 4.46 | 0.000 | -0.40 | out_degree (0.65) |  |
| neighbour_risk_fraction | unit | [0, 1] | 0.34 | 0.53 | 0.34 | 0.000 | 0.00 | ack_ratio (0.87) |  |
| local_clustering_coeff | unit | [0, 1] | 16.04 | 0.97 | 16.04 | 0.000 | 0.00 | retrans_count (0.20) |  |
| reciprocity | unit | [0, 1] | 0.12 | 0.47 | 0.12 | 0.000 | 0.00 | neighbour_risk_fraction (0.82) |  |
| d_syn_ratio | unit | [-0.5, 0.5] | 0.95 | 0.82 | 0.95 | 0.000 | 0.00 | syn_ratio (0.74) |  |
| d_dst_port_entropy | zscore | [-3.95, 5.72] | 0.77 | 0.72 | 0.76 | 0.000 | -0.12 | slope3_dst_port_entropy (0.63) |  |
| d_out_degree | asinh | [-180, 178] | 2.72 | 0.48 | 0.42 | 0.000 | -0.18 | d_new_peer_count (0.72) |  |
| d_new_peer_count | asinh | [-135, 119] | 1.14 | 0.74 | 0.10 | 0.000 | -0.09 | d_out_degree (0.72) |  |
| d_iat_var | asinh | [-1.45e+15, 1.12e+15] | 2.57 | 0.69 | 0.05 | 0.000 | -0.06 | slope3_iat_var (0.40) |  |
| d_retrans_rate | unit | [-0.773, 1] | 0.30 | 0.58 | 0.30 | 0.000 | 0.00 | retrans_rate (0.61) |  |
| slope3_syn_ratio | unit | [-0.25, 0.25] | 2.07 | 0.84 | 2.07 | 0.000 | 0.00 | syn_ratio (0.80) |  |
| slope3_dst_port_entropy | zscore | [-2.17, 2.86] | 0.99 | 0.73 | 0.98 | 0.000 | -0.15 | d_dst_port_entropy (0.63) |  |
| slope3_out_degree | asinh | [-84, 94.5] | 2.07 | 0.53 | 0.32 | 0.000 | -0.16 | d_out_degree (0.45) |  |
| slope3_iat_var | asinh | [-8.9e+14, 8.9e+14] | 2.62 | 0.70 | 0.01 | 0.000 | -0.08 | d_iat_var (0.40) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
