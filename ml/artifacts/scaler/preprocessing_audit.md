# Preprocessing audit

Fit rows: 500,000 (active: 26,417). Clip: ±10. Dropped: none. Flagged: 4.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 0.5] | 3.62 | 0.87 | 3.62 | 0.000 | 0.00 | slope3_syn_ratio (0.81) |  |
| ack_ratio | unit | [0, 0.5] | 0.43 | 0.50 | 0.43 | 0.000 | 0.00 | neighbour_risk_fraction (0.88) |  |
| rst_ratio | unit | [0, 0.0455] | 73.39 | 1.00 | 73.39 | 0.000 | 0.00 | iat_var (0.07) |  |
| fin_ratio | unit | [0, 0.5] | 18.12 | 0.97 | 18.12 | 0.000 | 0.00 | dst_port_entropy (0.16) |  |
| psh_ratio | unit | [0, 0.295] | 10.60 | 0.93 | 10.60 | 0.000 | 0.00 | flow_duration_var (0.55) |  |
| urg_ratio | unit | [0, 0.5] | 1.53 | 0.73 | 1.53 | 0.000 | 0.00 | ack_ratio (0.65) |  |
| bytes_total | log1p | [0, 6.15e+08] | 51.96 | 0.66 | 2.09 | 0.000 | -0.56 | active_flow_count (0.84) |  |
| bytes_up_down_ratio | log1p | [0, 2.26e+04] | 152.98 | 0.79 | 4.44 | 0.000 | -0.35 | bytes_total (0.43) |  |
| pkts_per_flow_mean | log1p | [0, 4.02e+04] | 84.05 | 0.47 | 1.64 | 0.000 | -0.85 | flow_duration_mean (0.86) |  |
| flow_duration_mean | log1p | [-1, 1.2e+08] | 8.49 | 0.47 | 1.62 | 0.000 | -0.70 | iat_max (1.00) |  |
| flow_duration_var | log1p | [0, 3.56e+15] | 5.64 | 0.74 | 2.23 | 0.000 | -0.45 | iat_var (0.99) |  |
| iat_mean | log1p | [-1, 6.41e+07] | 16.03 | 0.47 | 1.48 | 0.000 | -0.73 | flow_duration_mean (0.99) |  |
| iat_var | log1p | [0, 1.78e+15] | 29.94 | 0.74 | 2.18 | 0.000 | -0.46 | flow_duration_var (0.99) |  |
| iat_max | log1p | [-1, 1.2e+08] | 4.96 | 0.47 | 1.61 | 0.000 | -0.70 | flow_duration_mean (1.00) |  |
| active_flow_count | log1p | [0, 1.78e+04] | 38.63 | 0.47 | 2.56 | 0.000 | -0.67 | out_degree (0.86) |  |
| ttl_mean | zscore | [0, 255] | 0.99 | 0.00 | 0.99 | 0.000 | -1.32 | tcp_window_entropy (0.18) | log1p would reduce |skew| 0.99 -> 0.36 |
| ttl_var | log1p | [0, 9.38e+03] | 10.87 | 0.90 | 5.89 | 0.000 | -0.18 | tcp_window_entropy (0.19) |  |
| tcp_window_mean | log1p | [0, 6.55e+04] | 3.80 | 0.02 | 0.86 | 0.000 | -3.58 | payload_size_var (0.50) |  |
| tcp_window_entropy | zscore | [0, 2.69] | 1.53 | 0.32 | 1.53 | 0.000 | -0.89 | tcp_window_mean (0.35) | log1p would reduce |skew| 1.53 -> 0.85 |
| frag_flag_rate | unit | [0, 0.0119] | 38.59 | 1.00 | 38.59 | 0.000 | 0.00 | retrans_count (0.10) |  |
| payload_size_mean | log1p | [0, 3.76e+03] | 2.04 | 0.19 | 0.48 | 0.000 | -1.64 | payload_size_p95 (0.98) |  |
| payload_size_var | log1p | [0, 2.53e+07] | 18.59 | 0.25 | 0.49 | 0.000 | -1.50 | payload_size_p95 (0.94) |  |
| payload_size_p95 | log1p | [0, 7.45e+03] | 1.14 | 0.20 | 0.67 | 0.000 | -1.71 | payload_size_mean (0.98) |  |
| payload_size_entropy | zscore | [0, 2.52] | 0.14 | 0.25 | 0.14 | 0.000 | -1.32 | payload_size_var (0.81) |  |
| retrans_count | log1p | [0, 7.79e+04] | 93.20 | 0.71 | 2.80 | 0.000 | -0.48 | retrans_rate (0.69) |  |
| retrans_rate | unit | [0, 1] | 3.52 | 0.71 | 3.52 | 0.000 | 0.00 | retrans_count (0.69) |  |
| out_degree | log1p | [0, 194] | 11.90 | 0.47 | 2.44 | 0.000 | -0.75 | active_flow_count (0.86) |  |
| in_degree | log1p | [0, 114] | 9.89 | 0.31 | 2.06 | 0.000 | -1.04 | dst_ip_entropy (0.65) |  |
| dst_ip_entropy | zscore | [-0, 4.09] | 3.69 | 0.87 | 3.69 | 0.000 | -0.31 | out_degree (0.85) | log1p would reduce |skew| 3.69 -> 3.01 |
| dst_port_entropy | zscore | [-0, 5.9] | 2.58 | 0.77 | 2.57 | 0.000 | -0.46 | active_flow_count (0.64) | log1p would reduce |skew| 2.58 -> 1.79 |
| new_peer_count | log1p | [0, 171] | 18.72 | 0.80 | 4.37 | 0.000 | -0.40 | d_new_peer_count (0.65) |  |
| neighbour_risk_fraction | unit | [0, 1] | 0.35 | 0.53 | 0.35 | 0.000 | 0.00 | ack_ratio (0.88) |  |
| local_clustering_coeff | unit | [0, 1] | 15.88 | 0.97 | 15.88 | 0.000 | 0.00 | retrans_count (0.20) |  |
| reciprocity | unit | [0, 1] | 0.11 | 0.47 | 0.11 | 0.000 | 0.00 | neighbour_risk_fraction (0.82) |  |
| d_syn_ratio | unit | [-0.5, 0.5] | 1.01 | 0.83 | 1.01 | 0.000 | 0.00 | syn_ratio (0.75) |  |
| d_dst_port_entropy | zscore | [-3.68, 4.93] | 0.65 | 0.72 | 0.65 | 0.000 | -0.12 | slope3_dst_port_entropy (0.61) |  |
| d_out_degree | asinh | [-156, 185] | 2.74 | 0.48 | 0.45 | 0.000 | -0.18 | d_new_peer_count (0.73) |  |
| d_new_peer_count | asinh | [-138, 171] | 1.68 | 0.74 | 0.14 | 0.000 | -0.09 | d_out_degree (0.73) |  |
| d_iat_var | asinh | [-1.42e+15, 1e+15] | 2.51 | 0.69 | 0.00 | 0.000 | -0.06 | slope3_iat_var (0.41) |  |
| d_retrans_rate | unit | [-1, 1] | 0.26 | 0.59 | 0.26 | 0.000 | 0.00 | retrans_rate (0.63) |  |
| slope3_syn_ratio | unit | [-0.25, 0.25] | 2.05 | 0.84 | 2.05 | 0.000 | 0.00 | syn_ratio (0.81) |  |
| slope3_dst_port_entropy | zscore | [-1.73, 2.47] | 0.95 | 0.73 | 0.95 | 0.000 | -0.15 | dst_port_entropy (0.62) |  |
| slope3_out_degree | asinh | [-89, 94.5] | 0.56 | 0.53 | 0.58 | 0.000 | -0.16 | d_out_degree (0.43) |  |
| slope3_iat_var | asinh | [-8.9e+14, 8.9e+14] | 6.74 | 0.70 | 0.03 | 0.000 | -0.08 | d_iat_var (0.41) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
