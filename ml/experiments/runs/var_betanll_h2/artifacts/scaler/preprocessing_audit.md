# Preprocessing audit

Fit rows: 500,000 (active: 26,454). Clip: ±10. Dropped: none. Flagged: 4.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 0.5] | 3.63 | 0.87 | 3.63 | 0.000 | 0.00 | slope3_syn_ratio (0.81) |  |
| ack_ratio | unit | [0, 0.5] | 0.43 | 0.50 | 0.43 | 0.000 | 0.00 | neighbour_risk_fraction (0.88) |  |
| rst_ratio | unit | [0, 0.0233] | 42.96 | 1.00 | 42.96 | 0.000 | 0.00 | iat_var (0.10) |  |
| fin_ratio | unit | [0, 0.5] | 18.37 | 0.97 | 18.37 | 0.000 | 0.00 | dst_port_entropy (0.17) |  |
| psh_ratio | unit | [0, 0.143] | 6.91 | 0.93 | 6.91 | 0.000 | 0.00 | flow_duration_var (0.59) |  |
| urg_ratio | unit | [0, 0.5] | 1.53 | 0.73 | 1.53 | 0.000 | 0.00 | ack_ratio (0.65) |  |
| bytes_total | log1p | [0, 5.88e+08] | 60.68 | 0.66 | 2.10 | 0.000 | -0.56 | active_flow_count (0.84) |  |
| bytes_up_down_ratio | log1p | [0, 2.96e+03] | 49.89 | 0.79 | 4.33 | 0.000 | -0.35 | bytes_total (0.43) |  |
| pkts_per_flow_mean | log1p | [0, 1.89e+04] | 102.64 | 0.47 | 1.49 | 0.000 | -0.86 | flow_duration_mean (0.86) |  |
| flow_duration_mean | log1p | [0, 1.2e+08] | 8.56 | 0.48 | 1.65 | 0.000 | -0.70 | iat_max (1.00) |  |
| flow_duration_var | log1p | [0, 3.54e+15] | 5.70 | 0.74 | 2.26 | 0.000 | -0.45 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 6.14e+07] | 15.34 | 0.48 | 1.50 | 0.000 | -0.73 | flow_duration_mean (0.99) |  |
| iat_var | log1p | [0, 1.78e+15] | 33.81 | 0.74 | 2.20 | 0.000 | -0.46 | flow_duration_var (0.99) |  |
| iat_max | log1p | [0, 1.2e+08] | 5.01 | 0.48 | 1.64 | 0.000 | -0.70 | flow_duration_mean (1.00) |  |
| active_flow_count | log1p | [0, 1.78e+04] | 38.61 | 0.47 | 2.60 | 0.000 | -0.66 | out_degree (0.86) |  |
| ttl_mean | zscore | [0, 255] | 0.98 | 0.00 | 0.98 | 0.000 | -1.32 | tcp_window_mean (0.17) | log1p would reduce |skew| 0.98 -> 0.14 |
| ttl_var | log1p | [0, 9.52e+03] | 10.72 | 0.90 | 5.73 | 0.000 | -0.18 | tcp_window_entropy (0.20) |  |
| tcp_window_mean | log1p | [0, 6.55e+04] | 3.76 | 0.02 | 0.82 | 0.000 | -3.59 | payload_size_var (0.48) |  |
| tcp_window_entropy | zscore | [0, 2.76] | 1.57 | 0.32 | 1.57 | 0.000 | -0.88 | tcp_window_mean (0.34) | log1p would reduce |skew| 1.57 -> 0.87 |
| frag_flag_rate | unit | [0, 0.0182] | 38.54 | 1.00 | 38.54 | 0.000 | 0.00 | active_flow_count (0.10) |  |
| payload_size_mean | log1p | [0, 3.02e+03] | 2.05 | 0.19 | 0.48 | 0.000 | -1.64 | payload_size_p95 (0.98) |  |
| payload_size_var | log1p | [0, 1.22e+07] | 5.78 | 0.24 | 0.50 | 0.000 | -1.50 | payload_size_p95 (0.94) |  |
| payload_size_p95 | log1p | [0, 7.3e+03] | 1.10 | 0.19 | 0.68 | 0.000 | -1.71 | payload_size_mean (0.98) |  |
| payload_size_entropy | zscore | [0, 2.55] | 0.14 | 0.24 | 0.14 | 0.000 | -1.32 | payload_size_var (0.81) |  |
| retrans_count | log1p | [0, 7.79e+04] | 44.20 | 0.71 | 3.02 | 0.000 | -0.48 | retrans_rate (0.70) |  |
| retrans_rate | unit | [0, 1] | 3.38 | 0.71 | 3.38 | 0.000 | 0.00 | retrans_count (0.70) |  |
| out_degree | log1p | [0, 182] | 11.26 | 0.47 | 2.51 | 0.000 | -0.74 | active_flow_count (0.86) |  |
| in_degree | log1p | [0, 121] | 11.36 | 0.31 | 2.14 | 0.000 | -1.04 | dst_ip_entropy (0.64) |  |
| dst_ip_entropy | zscore | [-0, 4.08] | 3.79 | 0.88 | 3.79 | 0.000 | -0.31 | out_degree (0.85) | log1p would reduce |skew| 3.79 -> 3.07 |
| dst_port_entropy | zscore | [-0, 5.8] | 2.55 | 0.77 | 2.54 | 0.000 | -0.46 | active_flow_count (0.63) | log1p would reduce |skew| 2.55 -> 1.81 |
| new_peer_count | log1p | [0, 149] | 16.77 | 0.80 | 4.45 | 0.000 | -0.40 | out_degree (0.66) |  |
| neighbour_risk_fraction | unit | [0, 1] | 0.33 | 0.53 | 0.33 | 0.000 | 0.00 | ack_ratio (0.88) |  |
| local_clustering_coeff | unit | [0, 1] | 15.07 | 0.97 | 15.07 | 0.000 | 0.00 | retrans_count (0.19) |  |
| reciprocity | unit | [0, 1] | 0.10 | 0.47 | 0.10 | 0.000 | 0.00 | neighbour_risk_fraction (0.83) |  |
| d_syn_ratio | unit | [-0.5, 0.5] | 1.10 | 0.83 | 1.10 | 0.000 | 0.00 | syn_ratio (0.76) |  |
| d_dst_port_entropy | zscore | [-3.54, 5.02] | 0.68 | 0.72 | 0.68 | 0.000 | -0.12 | slope3_dst_port_entropy (0.62) |  |
| d_out_degree | asinh | [-119, 175] | 2.72 | 0.48 | 0.35 | 0.000 | -0.19 | d_new_peer_count (0.73) |  |
| d_new_peer_count | asinh | [-123, 149] | 1.58 | 0.74 | 0.00 | 0.000 | -0.09 | d_out_degree (0.73) |  |
| d_iat_var | asinh | [-1.42e+15, 1e+15] | 5.78 | 0.69 | 0.03 | 0.000 | -0.06 | d_dst_port_entropy (0.40) |  |
| d_retrans_rate | unit | [-1, 0.927] | 0.19 | 0.59 | 0.19 | 0.000 | 0.00 | retrans_rate (0.61) |  |
| slope3_syn_ratio | unit | [-0.25, 0.25] | 2.03 | 0.84 | 2.03 | 0.000 | 0.00 | syn_ratio (0.81) |  |
| slope3_dst_port_entropy | zscore | [-2.17, 2.51] | 0.89 | 0.74 | 0.89 | 0.000 | -0.15 | d_dst_port_entropy (0.62) |  |
| slope3_out_degree | asinh | [-84, 85.5] | 1.96 | 0.53 | 0.23 | 0.000 | -0.16 | new_peer_count (0.46) |  |
| slope3_iat_var | asinh | [-8.9e+14, 8.9e+14] | 2.88 | 0.71 | 0.02 | 0.000 | -0.08 | d_iat_var (0.38) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
