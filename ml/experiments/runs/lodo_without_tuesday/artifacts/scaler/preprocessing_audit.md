# Preprocessing audit

Fit rows: 500,000 (active: 26,330). Clip: ±10. Dropped: none. Flagged: 4.

| feature | kind | raw range | raw \|skew\| | zero frac | scaled \|skew\| | saturation | silent-row value | most correlated | flags |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| syn_ratio | unit | [0, 0.5] | 3.48 | 0.86 | 3.48 | 0.000 | 0.00 | slope3_syn_ratio (0.81) |  |
| ack_ratio | unit | [0, 0.5] | 0.43 | 0.50 | 0.43 | 0.000 | 0.00 | neighbour_risk_fraction (0.88) |  |
| rst_ratio | unit | [0, 0.0201] | 52.83 | 1.00 | 52.83 | 0.000 | 0.00 | iat_var (0.08) |  |
| fin_ratio | unit | [0, 0.5] | 16.43 | 0.97 | 16.43 | 0.000 | 0.00 | dst_port_entropy (0.15) |  |
| psh_ratio | unit | [0, 0.333] | 9.84 | 0.93 | 9.84 | 0.000 | 0.00 | flow_duration_var (0.56) |  |
| urg_ratio | unit | [0, 0.5] | 1.53 | 0.73 | 1.53 | 0.000 | 0.00 | ack_ratio (0.65) |  |
| bytes_total | log1p | [0, 5.85e+08] | 65.01 | 0.61 | 1.99 | 0.000 | -0.61 | active_flow_count (0.86) |  |
| bytes_up_down_ratio | log1p | [0, 1.1e+04] | 126.22 | 0.74 | 3.73 | 0.000 | -0.40 | bytes_total (0.44) |  |
| pkts_per_flow_mean | log1p | [0, 4.67e+04] | 102.65 | 0.47 | 1.49 | 0.000 | -0.86 | flow_duration_mean (0.86) |  |
| flow_duration_mean | log1p | [0, 1.2e+08] | 8.42 | 0.47 | 1.61 | 0.000 | -0.70 | iat_max (1.00) |  |
| flow_duration_var | log1p | [0, 3.6e+15] | 5.59 | 0.74 | 2.20 | 0.000 | -0.46 | iat_var (0.99) |  |
| iat_mean | log1p | [0, 6.41e+07] | 16.90 | 0.47 | 1.46 | 0.000 | -0.73 | flow_duration_mean (0.99) |  |
| iat_var | log1p | [0, 1.78e+15] | 38.22 | 0.74 | 2.14 | 0.000 | -0.47 | flow_duration_var (0.99) |  |
| iat_max | log1p | [0, 1.2e+08] | 4.84 | 0.47 | 1.60 | 0.000 | -0.70 | flow_duration_mean (1.00) |  |
| active_flow_count | log1p | [0, 1.78e+04] | 38.21 | 0.47 | 2.58 | 0.000 | -0.67 | out_degree (0.87) |  |
| ttl_mean | zscore | [0, 255] | 1.01 | 0.00 | 1.01 | 0.000 | -1.32 | tcp_window_entropy (0.19) | log1p would reduce |skew| 1.01 -> 0.23 |
| ttl_var | log1p | [0, 9.7e+03] | 11.48 | 0.90 | 5.91 | 0.000 | -0.19 | tcp_window_entropy (0.20) |  |
| tcp_window_mean | log1p | [0, 6.55e+04] | 3.79 | 0.02 | 0.92 | 0.000 | -3.58 | payload_size_var (0.49) |  |
| tcp_window_entropy | zscore | [0, 2.73] | 1.56 | 0.32 | 1.56 | 0.000 | -0.90 | tcp_window_mean (0.35) | log1p would reduce |skew| 1.56 -> 0.86 |
| frag_flag_rate | unit | [0, 0.0164] | 36.48 | 1.00 | 36.48 | 0.000 | 0.00 | local_clustering_coeff (0.11) |  |
| payload_size_mean | log1p | [0, 2.82e+03] | 2.00 | 0.18 | 0.53 | 0.000 | -1.66 | payload_size_p95 (0.98) |  |
| payload_size_var | log1p | [0, 6.03e+06] | 3.83 | 0.24 | 0.55 | 0.000 | -1.53 | payload_size_p95 (0.94) |  |
| payload_size_p95 | log1p | [0, 5.84e+03] | 1.02 | 0.19 | 0.72 | 0.000 | -1.74 | payload_size_mean (0.98) |  |
| payload_size_entropy | zscore | [0, 2.52] | 0.11 | 0.24 | 0.11 | 0.000 | -1.34 | payload_size_var (0.81) |  |
| retrans_count | log1p | [0, 5.34e+04] | 87.79 | 0.70 | 2.79 | 0.000 | -0.48 | retrans_rate (0.69) |  |
| retrans_rate | unit | [0, 1] | 3.48 | 0.70 | 3.48 | 0.000 | 0.00 | retrans_count (0.69) |  |
| out_degree | log1p | [0, 194] | 10.83 | 0.47 | 2.54 | 0.000 | -0.74 | active_flow_count (0.87) |  |
| in_degree | log1p | [0, 121] | 10.36 | 0.30 | 2.15 | 0.000 | -1.04 | dst_ip_entropy (0.66) |  |
| dst_ip_entropy | zscore | [-0, 4.07] | 3.72 | 0.87 | 3.72 | 0.000 | -0.31 | out_degree (0.86) | log1p would reduce |skew| 3.72 -> 3.01 |
| dst_port_entropy | zscore | [-0, 6.13] | 2.69 | 0.76 | 2.67 | 0.000 | -0.47 | active_flow_count (0.64) | log1p would reduce |skew| 2.69 -> 1.79 |
| new_peer_count | log1p | [0, 149] | 15.84 | 0.81 | 4.60 | 0.000 | -0.38 | out_degree (0.67) |  |
| neighbour_risk_fraction | unit | [0, 1] | 0.35 | 0.53 | 0.35 | 0.000 | 0.00 | ack_ratio (0.88) |  |
| local_clustering_coeff | unit | [0, 1] | 14.51 | 0.97 | 14.51 | 0.000 | 0.00 | retrans_count (0.21) |  |
| reciprocity | unit | [0, 1] | 0.11 | 0.47 | 0.11 | 0.000 | 0.00 | pkts_per_flow_mean (0.82) |  |
| d_syn_ratio | unit | [-0.5, 0.5] | 1.05 | 0.82 | 1.05 | 0.000 | 0.00 | syn_ratio (0.76) |  |
| d_dst_port_entropy | zscore | [-4.37, 5.46] | 0.63 | 0.71 | 0.63 | 0.000 | -0.13 | slope3_dst_port_entropy (0.63) |  |
| d_out_degree | asinh | [-180, 179] | 2.58 | 0.48 | 0.34 | 0.000 | -0.19 | d_new_peer_count (0.72) |  |
| d_new_peer_count | asinh | [-135, 149] | 2.45 | 0.75 | 0.13 | 0.000 | -0.09 | d_out_degree (0.72) |  |
| d_iat_var | asinh | [-1.42e+15, 1e+15] | 1.82 | 0.68 | 0.04 | 0.000 | -0.07 | slope3_iat_var (0.41) |  |
| d_retrans_rate | unit | [-1, 1] | 0.34 | 0.58 | 0.34 | 0.000 | 0.00 | retrans_rate (0.63) |  |
| slope3_syn_ratio | unit | [-0.25, 0.25] | 1.89 | 0.83 | 1.89 | 0.000 | 0.00 | syn_ratio (0.81) |  |
| slope3_dst_port_entropy | zscore | [-2.13, 2.73] | 0.90 | 0.73 | 0.90 | 0.000 | -0.15 | d_dst_port_entropy (0.63) |  |
| slope3_out_degree | asinh | [-80, 94.5] | 3.08 | 0.52 | 0.34 | 0.000 | -0.16 | d_out_degree (0.45) |  |
| slope3_iat_var | asinh | [-9.81e+14, 8.9e+14] | 13.72 | 0.69 | 0.01 | 0.000 | -0.09 | d_iat_var (0.41) |  |
| is_active | unit | [1, 1] | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | syn_ratio (0.00) |  |
