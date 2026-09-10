
## epsilon x backbone sweep

`kappa` is the exponent in  worst-group error ~ eps^(-kappa)  at matched z_t.

| backbone | n | kappa | regime | predicted | DFR gain (measured) |
|---|---|---|---|---|---|
| waterbirds_clip_train | 4578 | -0.924 | alpha>1 (majority binding; balancing cannot accelerate) | no help | +0.273 ± 0.063 |
| waterbirds_dinov2_train | 4578 | 0.066 | alpha>1 (majority binding; balancing cannot accelerate) | no help | +0.058 ± 0.027 |
| waterbirds_erm_rn50_train | 4578 | -0.257 | alpha>1 (majority binding; balancing cannot accelerate) | no help | +0.385 ± 0.052 |
| waterbirds_under_rn50_train | 4578 | -0.958 | alpha>1 (majority binding; balancing cannot accelerate) | no help | +0.273 ± 0.075 |

### waterbirds_clip_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.005 | 4578 | 23 | 0.023 | 0.600 | 0.241 | 0.605 |
| 0.01 | 4601 | 46 | 0.107 | 0.368 | 0.235 | 0.396 |
| 0.02 | 4648 | 93 | 0.111 | 0.148 | 0.175 | 0.136 |
| 0.035 | 4720 | 165 | 0.112 | 0.046 | 0.157 | 0.053 |
| 0.05 | 4795 | 240 | 0.114 | 0.025 | 0.150 | 0.031 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.

### waterbirds_dinov2_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.005 | 4578 | 23 | 0.574 | 0.958 | 0.737 | 0.959 |
| 0.01 | 4601 | 46 | 0.570 | 0.956 | 0.725 | 0.957 |
| 0.02 | 4648 | 93 | 0.576 | 0.956 | 0.697 | 0.960 |
| 0.035 | 4720 | 165 | 0.588 | 0.953 | 0.680 | 0.958 |
| 0.05 | 4795 | 240 | 0.529 | 0.954 | 0.623 | 0.958 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.

### waterbirds_erm_rn50_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.005 | 4578 | 23 | 0.072 | 0.327 | 0.174 | 0.377 |
| 0.01 | 4601 | 46 | 0.052 | 0.275 | 0.158 | 0.350 |
| 0.02 | 4648 | 93 | 0.078 | 0.200 | 0.159 | 0.293 |
| 0.035 | 4720 | 165 | 0.074 | 0.197 | 0.150 | 0.276 |
| 0.05 | 4795 | 240 | 0.070 | 0.165 | 0.135 | 0.246 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.

### waterbirds_under_rn50_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.005 | 4578 | 23 | 0.113 | 0.769 | 0.236 | 0.810 |
| 0.01 | 4601 | 46 | 0.129 | 0.754 | 0.231 | 0.780 |
| 0.02 | 4648 | 93 | 0.099 | 0.534 | 0.183 | 0.619 |
| 0.035 | 4720 | 165 | 0.084 | 0.432 | 0.148 | 0.508 |
| 0.05 | 4795 | 240 | 0.078 | 0.321 | 0.134 | 0.430 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.
