
## epsilon x backbone sweep

mode: **reweight**  (reweight = eps in the loss weights, sample held fixed; subsample = minority rows deleted, carries a margin confound)

`kappa` is the exponent in  worst-group error ~ eps^(-kappa)  at matched z_t. `kappa_per_z` in the json gives it at three horizons (0.25/0.5/1.0 of z_max): if those drift, kappa depends on how long the run was and is not yet an asymptotic quantity.

| backbone | n | kappa | regime | predicted | DFR gain (measured) |
|---|---|---|---|---|---|
| waterbirds_dinov2_train | 4795 | 0.734 | alpha<1 (balancing is the lever) | helps | +0.058 ± 0.027 |
| waterbirds_erm_rn50_train | 4795 | 0.476 | ambiguous (near the transition) | unclear | +0.385 ± 0.052 |
| waterbirds_under_rn50_train | 4795 | 0.609 | alpha<1 (balancing is the lever) | helps | +0.273 ± 0.075 |

### waterbirds_dinov2_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.01 | 4795 | 240 | 0.309 | 0.940 | 0.664 | 0.955 |
| 0.03 | 4795 | 240 | 0.466 | 0.953 | 0.628 | 0.963 |
| 0.08 | 4795 | 240 | 0.582 | 0.959 | 0.630 | 0.957 |
| 0.2 | 4795 | 240 | 0.651 | 0.971 | 0.649 | 0.956 |
| 0.5 | 4795 | 240 | 0.664 | 0.984 | 0.634 | 0.952 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.

### waterbirds_erm_rn50_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.01 | 4795 | 240 | 0.024 | 0.116 | 0.153 | 0.281 |
| 0.03 | 4795 | 240 | 0.048 | 0.152 | 0.134 | 0.258 |
| 0.08 | 4795 | 240 | 0.091 | 0.176 | 0.139 | 0.236 |
| 0.2 | 4795 | 240 | 0.132 | 0.208 | 0.152 | 0.227 |
| 0.5 | 4795 | 240 | 0.175 | 0.251 | 0.154 | 0.235 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.

### waterbirds_under_rn50_train

| eps | n | n_min | beta_min w0 | beta_min w1 | beta_maj w0 | beta_maj w1 |
|---|---|---|---|---|---|---|
| 0.01 | 4795 | 240 | 0.034 | 0.179 | 0.199 | 0.370 |
| 0.03 | 4795 | 240 | 0.062 | 0.280 | 0.149 | 0.414 |
| 0.08 | 4795 | 240 | 0.098 | 0.353 | 0.130 | 0.439 |
| 0.2 | 4795 | 240 | 0.143 | 0.401 | 0.138 | 0.445 |
| 0.5 | 4795 | 240 | 0.194 | 0.395 | 0.146 | 0.414 |

beta is reported over two z-windows. It converges to max(alpha,1) from BELOW and slowly, so read the DRIFT between w0 and w1, never w1 alone. The majority columns are the control: they should sit near 1 and should NOT move with eps.
