# DFR bridge: dinov2 on waterbirds, last layer retrained on val

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 1199, separable = True, margin (lower bound) = 0.865141085065416; subsample size = 532, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.75428 | 0.628 | 0.714 | 0.873 | 0.920 |
| 10 | 0.78615 | 0.690 | 0.848 | 0.940 | 0.939 |
| 99.8 | 0.83954 | 0.831 | 0.927 | 0.949 | 0.947 |
| 998 | 0.94404 | 0.925 | 0.942 | 0.952 | 0.949 |
| 9.98e+03 | 0.97850 | 0.940 | 0.947 | 0.950 | 0.949 |
| 1e+05 | 0.98686 | 0.942 | 0.950 | 0.949 | 0.949 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.8607, 0.8407 | 0.881 +- 0.014 | 0.784 |
| 10 | 0.8051, 0.7834 | 0.906 +- 0.014 | 0.854 |
| 99.8 | 0.7918, 0.7650 | 0.907 +- 0.014 | 0.913 |
| 998 | 0.7767, 0.7497 | 0.906 +- 0.014 | 0.942 |
| 9.98e+03 | 0.7619, 0.7375 | 0.904 +- 0.016 | 0.946 |
| 1e+05 | 0.7519, 0.7296 | 0.903 +- 0.016 | 0.947 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.556 | 0.560 | 0.7296 | True |
| 1 | 0.556 | 0.540 | 0.7664 | True |
| 2 | 0.556 | 0.560 | 0.7773 | True |
| 3 | 0.556 | 0.680 | 0.7519 | True |
| 4 | 0.556 | 0.540 | 0.7343 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.945 | 0.955 | 0.4246 | nan |
| 1 | 0.950 | 0.966 | 0.5628 | nan |
| 10 | 0.958 | 0.966 | 0.6286 | nan |
| 100 | 0.960 | 0.967 | 0.7408 | nan |
| 1000 | 0.957 | 0.961 | 0.7629 | nan |

No val bundle: no C selected (reporting all is not a tuned DFR).
