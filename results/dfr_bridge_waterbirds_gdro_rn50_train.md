# DFR bridge: gdro_rn50 on waterbirds, last layer retrained on train

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 4795, separable = True, margin (lower bound) = 0.04256380454409016; subsample size = 224, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.70755 | 0.486 | 0.506 | 0.547 | 0.587 |
| 10 | 0.66653 | 0.439 | 0.492 | 0.562 | 0.607 |
| 99.8 | 0.81420 | 0.441 | 0.493 | 0.544 | 0.584 |
| 998 | 0.89664 | 0.465 | 0.501 | 0.529 | 0.553 |
| 9.98e+03 | 0.96104 | 0.496 | 0.509 | 0.525 | 0.534 |
| 1e+05 | 0.98993 | 0.511 | 0.515 | 0.522 | 0.526 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.6742, 0.6233 | 0.526 +- 0.024 | 0.531 |
| 10 | 0.4001, 0.3664 | 0.540 +- 0.031 | 0.525 |
| 99.8 | 0.2510, 0.2296 | 0.531 +- 0.034 | 0.516 |
| 998 | 0.1560, 0.1423 | 0.525 +- 0.036 | 0.512 |
| 9.98e+03 | 0.1062, 0.0838 | 0.522 +- 0.036 | 0.516 |
| 1e+05 | 0.0900, 0.0693 | 0.520 +- 0.038 | 0.518 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.953 | 0.960 | 0.1130 | True |
| 1 | 0.953 | 0.960 | 0.0905 | True |
| 2 | 0.953 | 0.980 | 0.0954 | True |
| 3 | 0.953 | 0.940 | 0.0693 | True |
| 4 | 0.953 | 0.980 | 0.0817 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.568 | 0.701 | 0.0652 | 0.569 |
| 1 | 0.613 | 0.738 | 0.1159 | 0.622 |
| 10 | 0.604 | 0.731 | 0.1150 | 0.607 |
| 100 | 0.577 | 0.709 | 0.0836 | 0.584 |
| 1000 | 0.555 | 0.680 | 0.0517 | 0.534 |

C selected on val: 1
