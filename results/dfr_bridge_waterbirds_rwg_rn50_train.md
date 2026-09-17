# DFR bridge: rwg_rn50 on waterbirds, last layer retrained on train

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 4795, separable = True, margin (lower bound) = 0.012581682643267473; subsample size = 224, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.85893 | 0.617 | 0.617 | 0.621 | 0.639 |
| 10 | 0.75902 | 0.537 | 0.536 | 0.558 | 0.583 |
| 99.8 | 0.77455 | 0.506 | 0.509 | 0.526 | 0.531 |
| 998 | 0.86467 | 0.506 | 0.525 | 0.544 | 0.551 |
| 9.98e+03 | 0.93444 | 0.495 | 0.497 | 0.506 | 0.512 |
| 1e+05 | 0.95149 | 0.495 | 0.491 | 0.491 | 0.494 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.7887, 0.7412 | 0.649 +- 0.019 | 0.623 |
| 10 | 0.5036, 0.3998 | 0.637 +- 0.009 | 0.553 |
| 99.8 | 0.3139, 0.2576 | 0.625 +- 0.011 | 0.518 |
| 998 | 0.1620, 0.1353 | 0.617 +- 0.015 | 0.532 |
| 9.98e+03 | 0.0957, 0.0872 | 0.613 +- 0.018 | 0.503 |
| 1e+05 | 0.0802, 0.0673 | 0.609 +- 0.018 | 0.493 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.953 | 0.960 | 0.0901 | True |
| 1 | 0.953 | 0.940 | 0.0798 | True |
| 2 | 0.953 | 0.920 | 0.0940 | True |
| 3 | 0.953 | 1.000 | 0.0673 | True |
| 4 | 0.953 | 0.940 | 0.0699 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.623 | 0.787 | 0.0903 | 0.639 |
| 1 | 0.634 | 0.781 | 0.1255 | 0.617 |
| 10 | 0.651 | 0.764 | 0.0892 | 0.647 |
| 100 | 0.643 | 0.743 | 0.0710 | 0.684 |
| 1000 | 0.588 | 0.697 | 0.0465 | 0.612 |

C selected on val: 100
