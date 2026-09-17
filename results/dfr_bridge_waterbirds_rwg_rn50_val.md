# DFR bridge: rwg_rn50 on waterbirds, last layer retrained on val

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 1199, separable = True, margin (lower bound) = 0.09350453008273484; subsample size = 532, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.68022 | 0.614 | 0.612 | 0.587 | 0.631 |
| 10 | 0.62569 | 0.555 | 0.551 | 0.548 | 0.621 |
| 99.8 | 0.65063 | 0.523 | 0.526 | 0.576 | 0.626 |
| 998 | 0.71176 | 0.536 | 0.569 | 0.597 | 0.617 |
| 9.98e+03 | 0.83595 | 0.579 | 0.597 | 0.595 | 0.601 |
| 1e+05 | 0.89925 | 0.578 | 0.592 | 0.600 | 0.606 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.6202, 0.5861 | 0.691 +- 0.017 | 0.611 |
| 10 | 0.5120, 0.4914 | 0.678 +- 0.010 | 0.569 |
| 99.8 | 0.5076, 0.4810 | 0.642 +- 0.013 | 0.563 |
| 998 | 0.5163, 0.4881 | 0.617 +- 0.014 | 0.579 |
| 9.98e+03 | 0.5096, 0.4807 | 0.606 +- 0.016 | 0.593 |
| 1e+05 | 0.5155, 0.4775 | 0.601 +- 0.016 | 0.594 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.556 | 0.360 | 0.5321 | True |
| 1 | 0.556 | 0.320 | 0.5429 | True |
| 2 | 0.556 | 0.320 | 0.5290 | True |
| 3 | 0.556 | 0.340 | 0.4958 | True |
| 4 | 0.556 | 0.360 | 0.4775 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.702 | 0.777 | 0.1920 | nan |
| 1 | 0.718 | 0.729 | 0.5211 | nan |
| 10 | 0.681 | 0.702 | 0.5653 | nan |
| 100 | 0.646 | 0.678 | 0.5878 | nan |
| 1000 | 0.606 | 0.649 | 0.5632 | nan |

No val bundle: no C selected (reporting all is not a tuned DFR).
