# DFR bridge: dinov2 on waterbirds, last layer retrained on train

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 4795, separable = True, margin (lower bound) = 0.5506020503059962; subsample size = 224, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.75086 | 0.639 | 0.725 | 0.882 | 0.938 |
| 10 | 0.82736 | 0.717 | 0.845 | 0.924 | 0.954 |
| 99.8 | 0.89664 | 0.804 | 0.884 | 0.924 | 0.938 |
| 998 | 0.96338 | 0.855 | 0.887 | 0.906 | 0.921 |
| 9.98e+03 | 0.98512 | 0.869 | 0.884 | 0.900 | 0.909 |
| 1e+05 | 0.99073 | 0.873 | 0.885 | 0.894 | 0.902 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.8041, 0.7935 | 0.854 +- 0.015 | 0.796 |
| 10 | 0.6993, 0.6821 | 0.857 +- 0.013 | 0.860 |
| 99.8 | 0.6227, 0.6031 | 0.851 +- 0.014 | 0.888 |
| 998 | 0.5540, 0.5333 | 0.849 +- 0.014 | 0.892 |
| 9.98e+03 | 0.5197, 0.4990 | 0.848 +- 0.013 | 0.890 |
| 1e+05 | 0.5022, 0.4805 | 0.847 +- 0.014 | 0.888 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.953 | 0.920 | 0.5296 | True |
| 1 | 0.953 | 0.940 | 0.5057 | True |
| 2 | 0.953 | 0.920 | 0.5076 | True |
| 3 | 0.953 | 0.940 | 0.4874 | True |
| 4 | 0.953 | 0.940 | 0.4805 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.883 | 0.916 | 0.3205 | 0.873 |
| 1 | 0.941 | 0.955 | 0.4032 | 0.923 |
| 10 | 0.945 | 0.958 | 0.4388 | 0.942 |
| 100 | 0.946 | 0.957 | 0.5130 | 0.940 |
| 1000 | 0.945 | 0.953 | 0.5137 | 0.931 |

C selected on val: 10
