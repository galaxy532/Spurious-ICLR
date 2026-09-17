# DFR bridge: erm_rn50 on waterbirds, last layer retrained on train

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 4795, separable = True, margin (lower bound) = 0.02659733836780392; subsample size = 224, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.15126 | 0.102 | 0.095 | 0.132 | 0.512 |
| 10 | 0.33199 | 0.123 | 0.125 | 0.215 | 0.441 |
| 99.8 | 0.50481 | 0.149 | 0.187 | 0.288 | 0.357 |
| 998 | 0.75086 | 0.209 | 0.238 | 0.277 | 0.304 |
| 9.98e+03 | 0.93153 | 0.245 | 0.269 | 0.279 | 0.282 |
| 1e+05 | 0.98452 | 0.279 | 0.285 | 0.290 | 0.293 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.3907, 0.3684 | 0.495 +- 0.032 | 0.210 |
| 10 | 0.3514, 0.3321 | 0.512 +- 0.016 | 0.226 |
| 99.8 | 0.2525, 0.2385 | 0.505 +- 0.013 | 0.245 |
| 998 | 0.1421, 0.1364 | 0.503 +- 0.014 | 0.257 |
| 9.98e+03 | 0.0956, 0.0917 | 0.502 +- 0.014 | 0.269 |
| 1e+05 | 0.0860, 0.0806 | 0.502 +- 0.015 | 0.287 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.953 | 0.880 | 0.0829 | True |
| 1 | 0.953 | 0.860 | 0.0841 | True |
| 2 | 0.953 | 0.880 | 0.0806 | True |
| 3 | 0.953 | 0.860 | 0.0869 | True |
| 4 | 0.953 | 0.860 | 0.0954 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.522 | 0.582 | 0.0713 | 0.496 |
| 1 | 0.565 | 0.584 | 0.0846 | 0.549 |
| 10 | 0.569 | 0.581 | 0.0816 | 0.541 |
| 100 | 0.550 | 0.561 | 0.0890 | 0.528 |
| 1000 | 0.534 | 0.549 | 0.0879 | 0.528 |

C selected on val: 1
