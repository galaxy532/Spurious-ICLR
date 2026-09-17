# DFR bridge: erm_rn50 on waterbirds, last layer retrained on val

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 1199, separable = True, margin (lower bound) = 0.23921285397971614; subsample size = 532, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.28413 | 0.137 | 0.145 | 0.229 | 0.269 |
| 10 | 0.51584 | 0.173 | 0.215 | 0.425 | 0.377 |
| 99.8 | 0.64719 | 0.231 | 0.388 | 0.445 | 0.425 |
| 998 | 0.82269 | 0.385 | 0.444 | 0.480 | 0.452 |
| 9.98e+03 | 0.95169 | 0.441 | 0.475 | 0.472 | 0.456 |
| 1e+05 | 0.98497 | 0.461 | 0.478 | 0.464 | 0.461 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.5218, 0.5129 | 0.491 +- 0.015 | 0.195 |
| 10 | 0.5185, 0.5017 | 0.486 +- 0.016 | 0.298 |
| 99.8 | 0.5278, 0.5062 | 0.488 +- 0.016 | 0.372 |
| 998 | 0.5366, 0.5198 | 0.489 +- 0.015 | 0.440 |
| 9.98e+03 | 0.5365, 0.5161 | 0.489 +- 0.015 | 0.461 |
| 1e+05 | 0.5332, 0.5114 | 0.488 +- 0.017 | 0.466 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.556 | 0.340 | 0.5305 | True |
| 1 | 0.556 | 0.320 | 0.5213 | True |
| 2 | 0.556 | 0.340 | 0.5114 | True |
| 3 | 0.556 | 0.380 | 0.5410 | True |
| 4 | 0.556 | 0.320 | 0.5620 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.484 | 0.543 | 0.3062 | nan |
| 1 | 0.513 | 0.545 | 0.5099 | nan |
| 10 | 0.512 | 0.543 | 0.5470 | nan |
| 100 | 0.502 | 0.535 | 0.5777 | nan |
| 1000 | 0.500 | 0.535 | 0.5918 | nan |

No val bundle: no C selected (reporting all is not a tuned DFR).
