# DFR bridge: gdro_rn50 on waterbirds, last layer retrained on val

Definitions and predictions: docstring of `dfr_bridge.py` and `THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style group-balanced subsample (points deleted). Plain GD, logistic loss, no intercept, no regularisation. z = h * steps.

retraining set: n = 1199, separable = True, margin (lower bound) = 0.20359022297679627; subsample size = 532, 5 draws

## P1 -- do the reweighted runs end in the same place?

| z | min cosine among A | worst-cell test acc, eps=0.01 | worst-cell test acc, eps=0.05 | worst-cell test acc, eps=0.2 | worst-cell test acc, eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.46649 | 0.488 | 0.518 | 0.594 | 0.648 |
| 10 | 0.53158 | 0.398 | 0.489 | 0.639 | 0.657 |
| 99.8 | 0.63092 | 0.467 | 0.586 | 0.595 | 0.620 |
| 998 | 0.78869 | 0.564 | 0.562 | 0.578 | 0.586 |
| 9.98e+03 | 0.92927 | 0.551 | 0.562 | 0.578 | 0.586 |
| 1e+05 | 0.97887 | 0.548 | 0.572 | 0.575 | 0.578 |

## P2 -- do the DFR-style subsamples end somewhere else?

| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, B (mean +- sd) | worst-cell test acc, A (mean) |
|---|---|---|---|
| 1 | 0.5949, 0.5705 | 0.652 +- 0.018 | 0.562 |
| 10 | 0.5267, 0.4907 | 0.650 +- 0.014 | 0.546 |
| 99.8 | 0.5329, 0.5030 | 0.643 +- 0.022 | 0.567 |
| 998 | 0.5439, 0.5178 | 0.637 +- 0.024 | 0.572 |
| 9.98e+03 | 0.5368, 0.5130 | 0.634 +- 0.025 | 0.569 |
| 1e+05 | 0.5286, 0.5051 | 0.633 +- 0.027 | 0.568 |

Tightest 50 retraining points under the final A direction (approximate support vectors): how many did each draw delete?

| draw | fraction of ALL points deleted | fraction of TIGHT points deleted | cosine to A | subsample separable |
|---|---|---|---|---|
| 0 | 0.556 | 0.480 | 0.5249 | True |
| 1 | 0.556 | 0.480 | 0.5571 | True |
| 2 | 0.556 | 0.440 | 0.5051 | True |
| 3 | 0.556 | 0.380 | 0.5380 | True |
| 4 | 0.556 | 0.420 | 0.5181 | True |

## C -- DFR itself (l1 logistic regression with intercept, weights averaged over the same subsamples)

| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |
|---|---|---|---|---|
| 0.1 | 0.713 | 0.758 | 0.2711 | nan |
| 1 | 0.679 | 0.726 | 0.4764 | nan |
| 10 | 0.659 | 0.716 | 0.5135 | nan |
| 100 | 0.684 | 0.709 | 0.5447 | nan |
| 1000 | 0.669 | 0.692 | 0.5225 | nan |

No val bundle: no C selected (reporting all is not a tuned DFR).
