# Per-group hard margins

`gamma_g` are the manuscript's group hard-margins (supplementary Eq eq:margins):
the minimum of `y * w_hat . x` within each group, under the global minimum-norm
separator `w_hat`, normalised so the SMALLER of the two is 1. `w_hat` has no
intercept, matching the GD in `long_horizon.py`.

The alignment identity `gamma_min ~= max(1, alpha)` makes the last column a
prediction: the LARGER-margin group's error exponent `beta` should equal it, and
the other group's should be 1. `long_horizon.py` measures both independently.

Read `plateau` first. It is the relative change of the margin across the top two
C values of the solver ladder. Below 1e-3 the margin is settled; above 1e-2 the
row is not a measurement yet and must not be quoted.

| bundle | degradation | n | d | margin | gamma(g=0) | gamma(g=1) | larger | beta predicted | plateau |
|---|---|---|---|---|---|---|---|---|---|
| features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz | none | 4795 | 768 | 0.6312 | 1.0000 | 1.0000 | tie | 1.0000 | 2.9e-10 |
| features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz | none | 4795 | 768 | 0.6312 | 1.0000 | 1.0000 | tie | 1.0000 | 7.4e-10 |
| features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz | none | 4795 | 768 | 0.6312 | 1.0000 | 1.0000 | tie | 1.0000 | 1.3e-09 |

## Cross-checks and fragility

`margin_logistic` is the same geometric margin from logistic regression at huge C
(a different loss and optimiser). `rho_lp` is the exact LP of `separability_check`,
which uses a BOX constraint and is therefore NOT comparable in value -- it is here
only because a positive value PROVES separability. `n at margin` counts the points
within 1% of their group's minimum: a `gamma` ratio resting on one point is fragile.

| bundle | margin (svc) | margin (logistic) | rho_lp | lp separable | n at margin g=0 | g=1 | C used | converged |
|---|---|---|---|---|---|---|---|---|
| features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz | 0.6312 | - | - | None | 384 | 37 | 1e+06 | True |
| features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz | 0.6312 | - | - | None | 110 | 311 | 1e+06 | True |
| features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz | 0.6312 | - | - | None | 398 | 23 | 1e+06 | True |
