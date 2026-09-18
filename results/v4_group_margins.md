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
| features_v4_waterbirds_dinov2_res100g0_train.npz | resolution 0.1 on g=0 (both) | 4795 | 768 | 0.3517 | 1.0000 | 1.0000 | tie | 1.0000 | 4.9e-10 |
| features_v4_waterbirds_dinov2_res140g0_train.npz | resolution 0.14 on g=0 (both) | 4795 | 768 | 0.4091 | 1.0000 | 1.0000 | tie | 1.0000 | 8.7e-11 |
| features_v4_waterbirds_dinov2_res200g0_train.npz | resolution 0.2 on g=0 (both) | 4795 | 768 | 0.5019 | 1.0000 | 1.0000 | tie | 1.0000 | 7.2e-11 |
| features_v4_waterbirds_dinov2_res280g0_train.npz | resolution 0.28 on g=0 (both) | 4795 | 768 | 0.5614 | 1.0000 | 1.0000 | tie | 1.0000 | 2.4e-09 |
| features_v4_waterbirds_dinov2_res400g0_train.npz | resolution 0.4 on g=0 (both) | 4795 | 768 | 0.6032 | 1.0000 | 1.0000 | tie | 1.0000 | 7.9e-10 |
| features_v4_waterbirds_dinov2_res600g0_train.npz | resolution 0.6 on g=0 (both) | 4795 | 768 | 0.6375 | 1.0000 | 1.0000 | tie | 1.0000 | 2.2e-09 |
| features_v4_waterbirds_dinov2_train.npz | none | 4795 | 768 | 0.6312 | 1.0000 | 1.0000 | tie | 1.0000 | 2.0e-10 |

## Cross-checks and fragility

`margin_logistic` is the same geometric margin from logistic regression at huge C
(a different loss and optimiser). `rho_lp` is the exact LP of `separability_check`,
which uses a BOX constraint and is therefore NOT comparable in value -- it is here
only because a positive value PROVES separability. `n at margin` counts the points
within 1% of their group's minimum: a `gamma` ratio resting on one point is fragile.

| bundle | margin (svc) | margin (logistic) | rho_lp | lp separable | n at margin g=0 | g=1 | C used | converged |
|---|---|---|---|---|---|---|---|---|
| features_v4_waterbirds_dinov2_res100g0_train.npz | 0.3517 | - | - | None | 514 | 52 | 1e+06 | True |
| features_v4_waterbirds_dinov2_res140g0_train.npz | 0.4091 | - | - | None | 473 | 49 | 1e+06 | True |
| features_v4_waterbirds_dinov2_res200g0_train.npz | 0.5019 | - | - | None | 448 | 63 | 1e+06 | True |
| features_v4_waterbirds_dinov2_res280g0_train.npz | 0.5614 | - | - | None | 395 | 58 | 1e+06 | True |
| features_v4_waterbirds_dinov2_res400g0_train.npz | 0.6032 | - | - | None | 361 | 67 | 1e+06 | True |
| features_v4_waterbirds_dinov2_res600g0_train.npz | 0.6375 | - | - | None | 359 | 69 | 1e+06 | True |
| features_v4_waterbirds_dinov2_train.npz | 0.6312 | - | - | None | 350 | 71 | 1e+06 | True |
