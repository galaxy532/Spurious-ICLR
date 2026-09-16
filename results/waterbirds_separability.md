
## linear separability of each arm-A cell

Separability is through the ORIGIN (`common.logistic_gd` has no intercept).
`margin` is geometric, min_i y_i(w.x_i)/||w||; it is a LOWER BOUND on the max
margin (the separator comes from weakly-regularised logistic regression, which
converges in direction to max-margin but is not run to convergence).
`margin_n` divides by mean ||x_i|| so it compares across backbones.

### waterbirds_clip_train

| cell | n | n_min | separable | margin | margin_n | decided by |
|---|---|---|---|---|---|---|
| full | 4795 | 240 | **no** | 0 | 0 | LP proof (max margin is 0) |
| 0.005 | 4578 | 23 | yes | 0.08611 | 0.003829 | logistic (constructive) |
| 0.01 | 4601 | 46 | yes | 0.0497 | 0.00221 | logistic (constructive) |
| 0.02 | 4648 | 93 | yes | 0.007242 | 0.0003219 | logistic (constructive) |
| 0.035 | 4720 | 165 | **no** | 0 | 0 | LP proof (max margin is 0) |
| 0.05 | 4795 | 240 | **no** | 0 | 0 | LP proof (max margin is 0) |

**Verdict.** FULL SPLIT IS NOT SEPARABLE but cells ['0.005', '0.01', '0.02'] ARE. The grid STRADDLES the boundary: beta is a real exponent at those eps and is identically 0 at the others, so the kappa fit mixes two different quantities and cannot be interpreted.

### waterbirds_dinov2_train

| cell | n | n_min | separable | margin | margin_n | decided by |
|---|---|---|---|---|---|---|
| full | 4795 | 240 | yes | 0.5506 | 0.01988 | logistic (constructive) |
| 0.005 | 4578 | 23 | yes | 0.7739 | 0.02795 | logistic (constructive) |
| 0.01 | 4601 | 46 | yes | 0.7481 | 0.02701 | logistic (constructive) |
| 0.02 | 4648 | 93 | yes | 0.681 | 0.02459 | logistic (constructive) |
| 0.035 | 4720 | 165 | yes | 0.6279 | 0.02267 | logistic (constructive) |
| 0.05 | 4795 | 240 | yes | 0.5506 | 0.01988 | logistic (constructive) |

**Verdict.** FULL SPLIT IS SEPARABLE, so every eps cell is too (subsets of a separable set are separable). A separability flip CANNOT explain this backbone's eps-dependence. Margin varies 1.4x across the grid -- if that is large, the curves differ in CONVERGENCE SPEED, not in regime, and the remedy is a longer run.

### waterbirds_erm_rn50_train

| cell | n | n_min | separable | margin | margin_n | decided by |
|---|---|---|---|---|---|---|
| full | 4795 | 240 | yes | 0.003726 | 0.0001324 | logistic (constructive) |
| 0.005 | 4578 | 23 | yes | 0.0238 | 0.0008529 | logistic (constructive) |
| 0.01 | 4601 | 46 | yes | 0.01701 | 0.0006087 | logistic (constructive) |
| 0.02 | 4648 | 93 | yes | 0.01105 | 0.0003941 | logistic (constructive) |
| 0.035 | 4720 | 165 | yes | 0.007365 | 0.0002628 | logistic (constructive) |
| 0.05 | 4795 | 240 | yes | 0.003733 | 0.0001326 | logistic (constructive) |

**Verdict.** FULL SPLIT IS SEPARABLE, so every eps cell is too (subsets of a separable set are separable). A separability flip CANNOT explain this backbone's eps-dependence. Margin varies 6.4x across the grid -- if that is large, the curves differ in CONVERGENCE SPEED, not in regime, and the remedy is a longer run.

### waterbirds_under_rn50_train

| cell | n | n_min | separable | margin | margin_n | decided by |
|---|---|---|---|---|---|---|
| full | 4795 | 240 | yes | 0.01632 | 0.0004141 | logistic (constructive) |
| 0.005 | 4578 | 23 | yes | 0.09678 | 0.002447 | logistic (constructive) |
| 0.01 | 4601 | 46 | yes | 0.08611 | 0.002178 | logistic (constructive) |
| 0.02 | 4648 | 93 | yes | 0.04485 | 0.001135 | logistic (constructive) |
| 0.035 | 4720 | 165 | yes | 0.02711 | 0.0006872 | logistic (constructive) |
| 0.05 | 4795 | 240 | yes | 0.01632 | 0.0004141 | logistic (constructive) |

**Verdict.** FULL SPLIT IS SEPARABLE, so every eps cell is too (subsets of a separable set are separable). A separability flip CANNOT explain this backbone's eps-dependence. Margin varies 5.9x across the grid -- if that is large, the curves differ in CONVERGENCE SPEED, not in regime, and the remedy is a longer run.
