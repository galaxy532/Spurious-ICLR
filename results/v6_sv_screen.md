# Session 6 -- does any image property avoid the margin set?

- bundle: `features_v4_waterbirds_dinov2_train.npz`   n = 4795
- separator: LinearSVC hinge, no intercept, C = 1e+06, margin u* = 0.6312, plateau 1.3e-09
- **margin set S** (u within 0.001 relative of u*): **418** points (45 waterbirds, 373 landbirds)
- family: 650 candidates (A_attribute 624, B_proxy_tail 24, C_binary 2); **323 eligible** after the size / label-balance / label-proxy filters (minimum group size 48)
- **detectability floor: m >= 96** (unstratified bound). A margin-free group of at least this size, with the data's label mix, survives Holm. Below it, a margin-free group can arise by chance.

## Answer

**NO HIT.** No eligible candidate contains zero margin points beyond what chance allows. See the graded table below for how close the best came.

## The closest candidates, by depletion of margin points

`margin pts in G` against `expected` is the graded statistic: how many of the S points fall in G, against the label-matched expectation. `p (depletion)` is the exact probability of that few or fewer under the null. `ratio` exceeds 1 only when the count is 0.

| candidate G | m | waterbird/landbird | ratio | larger | margin pts in G | expected | p (min) | Holm p (min) | p (depletion) | Holm p (depl.) | AUC y | phi y | AUC place | AUC old g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| has_head_pattern::capped = 1 | 565 | 142/423 | 1.0000 | tie | 33 | 48.6 | 1.06e-01 | 1.00e+00 | 5.94e-03 | 1.00e+00 | 0.506 | +0.017 | 0.506 | 0.495 |
| has_underparts_color::grey = 0 | 3891 | 910/2981 | 1.0000 | tie | 320 | 338.8 | 1.00e+00 | 1.00e+00 | 9.36e-03 | 1.00e+00 | 0.504 | +0.009 | 0.502 | 0.496 |
| has_primary_color::brown = 1 | 1190 | 195/995 | 1.0000 | tie | 90 | 108.7 | 7.30e-01 | 1.00e+00 | 1.51e-02 | 1.00e+00 | 0.452 | -0.093 | 0.452 | 0.479 |
| has_back_color::brown = 1 | 959 | 146/813 | 1.0000 | tie | 71 | 88.3 | 7.79e-01 | 1.00e+00 | 1.55e-02 | 1.00e+00 | 0.455 | -0.095 | 0.459 | 0.500 |
| has_upper_tail_color::orange = 0 | 4728 | 1103/3625 | 1.0000 | tie | 406 | 411.8 | 1.00e+00 | 1.00e+00 | 1.76e-02 | 1.00e+00 | 0.503 | +0.023 | 0.503 | 0.496 |
| has_wing_pattern::multi-colored = 0 | 3430 | 842/2588 | 1.0000 | tie | 277 | 296.2 | 1.00e+00 | 1.00e+00 | 1.79e-02 | 1.00e+00 | 0.527 | +0.050 | 0.518 | 0.483 |
| has_tail_pattern::solid = 0 | 3079 | 685/2394 | 1.0000 | tie | 251 | 270.2 | 1.00e+00 | 1.00e+00 | 2.27e-02 | 1.00e+00 | 0.483 | -0.031 | 0.488 | 0.515 |
| has_crown_color::buff = 0 | 4208 | 1031/3177 | 1.0000 | tie | 350 | 363.5 | 1.00e+00 | 1.00e+00 | 2.53e-02 | 1.00e+00 | 0.532 | +0.082 | 0.527 | 0.475 |
| has_belly_pattern::multi-colored = 0 | 4089 | 1014/3075 | 1.0000 | tie | 338 | 352.5 | 1.00e+00 | 1.00e+00 | 2.54e-02 | 1.00e+00 | 0.538 | +0.090 | 0.524 | 0.472 |
| has_breast_pattern::striped = 1 | 540 | 89/451 | 1.0000 | tie | 37 | 49.3 | 3.16e-03 | 1.00e+00 | 2.68e-02 | 1.00e+00 | 0.479 | -0.057 | 0.480 | 0.507 |
| has_breast_pattern::solid = 0 | 2127 | 417/1710 | 1.0000 | tie | 171 | 190.1 | 4.40e-02 | 1.00e+00 | 2.70e-02 | 1.00e+00 | 0.455 | -0.076 | 0.458 | 0.497 |
| has_forehead_color::rufous = 0 | 4721 | 1111/3610 | 1.0000 | tie | 405 | 410.6 | 1.00e+00 | 1.00e+00 | 2.90e-02 | 1.00e+00 | 0.509 | +0.061 | 0.507 | 0.497 |
| has_eye_color::yellow = 1 | 85 | 34/51 | 1.0000 | tie | 2 | 6.5 | 4.60e-02 | 1.00e+00 | 3.45e-02 | 1.00e+00 | 0.508 | +0.053 | 0.506 | 0.493 |
| has_head_pattern::masked = 1 | 193 | 60/133 | 1.0000 | tie | 9 | 15.9 | 4.20e-01 | 1.00e+00 | 3.58e-02 | 1.00e+00 | 0.509 | +0.038 | 0.508 | 0.499 |
| has_under_tail_color::orange = 0 | 4731 | 1107/3624 | 1.0000 | tie | 407 | 411.9 | 1.00e+00 | 1.00e+00 | 3.77e-02 | 1.00e+00 | 0.505 | +0.038 | 0.504 | 0.494 |
| has_back_pattern::spotted = 1 | 309 | 58/251 | 1.0000 | tie | 19 | 27.8 | 1.00e+00 | 1.00e+00 | 3.88e-02 | 1.00e+00 | 0.492 | -0.028 | 0.493 | 0.506 |
| has_head_pattern::striped = 1 | 332 | 59/273 | 1.0000 | tie | 21 | 30.0 | 1.81e-01 | 1.00e+00 | 3.97e-02 | 1.00e+00 | 0.489 | -0.035 | 0.489 | 0.494 |
| has_shape::hawk-like = 0 | 4648 | 1076/3572 | 1.0000 | tie | 399 | 405.4 | 1.00e+00 | 1.00e+00 | 4.51e-02 | 1.00e+00 | 0.498 | -0.008 | 0.499 | 0.496 |
| has_breast_color::grey = 0 | 3896 | 892/3004 | 1.0000 | tie | 327 | 340.4 | 1.00e+00 | 1.00e+00 | 4.62e-02 | 1.00e+00 | 0.493 | -0.016 | 0.493 | 0.489 |
| has_throat_color::red = 0 | 4603 | 1099/3504 | 1.0000 | tie | 392 | 399.4 | 1.00e+00 | 1.00e+00 | 4.67e-02 | 1.00e+00 | 0.518 | +0.077 | 0.515 | 0.499 |
| has_nape_color::brown = 1 | 821 | 123/698 | 1.0000 | tie | 64 | 75.7 | 1.65e-01 | 1.00e+00 | 6.51e-02 | 1.00e+00 | 0.460 | -0.089 | 0.465 | 0.495 |
| has_breast_color::black = 0 | 3830 | 887/2943 | 1.0000 | tie | 322 | 334.0 | 1.00e+00 | 1.00e+00 | 7.16e-02 | 1.00e+00 | 0.499 | -0.002 | 0.496 | 0.483 |
| has_underparts_color::black = 0 | 3950 | 931/3019 | 1.0000 | tie | 332 | 343.5 | 1.00e+00 | 1.00e+00 | 7.18e-02 | 1.00e+00 | 0.508 | +0.018 | 0.505 | 0.481 |
| has_throat_color::black = 0 | 3727 | 842/2885 | 1.0000 | tie | 314 | 326.3 | 1.01e-02 | 1.00e+00 | 7.27e-02 | 1.00e+00 | 0.486 | -0.027 | 0.489 | 0.488 |
| has_breast_pattern::spotted = 1 | 302 | 54/248 | 1.0000 | tie | 20 | 27.3 | 4.18e-02 | 1.00e+00 | 7.40e-02 | 1.00e+00 | 0.491 | -0.033 | 0.491 | 0.500 |

## Confirmation bundles written

Measured next by the unmodified `group_margins.py`; `--compare` checks agreement.

- `features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz` -- has_breast_pattern::striped = 1 (screen ratio 1.000000, larger tie, reason: best-ranked)
- `features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz` -- has_nape_color::black = 0 (screen ratio 1.000000, larger tie, reason: best-ranked)
- `features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz` -- has_upperparts_color::blue = 1 (screen ratio 1.000000, larger tie, reason: best-ranked)

Every candidate, eligible or not, is in the `.json` next to this file.
