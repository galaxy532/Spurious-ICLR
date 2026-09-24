# Session 6 -- screen vs the unmodified group_margins.py

The separator does not depend on g, so the two must agree to solver precision (LinearSVC shuffles its coordinates, so ~1e-9, checked at 1e-4).

| bundle | candidate | screen ratio | group_margins ratio | screen larger | group_margins larger | agree |
|---|---|---|---|---|---|---|
| features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz | has_breast_pattern::striped = 1 | 1.000000 | 1.000000 | tie | tie | yes |
| features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz | has_nape_color::black = 0 | 1.000000 | 1.000000 | tie | tie | yes |
| features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz | has_upperparts_color::blue = 1 | 1.000000 | 1.000000 | tie | tie | yes |

**AGREEMENT OK**
