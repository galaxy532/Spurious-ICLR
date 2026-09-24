# Session 6 -- do the train-split hits replicate on the test split?

- train: `features_v4_waterbirds_dinov2_train.npz`, |S| = 418
- test: separator re-fitted on the test photographs; same candidate definitions (proxy cuts re-computed on the test split's own values).

**Nothing to replicate: the train screen had no hit.**

For information only, the three best-ranked train candidates on test:

| candidate G | train: m, margin pts, ratio | test: m | test margin pts / expected | test ratio | test larger | test p (min) | Holm over hits | replicates |
|---|---|---|---|---|---|---|---|---|
| has_breast_pattern::striped = 1 | 540, 37, 1.0000 | 682 | 60 / 61.8 | 1.0000 | tie | 6.09e-01 | 1.00e+00 | no |
| has_nape_color::black = 0 | 3505, 309, 1.0000 | 4287 | 374 / 382.8 | 1.0000 | tie | 1.00e+00 | 1.00e+00 | no |
| has_upperparts_color::blue = 1 | 244, 23, 1.0000 | 301 | 18 / 28.0 | 1.0000 | tie | 1.54e-01 | 4.61e-01 | no |
