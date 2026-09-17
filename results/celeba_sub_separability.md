## CelebA nested subsamples -- separability of the full subsample

Through the origin (no intercept). margin = min_i y_i w.x_i / ||w|| for the separator found; a lower bound on the maximum margin. 'LP proof' = proven.

| bundle | n | n_min (g=1) | separable | margin | decided by | cells |
|---|---|---|---|---|---|---|
| celeba_clip_n5000_train | 5000 | 2097 | **no** | 0 | LP proof (max margin is 0) | {'y=-1,g=0': 2200, 'y=-1,g=1': 2054, 'y=1,g=0': 703, 'y=1,g=1': 43} |
| celeba_clip_n10000_train | 10000 | 4193 | **no** | 0 | LP proof (max margin is 0) | {'y=-1,g=0': 4401, 'y=-1,g=1': 4108, 'y=1,g=0': 1406, 'y=1,g=1': 85} |
| celeba_clip_n20000_train | 20000 | 8388 | **no** | 0 | LP proof (max margin is 0) | {'y=-1,g=0': 8801, 'y=-1,g=1': 8217, 'y=1,g=0': 2811, 'y=1,g=1': 171} |
| celeba_dinov2_n5000_train | 5000 | 2097 | ? | -- | 851 violations left; LP hit time limit | {'y=-1,g=0': 2200, 'y=-1,g=1': 2054, 'y=1,g=0': 703, 'y=1,g=1': 43} |
| celeba_dinov2_n10000_train | 10000 | 4193 | ? | -- | 1797 violations left; LP hit time limit | {'y=-1,g=0': 4401, 'y=-1,g=1': 4108, 'y=1,g=0': 1406, 'y=1,g=1': 85} |
| celeba_dinov2_n20000_train | 20000 | 8388 | ? | -- | 3681 violations left; LP hit time limit | {'y=-1,g=0': 8801, 'y=-1,g=1': 8217, 'y=1,g=0': 2811, 'y=1,g=1': 171} |
| celeba_erm_rn50_n5000_train | 5000 | 2097 | yes | 0.02569 | logistic (constructive) | {'y=-1,g=0': 2200, 'y=-1,g=1': 2054, 'y=1,g=0': 703, 'y=1,g=1': 43} |
| celeba_erm_rn50_n10000_train | 10000 | 4193 | yes | 0.006463 | logistic (constructive) | {'y=-1,g=0': 4401, 'y=-1,g=1': 4108, 'y=1,g=0': 1406, 'y=1,g=1': 85} |
| celeba_erm_rn50_n20000_train | 20000 | 8388 | yes | 0.001235 | logistic (constructive) | {'y=-1,g=0': 8801, 'y=-1,g=1': 8217, 'y=1,g=0': 2811, 'y=1,g=1': 171} |
| celeba_under_rn50_n5000_train | 5000 | 2097 | yes | 0.008284 | logistic (constructive) | {'y=-1,g=0': 2200, 'y=-1,g=1': 2054, 'y=1,g=0': 703, 'y=1,g=1': 43} |
| celeba_under_rn50_n10000_train | 10000 | 4193 | yes | 0.000463 | logistic (constructive) | {'y=-1,g=0': 4401, 'y=-1,g=1': 4108, 'y=1,g=0': 1406, 'y=1,g=1': 85} |
| celeba_under_rn50_n20000_train | 20000 | 8388 | ? | -- | 323 violations left; LP hit time limit | {'y=-1,g=0': 8801, 'y=-1,g=1': 8217, 'y=1,g=0': 2811, 'y=1,g=1': 171} |
