# Session 1 summary (20260917_050006)

Started 2026-09-17 05:00:06 UTC on ng9wohn0xv.

## 00_validate_gd

- command: `python long_horizon.py --validate-only --device cuda`
- start: 05:00:10, end: 05:00:14, duration: 0 min 4 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/00_validate_gd.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): 

```
validating the batched engine on cuda ...
  [cuda float64] batched vs per-eps reference: max rel diff 3.93e-16  (tol 1e-09)  OK
  [cuda float64] iterate margin recomputed directly: max abs diff 8.33e-17  OK
  [cuda float32] precision budget: max rel diff 5.13e-07
VALIDATION OK
```

## 01_smoke_extract

- command: `python extract_features.py --dataset waterbirds --backbone gdro_rn50 --epochs 1 --max-batches 5 --splits train --out-prefix smoke --no-save-model`
- start: 05:00:14, end: 05:00:36, duration: 0 min 22 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/01_smoke_extract.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): 

```
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
100%|██████████| 97.8M/97.8M [00:00<00:00, 279MB/s]
backbone gdro_rn50: d = 2048, task epochs = 1, train mode = gdro

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  training (gdro) ...
  train mode gdro: cell sizes [3498, 184, 1057, 56], dro eta 0.01
  shuffle = True
    epoch 1/1  ce 0.5031  acc 0.7250  q [0.25, 0.24899999797344208, 0.25099998712539673, 0.25]

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...
  embedding: 100%|██████████| 75/75 [00:06<00:00, 11.00batch/s]
  wrote smoke_waterbirds_gdro_rn50_train.npz  phi (4795, 2048)  eps = 0.0501
```

## 20_extract_rwg_rn50

- command: `python extract_features.py --dataset waterbirds --backbone rwg_rn50 --splits train,val,test --out-prefix features_v2`
- start: 05:00:36, end: 05:05:08, duration: 4 min 32 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/20_extract_rwg_rn50.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): state_waterbirds_speed 

```
    epoch 5/10  ce 0.4423  acc 0.8192
    epoch 6/10  ce 0.3571  acc 0.8611
    epoch 7/10  ce 0.2141  acc 0.9233
    epoch 8/10  ce 0.2753  acc 0.8966
    epoch 9/10  ce 0.4319  acc 0.8086
    epoch 10/10  ce 0.2527  acc 0.9109
  saved trained backbone to models/waterbirds_rwg_rn50.pt

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...
  embedding: 100%|██████████| 75/75 [00:13<00:00,  5.71batch/s]
  wrote features_v2_waterbirds_rwg_rn50_train.npz  phi (4795, 2048)  eps = 0.0501

waterbirds/val  n = 1199
  eps = P(g=1) = 0.4996
  P(y=1 | g=0) = 0.2217   P(y=1 | g=1) = 0.2220
  cells (y, attr):
    y=0:  attr=0     467   attr=1     466
    y=1:  attr=0     133   attr=1     133
  smallest cell = 133   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 1199 images ...
  embedding: 100%|██████████| 19/19 [00:07<00:00,  2.65batch/s]
  wrote features_v2_waterbirds_rwg_rn50_val.npz  phi (1199, 2048)  eps = 0.4996

waterbirds/test  n = 5794
  eps = P(g=1) = 0.5000
  P(y=1 | g=0) = 0.2216   P(y=1 | g=1) = 0.2216
  cells (y, attr):
    y=0:  attr=0    2255   attr=1    2255
    y=1:  attr=0     642   attr=1     642
  smallest cell = 642
  embedding 5794 images ...
  embedding: 100%|██████████| 91/91 [00:21<00:00,  4.17batch/s]
  wrote features_v2_waterbirds_rwg_rn50_test.npz  phi (5794, 2048)  eps = 0.5000
```

## 20_extract_gdro_rn50

- command: `python extract_features.py --dataset waterbirds --backbone gdro_rn50 --splits train,val,test --out-prefix features_v2`
- start: 05:05:08, end: 05:10:27, duration: 5 min 19 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/20_extract_gdro_rn50.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): waterbirds_speed.json waterbirds_speed.md waterbirds_speed_curves.npz 

```
    epoch 5/10  ce 0.4272  acc 0.8140  q [0.2070000022649765, 0.28700000047683716, 0.32100000977516174, 0.1850000023841858]
    epoch 6/10  ce 0.3880  acc 0.8302  q [0.20999999344348907, 0.28999999165534973, 0.3310000002384186, 0.16899999976158142]
    epoch 7/10  ce 0.2693  acc 0.8872  q [0.2199999988079071, 0.2849999964237213, 0.3400000035762787, 0.15600000321865082]
    epoch 8/10  ce 0.2111  acc 0.9160  q [0.21799999475479126, 0.2849999964237213, 0.35100001096725464, 0.1459999978542328]
    epoch 9/10  ce 0.1879  acc 0.9281  q [0.2280000001192093, 0.2750000059604645, 0.3529999852180481, 0.14300000667572021]
    epoch 10/10  ce 0.2101  acc 0.9203  q [0.23499999940395355, 0.27000001072883606, 0.3569999933242798, 0.1379999965429306]
  saved trained backbone to models/waterbirds_gdro_rn50.pt

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...
  embedding: 100%|██████████| 75/75 [00:14<00:00,  5.24batch/s]
  wrote features_v2_waterbirds_gdro_rn50_train.npz  phi (4795, 2048)  eps = 0.0501

waterbirds/val  n = 1199
  eps = P(g=1) = 0.4996
  P(y=1 | g=0) = 0.2217   P(y=1 | g=1) = 0.2220
  cells (y, attr):
    y=0:  attr=0     467   attr=1     466
    y=1:  attr=0     133   attr=1     133
  smallest cell = 133   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 1199 images ...
  embedding: 100%|██████████| 19/19 [00:06<00:00,  3.13batch/s]
  wrote features_v2_waterbirds_gdro_rn50_val.npz  phi (1199, 2048)  eps = 0.4996

waterbirds/test  n = 5794
  eps = P(g=1) = 0.5000
  P(y=1 | g=0) = 0.2216   P(y=1 | g=1) = 0.2216
  cells (y, attr):
    y=0:  attr=0    2255   attr=1    2255
    y=1:  attr=0     642   attr=1     642
  smallest cell = 642
  embedding 5794 images ...
  embedding: 100%|██████████| 91/91 [00:14<00:00,  6.35batch/s]
  wrote features_v2_waterbirds_gdro_rn50_test.npz  phi (5794, 2048)  eps = 0.5000
```

## 20_extract_erm_rn50

- command: `python extract_features.py --dataset waterbirds --backbone erm_rn50 --splits train,val,test --out-prefix features_v2`
- start: 05:10:27, end: 05:15:56, duration: 5 min 29 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/20_extract_erm_rn50.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): celeba_sub_separability.json celeba_sub_separability.md 

```
    epoch 5/10  ce 0.5602  acc 0.7679
    epoch 6/10  ce 0.4123  acc 0.8346
    epoch 7/10  ce 0.2770  acc 0.8913
    epoch 8/10  ce 0.3750  acc 0.8507
    epoch 9/10  ce 0.2807  acc 0.8916
    epoch 10/10  ce 0.3301  acc 0.8644
  saved trained backbone to models/waterbirds_erm_rn50.pt

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...
  embedding: 100%|██████████| 75/75 [00:14<00:00,  5.29batch/s]
  wrote features_v2_waterbirds_erm_rn50_train.npz  phi (4795, 2048)  eps = 0.0501

waterbirds/val  n = 1199
  eps = P(g=1) = 0.4996
  P(y=1 | g=0) = 0.2217   P(y=1 | g=1) = 0.2220
  cells (y, attr):
    y=0:  attr=0     467   attr=1     466
    y=1:  attr=0     133   attr=1     133
  smallest cell = 133   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 1199 images ...
  embedding: 100%|██████████| 19/19 [00:05<00:00,  3.35batch/s]
  wrote features_v2_waterbirds_erm_rn50_val.npz  phi (1199, 2048)  eps = 0.4996

waterbirds/test  n = 5794
  eps = P(g=1) = 0.5000
  P(y=1 | g=0) = 0.2216   P(y=1 | g=1) = 0.2216
  cells (y, attr):
    y=0:  attr=0    2255   attr=1    2255
    y=1:  attr=0     642   attr=1     642
  smallest cell = 642
  embedding 5794 images ...
  embedding: 100%|██████████| 91/91 [00:16<00:00,  5.57batch/s]
  wrote features_v2_waterbirds_erm_rn50_test.npz  phi (5794, 2048)  eps = 0.5000
```

## 10_waterbirds_speed

- command: `python long_horizon.py --device cuda --bundles features_waterbirds_*_train.npz --tag waterbirds_speed --max-hours 5.0`
- start: 05:00:36, end: 05:26:00, duration: 25 min 24 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/10_waterbirds_speed.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): celeba_sub_separability.json celeba_sub_separability.md state_waterbirds_speed waterbirds_speed.json waterbirds_speed.md waterbirds_speed_curves.npz 

```
| 998 | 54.5 | 82.9 | 106 | 114 | 95.6 |
| 3.15e+03 | 128 | 190 | 225 | 241 | 208 |
| 9.98e+03 | 282 | 373 | 440 | 477 | 423 |
| 3.15e+04 | 520 | 660 | 783 | 840 | 776 |
| 1e+05 | 844 | 1.04e+03 | 1.15e+03 | 1.19e+03 | 1.18e+03 |

### iterate margin and training accuracy

margin = min over training points of y w.x / ||w|| (negative: some point misclassified). Rises to the maximum margin at rate O(1/log t) (Soudry et al. 2018, Thm 5).

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | -24.7 | -22.6 | -15.3 | -7.21 | -6.27 |
| 3.15 | -19.6 | -14.8 | -9.57 | -4.34 | -5.04 |
| 10 | -12.1 | -9.43 | -6.02 | -3.2 | -3.14 |
| 31.6 | -9.51 | -7.19 | -3.98 | -2.77 | -2.99 |
| 99.8 | -7.65 | -5.46 | -2.4 | -2.22 | -2.72 |
| 316 | -6.27 | -4.57 | -1.38 | -1.38 | -2.21 |
| 998 | -6.05 | -3.49 | -0.613 | -0.796 | -1.13 |
| 3.15e+03 | -4.97 | -1.17 | -0.277 | -0.25 | -0.552 |
| 9.98e+03 | -2.69 | -0.295 | -0.127 | -0.0835 | -0.255 |
| 3.15e+04 | -0.332 | -0.153 | -0.0426 | -0.0232 | -0.0647 |
| 1e+05 | -0.185 | -0.0607 | -0.00893 | -0.00166 | -0.0167 |

minority training accuracy:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.104 | 0.108 | 0.138 | 0.338 | 0.983 |
| 3.15 | 0.096 | 0.113 | 0.200 | 0.471 | 0.967 |
| 10 | 0.096 | 0.154 | 0.321 | 0.600 | 0.971 |
| 31.6 | 0.129 | 0.212 | 0.396 | 0.700 | 0.983 |
| 99.8 | 0.183 | 0.300 | 0.467 | 0.792 | 0.992 |
| 316 | 0.229 | 0.367 | 0.554 | 0.867 | 0.992 |
| 998 | 0.300 | 0.425 | 0.683 | 0.921 | 1.000 |
| 3.15e+03 | 0.379 | 0.546 | 0.817 | 0.958 | 1.000 |
| 9.98e+03 | 0.462 | 0.721 | 0.900 | 0.983 | 1.000 |
| 3.15e+04 | 0.621 | 0.833 | 0.963 | 0.996 | 1.000 |
| 1e+05 | 0.771 | 0.950 | 0.992 | 1.000 | 1.000 |

```

## 20_extract_dinov2

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train,val,test --out-prefix features_v2`
- start: 05:15:56, end: 05:26:50, duration: 10 min 54 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/20_extract_dinov2.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): 

```
backbone dinov2: d = 768, task epochs = 0, train mode = erm

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...
  embedding: 100%|██████████| 75/75 [04:28<00:00,  3.58s/batch]
  wrote features_v2_waterbirds_dinov2_train.npz  phi (4795, 768)  eps = 0.0501

waterbirds/val  n = 1199
  eps = P(g=1) = 0.4996
  P(y=1 | g=0) = 0.2217   P(y=1 | g=1) = 0.2220
  cells (y, attr):
    y=0:  attr=0     467   attr=1     466
    y=1:  attr=0     133   attr=1     133
  smallest cell = 133   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 1199 images ...
  embedding: 100%|██████████| 19/19 [01:07<00:00,  3.53s/batch]
  wrote features_v2_waterbirds_dinov2_val.npz  phi (1199, 768)  eps = 0.4996

waterbirds/test  n = 5794
  eps = P(g=1) = 0.5000
  P(y=1 | g=0) = 0.2216   P(y=1 | g=1) = 0.2216
  cells (y, attr):
    y=0:  attr=0    2255   attr=1    2255
    y=1:  attr=0     642   attr=1     642
  smallest cell = 642
  embedding 5794 images ...
  embedding: 100%|██████████| 91/91 [04:25<00:00,  2.92s/batch]
  wrote features_v2_waterbirds_dinov2_test.npz  phi (5794, 768)  eps = 0.5000
```

## 29_check_v2_bundles

- command: `python -`
- start: 05:26:50, end: 05:27:01, duration: 0 min 11 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/29_check_v2_bundles.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): 

```
features_v2_waterbirds_dinov2_test.npz: phi (5794, 768), eps 0.5000, cells {'y=-1,g=0': 2255, 'y=-1,g=1': 2255, 'y=1,g=0': 642, 'y=1,g=1': 642}, standardized_with=train statistics, train_mode=erm, std stats stored=True, finite=True
features_v2_waterbirds_dinov2_train.npz: phi (4795, 768), eps 0.0501, cells {'y=-1,g=0': 3498, 'y=-1,g=1': 184, 'y=1,g=0': 1057, 'y=1,g=1': 56}, standardized_with=train statistics, train_mode=erm, std stats stored=True, finite=True
features_v2_waterbirds_dinov2_val.npz: phi (1199, 768), eps 0.4996, cells {'y=-1,g=0': 467, 'y=-1,g=1': 466, 'y=1,g=0': 133, 'y=1,g=1': 133}, standardized_with=train statistics, train_mode=erm, std stats stored=True, finite=True
features_v2_waterbirds_erm_rn50_test.npz: phi (5794, 2048), eps 0.5000, cells {'y=-1,g=0': 2255, 'y=-1,g=1': 2255, 'y=1,g=0': 642, 'y=1,g=1': 642}, standardized_with=train statistics, train_mode=erm, std stats stored=True, finite=True
features_v2_waterbirds_erm_rn50_train.npz: phi (4795, 2048), eps 0.0501, cells {'y=-1,g=0': 3498, 'y=-1,g=1': 184, 'y=1,g=0': 1057, 'y=1,g=1': 56}, standardized_with=train statistics, train_mode=erm, std stats stored=True, finite=True
features_v2_waterbirds_erm_rn50_val.npz: phi (1199, 2048), eps 0.4996, cells {'y=-1,g=0': 467, 'y=-1,g=1': 466, 'y=1,g=0': 133, 'y=1,g=1': 133}, standardized_with=train statistics, train_mode=erm, std stats stored=True, finite=True
features_v2_waterbirds_gdro_rn50_test.npz: phi (5794, 2048), eps 0.5000, cells {'y=-1,g=0': 2255, 'y=-1,g=1': 2255, 'y=1,g=0': 642, 'y=1,g=1': 642}, standardized_with=train statistics, train_mode=gdro, std stats stored=True, finite=True
features_v2_waterbirds_gdro_rn50_train.npz: phi (4795, 2048), eps 0.0501, cells {'y=-1,g=0': 3498, 'y=-1,g=1': 184, 'y=1,g=0': 1057, 'y=1,g=1': 56}, standardized_with=train statistics, train_mode=gdro, std stats stored=True, finite=True
features_v2_waterbirds_gdro_rn50_val.npz: phi (1199, 2048), eps 0.4996, cells {'y=-1,g=0': 467, 'y=-1,g=1': 466, 'y=1,g=0': 133, 'y=1,g=1': 133}, standardized_with=train statistics, train_mode=gdro, std stats stored=True, finite=True
features_v2_waterbirds_rwg_rn50_test.npz: phi (5794, 2048), eps 0.5000, cells {'y=-1,g=0': 2255, 'y=-1,g=1': 2255, 'y=1,g=0': 642, 'y=1,g=1': 642}, standardized_with=train statistics, train_mode=rwg, std stats stored=True, finite=True
features_v2_waterbirds_rwg_rn50_train.npz: phi (4795, 2048), eps 0.0501, cells {'y=-1,g=0': 3498, 'y=-1,g=1': 184, 'y=1,g=0': 1057, 'y=1,g=1': 56}, standardized_with=train statistics, train_mode=rwg, std stats stored=True, finite=True
features_v2_waterbirds_rwg_rn50_val.npz: phi (1199, 2048), eps 0.4996, cells {'y=-1,g=0': 467, 'y=-1,g=1': 466, 'y=1,g=0': 133, 'y=1,g=1': 133}, standardized_with=train statistics, train_mode=rwg, std stats stored=True, finite=True
```

## 30_celeba_subsample

- command: `python celeba_subsample.py --bundles features_celeba_*_train.npz --sizes 5000,10000,20000`
- start: 05:00:36, end: 07:12:20, duration: 131 min 44 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/30_celeba_subsample.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): celeba_sub_separability.json celeba_sub_separability.md state_waterbirds_speed waterbirds_speed.json waterbirds_speed.md waterbirds_speed_curves.npz 

```
  sizes celeba_erm_rn50:  67%|██████▋   | 2/3 [13:11<06:37, 397.03s/size][A
  sizes celeba_erm_rn50: 100%|██████████| 3/3 [20:33<00:00, 411.02s/size]
bundles:  75%|███████▌  | 3/4 [1:20:54<26:06, 1566.33s/bundle]    wrote ./features_celeba_erm_rn50_n5000_train.npz  cells {'y=-1,g=0': 2200, 'y=-1,g=1': 2054, 'y=1,g=0': 703, 'y=1,g=1': 43}
  separable=True  margin=0.02569  [logistic (constructive), 385.6s]
  wrote ./features_celeba_erm_rn50_n10000_train.npz  cells {'y=-1,g=0': 4401, 'y=-1,g=1': 4108, 'y=1,g=0': 1406, 'y=1,g=1': 85}
  separable=True  margin=0.006463  [logistic (constructive), 398.7s]
  wrote ./features_celeba_erm_rn50_n20000_train.npz  cells {'y=-1,g=0': 8801, 'y=-1,g=1': 8217, 'y=1,g=0': 2811, 'y=1,g=1': 171}
  separable=True  margin=0.001235  [logistic (constructive), 432.4s]
[celeba_under_rn50] loading features_celeba_under_rn50_train.npz

  sizes celeba_under_rn50:   0%|          | 0/3 [00:00<?, ?size/s][A
  sizes celeba_under_rn50:  33%|███▎      | 1/3 [06:33<13:07, 393.78s/size][A
  sizes celeba_under_rn50:  67%|██████▋   | 2/3 [13:25<06:44, 404.26s/size][A
  sizes celeba_under_rn50: 100%|██████████| 3/3 [50:34<00:00, 1011.63s/size]
bundles: 100%|██████████| 4/4 [2:11:42<00:00, 1975.70s/bundle]
  wrote ./features_celeba_under_rn50_n5000_train.npz  cells {'y=-1,g=0': 2200, 'y=-1,g=1': 2054, 'y=1,g=0': 703, 'y=1,g=1': 43}
  separable=True  margin=0.008284  [logistic (constructive), 391.3s]
  wrote ./features_celeba_under_rn50_n10000_train.npz  cells {'y=-1,g=0': 4401, 'y=-1,g=1': 4108, 'y=1,g=0': 1406, 'y=1,g=1': 85}
  separable=True  margin=0.000463  [logistic (constructive), 406.7s]
  wrote ./features_celeba_under_rn50_n20000_train.npz  cells {'y=-1,g=0': 8801, 'y=-1,g=1': 8217, 'y=1,g=0': 2811, 'y=1,g=1': 171}
  separable=None  margin=nan  [323 violations left; LP hit time limit, 2219.7s]
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

```

## 40_curves_to_json

- command: `python -`
- start: 07:12:20, end: 07:12:20, duration: 0 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session1_20260917_050006/40_curves_to_json.log`
- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): waterbirds_speed_curves.json 

```
results/waterbirds_speed_curves.npz -> results/waterbirds_speed_curves.json (27 arrays)
```

---

Finished 2026-09-17 07:12:20 UTC, total 132 min.

Result files for review: results/waterbirds_speed.{md,json}, results/waterbirds_speed_curves.json,
results/celeba_sub_separability.{md,json}, results/logs/session1_20260917_050006/bundles_v2.txt, and this file.
