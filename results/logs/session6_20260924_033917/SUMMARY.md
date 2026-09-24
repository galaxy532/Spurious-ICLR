# Session 6 summary (20260924_033917)

Started 2026-09-24 03:39:17 UTC on n3shwd4y5g.

- train bundle: `features_v4_waterbirds_dinov2_train.npz`
- test bundle (replication): `features_v3_waterbirds_dinov2_test.npz`
- minimum group size: 0.01 of n
- no GPU work in this session.

## 00_selftest_cub_meta

- command: `python cub_meta.py --self-test`
- start: 03:39:18, end: 03:39:18, duration: 0 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/00_selftest_cub_meta.log`
- new files in results/: 

```
cub_meta self-test
  ok   find_cub locates a real-shaped tree
attribute labels: 100%|██████████| 36/36 [00:00<00:00, 289262.34line/s]
  ok   attributes recovered exactly; the six-field line is counted, not dropped
  ok   part visibility recovered exactly
  ok   attribute names read from the archive root
attribute labels: 100%|██████████| 35/35 [00:00<00:00, 457322.87line/s]
  ok   a missing (image, attribute) pair is caught
scan archive: 11member [00:00, 8396.24member/s]
###################################################################################################################################################################### 100.0%
  ok   archive extraction takes the 5 text files and no image
  ok   the hand-extracted layout (tar with the member list) is found, names included
download: 100%|██████████| 300000/300000 [00:00<00:00, 781062197.39B/s]
  ok   download delivers identical bytes via curl, urllib
SELF-TEST OK
```

## 01_selftest_sv_screen

- command: `python sv_screen.py --self-test`
- start: 03:39:18, end: 03:39:21, duration: 0 min 3 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/01_selftest_sv_screen.log`
- new files in results/: 

```
sv_screen self-test
  ok   exact label-stratified p (min) 0.0491 matches Monte Carlo 0.0507
  ok   exact depletion p 0.0841 matches Monte Carlo 0.0838 (k=1, expected 3.68)
  ok   Holm step-down on a known vector
  ok   small label-pure group: AUC 0.545 passes, phi +0.265 catches it
candidates: 100%|██████████| 41/41 [00:00<00:00, 1695.04cand/s]
  ok   planted margin-free group found (ratio 15.093, Holm p 4.4e-26); 0 of 40 random decoys called (|S| = 20)
  c.npz                             : 100%|██████████| 5/5 [00:00<00:00, 19599.55C/s]
  ok   unmodified group_margins.py reports the same ratio (15.093054)
  ok   detectability floor at n=4795, |S|=421, 650 tests: m >= 102
SELF-TEST OK
```

## 02_integration

- command: `python validate_session6.py`
- start: 03:39:21, end: 03:39:29, duration: 0 min 8 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/02_integration.log`
- new files in results/: 

```
session 6 integration check
checks: 100%|██████████| 7/7 [00:07<00:00,  1.12s/check]

  cub_meta.py CLI joins a mini tree
  sv_screen.py CLI finds the planted attribute, no coin-flip one
  unmodified group_margins.py measures the confirmation bundles
  sv_screen.py --compare reports agreement
  confirmation bundle features are bit-identical to the source
  the replication run on the test split confirms the planted hit
  cub_meta.py refuses a tree with a Waterbirds image missing

INTEGRATION OK -- the real command line joins the CUB metadata, finds a planted margin-free attribute and no coin-flip one, and the unmodified group_margins.py agrees with the screen.
```

## 10_bird_fraction

- reused `results/v5_bird_fraction.csv` from session 5.

## 20_cub_meta

- command: `python cub_meta.py --fractions results/v5_bird_fraction.csv`
- start: 03:39:29, end: 03:39:36, duration: 0 min 7 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/20_cub_meta.log`
- new files in results/: v6_cub_meta.json v6_cub_meta.md v6_cub_meta.npz 

```
md5: 100%|██████████| 1150585339/1150585339 [00:01<00:00, 589546027.12B/s]
CUB directory: /notebooks/Spurious-ICLR/../data/CUB_200_2011
attribute labels: 100%|██████████| 3677856/3677856 [00:02<00:00, 1358910.61line/s]
# CUB-200-2011 metadata joined to Waterbirds (session 6)

Read-only join by filename. Nothing is modified; no image is opened.

## Verdict

**OK -- every check passed**

| check | value |
|---|---|
| Waterbirds images | 11788 |
| found in CUB images.txt | 11788 |
| missing from v5_bird_fraction.csv | 0 |
| attributes per image | 312 |
| attribute lines / irregular / unparsable | 3677856 / 606 / 0 |
| every (image, attribute) pair present | True |
| no duplicated (image, attribute) pair | True |
| every (image, part) pair present exactly once | True |
| every image has a bounding box | True |
| mask/composite dimensions all match (session 5) | True |
| attribute names file | `/notebooks/data/attributes.txt` |

- metadata source: existing extraction
- archive md5: `97eceeb196236b17998738112f37df78` -- matches the value CaltechDATA publishes

## The difficulty proxies (all splits)

| proxy | min | q05 | q25 | median | q75 | q95 | max |
|---|---|---|---|---|---|---|---|
| bird_frac | 0.01005 | 0.04044 | 0.07711 | 0.1158 | 0.1666 | 0.2699 | 0.7346 |
| bbox_area_frac | 0.05005 | 0.1121 | 0.2075 | 0.3174 | 0.4541 | 0.6896 | 1 |
| n_visible_parts | 3 | 9 | 11 | 12 | 13 | 13 | 15 |
| frac_attr_not_visible | 0 | 0 | 0 | 0.04808 | 0.1571 | 0.3622 | 0.891 |

- `bird_in_frame` (bounding box at least 2 px from every edge): 93.6% of images


wrote results/v6_cub_meta.npz
```

## 30_sv_screen

- command: `python sv_screen.py --bundle features_v4_waterbirds_dinov2_train.npz --meta results/v6_cub_meta.npz --min-frac 0.01`
- start: 03:39:36, end: 03:39:50, duration: 0 min 14 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/30_sv_screen.log`
- new files in results/: features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz v6_confirm_list.txt v6_sv_screen.json v6_sv_screen.md v6_sv_screen_full.md 

```
row order check: ok (4795 rows, y and place both match)
separator: margin 0.6312 at C = 1e+06, plateau 1.3e-09
candidates: 100%|██████████| 650/650 [00:02<00:00, 246.14cand/s]
|S| = 418; 323 of 650 candidates eligible; floor m >= 96; hits: 0
confirm bundles: 100%|██████████| 3/3 [00:03<00:00,  1.22s/bundle]
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

```

## 40_confirm_margins

- command: `python group_margins.py --no-lp --tag v6_confirm_margins --bundles results/features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz results/features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz results/features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz`
- start: 03:39:50, end: 03:40:10, duration: 0 min 20 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/40_confirm_margins.log`
- new files in results/: v6_confirm_margins.json v6_confirm_margins.md 

```
bundles:   0%|          | 0/3 [00:00<?, ?bundle/s]
results/features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz

  features_v6_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 1579.30C/s]
bundles:  33%|███▎      | 1/3 [00:06<00:12,  6.42s/bundle]  margin 0.6312   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

results/features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz

  features_v6_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 1678.26C/s]
bundles:  67%|██████▋   | 2/3 [00:12<00:06,  6.09s/bundle]  margin 0.6312   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

results/features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz

  features_v6_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 1495.51C/s]
bundles: 100%|██████████| 3/3 [00:18<00:00,  6.17s/bundle]
  margin 0.6312   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

wrote results/v6_confirm_margins.json
wrote results/v6_confirm_margins.md
```

## 41_confirm_check

- command: `python sv_screen.py --compare results/v6_confirm_margins.json`
- start: 03:40:10, end: 03:40:11, duration: 0 min 1 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/41_confirm_check.log`
- new files in results/: v6_confirm_check.md 

```
# Session 6 -- screen vs the unmodified group_margins.py

The separator does not depend on g, so the two must agree to solver precision (LinearSVC shuffles its coordinates, so ~1e-9, checked at 1e-4).

| bundle | candidate | screen ratio | group_margins ratio | screen larger | group_margins larger | agree |
|---|---|---|---|---|---|---|
| features_v6_waterbirds_dinov2_train_has-breast-pattern-striped-1.npz | has_breast_pattern::striped = 1 | 1.000000 | 1.000000 | tie | tie | yes |
| features_v6_waterbirds_dinov2_train_has-nape-color-black-0.npz | has_nape_color::black = 0 | 1.000000 | 1.000000 | tie | tie | yes |
| features_v6_waterbirds_dinov2_train_has-upperparts-color-blue-1.npz | has_upperparts_color::blue = 1 | 1.000000 | 1.000000 | tie | tie | yes |

**AGREEMENT OK**

```

## 50_sv_screen_test

- command: `python sv_screen.py --bundle features_v3_waterbirds_dinov2_test.npz --split test --meta results/v6_cub_meta.npz --min-frac 0.01 --tag v6_sv_screen_test --no-confirm`
- start: 03:40:11, end: 03:40:29, duration: 0 min 18 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/50_sv_screen_test.log`
- new files in results/: v6_sv_screen_test.json v6_sv_screen_test.md v6_sv_screen_test_full.md 

```
row order check: ok (5794 rows, y and place both match)
separator: margin 0.3505 at C = 1e+06, plateau 6.0e-10
candidates: 100%|██████████| 650/650 [00:03<00:00, 179.55cand/s]
|S| = 515; 362 of 650 candidates eligible; floor m >= 95; hits: 0
confirm bundles: 0bundle [00:00, ?bundle/s]
# Session 6 -- does any image property avoid the margin set?

- bundle: `features_v3_waterbirds_dinov2_test.npz`   n = 5794
- separator: LinearSVC hinge, no intercept, C = 1e+06, margin u* = 0.3505, plateau 6.0e-10
- **margin set S** (u within 0.001 relative of u*): **515** points (91 waterbirds, 424 landbirds)
- family: 650 candidates (A_attribute 624, B_proxy_tail 24, C_binary 2); **362 eligible** after the size / label-balance / label-proxy filters (minimum group size 58)
- **detectability floor: m >= 95** (unstratified bound). A margin-free group of at least this size, with the data's label mix, survives Holm. Below it, a margin-free group can arise by chance.

## Answer

**NO HIT.** No eligible candidate contains zero margin points beyond what chance allows. See the graded table below for how close the best came.

## The closest candidates, by depletion of margin points

`margin pts in G` against `expected` is the graded statistic: how many of the S points fall in G, against the label-matched expectation. `p (depletion)` is the exact probability of that few or fewer under the null. `ratio` exceeds 1 only when the count is 0.

| candidate G | m | waterbird/landbird | ratio | larger | margin pts in G | expected | p (min) | Holm p (min) | p (depletion) | Holm p (depl.) | AUC y | phi y | AUC place | AUC old g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| has_bill_color::brown = 0 | 5306 | 1210/4096 | 1.0000 | tie | 444 | 470.8 | 1.00e+00 | 1.00e+00 | 2.19e-05 | 7.93e-03 | 0.517 | +0.051 | 0.508 | 0.508 |
| has_throat_color::brown = 0 | 5318 | 1183/4135 | 1.0000 | tie | 449 | 472.6 | 1.00e+00 | 1.00e+00 | 1.25e-04 | 4.53e-02 | 0.502 | +0.007 | 0.506 | 0.505 |
| has_upper_tail_color::brown = 0 | 4966 | 1146/3820 | 1.0000 | tie | 413 | 440.4 | 1.00e+00 | 1.00e+00 | 3.32e-04 | 1.19e-01 | 0.523 | +0.054 | 0.504 | 0.504 |
| has_under_tail_color::brown = 0 | 4903 | 1160/3743 | 1.0000 | tie | 410 | 434.1 | 1.00e+00 | 1.00e+00 | 1.73e-03 | 6.20e-01 | 0.537 | +0.085 | 0.508 | 0.505 |
| has_forehead_color::black = 0 | 3930 | 778/3152 | 1.0000 | tie | 322 | 351.5 | 1.00e+00 | 1.00e+00 | 2.18e-03 | 7.82e-01 | 0.454 | -0.083 | 0.491 | 0.488 |
| bbox_area_frac top 0.25 (>= 0.4512) | 1449 | 382/1067 | 1.0000 | tie | 101 | 127.4 | 5.83e-01 | 1.00e+00 | 2.33e-03 | 8.32e-01 | 0.530 | +0.058 | 0.499 | 0.502 |
| has_shape::hawk-like = 0 | 5605 | 1243/4362 | 1.0000 | tie | 486 | 498.2 | 1.00e+00 | 1.00e+00 | 2.36e-03 | 8.39e-01 | 0.500 | +0.002 | 0.499 | 0.502 |
| has_wing_color::brown = 0 | 4237 | 1026/3211 | 1.0000 | tie | 348 | 374.6 | 1.00e+00 | 1.00e+00 | 3.70e-03 | 1.00e+00 | 0.544 | +0.082 | 0.509 | 0.509 |
| has_back_color::brown = 0 | 4689 | 1092/3597 | 1.0000 | tie | 392 | 415.6 | 1.00e+00 | 1.00e+00 | 4.05e-03 | 1.00e+00 | 0.526 | +0.056 | 0.504 | 0.505 |
| has_head_pattern::masked = 1 | 252 | 77/175 | 1.0000 | tie | 11 | 21.9 | 1.88e-01 | 1.00e+00 | 5.26e-03 | 1.00e+00 | 0.511 | +0.043 | 0.505 | 0.500 |
| has_primary_color::brown = 0 | 4357 | 1034/3323 | 1.0000 | tie | 361 | 385.7 | 2.63e-01 | 1.00e+00 | 5.46e-03 | 1.00e+00 | 0.534 | +0.066 | 0.508 | 0.507 |
| has_shape::pigeon-like = 0 | 5424 | 1199/4225 | 1.0000 | tie | 468 | 482.2 | 1.00e+00 | 1.00e+00 | 6.58e-03 | 1.00e+00 | 0.498 | -0.005 | 0.497 | 0.500 |
| has_belly_pattern::multi-colored = 0 | 4874 | 1152/3722 | 1.0000 | tie | 411 | 431.6 | 1.00e+00 | 1.00e+00 | 6.78e-03 | 1.00e+00 | 0.536 | +0.082 | 0.506 | 0.509 |
| has_bill_shape::dagger = 1 | 786 | 235/551 | 1.0000 | tie | 51 | 68.5 | 2.23e-02 | 1.00e+00 | 8.70e-03 | 1.00e+00 | 0.530 | +0.074 | 0.495 | 0.499 |
| has_head_pattern::eyeline = 1 | 670 | 122/548 | 1.0000 | tie | 44 | 60.2 | 6.98e-01 | 1.00e+00 | 1.00e-02 | 1.00e+00 | 0.487 | -0.034 | 0.498 | 0.497 |
| has_bill_color::buff = 1 | 823 | 189/634 | 1.0000 | tie | 56 | 73.0 | 4.73e-02 | 1.00e+00 | 1.25e-02 | 1.00e+00 | 0.503 | +0.008 | 0.498 | 0.493 |
| has_upperparts_color::brown = 0 | 4411 | 1050/3361 | 1.0000 | tie | 369 | 390.4 | 2.55e-01 | 1.00e+00 | 1.28e-02 | 1.00e+00 | 0.536 | +0.071 | 0.507 | 0.508 |
| has_belly_pattern::solid = 1 | 3321 | 731/2590 | 1.0000 | tie | 271 | 295.3 | 1.00e+00 | 1.00e+00 | 1.33e-02 | 1.00e+00 | 0.498 | -0.004 | 0.505 | 0.508 |
| has_crown_color::brown = 0 | 4716 | 1133/3583 | 1.0000 | tie | 399 | 417.1 | 1.00e+00 | 1.00e+00 | 1.99e-02 | 1.00e+00 | 0.544 | +0.094 | 0.504 | 0.506 |
| has_forehead_color::brown = 0 | 4796 | 1141/3655 | 1.0000 | tie | 407 | 424.5 | 3.59e-02 | 1.00e+00 | 2.08e-02 | 1.00e+00 | 0.539 | +0.086 | 0.508 | 0.502 |
| has_primary_color::blue = 1 | 367 | 24/343 | 1.0000 | tie | 23 | 33.9 | 1.04e-01 | 1.00e+00 | 2.13e-02 | 1.00e+00 | 0.471 | -0.098 | 0.501 | 0.502 |
| has_leg_color::brown = 0 | 5348 | 1230/4118 | 1.0000 | tie | 462 | 474.3 | 1.00e+00 | 1.00e+00 | 2.40e-02 | 1.00e+00 | 0.522 | +0.070 | 0.504 | 0.507 |
| has_shape::upland-ground-like = 0 | 5662 | 1257/4405 | 1.0000 | tie | 496 | 503.2 | 1.00e+00 | 1.00e+00 | 2.44e-02 | 1.00e+00 | 0.501 | +0.006 | 0.498 | 0.500 |
| has_wing_color::blue = 1 | 296 | 13/283 | 1.0000 | tie | 18 | 27.5 | 6.22e-01 | 1.00e+00 | 2.63e-02 | 1.00e+00 | 0.474 | -0.099 | 0.499 | 0.500 |
| has_bill_color::black = 0 | 3219 | 825/2394 | 1.0000 | tie | 263 | 283.5 | 4.69e-01 | 1.00e+00 | 3.11e-02 | 1.00e+00 | 0.556 | +0.093 | 0.497 | 0.501 |

Every candidate, eligible or not, is in the `.json` next to this file.

```

## 51_replicate

- command: `python sv_screen.py --replicate results/v6_sv_screen_test.json`
- start: 03:40:29, end: 03:40:31, duration: 0 min 2 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_033917/51_replicate.log`
- new files in results/: v6_replication.md 

```
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

```

## WHAT TO DO NEXT

Read in this order.

1. `results/v6_cub_meta.md` -- the verdict must be OK (every Waterbirds image
   found in CUB, every attribute and part present exactly once). If not, nothing
   below was run.
2. `results/v6_confirm_check.md` -- must say AGREEMENT OK. If not, the screen
   is not measuring what group_margins.py measures and must not be read.
3. `results/v6_sv_screen.md` -- the Answer section, then the depletion table.
   Note |S| and the detectability floor at the top.
4. `results/v6_replication.md` -- only meaningful if step 3 has a hit. A hit
   that does not replicate on the test split is not a finding.

Total wall clock: 1 min.

