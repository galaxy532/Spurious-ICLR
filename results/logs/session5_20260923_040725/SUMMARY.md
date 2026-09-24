# Session 5 summary (20260923_040725)

Started 2026-09-23 04:07:25 UTC on nih0mwal0a.

- source bundle: `features_v4_waterbirds_dinov2_train.npz`
- partition rules: `median tercile`
- no GPU work in this session; everything below is CPU.

## 00_selftest_cub_masks

- command: `python cub_masks.py --self-test`
- start: 04:07:27, end: 04:07:27, duration: 0 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/00_selftest_cub_masks.log`
- new files in results/: 

```
cub_masks self-test
masks: 100%|██████████| 3/3 [00:00<00:00, 3793.46img/s]
  ok   bird fraction exact on a known mask (0.20)
  ok   matching dimensions accepted
  ok   dimension mismatch caught
  ok   missing mask reported, no crash
  ok   alignment refused when a mismatch is present
  ok   AUC = 1.0 / 0.0 / 0.5 on separated, anti-separated, tied
  ok   find_segmentations locates a real-shaped tree
masks: 100%|██████████| 12/12 [00:00<00:00, 5311.49img/s]
  ok   end-to-end run over a mini dataset, all columns present
SELF-TEST OK
```

## 01_selftest_regroup

- command: `python regroup.py --self-test`
- start: 04:07:27, end: 04:07:28, duration: 0 min 1 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/01_selftest_regroup.log`
- new files in results/: 

```
regroup self-test
  ok   median keeps every row and puts the large birds in g=1
  ok   tercile drops the middle third, large birds in g=1
  ok   --invert flips the assignment
  ok   aligned bundle accepted
  ok   shuffled bundle caught
  ok   round-trip keeps the right rows, g, and re-standardises
SELF-TEST OK
```

## 02_selftest_margin_power

- command: `python margin_power.py --self-test`
- start: 04:07:28, end: 04:07:30, duration: 0 min 2 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/02_selftest_margin_power.log`
- new files in results/: 

```
margin_power self-test
  ok   translation adds exactly delta to the chosen margins
  ok   every other point is untouched
  ok   delta = 0 is exactly the identity
  ok   pinned control recovers its planted ratio (1.6000)
  ok   an unplanted random split is a tie
  ok   oracle target 2.266 before re-solving; after re-solving 1.784073 (non-tie) -- informational
SELF-TEST OK
```

## 03_integration

- command: `python validate_session5.py`
- start: 04:07:30, end: 04:07:36, duration: 0 min 6 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/03_integration.log`
- new files in results/: 

```
session 5 integration check
checks:   0%|          | 0/5 [00:00<?, ?check/s]
  regrouped.npz                     : 100%|██████████| 3/3 [00:00<00:00, 14446.51C/s]

  planted.npz                       : 100%|██████████| 3/3 [00:00<00:00, 17236.87C/s]
checks: 100%|██████████| 5/5 [00:04<00:00,  1.00check/s]

  regrouped bundle loads in group_margins unchanged
  phi and y pass through untouched when no row is dropped
  row-order guard fires on a shuffled bundle
  a planted asymmetry survives the session-5 path
  the real command line runs end to end on a mini dataset

INTEGRATION OK -- a re-partitioned bundle is read by the unmodified group_margins.py, the features are untouched, a shuffled bundle is refused, and a planted ratio survives the path.
```

## 10_margin_power

- command: `python margin_power.py --bundle features_v4_waterbirds_dinov2_train.npz --tag v5_margin_power`
- start: 04:07:36, end: 04:10:20, duration: 2 min 44 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/10_margin_power.log`
- new files in results/: v5_margin_power.json v5_margin_power.md 

```
# Can the per-group margin ratio be anything but 1 on these features?

An instrument calibration, not a result. See the module docstring for why
`validate_group_margins.py`'s 1.6000 does not answer this.

- bundle: `features_v4_waterbirds_dinov2_train.npz`   n = 4795, d = 768
- baseline margin: **0.6312**
- tie tolerance `0.001`, C ladder [100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

## Answer

**YES.** The oracle partition reaches a ratio of 9.9815 at frac=0.05. A non-tie is attainable on these features, so a real partition is worth trying; compare its ratio against this ceiling.

## Every configuration

`target` is the ratio under the BASELINE separator, before re-solving.
`ratio` is what `group_margins.py` reports after re-solving. The gap
between them is how much the optimiser undoes.

| sweep | knob | n_g1 | target | ratio | verdict | q01 ratio | q05 ratio | SV g0 | SV g1 |
|---|---|---|---|---|---|---|---|---|---|
| oracle | frac=0.05 | 240 | 9.9815 | 9.981470 | g=1 | 9.9894 | 10.1040 | 421 | 10 |
| oracle | frac=0.1 | 480 | 7.2260 | 7.225966 | g=1 | 7.2538 | 7.4007 | 421 | 9 |
| oracle | frac=0.25 | 1199 | 3.4425 | 3.442479 | g=1 | 3.4672 | 3.5860 | 421 | 17 |
| oracle | frac=0.5 | 2398 | 2.4033 | 2.403320 | g=1 | 2.4162 | 2.4830 | 421 | 39 |
| oracle | frac=0.75 | 3596 | 1.6702 | 1.670204 | g=1 | 1.6904 | 1.7824 | 421 | 33 |
| oracle | frac=0.9 | 4316 | 1.0775 | 1.077547 | g=1 | 1.1393 | 1.3103 | 421 | 8 |
| pinned | m1=1.05 | 1196 | 1.0500 | 1.050000 | g=1 | 1.0500 | 1.0500 | 3596 | 1196 |
| pinned | m1=1.6 | 1196 | 1.6000 | 1.600000 | g=1 | 1.6000 | 1.6000 | 3596 | 1196 |
| translate | delta=0.05 seed=0 | 2397 | 1.0500 | 1.000000 | tie | 1.0000 | 1.0000 | 236 | 183 |
| translate | delta=0.2 seed=0 | 2397 | 1.2000 | 1.000000 | tie | 1.0000 | 1.0000 | 251 | 148 |
| translate | delta=0.8 seed=0 | 2397 | 1.8000 | 1.000000 | tie | 1.0000 | 1.2338 | 312 | 74 |
| translate | delta=2 seed=0 | 2397 | 3.0000 | 1.000000 | tie | 1.1753 | 1.9215 | 331 | 16 |
| heavy_tail | q=1 seed=0 | 1199 | 3.4425 | 3.442479 | g=1 | 3.4672 | 3.5860 | 421 | 17 |
| heavy_tail | q=0.999 seed=0 | 1199 | 1.8121 | 1.812081 | g=1 | 3.4672 | 3.5860 | 421 | 1 |
| heavy_tail | q=0.99 seed=0 | 1199 | 1.0000 | 1.000000 | tie | 3.4588 | 3.5860 | 419 | 2 |
| heavy_tail | q=0.95 seed=0 | 1199 | 1.0000 | 1.000000 | tie | 1.2655 | 3.5591 | 413 | 8 |
| translate | delta=0.05 seed=1 | 2397 | 1.0500 | 1.000000 | tie | 1.0000 | 1.0000 | 227 | 195 |
| translate | delta=0.2 seed=1 | 2397 | 1.2000 | 1.000000 | tie | 1.0000 | 1.0000 | 246 | 156 |
| translate | delta=0.8 seed=1 | 2397 | 1.8000 | 1.000000 | tie | 1.0000 | 1.2443 | 302 | 69 |
| translate | delta=2 seed=1 | 2397 | 3.0000 | 1.000000 | tie | 1.1302 | 1.9266 | 339 | 21 |
| heavy_tail | q=1 seed=1 | 1199 | 3.4425 | 3.442479 | g=1 | 3.4672 | 3.5860 | 421 | 17 |
| heavy_tail | q=0.999 seed=1 | 1199 | 1.2679 | 1.267933 | g=1 | 3.4672 | 3.5860 | 421 | 1 |
| heavy_tail | q=0.99 seed=1 | 1199 | 1.0000 | 1.000000 | tie | 3.4628 | 3.5860 | 418 | 3 |
| heavy_tail | q=0.95 seed=1 | 1199 | 1.0000 | 1.000000 | tie | 1.2379 | 3.5642 | 412 | 9 |

## How to read this

- `pinned` must recover its `m1`. If it does not, the instrument is broken
  at this n and d and nothing else on the page can be read.
- `oracle` is the ceiling. Every row a tie means no group variable can
  produce a non-tie on this representation, bird size included.
- `translate` shows how much a naive planted gap is undone by re-solving.
  A large `target` next to a `ratio` of 1 is the optimiser rotating away
  from the asymmetry -- the same mechanism that makes real partitions tie.
- `heavy_tail` shows what one hard member does to a group's ess inf.
- If the `q01`/`q05` ratios move while `ratio` stays at 1, the asymmetry is
  real and the ess inf is hiding it. That is a question for the theory, not
  for this code.

```

## 20_cub_masks

- command: `python cub_masks.py --tag v5_bird_fraction`
- start: 04:10:20, end: 04:11:53, duration: 1 min 33 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/20_cub_masks.log`
- new files in results/: v5_bird_fraction.csv v5_bird_fraction.json v5_bird_fraction.md 

```
segmentations: /notebooks/Spurious-ICLR/../data/segmentations
masks: 100%|██████████| 11788/11788 [01:30<00:00, 129.58img/s]
# Bird-pixel fraction from the CUB-200-2011 segmentations

The fraction of each Waterbirds image's pixels that belong to the bird,
measured from CUB's own segmentation mask joined by filename. Nothing is
modified; this is a property of the photograph, used in `regroup.py` to
build a group variable that tracks core-task DIFFICULTY instead of the
background nuisance.

## Alignment

- images: **11788**
- masks found: **11788**
- mask dimensions == composite dimensions: **11788** (100.00%)
- usable: **11788**

**OK -- every mask has the same pixel dimensions as its composite, so the CUB segmentation describes the Waterbirds image**

## Distribution of the bird fraction

| min | q10 | q33 | median | q67 | q90 | max | mean |
|---|---|---|---|---|---|---|---|
| 0.0101 | 0.0512 | 0.0900 | 0.1158 | 0.1472 | 0.2247 | 0.7346 | 0.1308 |

## Leakage checks -- READ THESE BEFORE USING THE PARTITION

The new group variable must track how hard the bird is to see, NOT the
label and NOT the old group. An AUC near 0.5 means no information; far
from 0.5 means bird size is a proxy for that variable and the partition
would be confounded.

| against | AUC | mean if 1 | mean if 0 | verdict |
|---|---|---|---|---|
| y (label) | 0.5425 | 0.1426 | 0.1274 | clean |
| place (background) | 0.5099 | 0.1328 | 0.1295 | clean |
| g = 1[place != y] | 0.4962 | 0.1298 | 0.1313 | clean |

Per-image values: `results/v5_bird_fraction.csv`

wrote results/v5_bird_fraction.csv
```

## 30_regroup_median

- command: `python regroup.py --bundle features_v4_waterbirds_dinov2_train.npz --fractions results/v5_bird_fraction.csv --rule median`
- start: 04:11:53, end: 04:11:57, duration: 0 min 4 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/30_regroup_median.log`
- new files in results/: features_v5_waterbirds_dinov2_train_bfmedian.npz features_v5_waterbirds_dinov2_train_bfmedian_regroup.json 

```
row order check: ok (4795 rows, y and place both match)
leakage of bird fraction into y: AUC 0.5392 (deviation 0.0392)
bundles: 100%|██████████| 1/1 [00:01<00:00,  1.32s/bundle]
  treatment        features_v5_waterbirds_dinov2_train_bfmedian.npz  n=4795  g0=2398 g1=2397  eps=0.4999
wrote results/features_v5_waterbirds_dinov2_train_bfmedian_regroup.json
```

## 31_margins_median

- command: `python group_margins.py --no-lp --tag v5_group_margins_median --bundles results/features_v5_*_bfmedian*.npz`
- start: 04:11:57, end: 04:12:05, duration: 0 min 8 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/31_margins_median.log`
- new files in results/: v5_group_margins_median.json v5_group_margins_median.md 

```
bundles:   0%|          | 0/1 [00:00<?, ?bundle/s]
results/features_v5_waterbirds_dinov2_train_bfmedian.npz

  features_v5_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 660.23C/s]
bundles: 100%|██████████| 1/1 [00:06<00:00,  6.71s/bundle]
  margin 0.6312   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

wrote results/v5_group_margins_median.json
wrote results/v5_group_margins_median.md
```

## 30_regroup_tercile

- command: `python regroup.py --bundle features_v4_waterbirds_dinov2_train.npz --fractions results/v5_bird_fraction.csv --rule tercile`
- start: 04:12:05, end: 04:12:09, duration: 0 min 4 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/30_regroup_tercile.log`
- new files in results/: features_v5_waterbirds_dinov2_train_bftercile.npz features_v5_waterbirds_dinov2_train_bftercile_control.npz features_v5_waterbirds_dinov2_train_bftercile_regroup.json 

```
row order check: ok (4795 rows, y and place both match)
leakage of bird fraction into y: AUC 0.5392 (deviation 0.0392)
bundles: 100%|██████████| 2/2 [00:02<00:00,  1.13s/bundle]
  treatment        features_v5_waterbirds_dinov2_train_bftercile.npz  n=3198  g0=1599 g1=1599  eps=0.5000  (dropped 1597)
  matched_control  features_v5_waterbirds_dinov2_train_bftercile_control.npz  n=3198  g0=3039 g1=159  eps=0.0497  (dropped 1597)
wrote results/features_v5_waterbirds_dinov2_train_bftercile_regroup.json
```

## 31_margins_tercile

- command: `python group_margins.py --no-lp --tag v5_group_margins_tercile --bundles results/features_v5_*_bftercile*.npz`
- start: 04:12:09, end: 04:12:18, duration: 0 min 9 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_040725/31_margins_tercile.log`
- new files in results/: v5_group_margins_tercile.json v5_group_margins_tercile.md 

```
bundles:   0%|          | 0/2 [00:00<?, ?bundle/s]
results/features_v5_waterbirds_dinov2_train_bftercile.npz

  features_v5_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 2605.48C/s]
bundles:  50%|█████     | 1/2 [00:03<00:03,  3.76s/bundle]  margin 0.7938   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

results/features_v5_waterbirds_dinov2_train_bftercile_control.npz

  features_v5_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 397.22C/s]
bundles: 100%|██████████| 2/2 [00:07<00:00,  3.85s/bundle]
  margin 0.7938   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

wrote results/v5_group_margins_tercile.json
wrote results/v5_group_margins_tercile.md
```

## WHAT TO DO NEXT

Read in this order. The first one decides how to read the rest.

1. `results/v5_margin_power.md` -- **the calibration**. Its "Answer"
   section says whether ANY partition of these features can give a ratio
   other than 1. If it says NO, then every tie below is explained by the
   instrument and not by Waterbirds, and that is the finding of the session.
   Check the `pinned` rows first: they must recover their planted ratio, or
   nothing else on that page can be read.

2. `results/v5_bird_fraction.md` -- the alignment verdict must be OK, and
   the three leakage AUCs should sit near 0.5. An AUC far from 0.5 against
   `y` means bird size is a proxy for the label and the partition is void.

3. `results/v5_group_margins_median.md` -- the headline re-partition. Every
   row is the SAME features as the session-4 control; only `g` differs.

4. `results/v5_group_margins_tercile.md` -- read the treatment row against
   its `_control` row, NOT against session 4. The tercile rule drops the
   middle third and re-standardises, so only the matched control is comparable.

Total wall clock: 4 min.

