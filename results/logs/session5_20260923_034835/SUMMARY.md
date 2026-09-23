# Session 5 summary (20260923_034835)

Started 2026-09-23 03:48:35 UTC on nih0mwal0a.

- source bundle: `results/features_v4_waterbirds_dinov2_train.npz`
- partition rules: `median tercile`
- no GPU work in this session; everything below is CPU.

## 00_selftest_cub_masks

- command: `python cub_masks.py --self-test`
- start: 03:48:37, end: 03:48:38, duration: 0 min 1 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_034835/00_selftest_cub_masks.log`
- new files in results/: 

```
cub_masks self-test
masks: 100%|██████████| 3/3 [00:00<00:00, 1452.66img/s]
  ok   bird fraction exact on a known mask (0.20)
  ok   matching dimensions accepted
  ok   dimension mismatch caught
  ok   missing mask reported, no crash
  ok   alignment refused when a mismatch is present
  ok   AUC = 1.0 / 0.0 / 0.5 on separated, anti-separated, tied
  ok   find_segmentations locates a real-shaped tree
masks: 100%|██████████| 12/12 [00:00<00:00, 5221.67img/s]
  ok   end-to-end run over a mini dataset, all columns present
SELF-TEST OK
```

## 01_selftest_regroup

- command: `python regroup.py --self-test`
- start: 03:48:38, end: 03:48:39, duration: 0 min 1 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_034835/01_selftest_regroup.log`
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
- start: 03:48:39, end: 03:48:40, duration: 0 min 1 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_034835/02_selftest_margin_power.log`
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
- start: 03:48:40, end: 03:48:47, duration: 0 min 7 s
- exit code: **0** (ok)
- log: `results/logs/session5_20260923_034835/03_integration.log`
- new files in results/: 

```
session 5 integration check
checks:   0%|          | 0/5 [00:00<?, ?check/s]
  regrouped.npz                     : 100%|██████████| 3/3 [00:00<00:00, 13691.96C/s]

  planted.npz                       : 100%|██████████| 3/3 [00:00<00:00, 17549.39C/s]
checks: 100%|██████████| 5/5 [00:05<00:00,  1.00s/check]

  regrouped bundle loads in group_margins unchanged
  phi and y pass through untouched when no row is dropped
  row-order guard fires on a shuffled bundle
  a planted asymmetry survives the session-5 path
  the real command line runs end to end on a mini dataset

INTEGRATION OK -- a re-partitioned bundle is read by the unmodified group_margins.py, the features are untouched, a shuffled bundle is refused, and a planted ratio survives the path.
```

## ABORTED

Source bundle `results/features_v4_waterbirds_dinov2_train.npz` not found.
