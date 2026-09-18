# Session 4 stage A summary (20260918_215241)

Started 2026-09-18 21:52:41 UTC on nmh9h2rf20.

- backbone: `dinov2`   degradation: `resolution` on group `g == 0`
- levels: `1.0 0.60 0.40 0.28 0.20 0.14 0.10`

## 00_validate_degrade

- command: `python degrade.py`
- start: 21:52:44, end: 21:52:44, duration: 0 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/00_validate_degrade.log`
- new files in results/: 

```
  resolution ladder: 100%|██████████| 7/7 [00:00<00:00, 1003.94level/s]
  blur ladder: 100%|██████████| 4/4 [00:00<00:00, 2588.28level/s]

resolution ladder (high-frequency energy, must decrease):
  level 1.0        49287.4
  level 0.75        3849.0
  level 0.5          467.2
  level 0.35          67.6
  level 0.25          11.2
  level 0.15           3.0
  level 0.08           1.8
blur ladder:
  level 0.0        49287.4
  level 0.01         364.7
  level 0.03           3.1
  level 0.08           1.5

SELF-TEST OK
```

## 01_validate_degrade_path

- command: `python validate_degrade_path.py`
- start: 21:52:44, end: 21:52:45, duration: 0 min 1 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/01_validate_degrade_path.log`
- new files in results/: 

```
degraded 3 of 6; untouched 3
tags: {'resolution@1.0': '', 'resolution@0.2': 'res200', 'resolution@0.35': 'res350', 'blur@0.0': '', 'blur@0.03': 'blur030'}
INTEGRATION OK
```

## 02_validate_margins

- command: `python validate_group_margins.py`
- start: 21:52:45, end: 21:53:18, duration: 0 min 33 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/02_validate_margins.log`
- new files in results/: 

```
cases:   0%|          | 0/4 [00:00<?, ?case/s]
  asymmetric.npz                    : 100%|██████████| 5/5 [00:00<00:00, 15174.76C/s]

  symmetric.npz                     : 100%|██████████| 5/5 [00:00<00:00, 13733.80C/s]

  reversed.npz                      : 100%|██████████| 5/5 [00:00<00:00, 14533.28C/s]

  nonseparable.npz                  : 100%|██████████| 5/5 [00:00<00:00, 11293.23C/s]
cases: 100%|██████████| 4/4 [00:31<00:00,  7.78s/case]

| case | m0 | m1 | margin g0 | margin g1 | ratio | plateau | larger |
|---|---|---|---|---|---|---|---|
| asymmetric | 1.0 | 1.6 | 1.0000 | 1.6000 | 1.6000 | 6.1e-10 | g=1 |
| symmetric | 1.0 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.1e-11 | - |
| reversed | 1.5 | 1.0 | 1.5000 | 1.0000 | 0.6667 | 7.8e-10 | g=0 |
| nonseparable | 1.0 | 1.2 | - | - | - | - | - |

VALIDATION OK -- per-group margins and their ratio are recovered to within 2%, a tie is reported as a tie, and non-separable data is refused.
```

## 03_smoke_degrade

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --degrade-kind resolution --degrade-level 0.2 --degrade-group 0 --splits train --out-prefix smoke4 --no-save-model --max-batches 3`
- start: 21:53:18, end: 21:56:18, duration: 3 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/03_smoke_degrade.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
DEGRADATION: resolution level 0.2 applied to group g == 0, at both  (tag res200)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:10<00:00,  1.74s/batch]
  wrote smoke4_waterbirds_dinov2_res200g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_10

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4`
- start: 21:56:18, end: 21:58:35, duration: 2 min 17 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_10.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...
  embedding: 100%|██████████| 75/75 [02:06<00:00,  1.68s/batch]
  wrote features_v4_waterbirds_dinov2_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_060

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4 --degrade-kind resolution --degrade-level 0.60 --degrade-group 0`
- start: 21:58:35, end: 22:01:01, duration: 2 min 26 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_060.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
DEGRADATION: resolution level 0.6 applied to group g == 0, at both  (tag res600)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:09<00:00,  1.73s/batch]
  wrote features_v4_waterbirds_dinov2_res600g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_040

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4 --degrade-kind resolution --degrade-level 0.40 --degrade-group 0`
- start: 22:01:01, end: 22:03:16, duration: 2 min 15 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_040.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
DEGRADATION: resolution level 0.4 applied to group g == 0, at both  (tag res400)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:05<00:00,  1.68s/batch]
  wrote features_v4_waterbirds_dinov2_res400g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_028

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4 --degrade-kind resolution --degrade-level 0.28 --degrade-group 0`
- start: 22:03:16, end: 22:05:32, duration: 2 min 16 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_028.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
DEGRADATION: resolution level 0.28 applied to group g == 0, at both  (tag res280)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:07<00:00,  1.70s/batch]
  wrote features_v4_waterbirds_dinov2_res280g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_020

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4 --degrade-kind resolution --degrade-level 0.20 --degrade-group 0`
- start: 22:05:32, end: 22:07:48, duration: 2 min 16 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_020.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
DEGRADATION: resolution level 0.2 applied to group g == 0, at both  (tag res200)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:06<00:00,  1.69s/batch]
  wrote features_v4_waterbirds_dinov2_res200g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_014

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4 --degrade-kind resolution --degrade-level 0.14 --degrade-group 0`
- start: 22:07:48, end: 22:10:12, duration: 2 min 24 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_014.log`
- new files in results/: 

```
DEGRADATION: resolution level 0.14 applied to group g == 0, at both  (tag res140)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:08<00:00,  1.71s/batch]
  wrote features_v4_waterbirds_dinov2_res140g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 10_extract_010

- command: `python extract_features.py --dataset waterbirds --backbone dinov2 --splits train --out-prefix features_v4 --degrade-kind resolution --degrade-level 0.10 --degrade-group 0`
- start: 22:10:12, end: 22:12:29, duration: 2 min 17 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/10_extract_010.log`
- new files in results/: 

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
DEGRADATION: resolution level 0.1 applied to group g == 0, at both  (tag res100)
backbone dinov2: d = 768, task epochs = 0, train mode = erm, recipe = legacy

waterbirds/train  n = 4795
  eps = P(g=1) = 0.0501
  P(y=1 | g=0) = 0.2321   P(y=1 | g=1) = 0.2333
  cells (y, attr):
    y=0:  attr=0    3498   attr=1     184
    y=1:  attr=0      56   attr=1    1057
  smallest cell = 56   <-- BELOW 200; cell-conditional estimates will be refused
  embedding 4795 images ...  (4555 degraded)
  embedding: 100%|██████████| 75/75 [02:07<00:00,  1.71s/batch]
  wrote features_v4_waterbirds_dinov2_res100g0_train.npz  phi (4795, 768)  eps = 0.0501
```

## 20_group_margins

- command: `python group_margins.py --no-lp --tag v4_group_margins --bundles features_v4_waterbirds_dinov2*_train.npz`
- start: 22:12:29, end: 22:13:24, duration: 0 min 55 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/20_group_margins.log`
- new files in results/: v4_group_margins.json v4_group_margins.md 

```
bundles:   0%|          | 0/7 [00:00<?, ?bundle/s]
features_v4_waterbirds_dinov2_res100g0_train.npz

  features_v4_waterbirds_dinov2_res1: 100%|██████████| 5/5 [00:00<00:00, 246.21C/s]
bundles:  14%|█▍        | 1/7 [00:13<01:21, 13.52s/bundle]  margin 0.3517   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

features_v4_waterbirds_dinov2_res140g0_train.npz

  features_v4_waterbirds_dinov2_res1: 100%|██████████| 5/5 [00:00<00:00, 437.26C/s]
bundles:  29%|██▊       | 2/7 [00:23<00:56, 11.22s/bundle]  margin 0.4091   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

features_v4_waterbirds_dinov2_res200g0_train.npz

  features_v4_waterbirds_dinov2_res2: 100%|██████████| 5/5 [00:00<00:00, 1395.31C/s]
bundles:  43%|████▎     | 3/7 [00:32<00:41, 10.37s/bundle]  margin 0.5019   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

features_v4_waterbirds_dinov2_res280g0_train.npz

  features_v4_waterbirds_dinov2_res2: 100%|██████████| 5/5 [00:00<00:00, 781.76C/s]
bundles:  57%|█████▋    | 4/7 [00:38<00:25,  8.65s/bundle]  margin 0.5614   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

features_v4_waterbirds_dinov2_res400g0_train.npz

  features_v4_waterbirds_dinov2_res4: 100%|██████████| 5/5 [00:00<00:00, 1669.71C/s]
bundles:  71%|███████▏  | 5/7 [00:43<00:14,  7.30s/bundle]  margin 0.6032   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

features_v4_waterbirds_dinov2_res600g0_train.npz

  features_v4_waterbirds_dinov2_res6: 100%|██████████| 5/5 [00:00<00:00, 2297.24C/s]
bundles:  86%|████████▌ | 6/7 [00:48<00:06,  6.48s/bundle]  margin 0.6375   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

features_v4_waterbirds_dinov2_train.npz

  features_v4_waterbirds_dinov2_trai: 100%|██████████| 5/5 [00:00<00:00, 535.29C/s]
bundles: 100%|██████████| 7/7 [00:54<00:00,  7.76s/bundle]
  margin 0.6312   gamma(g=0) 1.0000   gamma(g=1) 1.0000   tie -- the setting sits on the alpha = 1 transition   beta predicted 1.0000   plateau settled

wrote results/v4_group_margins.json
wrote results/v4_group_margins.md
```

## 21_pick_levels

- command: `python pick_levels.py --json results/v4_group_margins.json`
- start: 22:13:24, end: 22:13:24, duration: 0 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session4A_20260918_215241/21_pick_levels.log`
- new files in results/: 

```
  level    margin    ratio  larger   status
   1.00    0.6312   1.0000    None   CONTROL
   0.60    0.6375   1.0000    None   tie -- the knob did not move the branch condition here
   0.40    0.6032   1.0000    None   tie -- the knob did not move the branch condition here
   0.28    0.5614   1.0000    None   tie -- the knob did not move the branch condition here
   0.20    0.5019   1.0000    None   tie -- the knob did not move the branch condition here
   0.14    0.4091   1.0000    None   tie -- the knob did not move the branch condition here
   0.10    0.3517   1.0000    None   tie -- the knob did not move the branch condition here

NO CANDIDATE LEVEL. Every degraded level either left the margin ratio at 1 or collapsed the overall margin below the floor.

Do NOT run stage B -- it would spend six GPU-hours confirming the tie that
sessions 1-3 already measured. This is a reportable result: image degradation
does not move gamma_min/gamma_maj on this backbone, so the alpha > 1 branch
needs a group asymmetry these benchmarks do not have and that image quality
cannot manufacture.

Before accepting it, one cheap retry is worth it: rerun stage A with a
larger-margin backbone, which has more room before the floor bites:
    BACKBONE=dinov2_l nohup setsid bash run_session4.sh A > session4A_L.out 2>&1 &
```

## WHAT TO DO NEXT

`21_pick_levels` above applied the selection rules to the ladder and printed
either the exact stage-B command to paste, or a recommendation NOT to run stage B.
Read that block: nothing else has to be decided by hand.

The full ladder, with the cross-check columns, is in `results/v4_group_margins.md`.
Two rows there are worth checking by eye whatever the picker says:

- the `level = 1.0` CONTROL must report a **tie**. If it does not, something
  changed since session 3 and stage B is premature whatever the other rows show.
- any row with `plateau` above 1e-2 is not a measurement. The picker drops those,
  but if it dropped a row you wanted, rerun `group_margins.py` on that bundle
  alone with a longer `--C` ladder rather than quoting it.

## Wall clock

Total 20 min.

