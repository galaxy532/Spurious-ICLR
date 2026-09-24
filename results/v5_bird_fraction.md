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
