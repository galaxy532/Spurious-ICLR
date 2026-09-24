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

