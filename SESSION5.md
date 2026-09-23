# Session 5 — the alpha > 1 hunt, moved from the images to the PARTITION

Written 22 September 2026, after session 4 stage A came back a tie. Read
`THEORY_AND_DFR.md` for the four points and `SESSION4.md` for what was tried
before this. **There is no GPU work in this session.** It reuses the features
session 4 already extracted; everything here is CPU and takes 20–40 minutes.

---

## 1. Where things stand

| session | what was varied | result |
|---|---|---|
| 1–3 | the backbone (erm, rwg, gdro, dinov2, dinov2_l) | `\|exponent − 1\| ≲ 0.01`, all five |
| 4 stage A | the images (group-conditional resolution degradation, 7 levels) | margin ratio `1.0000` at every level, plateau ≤ 5e-9 |

Session 4 stage B was **not run**, correctly: `pick_levels.py` found no candidate
level. Its printed suggestion to retry with `BACKBONE=dinov2_l` is a hardcoded
string in the `if not candidates:` branch (`pick_levels.py:119`), not a
data-driven recommendation, and its stated rationale ("more room before the floor
bites") does not apply — the margin floor is 0.05 and the smallest margin on the
whole ladder was 0.3517, seven times above it. **No row was rejected by the
floor.** Every row was the other case: the knob moved the margin and not the
ratio.

### Why session 4 could not have worked, in hindsight

`g = 1[place != y]` (`datasets.py:211`), so `g=0` is the **spurious-aligned**
group — background predicts the label — and `g=1` (240 points) is the conflicting
group. The degradation operators are both low-pass filters (`resolution` is
bicubic down/up, `blur` is a Gaussian kernel; `degrade.py`'s own self-test
validates both against the *same* high-frequency-energy statistic and they bottom
out in the same place). A low-pass filter destroys the bird and **preserves the
background**. So degrading `g=0` never made `g=0` hard — it pushed `g=0` onto the
shortcut, which is untouched and, at `corr(y, place) = 0.867`, plenty. The
numbers agree: the margin fell only to 0.3517 while `g=0`'s support-vector count
*grew*, 350 → 514.

**Do not re-propose blur, a longer level ladder, or a larger backbone for this.**
All three are the same lever.

---

## 2. What session 5 does, and why there are two parts

### Part A — the calibration (`margin_power.py`). Read this first.

Every null this project has recorded on the margin ratio is only meaningful if
the instrument could have reported otherwise. Nothing establishes that.

`validate_group_margins.py` recovers a ratio of 1.6000, but look at how its
`asymmetric` case is built (`make_bundle`): every point is `x[0] = y*m` plus a
perturbation orthogonal to `e_1` **and mirrored** (`sgn = ±1`). The mirroring
punishes any tilt out of `e_1` symmetrically, so the max-margin direction is
*pinned* to `e_1` and the group margins are exactly `m0` and `m1` by
construction. That is a correctness check. It deliberately removes the one thing
that matters on real data: the solver's freedom to rotate.

`gamma_g` is measured under the **global** max-margin separator, and the ratio
exceeds 1 only if one group contributes **no support vector**. When the optimiser
can rotate — and at `d = 768` it can — it trades margin between the groups until
both bind. Both bind is what a tie is.

So the prior question is not "is Waterbirds asymmetric" but:

> On these features, does **any** partition into two groups produce a ratio other
> than 1 after the separator is re-solved?

`margin_power.py` answers it with four sweeps. The decisive one is **`oracle`**:
rank every point by its own margin under the baseline separator and put the top
`k` into group 1. That is the most margin-asymmetric partition the dataset
admits. If the oracle partition ties, **bird size cannot do better**, and the
whole line is settled for ten minutes of CPU.

The project has precedent for this. `validate_invariance.py` check C2b builds an
F1-vs-n power curve for the invariance rule, and that curve is why Waterbirds was
dropped from that rule: at n = 5,794 it is underpowered, so a null there would
have been uninterpretable. The margin ratio has never had the equivalent.

This is an instrument calibration, **not a result**. It is not the semi-synthetic
`Phi` with a tunable alpha rejected in August 2026 — that one manufactured an
exponent and reported it. Nothing here is evidence for or against Theorem 5.3.

### Part B — the re-partition (`cub_masks.py` + `regroup.py`)

Sessions 1–4 all kept `g = 1[place != y]`. The background is a **nuisance**,
chosen by the benchmark's designers to be uncorrelated with how hard the bird is
to recognise, so the two groups are equally hard *by construction* and the
setting sits on the `alpha = 1` transition. That is the property being changed.

Bird size in frame is a real difficulty axis — a bird covering 30% of the pixels
carries far more core signal than one covering 1% — and it is a property of the
photograph, fixed before any training. Session 5 re-partitions by it.

**Nothing is modified.** No image is touched, no feature is recomputed, no
backbone is run. `phi` and `y` pass through bit-identical (checked in
`validate_session5.py`); only `g` changes. That is the whole argument for
re-partitioning over re-degrading: session 4's manipulation created a confound,
and this one cannot.

#### Where the masks come from

**Not from Waterbirds.** No Waterbirds release ships segmentation masks; the
tarball is `metadata.csv` plus composited JPEGs. The masks are CUB-200-2011's
own, a separate 39.3 MB download:

```
https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1
md5 4d47ba1228eae64f2fa547c47bc65255
```

It belongs next to `waterbird_complete95_forest2water2/` under `DATA_ROOT`, as
`segmentations/<class folder>/<image>.png`. `cub_masks.py` fetches and verifies
it if it is missing; `--no-download` makes that an error instead.

Waterbirds preserves CUB's `<class folder>/<image name>` paths, so the join is by
filename — exact. **A learned segmenter was rejected on purpose**: its errors
would be largest exactly where the bird is small or low-contrast, which is the
quantity being measured, so that confound would be built into the instrument.

#### The alignment check is not optional

The CUB mask describes the *original CUB photograph*. Waterbirds pastes the
segmented bird onto a Places background. If that step cropped or resized
anything, the mask does not describe the composite and every number downstream is
meaningless. A crop or a resize changes the image dimensions, so the check is:
**the mask's pixel dimensions must equal the composite's, for every image.**
`cub_masks.py` refuses to report a usable result below 100%, and `regroup.py`
refuses to run on a split with any unusable row.

---

## 3. What was built

| file | what it is | validated by |
|---|---|---|
| `cub_masks.py` | fetch + md5-verify the CUB segmentations, join by filename, bird-pixel fraction per image, alignment check, three leakage AUCs | `python cub_masks.py --self-test` — exact fraction on a known mask (0.20), dimension mismatch caught, missing mask reported not crashed, alignment refused when a mismatch is present, AUC = 1.0/0.0/0.5 on separated/anti-separated/tied |
| `regroup.py` | replaces `g` in a bundle and writes a NEW bundle; median and tercile rules; row-order guard; matched control for any row-dropping rule | `python regroup.py --self-test` — both rules, `--invert`, shuffled bundle caught, round-trip keeps the right rows and re-standardises |
| `margin_power.py` | the four calibration sweeps | `python margin_power.py --self-test` — planted translation exact to 1e-9 and the identity at delta 0, pinned control recovers 1.6000, unplanted split is a tie |
| `validate_session5.py` | the JOIN between them | a regrouped bundle loads in the **unmodified** `group_margins.py` with matching group counts, `phi`/`y` bit-identical, shuffled bundle refused, a planted ratio survives the path |
| `run_session5.sh` | the runner | `bash -n` |

**Nothing existing was edited.** `group_margins.py`, `degrade.py`,
`extract_features.py`, `datasets.py` and `common.py` are untouched, which is the
point of writing new bundles rather than adding flags: the instrument that
measured the session-4 ties is the same instrument, byte for byte.

### Two design decisions worth knowing before reading the output

**The tercile rule needs its own control.** `median` keeps every row, so its
features are bit-identical to the bundle session 4 already measured and the
session-4 tie is its control. `tercile` drops the middle third to widen the gap,
which changes the row set — and `common.standardize` centres each coordinate, so
with no intercept in the margin problem (`fit_intercept=False`, matching
`long_horizon.py`) a change in the mean moves the solution. A tercile ratio is
therefore **not** comparable to session 4. `regroup.py` emits a matched control
alongside it: the same surviving rows, re-standardised identically, carrying the
original `g`. Read treatment against control, never against session 4.

**Large birds are `g=1`.** `FeatureBundle` documents `g=0` as G_maj and `g=1` as
G_min, and Theorem 5.3's WLOG labels G_min as the **larger-margin** group (size
is irrelevant to the branch — it only decides which group's error carries the
`1/eps`). A large bird is easier, so it should have the larger margin. Putting
large birds in `g=1` makes the script's "min" the theorem's "min" and the reports
read straight: `gamma(g=1) > gamma(g=0)` is the direction the branch predicts.
`--invert` flips it.

---

## 4. What to run

```bash
cd /notebooks/Spurious-ICLR
git pull
nohup setsid bash run_session5.sh > session5_nohup.out 2>&1 &
```

No GPU, no `MAX_HOURS` cap, single stage. Logs and `SUMMARY.md` land in
`results/logs/session5_<stamp>/`, so a `git pull` brings the full log back, not
just the tables.

Stage 0 runs all four self-checks and **aborts before touching any data** if any
fails. The calibration runs *before* the masks, deliberately: if it says no
partition can work, the re-partition below is already explained before it is
read.

Options: `BUNDLE=<path>`, `RULES="median tercile"`, `NO_DOWNLOAD=1`, `PUSH=1`.

---

## 5. How to read the results — in this order

### 5.1 `results/v5_margin_power.md` — the calibration

**Check the `pinned` rows first.** They must recover their planted `m1` (1.05 and
1.60). If they do not, the instrument is broken at this n and d and nothing else
on the page can be read.

Then the **Answer** section:

- **"NO"** — no partition produced a ratio other than 1, *including* the oracle
  partition built with full knowledge of the margins. Then bird size cannot work
  either, and the session's finding is about the estimand rather than about
  Waterbirds: `gamma_min/gamma_maj` measured under a re-solved global max-margin
  separator in high dimension is generically 1, because the optimiser equalises
  the groups. Look at the ceiling on the `q05` ratio in the same table: if that
  is well above 1, an asymmetry **is** present in the features and the ess inf is
  what hides it.
- **"YES"** — a non-tie is attainable. The best oracle ratio is then the
  **ceiling** any real partition is competing against, and §5.3 is worth reading
  properly.

Two rows to look at either way:

- **`translate`** — a rigid shift of half the points along `y·w/||w||`, which adds
  exactly `delta` to their margins under `w`. Compare `target` against `ratio`.
  A large target next to a ratio of 1 is the optimiser rotating away from the
  planted asymmetry. This is the same mechanism that makes real partitions tie,
  shown with numbers.
- **`heavy_tail`** — the oracle partition with a fraction `q` of group 1 swapped
  for random members. `gamma_g` is an ess inf, set by the single worst point in
  the group, so one hard member pulls the whole group's margin back down. Real
  difficulty asymmetries are heterogeneous. If the ratio collapses as soon as
  `q < 1` while the `q01`/`q05` ratios hold, the estimand cannot see a
  heterogeneous asymmetry **at all** — that is a definitional limit, not a noise
  problem, and it would reframe every null in sessions 1–4.

### 5.2 `results/v5_bird_fraction.md`

- The **alignment verdict must be OK** (100% dimension match). If not, stop —
  the Waterbirds generation resized or cropped, and the bird fraction does not
  describe these images.
- The **three leakage AUCs** should sit near 0.5. `regroup.py` hard-refuses at
  `|AUC(bird fraction → y) − 0.5| > 0.10`. An AUC far from 0.5 against `y` means
  bird size is a proxy for the label and the partition is void; against `place`
  or the old `g`, it means the new partition is partly the old one and the result
  would not be independent of sessions 1–4.

### 5.3 `results/v5_group_margins_median.md` and `..._tercile.md`

Same instrument, same columns, same reading rules as session 4: `plateau` first
(below 1e-3 settled, above 1e-2 not a measurement), then the ratio, then
`n at margin` per group.

- `median`: compare directly against the session-4 control (`features_v4_..._train.npz`,
  ratio 1.0000). Same features, only `g` differs.
- `tercile`: compare the treatment row against its `_control` row on the same
  page. Never against session 4.

A ratio above `1 + 1e-3` in either, with the larger-margin group being `g=1`
(large birds), is a genuine margin asymmetry on real images with **no image
manipulation of any kind** — strictly stronger than what session 4 was going to
provide. It would then be worth running `long_horizon.py` on that bundle to test
the alignment identity's prediction for `beta`, which session 5 does **not** do.

---

## 6. The outcomes, and what each is worth

1. **Calibration says NO.** The strongest negative available and the cheapest: it
   says the `alpha > 1` branch is not merely absent from these benchmarks but
   *unobservable* through this estimator on any high-dimensional representation,
   whatever the group variable. Combined with the `q05` ceiling it localises the
   problem to the ess inf. This is a claim about the measurement, so it belongs
   in the paper as a scope statement rather than as a failed experiment.
2. **Calibration says YES and bird size ties.** The partition is not the lever —
   the margin asymmetry that bird size induces is real but too small to clear the
   coupling ratio. Quantify it with the `q05` ratio and move on.
3. **Calibration says YES and bird size gives a ratio > 1.** The phase-transition
   figure the paper does not have, on real images, with nothing manufactured.
   Next step is `long_horizon.py` on that bundle.

Aser's constraint from 22 Sept stands: **a negative is not going into the ICLR
submission.** Outcome 1 is still worth having, because it stops the next session
spending a week on a dataset arm that could not have worked either — and because
it converts "we looked and did not find it" into "here is why it is not findable
this way", which is a scope sentence rather than a failed experiment.

---

## 7. Known limitations — state these, do not discover them later

- **The oracle sweep is an upper bound in spirit, not a theorem.** The ranking is
  optimal under the *baseline* separator, and re-solving changes the separator. A
  partition could in principle beat it. Nothing observed so far suggests one
  does, but it is not proved.
- **The alignment identity is still the load-bearing assumption** if outcome 3
  happens, exactly as flagged in `SESSION4.md` §3. It is the manuscript's own,
  but it should be named as such in any write-up.
- **Bird-pixel fraction is a proxy for difficulty, not difficulty.** A large
  blurry bird against a confusing background can be harder than a small sharp
  one. The leakage AUCs check that it is not a proxy for the *label*; they do not
  establish that it is a good proxy for difficulty. The margin measurement is
  what settles that, which is why the ratio is measured rather than assumed.
- **Only the train split is re-partitioned.** `long_horizon.py` would need val and
  test too, and `cub_masks.py` already computes the fraction for all three splits
  (`v5_bird_fraction.csv` carries a `split` column), so that is a `regroup.py`
  invocation away, not new code.
- **`alpha_gt_1_predicted` in `group_margins.py`'s JSON is `true` on every tie**,
  because it tests `ratio > 1` strictly and the float noise lands at +2e-9. It
  reads the opposite of the result. Untouched — it is not mine to change — but do
  not consume that field.

---

## 8. Corrections made while designing this — do not re-derive them

- **"On Waterbirds the group and the spurious concept coincide"** was quoted from
  `datasets.py:65` to argue about the branch condition. That is a misuse: the
  passage is about whether the *identification rules* have information
  independent of `g`, and its own body says Waterbirds' `g` is "DERIVED from" the
  spurious attribute while CelebA's `g` "IS it". `s` is `place`; `g = 1[place ≠ y]`
  is not `place`. The defensible statement is the conditional one: given `y`, `g`
  determines `place` and vice versa.
- **"A group variable good at one requirement is bad at the other by
  construction"** was overstated. Def 3.3 needs `g` to index a difference in the
  r→s map (`A ≠ B`); the branch needs `g` to index a difference in core margins.
  Those are two separate demands on one partition, and no partition tried so far
  has met both — but there is no argument that they conflict. Open question.
- **A rigid translation is the wrong way to plant an asymmetry.** It plants it
  along the one direction the optimiser can undo, and the `translate` sweep exists
  only to show that. The first draft of `margin_power.py` used it as the main
  instrument and its self-test failed accordingly — a 1.5× planted gap came back
  a tie. That failure is the reason the `oracle` and `pinned` sweeps exist.
