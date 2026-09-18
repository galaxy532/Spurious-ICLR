# Session 4 — the alpha > 1 hunt, moved from backbones to the dataset

Written 18 September 2026, after session 3. Read `THEORY_AND_DFR.md` first for the
four points; this file covers only the new experiment.

---

## 1. Why this experiment exists

Seven runs across two datasets and five representations found no case of the
`alpha > 1` branch of Theorem 5.3. Sessions 1-3 looked for it by varying the
**backbone** — ERM, reweighted, group-DRO, and two frozen DINOv2 encoders. None of
them moved anything: all five Waterbirds representations gave `|exponent − 1| ≲ 0.01`
for both groups.

The reason is that the backbone was never the right knob. The branch condition is

```
alpha > 1   <=>   gamma~_min / gamma~_maj  >  gamma_crit = (1 + mu_A) / (1 + mu)
```

— **the two groups must have sufficiently different r-margins**, one genuinely easier
to classify from the core feature than the other, with the gap exceeding the coupling
ratio. Two things follow, and the second is the one that was got wrong first:

- Group **size is irrelevant** to the branch. The WLOG in Theorem 5.3
  (`gamma~_maj = 1`, `gamma~_min >= 1`) labels the groups **by margin**; `eps = P(G_min)`
  is free in `[0, 1]` and may exceed 1/2. Size only decides which group's error carries
  the `1/eps`. (An earlier draft of this document said the branch needed the *rare*
  group to be the easier one. That was wrong.)
- The margin gap is a property of the **dataset**, fixed before any training happens.

And that is exactly what Waterbirds, CelebA and colored MNIST are built not to have.
Their group variable — background, sex, colour — is a **nuisance**, chosen to be
uncorrelated with how hard the core task is. So `gamma~_min/gamma~_maj ≈ 1` and the
setting sits *on* the phase transition. They also **maximise** spuriousness
(`corr(y, place) = 0.867` on Waterbirds), which pushes `gamma_crit` *up*. Two
independent design choices, both pointing away from the branch.

So: supply the missing margin asymmetry at the **data** level, keep everything else —
the real images, the real spurious structure, a real frozen backbone, a real max-margin
problem — and see whether the theory's own prediction comes true.

**This is not the semi-synthetic `Phi` with a tunable alpha that was proposed and
rejected in August 2026.** That one manufactured the exponent inside the representation,
which would have assumed the answer. This manufactures a property of the *images*, and
then lets the backbone and the margin problem decide what the exponent is.

---

## 2. The manipulation

`degrade.py` reduces the core feature in **one group only**, on the raw image, before
the backbone's own transform:

- `resolution` (default): downsample the short side to `level` times its length, then
  resample back, both bicubic. `level = 1.0` is the identity. Scale-free, so it means
  the same thing for a 224 px and a 518 px backbone.
- `blur`: Gaussian radius `level * short_side`. `level = 0.0` is the identity. A second
  operator with a different frequency profile, so a result that depends on which one
  was used is visible as such.

**Group 0 is degraded by default.** The bundle convention is `g == 0 = G_maj`,
`g == 1 = G_min`, and the theorem calls `G_min` the *larger*-margin group. Degrading
`g == 0` therefore makes the script's "min" the theorem's "min" and the reports read
straight.

### The honest limitation

This is **not** a surgical operation on `r` alone. Reducing resolution destroys fine
detail (bird shape and texture — the core feature) far faster than coarse colour and
layout statistics (land vs water — the spurious feature), but it touches both.

Nothing downstream assumes purity, which is why this is acceptable. The per-group hard
margins that result are **measured**, not assumed, and the manuscript's own alignment
identity turns that measurement into a prediction. The knob only has to *move* the
margin ratio; it does not have to move it for a reason we can name.

---

## 3. The prediction being tested — parameter-free

The manuscript defines the group hard-margins (supplementary Eq. `eq:margins`) as

```
gamma_g = ess inf over group g of  y (w_hat . x),      min(gamma_maj, gamma_min) = 1
```

with `w_hat` the minimum-norm hard-margin separator — no intercept, matching the GD in
`long_horizon.py`. This is a property of the frozen representation alone: no gradient
descent, no `z`, no `eps`. `group_margins.py` measures it.

Then the alignment identity from "Interpretation of the main results":

```
gamma_min = max(1, alpha_xi(1 + delta_maj) - delta_min)  ~=  max(1, alpha)
```

Read right to left, it makes the measured margin ratio a **prediction of `beta`**:

| step | script | quantity |
|---|---|---|
| measure | `group_margins.py` | `gamma_min / gamma_maj` from the frozen features |
| predict | the identity above | larger-margin group's `beta` = that ratio; other group's `beta` = exactly 1 |
| measure again | `long_horizon.py` | both `beta`s, independently, at `z = 1e6` |

Nothing is fitted in between, and none of it needs the `(r, s)` identification that the
Waterbirds arm in `Rebuttals/` could not settle. **The identity is the assumption doing
the work here** — it is the manuscript's, not an invention, but it is the load-bearing
step and should be named as such in any write-up.

A **tie** in the margin measurement (ratio `1.000`) predicts `beta = 1` for both groups
— i.e. the `alpha < 1` branch, and exactly what sessions 1-3 measured. That is the
control, and it must come out right before anything else is believed.

---

## 4. What was built

| file | what it is | validated by |
|---|---|---|
| `degrade.py` | the two degradation operators | `python degrade.py` — identity really is identity, size preserved, deterministic, high-frequency energy falls **monotonically** with the level (a knob that is not monotone cannot be swept), bad arguments raise |
| `extract_features.py` | additive flags `--degrade-kind/-level/-group/-at`; the level is in the output filename so a sweep cannot overwrite itself | `validate_degrade_path.py` — exactly the masked indices change, the rest are byte-identical, tags do not collide |
| `group_margins.py` | per-group hard margins, the ratio, the predicted `beta`, plus cross-checks | `validate_group_margins.py` — recovers known margins and their ratio to 4 decimals on four constructions, reports a **tie** as a tie instead of ranking floating-point noise, refuses non-separable data |
| `pick_levels.py` | **new** — reads the stage-A ladder and prints the stage-B command, or says not to run it | exercised on both outcomes (a ladder where the knob works and one where it does not) |
| `run_session4.sh` | the two-stage unattended runner | `bash -n` |

Defaults are unchanged everywhere: with no `--degrade-*` flag, `extract_features.py`
does exactly what it did on 18 September 2026.

### Two things in `group_margins.py` worth knowing before reading its output

- **`plateau` is the first column to look at.** There is no exact hard-margin solver
  here; `LinearSVC(loss="hinge", fit_intercept=False)` is run along a ladder of `C` and
  the whole ladder is reported. `plateau_rel` is the relative change of the margin across
  the top two `C` values. Below `1e-3` the number is settled; above `1e-2` it is **not a
  measurement** and must not be quoted — rerun with a longer `--C` ladder.
- **A tie is a result, not a missing value.** Margins within `TIE_TOL = 1e-3` of each
  other are reported as `tie`, and that means the setting sits on the `alpha = 1`
  transition. Every real bundle measured so far is a tie.

Cross-checks printed alongside: `margin_logistic` (same geometric margin, different loss
and optimiser) and `rho_lp` (the exact LP of `separability_check`, which uses a **box**
constraint so its *value* is not comparable — it is there because a positive value
*proves* separability, which the SVC cannot).

---

## 5. What to run on Paperspace

### Stage A — the margin ladder. Cheap, roughly an hour, decides whether stage B is worth it.

```bash
cd /notebooks/Spurious-ICLR
git pull
nohup setsid bash run_session4.sh A > session4A_nohup.out 2>&1 &
```

Frozen DINOv2, no backbone training at all, train split only, at
`level = 1.0 0.60 0.40 0.28 0.20 0.14 0.10`. Writes `results/v4_group_margins.{md,json}`.

**You do not choose the levels by hand.** Stage A ends by running `pick_levels.py`,
which applies the selection rules to the ladder and prints either the exact stage-B
command to paste or a recommendation not to run stage B at all. It lands in
`results/logs/session4A_<stamp>/SUMMARY.md` under `21_pick_levels`, and can be rerun
any time:

```bash
python pick_levels.py                      # reads results/v4_group_margins.json
```

Its rules, in order: drop rows that are not measurements (not separable, or
`plateau_rel >= 1e-2`); always keep the `level = 1.0` control; drop rows whose
**overall** margin is below `--margin-floor` (default 0.05) because a level with a fine
ratio and a collapsed margin never reaches the regime within `z = 1e6` whatever its
ratio; call a surviving row a candidate if its ratio exceeds `1 + 1e-3`; then pick the
control, the smallest candidate ratio (the near-transition case) and the largest
(the strongest signal).

If nothing is a candidate it says so and tells you not to spend the six GPU-hours —
that is the third outcome in §6 below, and stage A alone has already produced it.

Two rows are worth checking by eye in `results/v4_group_margins.md` whatever the picker
says: the `level = 1.0` control **must** report a **tie** (if not, something changed
since session 3 and stage B is premature), and any row with `plateau` above `1e-2` is
not a measurement — rerun `group_margins.py` on that bundle alone with a longer `--C`
ladder rather than quoting it.

### Stage B — full splits at the chosen levels, then the long GD run. Six GPU-hours or so.

```bash
LEVELS="1.0 0.35 0.20" nohup setsid bash run_session4.sh B > session4B_nohup.out 2>&1 &
```

Extracts train/val/test at those levels, re-measures the margins *with* the LP and
logistic cross-checks (`results/v4_group_margins_final.md` — these are the numbers the
prediction is read from), then runs `long_horizon.py` at `z = 1e6`
(`results/waterbirds_degrade_v4.{md,json}` and `_curves.json`).

If it stops on the `MAX_HOURS` cap rather than finishing, **rerun the identical command**
and it resumes from saved state.

### Options

| variable | default | what it does |
|---|---|---|
| `LEVELS` | stage A's ladder | the degradation levels |
| `BACKBONE` | `dinov2` | `dinov2_l` is ~3x slower per level but starts from a larger margin (0.839 vs 0.551), so it has more room before the overall margin collapses |
| `KIND` | `resolution` | or `blur`, to check the result is not an artefact of one operator |
| `GROUP` | `0` | which group is degraded |
| `MAX_HOURS` | `5.0` | caps the GD run below the 6 h Paperspace limit |
| `PUSH=1` | off | commit and push `results/` at the end |

A self-check failure in stage 0 aborts before any extraction: if `degrade.py`,
`validate_degrade_path.py` or `validate_group_margins.py` fails, nothing measured
afterwards would mean anything.

---

## 6. Reading the result

Per level, one comparison:

| from | quantity |
|---|---|
| `results/v4_group_margins_final.md` | `beta predicted` (the `gamma` ratio) and which group is larger |
| `results/waterbirds_degrade_v4.md` | `beta_min` and `beta_maj` at the largest **full-window** `z` — not the half-window last row, which is biased upward |

Theorem 5.3 predicts the larger-margin group's `beta` equals the predicted value, the
other group's equals 1, and **neither depends on `eps`**.

Before claiming an escape, apply the session-3 detector — all three, or it is not one:

1. `beta_min` exceeds 1 by **more than the `eps`-spread of `beta_min` itself**;
2. `beta_maj` stays within that spread of 1 (the `alpha > 1` branch pins the majority at
   exactly 1 — it is an asymptotic equivalence there, not a `Theta` with a free exponent);
3. both hold at **two consecutive `z` decades**.

On the session-3 runs the spread was three times the excess, plain ERM showed the same
pattern as the group-robust backbones, and the spread shrank ~15% per half-decade — which
is why none of them was an escape.

### The three outcomes, and what each one is worth

- **The ratio moves and `beta` follows it.** The phase-transition figure the paper does
  not currently have, on real images and a real backbone.
- **The ratio moves and `beta` does not.** A genuine discrepancy between Theorem 5.3 and
  experiment, localised to the alignment identity. Worth more than the first outcome.
- **The ratio does not move.** Degrading images is not the lever; the margin ratio is
  held at 1 by something more robust than image quality. Stage A alone says this, for an
  hour of GPU time, and it sharpens the scope claim: the `alpha > 1` branch needs a
  group asymmetry that these benchmarks do not have and that image degradation cannot
  manufacture.

All three are reportable. None of them requires the branch to exist.
