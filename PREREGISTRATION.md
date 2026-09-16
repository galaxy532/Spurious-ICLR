# Preregistration — Rebuttals_3

Written **11 August 2026, before any result was produced.** Its only value is
that it exists before the numbers do. Nothing here may be edited after a result
is looked at; corrections go in a dated **Amendments** section at the bottom,
with the reason.

Two prior arms on this project returned negative results
(`Rebuttals/`, `Rebuttals_2/`). The purpose of this file is to make it impossible
to convert a third negative into a positive by moving a threshold afterwards.

---

## Arm A — epsilon × backbone sweep

### Claim under test

The manuscript's rate predictions relate quantities observable **without** any
`Phi_r` / `Phi_s` decomposition. Holding the representation fixed and sweeping
the group proportion `eps` by subsampling the minority, at matched normalised
time `z_t`:

- worst-group error scaling as `1/eps` ⇒ `alpha < 1`; balancing is the lever
- worst-group error insensitive to `eps` ⇒ `alpha > 1`; the majority is binding
  and balancing cannot accelerate it

`alpha` is a property of the **representation**, not the dataset. So with the
dataset held fixed, varying the backbone should move the regime.

### The prediction, stated so it can fail

For each `(dataset, backbone)` we measure `kappa`, the exponent in
`worst-group error ~ eps^(-kappa)` at matched `z_t`, and classify:

| `kappa` | regime | prediction |
|---|---|---|
| `>= 0.60` | `alpha < 1` | group-balanced last-layer retraining **helps** |
| `<= 0.20` | `alpha > 1` | balancing gives **no meaningful gain** |
| in between | near the transition | **no prediction made** |

These cut points are fixed now. The `ambiguous` band is a real outcome, not a
failure: the theorem's transition is continuous, so systems near `alpha = 1`
*should* land there, and forcing them into a bin would misrepresent the theory.

Ground truth is measured independently by `dfr_gain` — worst-group accuracy of
group-balanced last-layer retraining minus that of ERM, on a held-out half,
averaged over 20 balanced resamples.

**"Helps" is defined in advance as `dfr_gain_mean >= 0.02`** (2 points of
worst-group accuracy), with `dfr_gain_std` reported alongside.

### What counts as which outcome

- **Success.** Across the `(dataset, backbone)` cells that are not `ambiguous`,
  the `kappa` classification agrees with the measured `dfr_gain` sign in **at
  least 6 of 8** cells, AND at least one cell of each predicted class occurs.
  A sweep that classifies everything into one regime is uninformative even if
  every prediction is "right", and will be reported as uninformative.
- **Negative.** Agreement at or below chance. This gets reported. It is a real
  finding about the theory's practical usefulness.
- **Uninformative.** All cells land in one regime, or more than half are
  `ambiguous`. Reported as such, not spun.

### Fixed analysis decisions

| decision | value | why fixed now |
|---|---|---|
| split | **train** | the sweep is about training dynamics under imbalance, so it needs the split where the imbalance lives |
| eps mechanism | ~~subsample the minority~~ → **reweight the loss** (`--mode reweight`) | see Amendment 2 |
| eps grid, Waterbirds | ~~0.01, 0.02, 0.05, 0.10, 0.25~~ → ~~0.005, 0.01, 0.02, 0.035, 0.05~~ → **0.01, 0.03, 0.08, 0.2, 0.5** | see Amendments 1 and 2 |
| eps grid, CelebA | 0.02, 0.05, 0.10, 0.20, 0.40 | natural eps ≈ 0.42 allows a wider range |
| step size `h` | 0.05 | as in `Rebuttals/` |
| `T` | 2,000,000 | the regime is asymptotic; long on purpose |
| exponent windows | fractional 0.30–0.60 and 0.60–1.00 of the run | reported as two numbers with their drift, never one |
| `kappa` fit | log-log least squares of worst-group error on `eps` at 3 matched `z_t` values, median taken | |
| backbones | erm_rn50, under_rn50, clip, dinov2 | all four are reported whatever they show; `clip` is **excluded from arm A on Waterbirds** by the separability precondition — see Amendment 2 |
| features | standardised | `z_t` is not comparable across backbones otherwise |

### Controls that must hold, or the run is void

1. ~~**`beta_maj` must not move systematically with `eps`.** Only the minority is
   subsampled, so the majority curve is a control. If it drifts, something other
   than the group proportion changed and the cell is discarded, not interpreted.~~
   **RETIRED for `--mode reweight` by Amendment 2** — under reweighting the
   majority's own weight changes with `eps`, so its curve is *expected* to move
   and this test would void every run. Replaced by the margin-invariance and
   uniform-`c` controls in Amendment 2. Still applies to `--mode subsample`,
   where it **failed**.
2. **`beta_maj` should sit near 1.** In `Rebuttals/` it held 0.94–0.98 while the
   minority exponent was still climbing; that is what says the machinery works.
3. **`beta_min` will look too low.** It converges to `max(alpha, 1)` from below
   and slowly (0.716 at `z_T` = 1,500; 0.752 at 6,000; target 1.0). Reported with
   its drift across the two windows. A single low number is **not** a refutation
   and will not be reported as one.

---

## Arm B — per-coordinate invariance rule

### Claim under test

Does the frozen representation contain coordinates whose relation to `y` is
group-dependent, after base-rate differences are removed? If yes, the
`(Phi_r, Phi_s)` split has empirical content in that representation. If no, it
does not, and that explains the earlier negatives rather than adding to them.

### Fixed thresholds

| parameter | value |
|---|---|
| bins | 8 quantile bins, plus a separate zero atom |
| `min_cell` | 10, on **all four** cells of each 2×2 table |
| FDR level `q` | 0.05, Benjamini–Hochberg |
| `tau_rel` | 0.20 log-odds |
| `tau_het` | 0.20 log-odds |
| null | asymptotic Breslow–Day, audited on 200 random coordinates with the exact conditional null |

Threshold sensitivity over `tau ∈ {0.05, …, 0.40}` is reported **every time**,
not only when it is favourable. A split that exists at one setting of a
convention is not a finding.

### Datasets, and a power statement made in advance

The validated power curve (magnitude-only case, mean F1 over 3 seeds) is:

| n | F1 |
|---|---|
| 6,000 | 0.37 |
| 12,000 | 0.66 |
| 20,000 | 0.84 |
| 60,000 | 0.96 |
| 160,000 | 1.00 |

Therefore, declared now:

- **CelebA train (n = 162,770) is the primary test.** Adequately powered.
- **Waterbirds is secondary and underpowered.** Test split is n = 5,794 → F1
  ≈ 0.37; all splits pooled is n = 11,788 → F1 ≈ 0.66. **A null result on
  Waterbirds will not be interpreted as evidence of absence**, because this
  table says it cannot be. Pooling splits is permitted for this arm only, since
  `Phi` is the same function whichever images pass through it.

### Outcomes

- **Positive.** On CelebA, `n_s` is a stable non-trivial fraction across the
  `tau` sweep, and the three rules' agreement is reported.
- **Negative.** `n_s ≈ 0` on CelebA at adequate power. This is a **publishable
  finding**: it says the representation preserves group-invariance of the
  `y`-relation, which would retroactively explain `Rebuttals/`'s `dim K = 0`.
- Either way, all three rules are run and their pairwise agreement reported.

### Stated limitations, fixed now so they cannot be quietly dropped

- The rule **cannot cleanly target `A != B`**. That is a statement about a map
  between blocks; testing it requires the r-block, which is what we are looking
  for. The rule targets the weaker property: coordinates whose relation to `y`
  is group-dependent after base-rate removal. **Necessary, not sufficient** —
  group-dependent label noise and a label-mediated `s'` also produce it.
- **Marginal invariance is not joint invariance.**
- On **CelebA the group and the spurious concept coincide** (`g = Male` *is* the
  attribute), so the two-concept rule's factorial is over `(y, g)` and the "two
  rules using independent information agree" argument is weaker there than on
  Waterbirds. No more will be claimed for CelebA than the construction supports.

---

## What gets reported regardless of outcome

1. Every `(dataset, backbone)` cell of arm A, including `ambiguous` and void ones.
2. `n_s` on both datasets for arm B, with the full `tau` sensitivity table.
3. The Waterbirds power caveat, stated wherever a Waterbirds null appears.
4. The two prior negative arms, in the related-work or limitations section.
   The paper does not get to present arm A as the first thing that was tried.

## Amendments

*(dated entries only, each with the reason.)*

### Amendment 1 — 14 August 2026 — Waterbirds eps grid

**Changed.** Arm A Waterbirds eps grid, from `0.01, 0.02, 0.05, 0.10, 0.25` to
**`0.005, 0.01, 0.02, 0.035, 0.05`**.

**Why.** A sample-size fact, not a judgement call. Waterbirds train contains
**240 minority samples** against 4,555 majority, so the largest reachable `eps`
is **0.0501**. The original grid's top two points were unreachable.

`common.subsample_to_eps` caps `n_min` at the number of minority samples that
exist rather than failing, so the original grid would have collapsed three of its
five points onto `eps = 0.0501`, two of them exact duplicates — silently, with
nothing in the output indicating it.

**Why this matters more than a lost data point.** The failure is directional. The
`kappa` fit would have paired `x = log(0.10)` and `log(0.25)` with `y` values
measured at `eps = 0.05`, flattening the regression slope toward zero and pushing
the classification toward `kappa ≈ 0` — i.e. toward "`alpha > 1`, balancing
cannot help". A **false negative that would have looked like a clean positive
result**, on the arm whose whole purpose is to avoid over-claiming.

**When.** Discovered 14 August 2026 from `datasets.py` cell counts, **before any
sweep was run and before any result existed.** Nothing was fitted, plotted or
looked at beforehand.

**Consequences accepted, and to be reported in the paper:**

- Waterbirds arm A now spans **one decade** of `eps` (0.005–0.05) instead of 1.4.
- `n_min` at the bottom point is **23 samples**. Every Waterbirds curve must be
  reported with its `n_min` column, and the two thinnest points flagged.
- **CelebA becomes the primary arm-A evidence** (0.02–0.40, `n_min` from 1,929 to
  63,003) and Waterbirds a secondary replication. Any claim resting on Waterbirds
  alone is not made.
- The majority-as-control design is **kept**. Sweeping by subsampling the
  majority instead would have reached `eps = 0.5`, but at `eps = 0.5` the run
  would train on 480 samples at `d = 2048`, and the majority would stop being a
  control. Rejected on those grounds, not on the result.

**Guard added.** `eps_backbone_sweep.py` now refuses unreachable `eps` values
with a hard error naming the maximum reachable value, and warns when any point
yields fewer than 50 minority samples. `common.py` was **not** modified — it is a
byte-for-byte copy whose sha256 transfers `Estimator_Validation/`'s
certification.

---

### Amendment 2 — 16 September 2026 — eps by reweighting, not subsampling

**Changed.**

1. Arm A moves `eps` by **reweighting the loss** rather than deleting minority
   rows. `eps_backbone_sweep.py --mode reweight` is now the default;
   `--mode subsample` retains the old behaviour and is still runnable.
2. Waterbirds `eps` grid → **0.01, 0.03, 0.08, 0.2, 0.5**.
3. `clip` is **excluded from arm A on Waterbirds** by a separability
   precondition, and reported as out of scope rather than as a result.
4. Arm-A control 1 (`beta_maj` must not move with `eps`) is **retired** for
   reweight mode and replaced by two others, below.

**WHEN — and this is the part that must not be glossed.** Unlike Amendment 1,
this amendment is made **after** arm A was run and after its results were seen.
The first Waterbirds sweep (`results/waterbirds.{md,json}`, 10 September 2026)
exists, was inspected, and is what motivated the change. That is exactly the
situation a preregistration is meant to make visible rather than hide, so:

- The subsampled results are **reported, not discarded.** They appear in the
  paper alongside the reweighted ones.
- The comparison between the two modes **is** one of the findings, not a
  robustness footnote.
- Nothing was changed to make a number come out better. The change was forced by
  a control failure and by a diagnostic run (`separability_check.py`) that was
  written for the purpose and whose output is in `results/`.

**WHY — the margin confound.**

`--mode subsample` moves `eps` by deleting minority rows. On Waterbirds
(`results/waterbirds_separability.md`) that moves the maximum margin of the
training set by up to **6.4×**, and for `clip` it moves the training set across
the separability boundary entirely — separable at `eps` = 0.005, 0.01, 0.02 and
**not separable** at 0.035, 0.05 (LP proof, maximum margin exactly 0).

The measured decay exponent is a function of that margin. Pooling all 18
separable cells across all four backbones:

    beta_maj = 0.362 * log10(margin) + 1.034      R^2 = 0.92
    r(log margin, beta_maj) = 0.96   r(log margin, beta_min) = 0.97
    r(log eps,    beta_maj) = -0.18

Adding `log10(eps)` to that regression alongside the margin buys **dR^2 =
+0.004**. Once the margin is in the model, `eps` has no explanatory power left.

**The bias has a forced sign, which is why it cannot be averaged away.** Any
subset of a separable set has maximum margin **≥** the full set's, because the
infimum is taken over fewer constraints. Minority points are the ones near the
boundary. So deleting them can only **raise** the margin, which can only **raise**
the measured exponent — lowering `eps` always biases `kappa` in the same
direction. This is combinatorial, not a property of Waterbirds, so CelebA would
inherit it unchanged.

**And it is a gap between the population and the sample, not a sampling bug.**
`common.subsample_to_eps` draws uniformly, so the group-conditional distribution
is preserved exactly. What changes is the sample *size*. In the population the
margin does not depend on `eps` at all — the support of the mixture is the union
of both groups' supports for every `eps` in (0,1) — while in the sample it moves
by 6.4×. The sweep was measuring the sample. Reweighting keeps the sample fixed
and moves only the mixture weight, which is the only thing the theory moves.

**What the confound did to the first run.** Three of four backbones returned
*negative* `kappa` (clip −0.924, under_rn50 −0.958, erm_rn50 −0.257): worst-group
error *rising* with `eps`, a direction **neither** regime of the theory predicts.
`classify()` has no branch for a wrong-sign `kappa`, so all three were reported
as `alpha > 1`. The one backbone whose control held, dinov2, reads "insensitive
to `eps`" only because its margin barely moves under subsampling (1.4× against
5.9× and 6.4×) — so that cell is not independent evidence either.

**Verification done before adopting reweighting.** The concern was that weights
might change *where* GD converges rather than only *how fast*. Measured, not
assumed: with positive fixed per-sample weights, logistic GD still converges in
direction to the same maximum-margin separator. Cosine to the hard-margin SVM
rises with `T` for every `eps`, and the cosine between the `eps` = 0.05 and
`eps` = 0.8 runs goes **0.999288 → 0.999664 → 0.999891** across
`T` = 1e4, 1e5, 1e6, with all ten weighting pairs moving monotonically toward 1.
The effect survives: worst-group error at matched `z_t` spanned **6.8×** across
the grid on that problem, `kappa` = 0.682 (R^2 = 0.99), with the sign the theory
predicts when the minority is binding — the **opposite** of the sign the
subsampled sweep produced on real data.

**Why the grid could widen.** Amendment 1 capped Waterbirds at `eps` ≤ 0.0501
because only 240 minority samples exist. Reweighting deletes nothing, so no
`eps` is unreachable and the cap does not apply. The grid spans **1.7 decades**
(0.01–0.5) instead of one, and brackets the natural proportion 0.0501. A
consequence worth stating: **Waterbirds is no longer range-limited, so it is no
longer dependent on CelebA for the width of the sweep.** Amendment 1's demotion
of Waterbirds to a secondary replication is relaxed on that specific ground —
and on that ground only. The `n_min` = 23 problem also disappears: every cell
now uses all 240 minority samples.

**Why `clip` is excluded rather than retrained.** Frozen CLIP features do not
linearly separate Waterbirds train at all — maximum margin exactly 0, by LP
proof on the full split. The implicit-bias phase this arm measures exponents in
only exists on separable data; off it, GD converges to a finite minimiser, every
`beta` is 0 by construction and `kappa` is **undefined rather than small**.
Reporting that as a scope condition of the theory is the honest result.
Fine-tuning CLIP until it separates was rejected: it would change `Phi`, which is
the experimental axis; it would select backbones on the validity condition, which
is selection bias; and it would destroy the `clip`-vs-`dinov2` contrast the
backbone set was built around. Noted for the paper: `backbones.py` predicted *a
priori* that CLIP would be "the most spuriously entangled of the four", and it
turns out to be the one whose features cannot separate the dataset at all.

**Controls, replacing the retired one.**

1. **Margin invariance.** The maximum margin and separability are constant across
   the whole `eps` grid *by construction* under reweighting — the sample is
   identical at every point. Verified once per bundle by
   `separability_check.py`. This is the control whose failure voided the
   subsampled run.
2. **Uniform-`c` identity.** At `c_i = 1/N` the weighted recurrence must
   reproduce `common.logistic_gd`. Checked by `gd_gpu.py --validate`, which now
   runs it explicitly; measured agreement in numpy is 1.1e-16. Note this is
   reached at `eps` equal to the natural proportion, where the weights are
   uniform by definition — so the grid contains its own consistency check.
3. The `beta_min` drift caveat is **unchanged**: it converges to `max(alpha, 1)`
   from below and slowly, is reported over two `z` windows, and a single low
   number is not a refutation.

**`kappa` stability in `T` is now read, not assumed.** `kappa_per_z` reports the
fit at three horizons (0.25/0.5/1.0 of `z_max`). If those drift, `kappa` depends
on run length and is not yet asymptotic; it is reported as such. In the first run
dinov2 gave [0.071, 0.066, 0.063] and clip gave [−0.61, −0.924, −1.259].

**Unchanged.** Every threshold in the success criteria: `kappa` ≥ 0.60 →
"balancing helps", ≤ 0.20 → "no help", between → no prediction; "helps" means
`dfr_gain_mean` ≥ 0.02 worst-group accuracy; success = agreement in ≥ 6 of 8
non-ambiguous cells **and** at least one cell of each class; an all-one-regime
sweep is UNINFORMATIVE. `h` = 0.05, `T` = 2,000,000, the two exponent windows,
standardisation, the train split, and all of arm B.

**Guards added.** `eps_backbone_sweep.py` now refuses any bundle whose full split
has no separator found, unless `--allow-nonseparable` is passed, and records
`mode` and `full_split_margin` in its output so the two modes can never be
confused after the fact. `common.py` and `identify_rs.py` were **not** modified —
their sha256 values still match those recorded in `README.md`, so
`Estimator_Validation/`'s certification still transfers. The weighted recurrence
lives in `gd_gpu.py` for exactly that reason.
