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
| eps grid, Waterbirds | ~~0.01, 0.02, 0.05, 0.10, 0.25~~ → **0.005, 0.01, 0.02, 0.035, 0.05** | see Amendment 1 |
| eps grid, CelebA | 0.02, 0.05, 0.10, 0.20, 0.40 | natural eps ≈ 0.42 allows a wider range |
| step size `h` | 0.05 | as in `Rebuttals/` |
| `T` | 2,000,000 | the regime is asymptotic; long on purpose |
| exponent windows | fractional 0.30–0.60 and 0.60–1.00 of the run | reported as two numbers with their drift, never one |
| `kappa` fit | log-log least squares of worst-group error on `eps` at 3 matched `z_t` values, median taken | |
| backbones | erm_rn50, under_rn50, clip, dinov2 | all four are reported whatever they show |
| features | standardised | `z_t` is not comparable across backbones otherwise |

### Controls that must hold, or the run is void

1. **`beta_maj` must not move systematically with `eps`.** Only the minority is
   subsampled, so the majority curve is a control. If it drifts, something other
   than the group proportion changed and the cell is discarded, not interpreted.
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
