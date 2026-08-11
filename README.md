# Rebuttals_3 — ICLR resubmission experiments

Third experiment repo for *Implicit Bias of Gradient Descent under Spurious
Correlations*. Companion to `Rebuttals/` (Waterbirds identification) and
`Rebuttals_2/` (tabular arm). **Both of those returned negative results**; this
one is built around that fact rather than around another attempt at the same
thing.

Two independent arms. Neither depends on the other — run them in either order,
and a null in one does not invalidate the other.

| arm | question | needs the `Phi_r`/`Phi_s` split? | critical path |
|---|---|---|---|
| **A. epsilon × backbone sweep** | does the theory *predict* when group balancing helps? | **no** | yes |
| **B. per-coordinate invariance** | is there group-dependent structure in `Phi` at all? | it *produces* one | no |

**Arm A is the paper.** It tests consequences of the theory rather than its
assumptions, which is the correct response to two failed attempts to verify the
assumptions directly. Arm B is cheap, CPU-only, and reportable whichever way it
comes out.

---

## Quick start

```bash
pip install numpy scipy scikit-learn pandas pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch timm            # for the clip / dinov2 backbones

python download_data.py                     # fetch both datasets
python download_data.py --check             # confirm they are complete
python validate_invariance.py --mutants     # CPU, ~15 s
python gd_gpu.py --validate                 # GPU, <1 min, before any real run
python datasets.py                          # cell sizes for both datasets
```

### What downloads automatically, and what does not

**Waterbirds — fully automatic.** One HTTPS GET from Stanford, one tar extract.
No login, no quota.

**CelebA — automatic *if Google Drive cooperates*, which it often does not.**
The official distribution is Drive-hosted and rate-limits by IP, serving a
virus-scan interstitial instead of the file once a quota is hit. `download_data.py`
tries torchvision's downloader (which carries the maintained Drive file IDs),
then `gdown` using those same IDs, and if both fail it prints the exact three
files to fetch by hand, where to put them, and what you should see afterwards.
Landing there is a browser download and a file move — not a debugging session.

`--check` verifies completeness rather than mere presence: it catches a
half-finished extract (annotations present, images missing) instead of letting
it fail three steps later as a confusing column error.

`validate_invariance.py` needs no data and no GPU. If it does not print
`VALIDATION PASSED`, nothing downstream is trustworthy.

`gd_gpu.py --validate` needs a GPU and no data. **`gd_gpu.py` has never been
executed by its author** — it was written in a sandbox where the PyTorch host is
blocked — so it verifies itself against the certified `common.logistic_gd`
instead of being trusted. `eps_backbone_sweep.py --device cuda` refuses to start
until that check passes, so you cannot skip it by accident.

## The pipeline

```bash
# 1. features: one bundle per (dataset, backbone).  GPU, ~1-2 h each for the
#    trained backbones, minutes for the frozen ones.
for B in erm_rn50 under_rn50 clip dinov2; do
  python extract_features.py --dataset waterbirds --backbone $B --splits train,test
  python extract_features.py --dataset celeba     --backbone $B --splits train,test
done

# 2. ARM A -- the deliverable.  --device cuda is NOT optional for CelebA.
python eps_backbone_sweep.py --bundles 'features_waterbirds_*_train.npz' \
    --device cuda --dfr
python eps_backbone_sweep.py --bundles 'features_celeba_*_train.npz' \
    --eps 0.02,0.05,0.10,0.20,0.40 --device cuda --dfr

# 3. ARM B -- the third identification rule.  CPU, minutes to ~10 min.
python invariance.py --bundle features_celeba_erm_rn50_train.npz
python invariance.py --bundle features_waterbirds_erm_rn50_test.npz
```

### Why `--device cuda` is mandatory for CelebA

Measured on CPU before the GPU path was written:

| | ms/step | `T=2e6`, per eps |
|---|---|---|
| Waterbirds train, rn50 | 3.8 | 2.1 h |
| CelebA train, clip | 51.2 | 28.4 h |
| CelebA train, rn50 | 140.5 | **78.0 h** |

CelebA at 5 eps × 4 backbones is **~1,560 CPU-hours, about 65 days**. That is a
feasibility problem, not a tuning one.

The step is memory-bandwidth bound — each iteration streams the design matrix
twice — so on an A6000 at ~768 GB/s the floor for CelebA/rn50 in float32 is
~3.5 ms/step: about **2 h per eps** instead of 78. Budget roughly 26 h for all
of CelebA and about an hour for Waterbirds. `T` is left at `2e6` because the
regime is asymptotic and `z_T` should be as large as the budget allows.

`float32` is the default; `gd_gpu.py --validate` reports the float32-vs-float64
divergence so you can see the precision budget rather than assume it. TF32 is
available and off — its ~1e-3 relative precision is the same order as the
quantities being measured late in the run.

Everything lands in `results/` as small `.json` + `.md`. **`results/` is
tracked** — see the note at the top of `.gitignore` for why that matters.

---

## Arm A — what the table means

`eps_backbone_sweep.py` produces one row per backbone:

| backbone | kappa | regime | predicted | DFR gain (measured) |
|---|---|---|---|---|

`kappa` is the exponent in *worst-group error ~ eps^(-kappa)* at matched `z_t`.
`kappa ≈ 1` is the `1/eps` law, meaning `alpha < 1` and balancing is the right
lever; `kappa ≈ 0` means insensitive to `eps`, so `alpha > 1`, the majority is
binding, and balancing cannot accelerate anything.

**The result is the agreement between the last two columns.** `predicted` comes
from `kappa` alone; `DFR gain` is measured by actually running group-balanced
last-layer retraining. If they track each other across backbones, the theory
supplies a usable decision rule — measure `kappa` on a pilot, then know whether
to spend on balancing. If they do not, that is a real negative about the
theory's practical value and gets reported as one.

Three controls that decide whether the numbers mean anything:

- **Matched `z_t`, never matched epochs.** Comparing at equal wall-clock or
  equal loss gives a clean-looking wrong answer.
- **Standardised features.** Rescaling `Phi` is exactly rescaling the step size,
  so `z_t` is not comparable across backbones without it. `extract_features.py`
  standardises and records it; the sweep warns if the flag is missing.
- **The majority curve is the control.** Only the minority is subsampled, so
  `beta_maj` should sit near 1 and should *not* move with `eps`. If it moves,
  something other than the group proportion changed.

**Expect `beta_min` to look too low.** It converges to `max(alpha, 1)` from
*below*, slowly — 0.716 at `z_T = 1,500`, 0.752 at `z_T = 6,000`, target 1.0.
Two z-windows are always reported; read the drift, never `w1` alone. A short run
reads as a refutation.

## Arm B — the third identification rule

`invariance.py` adds a rule alongside the two in `identify_rs.py`:

| rule | needs | fires when |
|---|---|---|
| two-concept | the spurious attribute annotated | the 2×2 factorial loads on the attribute |
| sign-flip | group index only | `rho_0 · rho_1 < 0` — a **sign reversal** |
| **invariance** | group index only | the relation to `y` differs across groups **in any way** |

It is per-coordinate Invariant Causal Prediction: a coordinate is causal iff
`y | Phi_k` has the same conditional law in both groups.

### Why it was worth writing

The sign-flip rule fires only on sign reversal. Definition 3.3's spuriousness is
`A != B`, which includes `A = 2B` with both positive — spurious, no sign flip,
**invisible to sign-flip**. Magnitude-only differences are most of the space.

Worse, the reason sign-flip works on Waterbirds is that `g := 1[place != y]`
manufactures the reversal. That is a *label*-mediated construction — the case
the paper distinguishes itself from. So the rule is tuned to the wrong setting
and blind to the paper's own.

`validate_invariance.py` measures this directly (F1 on the magnitude-only case,
mean of 3 seeds):

| n | invariance | sign-flip |
|---|---|---|
| 6,000 | 0.37 | **0.00** |
| 12,000 | 0.66 | **0.00** |
| 20,000 | 0.84 | **0.00** |
| 60,000 | 0.96 | **0.00** |
| 160,000 | 1.00 | **0.00** |

Sign-flip is flat zero at every sample size. On the sign-reversal case both
rules score 1.00, so nothing is lost.

### The power curve decides which dataset you can use

Read that table against what is actually available:

| | n | expected F1 |
|---|---|---|
| Waterbirds test | 5,794 | ~0.37 — **underpowered** |
| Waterbirds, all splits pooled | 11,788 | ~0.66 — marginal |
| CelebA train | 162,770 | ~1.00 — comfortable |

**A null result on Waterbirds alone would be uninterpretable.** Run CelebA.
Pooling Waterbirds splits is legitimate for this question — `Phi` is the same
function whichever images pass through it, the same argument that justified
measuring coupling on the test split in `Rebuttals/`.

### The trap it is built to survive

The naive test — is `P(y | bin, g=0) == P(y | bin, g=1)`? — is wrong, and wrong
in a way that produces a *confident* false positive. If the groups have
different class balance, Bayes' rule makes the within-bin probabilities differ
for **every** coordinate, causal ones included. On CelebA
`P(blond | female) ≈ 0.24` against `P(blond | male) ≈ 0.02`, a factor of twelve.

So the null is not "no group difference" but "a group difference that does not
depend on where you are along the coordinate". On the log-odds scale a pure
base-rate shift is a *constant offset*; a genuine change in the relation makes
that offset vary bin to bin. That is the Breslow–Day test for homogeneity of the
odds ratio, with Mantel–Haenszel supplying the common odds ratio. Validation
check C4, at three shift sizes:

| | invariance rejects | naive rejects |
|---|---|---|
| pure base-rate difference, `d = 80` | **0 / 80** | **80 / 80** |

The naive statistic rejects literally everything. This is the single strongest
argument for the construction and belongs in the paper.

### What it cannot do

Stated here so it gets stated in the paper:

- **It cannot cleanly target `A != B`.** That is a statement about a map
  *between blocks*, and testing it needs the r-block — which is what we are
  looking for. Circular. The rule targets the weaker, well-defined property:
  coordinates whose relation to `y` is group-dependent after base-rate effects
  are removed. Necessary, not sufficient — group-dependent label noise and a
  label-mediated `s'` also produce heterogeneity.
- **Marginal invariance is not joint invariance.** Full ICP intersects over
  subsets: `2^d` tests, and `k^|S|` bins to condition on `|S|` coordinates, so
  empty bins by `|S| ≈ 5`. Not an option at `d = 2048`.
- Both limits are shared with the sign-flip rule, which is also marginal and
  also only necessary. So this is not worse than what is already in the pipeline.

---

## Datasets

Read `datasets.py` — the declarations are there with their reasoning.

**Waterbirds**: `y = 1[waterbird]`, `g = 1[place != y]`, `eps ≈ 0.05`.
Unchanged from `Rebuttals/`.

**CelebA**: `y = 1[Blond_Hair]`, `g = 1[Male]`, `eps ≈ 0.42`.

> **This reverses a decision recorded in `Rebuttals/README.md`**, which called
> CelebA unusable because the natural `eps ≈ 0.45` means "no imbalance". That
> objection was right *for that use* — reading the natural `eps` off the data.
> The epsilon-sweep does the opposite: it **manufactures** `eps` by subsampling.
> For that, a naturally balanced group variable is the ideal substrate, because
> it is what gives the sweep room to move. A dataset starting at `eps = 0.05`
> can only be swept downward; one starting at `0.42` sweeps across an order of
> magnitude with thousands of samples at every point.

Two CelebA caveats that must be stated, not discovered:

1. The base rates differ by a factor of twelve across groups — the confound
   above. CelebA is the dataset that would have caught a naive implementation.
2. **The group and the spurious concept coincide** (`g = Male` *is* the
   attribute), unlike Waterbirds where `g` is derived from it. So the
   two-concept rule's factorial is over `(y, g)`, the same cells the coupling
   test conditions on, and the "two rules using independent information agree"
   argument is weaker on CelebA. Report the three-rule agreement for both and
   claim no more for CelebA than the construction supports.

The loader accepts **both CelebA layouts**, sniffed by extension:

- **Official / Google Drive** — `list_attr_celeba.txt`, whitespace-delimited,
  with a leading count line and a 40-name header, so rows carry one more field
  than there are names. This is the one currently planned.
- **Kaggle** (`jessicali9530/celeba-dataset`) — `list_attr_celeba.csv`, ordinary
  CSV with an `image_id` column.

Verified against a synthetic fixture in both layouts. If a mirror is swapped
mid-project, nothing downstream changes.

Data lives in a **sibling** directory, never inside the repo:

```
<paperspace root>/
├── NeurIPS-rebuttals-3/     <- run all commands from here
└── data/
    ├── waterbird_complete95_forest2water2/
    └── celeba/
```

Override with `SPURIOUS_DATA_ROOT`.

---

## Files

| file | role | runs where |
|---|---|---|
| `PREREGISTRATION.md` | thresholds and outcomes, fixed before results | — |
| `download_data.py` | fetch both datasets, verify completeness | CPU |
| `datasets.py` | y / group / eps declarations, cell reports | CPU |
| `backbones.py` | the four `Phi` and why each is there | CPU |
| `extract_features.py` | train-or-load, freeze, embed, standardise | **GPU** |
| `eps_backbone_sweep.py` | **arm A** — the deliverable table | CPU or **GPU** |
| `gd_gpu.py` | GPU logistic GD + its agreement check | **GPU** |
| `invariance.py` | **arm B** — the third rule | CPU |
| `validate_invariance.py` | ground-truth validation of arm B | CPU, 15 s |
| `common.py` | shared estimators, copied from `Rebuttals/` | CPU |
| `identify_rs.py` | the two existing rules, copied from `Rebuttals/` | CPU |

`common.py` and `identify_rs.py` are **byte-for-byte copies**, so the validation
recorded in `Estimator_Validation/README.md` (alpha recovery to 0.02%, coupling
power and calibration, isotropy diagnostic) certifies these copies too:

```
common.py       sha256 5bb39fad639a6e208a8890dbdf6399be53941ec622d850db688dced008a0aafe   1116 lines
identify_rs.py  sha256 dd0ce7f86f07444c64728e989efad0f01c1c0b95ecc830f82503036371608aa4    505 lines
```

If those stop matching `Rebuttals/`, the copies have drifted and the validation
no longer transfers. `sha256sum common.py ../Rebuttals/common.py`.

---

## Lessons carried forward from the earlier repos

1. **`results/` is tracked.** `Rebuttals_2/` ignored it wholesale, which broke
   the only channel for getting numbers off Paperspace and onto a laptop for
   auditing. Every run looked fine and produced nothing pullable.
2. **`.gitignore` rules go in before the first commit.** `Rebuttals/` still
   tracks `__pycache__` because the rule was added after those files were
   committed, and a `.gitignore` has no effect on files git already tracks.
   Check `git status --short` before the first commit.
3. **Run the download step early.** It is the one step that could not be tested
   from a sandbox in any of the three repos. Discovering a dead URL on deadline
   day has now nearly happened twice.
4. **Validate estimators against known ground truth before running them on real
   data.** `validate_invariance.py` caught a genuine bug in its own subject
   during development — see below.

## One bug worth knowing about

`stratified_tables` guards **all four cells** of every 2×2 table, not the row
totals. Guarding row totals alone was the first implementation and it was wrong.

In the separable regime the manuscript studies, the extreme bins of a coordinate
saturate — every sample in the top bin has `y = +1`, giving tables like
`a = 2294, b = 0, c = 206, d = 0`. That odds ratio is `0/0`. The Haldane
correction makes it finite, but the value it produces then grows like `log(n)` —
a number manufactured entirely by the correction. Effect sizes inflated 2.6×,
and the statistic scattered from 6.3 to 57.6 across coordinates that were
identical by construction.

**The top-level symptom was F1 that did not increase with `n`.** If that ever
reappears, look there first. Mutant `M4` in the validation suite reintroduces
the bug and asserts the suite catches it.
