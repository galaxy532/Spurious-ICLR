# HANDOVER — state of play, 16 September 2026

Read this before `README.md`. It records what happened in the 10–16 September
sessions, including one thing that was said wrongly and has been retracted.

The formal record of the design change is **`PREREGISTRATION.md`, Amendment 2**.
This file is the narrative around it: what is established, what was withdrawn,
what is still open, and what to ask for first.

**This file SUPERSEDES the project-memory note `project_first_sweep_results.md`**
on everything to do with arm A. That note was written on 10 September, before the
separability run and before the margin regression, and it treats the separability
story as an unconfirmed hypothesis and CelebA arm A as still worth costing. Both
have moved on. Where the two disagree, this file is right. (Project memory was
unavailable when this was written, which is why it could not simply be updated.)

---

## 0. A RETRACTION

**Claim made, and withdrawn:** *"training a backbone longer to reach separability
also makes the spurious signal more dominant, so there is a tension between
wanting separable features and wanting spuriously-entangled ones."*

That was asserted as though it were established. **It is not.** It was an
unverified plausibility argument, and no source was checked before stating it.

It may also be backwards. Two literatures point the other way:

- **Kirichenko, Izmailov & Wilson (ICLR 2023)**, the paper DFR comes from —
  whose thesis is that ERM-trained features *retain* the core information, which
  is why last-layer retraining recovers worst-group accuracy.
- **Simplicity bias / gradient starvation** (Shah et al. 2020; Pezeshki et al.
  2021) — the simple, spurious feature is learned *first*, which would predict
  `under_rn50` (1 epoch) is *more* spurious-dominated than `erm_rn50`, not less.

Both of those are stated from memory and were **not** re-checked either. Treat
them as pointers to verify, not citations. **Nothing in the paper should rest on
that tension claim.** If the framing matters, it needs a real literature search.

What *does* survive, and depends on no literature at all: the implicit-bias
regime requires linearly separable features; separability is a measurable
property of (`Phi`, sample); and nothing in the backbone design controls it.

---

## 1. Established results

### Arm B is POSITIVE — the strongest thing in the project right now

`invariance.py` on `features_celeba_erm_rn50_train.npz` (n = 162,770, d = 2,048):

- BH-rejected (heterogeneous relation): **820 of 2,048**
- **n_r = 475, n_s = 608**, n_weak = 965, 917 untestable (< 2 usable bins)
- Threshold sensitivity is smooth and monotone (tau 0.05 → 0.40 moves n_s
  815 → 204), so the split is not an artefact of tau = 0.20

The feared null (`n_s ≈ 0`, which would have retroactively explained
`Rebuttals/`'s `dim K = 0`) **did not happen**. Arm B has only been run on this
ONE bundle — extending it to Waterbirds and the other backbones is cheap CPU work
and probably worth doing.

### Arm A's subsampling sweep is confounded by the margin

Full argument and numbers: `PREREGISTRATION.md` Amendment 2. In brief:

- Deleting minority rows moves the training set's max margin by up to **6.4×**,
  and for CLIP across the separability boundary entirely.
- Pooling all 18 separable cells across four backbones:
  `beta = 0.362 log10(margin) + 1.034`, **R² = 0.92**. Adding `log10(eps)` buys
  **ΔR² = +0.004**. `r(log eps, beta) = −0.18`.
- The bias has a **forced sign**: any subset of a separable set has max margin
  ≥ the full set's, so lowering eps can only raise the margin and hence the
  measured exponent. Combinatorial — CelebA would inherit it unchanged.
- It is a population-vs-sample gap, not a sampling bug. `subsample_to_eps` draws
  uniformly, so the group-conditional distribution is preserved. What moves is
  the sample *size*. In the population the margin does not depend on eps at all.

Consequence: `results/waterbirds.{md,json}` (10 Sept, subsampled) does **not**
measure alpha. It is kept and reported — the mode comparison is itself a finding
— but no claim about alpha rests on it.

### CLIP does not linearly separate Waterbirds

Max margin exactly **0** on the full train split, by LP proof
(`results/waterbirds_separability.md`). Excluded from arm A by the separability
precondition and reported as **out of scope**, not retrained. Decided with Aser.
`backbones.py` predicted *a priori* that CLIP would be "the most spuriously
entangled of the four"; it turns out to be the one whose features cannot separate
the dataset at all. He wants this in the manuscript rewrite.

Full-split margins (the three that remain, spanning 150×):
dinov2 **0.551**, under_rn50 **0.0163**, erm_rn50 **0.00373**.

### Reweighting was verified before adoption

Weights change *how fast*, not *where*. With positive fixed per-sample weights,
logistic GD still converges in direction to the same max-margin separator:
cosine to the hard-margin SVM rises with T for every eps, and the cosine between
the eps = 0.05 and eps = 0.8 runs goes **0.999288 → 0.999664 → 0.999891** across
T = 1e4, 1e5, 1e6, with all ten pairs monotone toward 1. The effect survives:
worst-group error spanned **6.8×**, kappa = 0.682 (R² = 0.99), with the sign the
theory predicts when the minority is binding — the *opposite* of the sign the
subsampled sweep gave on real data.

Caveat, stated: one synthetic configuration, n = 2,000, d = 50.

### gd_gpu is validated at a realistic horizon

`--validate --T 200000`: float64 agrees to **5.9e-16**, float32 budget
**1.8e-3**. float32 is fine for these runs.

---

## 2. What changed in the repo

| file | change |
|---|---|
| `PREREGISTRATION.md` | **Amendment 2** — reweighting, new grid, CLIP exclusion, arm-A control 1 retired |
| `eps_backbone_sweep.py` | `--mode reweight\|subsample` (default reweight), `eps_weights()`, separability precondition, `--allow-nonseparable`, `mode` + `full_split_margin` in output |
| `gd_gpu.py` | `c=` per-sample weights; numpy `logistic_gd_weighted`; `--validate` also checks weighted-torch vs weighted-numpy and the uniform-`c` reduction |
| `separability_check.py` | NEW — per-cell separability and margin, constructive test then exact LP |
| `progress.py` | NEW — shared progress bars, tqdm-or-fallback, stderr only, `NO_PROGRESS=1` disables |
| `extract_features.py`, `invariance.py`, `validate_invariance.py`, `download_data.py` | progress bars |
| `common.py`, `identify_rs.py` | **UNTOUCHED** — sha256 `5bb39fad…` / `dd0ce7f8…` still match README, so `Estimator_Validation/`'s certification still transfers |

**Standing rule:** every script written for this repo shows a progress bar on its
long-running loops. Aser asked for this explicitly and it applies to all future
scripts, not just these.

**Arm-A control 1 is retired** for reweight mode (`beta_maj` must not move with
eps). Under reweighting the majority's own weight changes with eps, so its curve
is *expected* to move and the test would void every run. Replaced by margin
invariance and the uniform-`c` identity. Struck through rather than deleted, and
still applies to `--mode subsample`, where it failed. Aser agreed to this.

---

## 3. What to ask for first

The reweighted arm A was queued on 16 September:

```
python gd_gpu.py --validate --T 200000
python eps_backbone_sweep.py --bundles 'features_waterbirds_*_train.npz' \
    --eps 0.01,0.03,0.08,0.2,0.5 --mode reweight --device cuda --dfr --tag waterbirds_rw
```

Ask for `results/waterbirds_rw.{md,json}`. Then, in order:

1. **`kappa_per_z`** for each backbone — the fit at three horizons
   (0.25/0.5/1.0 of z_max). Three stable numbers means kappa is asymptotic;
   drift means it is not and T must go up. This is the only outstanding caveat
   from the reweighting decision. (First run, for reference: dinov2
   `[0.071, 0.066, 0.063]`, clip `[−0.61, −0.924, −1.259]`.)
2. **Whether CLIP was skipped** by the gate, and that the skip is in the log.
3. **`beta_min` drift across the two z-windows** — unchanged caveat, converges to
   `max(alpha, 1)` from below and slowly. A single low number is not a
   refutation.
4. Whether `erm_rn50` (full-split margin 0.00373) reached the regime at all. Its
   margin is ~150× smaller than dinov2's, so it may need a larger T even with
   the confound removed.

---

## 4. Open items

- **CelebA arm A is not run and is not scheduled.** It is a ~25–40 h GPU job
  against a 6 h Paperspace session cap, and reweighting removed the reason it was
  needed (Waterbirds now spans 1.7 decades of eps instead of one, so it is no
  longer range-limited). Before ever committing that compute, measure CelebA's
  margin response — it inherits the same confound in `--mode subsample`, and
  almost certainly has no separable cell at all (n ≈ 96k–158k against d ≤ 2,048),
  which would make arm A inapplicable there.
- **`eps_backbone_sweep.py` writes nothing until every bundle finishes.** This
  cost two killed CelebA runs. Per-bundle incremental writes plus a GD resume
  checkpoint are still unbuilt and are the highest-value robustness fix left.
  (`separability_check.py` already writes after every bundle.)
- **Batched-GEMM speedup, unbuilt.** Under reweighting every eps shares the
  identical design matrix, so all five can run as one GEMM —
  `G = Xy.T @ (P * C)`, one pass per step instead of five, ~5× and exact (no
  masks, no change to fp32 summation order). Only worth building if CelebA is
  ever revived.
- **Exact margins via Frank–Wolfe, unbuilt.** The margins in
  `results/waterbirds_separability.md` are lower bounds from a logistic fit. The
  min-norm point in the convex hull of `{y_i x_i}` gives the exact max margin and
  scales to CelebA where the LP does not.
- **Arm B on more bundles** — cheap CPU, and it is the positive result.
- **Project memory was unavailable** for most of the 16 September session, which
  is why this file exists.

---

## 5. Two standing rules from Aser

- He audits reasoning rather than accepting it, and has been right every time he
  has pushed back. Do not hedge, and do not assert anything unverified —
  especially citations and mathematical claims. Say "I have not checked this"
  when that is the case. Section 0 of this file exists because that rule was
  broken.
- Consult him before implementing changes. He leads this project.
