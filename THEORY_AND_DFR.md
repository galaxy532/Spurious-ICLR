# How the theory meets DFR

Written 17 September 2026. Sources checked in this session:
Kirichenko, Izmailov & Wilson, *Last Layer Re-Training is Sufficient for Robustness
to Spurious Correlations*, ICLR 2023 (`2204.02937v2.pdf` in the project folder),
and our manuscript (`neurips_2026.tex`, `supplementary.tex`, compiled PDF).
Anything not checked against one of these is marked **(not verified)** or
**(hypothesis)**.

---

## 0. The idea in five lines

1. Our theory is about **speed**: how fast each group's error falls under gradient
   descent (GD), and how the group proportion ε changes that speed.
2. DFR is about **destination**: which classifier you end up with, judged by
   worst-group test accuracy.
3. **Corollary 7.1** is the one place where our manuscript talks about the
   destination (test error), and it depends only on the max-margin direction ŵ.
4. ŵ does **not** depend on ε. So changing ε by **reweighting** (same data points)
   changes the speed but never the destination. Changing the destination requires
   changing **which points** are in the training set — which is exactly what DFR's
   **subsampling** does.
5. So the theory explains DFR's central design choice — why it balances by
   **deleting points** rather than by reweighting or balanced sampling. In the
   population, deleting points would change nothing (the support, hence ŵ, is the
   same). For deletion to matter it must change the SVM, i.e. remove points that
   **define the margin** (support vectors). Removing points that are not on the
   margin leaves the hard-margin SVM exactly unchanged.

---

## 1. What DFR actually says

**The claim (Section 1).** Networks trained with ordinary ERM still learn the "core"
features; they fail on minority groups because the **last layer** puts too much
weight on the spurious features. So retraining only the last layer can fix it.

**The method (Sections 5–6, footnote 3).**
- Freeze the feature extractor, discard the last layer, train a new one (logistic
  regression) on a **reweighting dataset D̂ where the groups are represented
  equally**.
- D̂ is built by subsampling: *"We keep all of the data from the smallest group,
  and subsample the data from the other groups to the same size."* (footnote 3)
- Main version, DFR^Val_Tr: D̂ is a group-balanced subset of the **validation**
  data (not the training data).
- Logistic regression is fitted **10 times** on different balanced subsets and the
  **weights are averaged**.
- **Strong ℓ1 regularisation**, with its strength tuned — because *"the size of the
  reweighting set D̂ is small relative to the number of features"* (Section 6).

**Two ablation findings that matter here (Appendix A and C.4).**
- **Subsampling, not sampling.** *"In DFR, we subsample the reweighting dataset to
  be group balanced instead of using class- or group-balanced sampling. […] This
  detail is hugely important, as group-balanced sampling does not produce
  classifiers robust to spurious correlations (e.g. see RWG and SUBG methods in
  Idrissi et al., 2021)."* (Appendix A.)
  Note: in Idrissi et al., SUBG is itself a subsampling method, so the example in
  that parenthesis is not obviously consistent with the sentence. **(Idrissi et al.
  not checked.)**
- **Held-out data beats training data.** Retraining the last layer on the training
  data (DFR^Tr_Tr) is worse. Their explanation (Appendix C.4): the feature extractor
  has seen the minority training points, so the features of those points are
  distributed differently from the features of unseen test points; the last layer
  is then trained under a distribution shift.

**What DFR says about separability or margins: nothing.** "Maximum-margin" appears
only once, in the related-work section, citing other papers. "Linearly separable"
appears only to describe the Dominoes images. The paper has no theory of the
last-layer retraining step; it is an empirical method with an empirical explanation.

---

## 2. What our theory says (only what is needed here)

**Setting.** Linear model on a fixed representation Φ = (r, s), binary labels,
logistic loss (the implicit-bias proposition allows any loss satisfying Soudry et
al.'s Assumptions 2–3, `supplementary.tex` l.2648), population gradient descent,
training data linearly separable (Assumption `ass:sep`). The training distribution
is a mixture: group 1 with probability 1−ε, group 2 with probability ε, ε ∈ [0,1].

**The max-margin direction ŵ** (`eq:full-separability`):
ŵ = argmin ‖w‖² subject to P_train(y w·Φ(x) ≥ 1) = 1.
The constraint must hold **almost surely** under the training mixture.

**Why ŵ does not depend on ε (our derivation from that definition).** For any ε
strictly between 0 and 1, the mixture gives positive probability to exactly the
same regions as each group does. So "holds almost surely" means the same set of
constraints for every such ε, and the minimiser ŵ is the same. (At ε = 0 or 1 one
group disappears and ŵ can change.) This matches the numerical check made before
adopting reweighting: the directions of runs at different ε converged to each other
(cosine 0.999891 at T = 1e6, synthetic data).

**GD goes to ŵ** (Proposition `prop:soudry-continuous-main`):
w^t = ŵ log z_t − ŵ log log z_t + O(1). The direction tends to ŵ; the size grows
like log z_t.

**Speed** (Theorem `thm:main-iso`): ε enters the **rates**. If α < 1, each group's
training error behaves like κ_g / (ε_g z_t). If α > 1, the group with the larger
r-margin escapes the ε-dependence and decays like z_t^(−α).

**Destination** (Corollary 7.1, `cor:test-shift-general-main`, "Accuracy ceiling
under correlation shift"). A test distribution where the spurious feature follows a
different map, s = C r + ξ. Define χ̂ = ŵ_r + Cᵀ ŵ_s and the test margin γ_test(C).
- (i) γ_test > 0: test error still goes to 0, at a rate governed by γ_test.
- (ii) γ_test < 0: test error tends to a floor P(χ̂·r + ŵ_s·ξ < 0) > 0 — in the
  compiled PDF's words, a floor that emerges *"irrespective of how long training
  continues"*.

The only training-side object in Corollary 7.1 is **ŵ**. ε does not appear.

---

## 3. Why Corollary 7.1 is the bridge

DFR is judged on **test** accuracy. Corollary 7.1 is the only result in our
manuscript about test error. Put it together with section 2:

1. The long-run test outcome (whether there is a floor, its value, and the decay
   exponent when there is none) depends only on ŵ and on the test distribution.
2. ŵ is the same for every ε in (0,1).
3. **So reweighting the groups cannot change the long-run test outcome.** It only
   changes how fast GD gets there (through the 1/ε_g factors of the theorem).
4. To change the long-run test outcome you have to change ŵ. There are only three
   ways to do that:
   - change **which points** the separator must satisfy — delete some
     (**subsampling**) or use different ones (**held-out data**);
   - **stop before convergence** (early stopping);
   - **leave plain GD** (e.g. add a regulariser such as ℓ1).

This sorts DFR's ingredients into two kinds:

| ingredient | changes speed | changes destination (ŵ) |
|---|---|---|
| group-balanced *sampling* / loss reweighting | yes | **no** |
| group-balanced *subsampling* (footnote 3) | — | yes |
| held-out D̂ instead of training data | — | yes |
| ℓ1 regularisation | — | yes (not the ℓ2 max margin any more) |
| averaging 10 fits | — | yes (average of different destinations) |

---

## 4. DFR's findings seen through this

**(a) "Group-balanced sampling does not produce robust classifiers."**
Consistent with section 3: sampling with balanced group probabilities is, in
expectation, the same as reweighting, so the points are the same, ŵ is the same,
and Corollary 7.1 gives the same destination. **Our theory supplies a mechanism that
DFR does not give.** Caveats: DFR fits a regularised model, not GD run to
convergence; and minibatch sampling equals reweighting only in expectation.

**(b) "Subsampling is hugely important."**
Consistent with section 3: deleting points changes the constraint set, so ŵ changes.
How much this can matter is something we measured ourselves on Waterbirds: deleting
rows moved the training set's maximum margin by up to **6.4×**
(`PREREGISTRATION.md`, Amendment 2). And it moves in a known direction: a subset of a
separable set has a maximum margin at least as large as the full set's.
**(Hypothesis:)** DFR's gain comes from this change of ŵ. It has not been separated
from the effect of ℓ1 regularisation or of using held-out data.

**This is the key point (Aser, 17 Sept).** In the **population**, subsampling at a
fixed proportion does not change which regions have positive probability either, so
by our theory it would have **no effect** on ŵ and no effect on the long-run test
outcome. Therefore, **for DFR's deletion to have any effect, it must change the SVM
solution — it must remove points that define the margin.** This is not a limitation
of the theory; it is what the theory says the mechanism of DFR has to be.

A standard property of the hard-margin SVM makes this sharp: deleting training points
that are **not** support vectors leaves the solution unchanged (the remaining
constraints and optimality conditions still hold). So:
- a DFR subsample that happens to keep every support vector gives the same ŵ as the
  full set, and the same long-run test outcome as reweighting;
- DFR can only differ when its deletions hit support vectors. Keeping all of the
  smallest (y, g) cell and cutting the large cells mostly deletes **majority** points;
  if majority points are on the margin, their removal moves ŵ.

This is also the effect that confounded our first ε-sweep (deleting rows moved the
max margin by up to 6.4×): the same mechanism, seen from the other side.

**(c) "Held-out data beats training data."**
DFR's explanation is a shift in feature distribution between seen and unseen points.
**(Hypothesis, not tested:)** a margin reading of the same fact — on the training
points the frozen features are already arranged around exactly those points, so the
separator they impose is fitted to them; a held-out D̂ imposes constraints from points
the extractor never saw, which are more like the test points.

**(d) "D̂ is small relative to the number of features."**
Then D̂ is typically linearly separable, so an **unregularised** logistic fit on D̂ is
exactly in the implicit-bias regime our theory describes. This is where the theory
applies most directly to DFR-style practice. DFR's ℓ1 moves away from that regime.

---

## 5. Where DFR is incomplete, and what our theory adds

- **DFR shows *that* last-layer retraining works, not *why* one balancing method works
  and another does not.** Corollary 7.1 plus the ε-independence of ŵ explains both
  halves: balanced sampling/reweighting cannot move ŵ; deletion can, and only through
  the support vectors it removes (section 4b).
- **(Secondary, optional.)** DFR's "group-balanced sampling does not help" is about the
  destination only. At a finite training budget reweighting still changes a group's
  error by the 1/ε_g factor, so it could help early and stop helping later. This is a
  side prediction; it is not needed for the main argument above.
- **The name is slightly misleading about the mechanism.** "Deep Feature
  *Reweighting*" works by *subsampling*; per our theory, true reweighting only changes
  speed.
- **What we cannot claim.** That DFR is wrong. Our theorem assumes a population, the
  isotropic regime, separability and unregularised GD; DFR violates several of these
  (finite D̂, ℓ1, averaging). Everything in sections 4–5 is a reading of DFR through
  our theory, not a proof about DFR.

---

## 6. The experiment that tests this — `dfr_bridge.py` (built 17 Sept 2026)

Frozen Waterbirds features (train and test bundles). Last layer trained by plain GD on
the logistic loss, **no regularisation**, recording worst-group **test** error and
accuracy at matched z.

- **A — reweighting.** Same training points, ε on the grid 0.01…0.5 by loss weights.
  Prediction: test curves differ at finite z, then converge to the same limit.
  Direct check: cosine between the weight directions across ε → 1.
- **B — DFR-style subsampling.** Group-balanced subsample built as in footnote 3,
  several draws. Prediction: converges to a **different** limit (different ŵ).
- **C — DFR itself** (B with ℓ1, averaged over the draws) as the reference point.
- **Support-vector check.** For each B draw: the fraction of the tightest training
  points (approximate support vectors under the A direction) it deleted, next to the
  fraction of all points it deleted, and its cosine to A. Prediction: draws that delete
  more margin points end further from A.

**Prerequisite found while building it.** Until 17 Sept, `extract_features.py`
standardised each split with its own mean and std — a different map on train and test.
Train-only analyses are unaffected, but a last layer fitted on train cannot be evaluated
on such test features. Fixed (all splits now use train statistics); `dfr_bridge.py`
refuses old bundles, so test/val bundles must be re-extracted.

Caveat: if the test groups are separated by ŵ (Corollary 7.1 case (i)), there is no
floor and test error goes to 0 in both A and B; then compare the decay rates, not the
floors.

---

## 7. What is verified and what is not

| statement | status |
|---|---|
| DFR quotes in section 1 | checked in `2204.02937v2.pdf` |
| DFR has no separability/margin theory | checked (text search of the whole paper) |
| Corollary 7.1 statement; ŵ definition; GD → ŵ proposition | checked in manuscript |
| ŵ is the same for every ε in (0,1) | our derivation from the definition; matches the synthetic cosine check |
| reweighting cannot change the long-run test outcome (in the theorem's setting) | follows from the two lines above |
| subsampling moves the max margin (up to 6.4× on Waterbirds) | measured, Amendment 2 |
| DFR's gain is caused by the change of ŵ | **hypothesis** |
| margin reading of "held-out beats training" | **hypothesis** |
| deleting non-support points leaves the hard-margin SVM unchanged | standard SVM property (not re-derived here) |
| DFR's effect goes through deleted support vectors | **prediction**, tested by `dfr_bridge.py` |
| reweighting helps early and not late | side **prediction**, not needed for the argument |
| Idrissi et al.'s RWG/SUBG results | **not checked** |
