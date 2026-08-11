"""Per-coordinate group-invariance rule -- the third identification rule.

WHAT THIS IS
============

A third way to split the frozen representation Phi into Phi_r (causal) and
Phi_s (spurious), to sit alongside the two rules already in `identify_rs.py`:

  1. two-concept   -- needs the spurious attribute annotated (Waterbirds
                      `place`, CelebA `Male`). 2x2 factorial over (y, attribute).
  2. sign-flip     -- needs only the group index. Keeps coordinate k when the
                      group-conditional correlation with y REVERSES SIGN,
                      rho_0 * rho_1 < 0.
  3. invariance    -- this file. Needs only the group index. Keeps coordinate k
                      when the RELATION between k and y differs across groups
                      in any way at all, sign reversal or not.

Rule 3 is the nonparametric, per-coordinate form of Invariant Causal Prediction
(Peters, Buehlmann & Meinshausen 2016): a predictor is causal iff the conditional
law of y given that predictor is the same in every environment. Here the
environments are the two groups G_maj and G_min.


WHY IT WAS ADDED -- the blind spot in the sign-flip rule
========================================================

This is the whole motivation, so it is worth being precise.

The manuscript's Definition 3.3 makes a coordinate spurious when the operator
carrying r to s DIFFERS between the groups:  A != B.  The sign-flip rule fires
only when the group-conditional correlation with y reverses sign. Those are not
the same condition. Take

        A = 2 B,   both positive.

The relation to y differs by a factor of two across groups, so the coordinate is
spurious under Definition 3.3 -- and no sign flip occurs, so the sign-flip rule
files it as CAUSAL. Magnitude-only differences are the bulk of the space of
(A, B) pairs with A != B, and the sign-flip rule is structurally blind to all of
them.

Worse, the reason the sign-flip rule works on Waterbirds at all is an artefact of
how that benchmark defines its groups. With  g := 1[place != y], recoding both to
+/-1 pins  place = y  in group 0 and  place = -y  in group 1. The sign reversal
is manufactured by the group definition. That is a LABEL-mediated construction --
precisely the setting the manuscript distinguishes itself from. So the rule is
tuned to the case the paper is not about and blind to the case it is about.

Against that, the statistic below fires on any heterogeneity of the relation:
slope change, curvature change, sign change. It has no such blind spot.


THE STATISTIC, AND THE TRAP IT AVOIDS
=====================================

The naive test is: for each bin of coordinate k, is

        P(y = +1 | Phi_k in bin b, g = 0)  ==  P(y = +1 | Phi_k in bin b, g = 1) ?

That test is WRONG here, and it fails in a way that would have produced a
confident false positive on both of our datasets. If the two groups simply have
different class balances -- P(y | g=0) != P(y | g=1), which is true on Waterbirds
and enormously true on CelebA, where P(blond | female) ~ 0.24 and
P(blond | male) ~ 0.02 -- then by Bayes' rule the within-bin probabilities differ
for EVERY coordinate, including perfectly causal ones. The test would reject
everything and hand back an empty r-block.

A base-rate difference is not spuriousness. Under Definition 3.3 spuriousness is
A != B, a statement about the r -> s relation, not about how many positives each
group happens to contain.

So the null is not "no group difference" but "a group difference that does not
depend on where you are along coordinate k". On the log-odds scale, a pure
base-rate shift is a CONSTANT offset between the two groups; a genuine change in
the relation makes that offset vary from bin to bin. Writing

        delta_b := logit P(y=+1 | bin b, g=0) - logit P(y=+1 | bin b, g=1)
                 = log of the group odds ratio inside bin b,

the two hypotheses are

        H_0 (invariant / causal)  :  delta_b is CONSTANT in b
        H_1 (spurious)            :  delta_b VARIES with b

and only H_1 is evidence of A != B. Testing constancy of a stratum-wise odds
ratio is a solved problem in categorical statistics: it is the Breslow-Day test
for homogeneity of the odds ratio, with the Mantel-Haenszel estimator supplying
the common odds ratio under H_0. That is what is implemented below.

Note what this buys, stated plainly so nobody over-claims later: the base-rate
confound is removed BY CONSTRUCTION, not by an assumption that it is small.


WHAT THIS RULE CANNOT DO
========================

A per-coordinate test cannot cleanly target A != B, and pretending otherwise
would be dishonest. A != B is a statement about a MAP BETWEEN BLOCKS -- how the
r-block drives coordinate k -- and testing it properly needs the r-block, which
is what we are trying to find. That is circular.

What this rule targets is the weaker, well-defined property: coordinates whose
relation to y is group-dependent after base-rate effects are removed. Two
consequences to keep in view:

  * It is NECESSARY but not SUFFICIENT for Definition 3.3. Group-dependent label
    noise, or a genuinely label-mediated shortcut feature s' (independent of r,
    predictive of y in a group-dependent way), also produce heterogeneity.
  * Marginal invariance is not joint invariance. A coordinate can pass the test
    one at a time and fail jointly. Full ICP intersects over SUBSETS, which is
    2^d tests and needs k^|S| bins to condition on |S| coordinates -- empty bins
    by |S| ~ 5, which is the known reason ICP does not scale past 10-15
    variables. At d = 2048 or 8192 it is not an option.

Both limits are shared with the sign-flip rule, which is also marginal and also
only necessary, so this does not make the rule worse than what is already in the
pipeline. It is stated here so it gets stated in the paper.


BINNING
=======

Quantile bins, not equal-width bins. Equal-width binning of SAE activations puts
almost every sample in the first bin -- the activations are non-negative and
heavy-tailed -- and leaves the rest empty. Quantile bins equalise counts by
construction, which is what the asymptotics need.

The ReLU zero atom is handled separately. An SAE coordinate is exactly zero for
most samples (the reference run has L0 = 419 active of 8192), so a large point
mass sits at 0 and quantile boundaries collapse onto it. Bin 0 is therefore
defined as {Phi_k == 0} and the quantile bins are computed on the NONZERO part
only. On a dense basis (raw Phi) the zero atom is empty and this reduces to plain
quantile binning, so one code path serves both.

Bins where either group has fewer than `min_cell` samples are dropped: a 2x2
table with an empty row carries no information about delta_b and only adds
variance. A coordinate left with fewer than 2 usable bins is undefined -- there
is nothing to test constancy across -- and is filed as `weak`.


EFFECT SIZE, AND WHY IT IS NOT OPTIONAL
=======================================

CelebA train has ~163,000 samples. At that n, a p-value-only rule rejects
essentially every coordinate, because a heterogeneity of 0.01 log-odds is
detectable and meaningless. Classification therefore requires BOTH statistical
significance (BH-FDR) AND an effect size above a threshold. The two effect sizes
are on the same scale (log-odds) and are read the same way:

    strength_k -- how much the log-odds of y swings as coordinate k moves.
                  This is RELEVANCE. A coordinate with strength below tau_rel
                  says nothing about y in either group and is `weak`.

    het_k      -- how much the GROUP EFFECT (delta_b) swings as coordinate k
                  moves. This is the heterogeneity that H_1 asserts.

    r-type  if  strength_k >= tau_rel  and not (het_k >= tau_het and rejected)
    s-type  if  strength_k >= tau_rel  and      het_k >= tau_het and rejected
    weak    otherwise

`weak` exists in the other two rules too (the reference run put 7401 of 8192
coordinates there), so the three rules produce comparable three-way splits and
`agreement()` in identify_rs.py can be pointed at this one unchanged.

Haldane-Anscombe: 0.5 is added to all four cells of every 2x2 table before any
log is taken. Without it a single empty cell sends delta_b to +/- infinity and
takes the coordinate's statistic with it.


NULL DISTRIBUTION
=================

Default is the asymptotic Breslow-Day null, chi-square on (n_usable_bins - 1)
degrees of freedom, which is standard and costs nothing. Because "standard" is
not the same as "calibrated on our data", `--null bootstrap` re-derives the
p-value for a random subset of coordinates by conditional resampling: hold every
2x2 table's margins fixed, resample each bin's count from Fisher's noncentral
hypergeometric at the fitted common odds ratio, recompute the statistic. That is
the exact conditional null under H_0. It is slow, which is why it runs on a
subset -- its job is to audit the asymptotic p-values, not to replace them.

`validate_invariance.py` checks the calibration end to end against known ground
truth, including the base-rate-difference case described above. Run it before
trusting any number this file prints.

Examples
--------
    # split a cached bundle, both bases
    python invariance.py --bundle features_waterbirds_erm_rn50_test.npz
    python invariance.py --bundle features_celeba_clip_train.npz --tau-het 0.30

    # audit the asymptotic p-values on 200 random coordinates
    python invariance.py --bundle features_waterbirds_erm_rn50_test.npz \
        --null bootstrap --n-boot 2000 --boot-subset 200
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def bin_coordinate(
    x: np.ndarray,
    n_bins: int = 8,
    zero_atom: bool = True,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Assign each sample to a bin of coordinate `x`. Returns int codes, -1 = unusable.

    Parameters
    ----------
    x : (N,)
        One column of Phi.
    n_bins : int
        Number of quantile bins for the non-atom part. The zero atom, when
        present and non-empty, is an ADDITIONAL bin on top of these.
    zero_atom : bool
        Treat {x == 0} as its own bin. True is right for SAE activations (ReLU
        output, mostly exact zeros) and harmless on a dense basis, where the
        atom is empty and the function reduces to plain quantile binning.
    zero_tol : float
        Tolerance for "exactly zero".

    Returns
    -------
    codes : (N,) int
        Bin index in 0..K-1, or -1 for samples that could not be binned (only
        possible when the non-atom part is too small to split).
    """
    x = np.asarray(x, dtype=np.float64)
    codes = np.full(x.shape[0], -1, dtype=np.int64)

    if zero_atom:
        is_zero = np.abs(x) <= zero_tol
    else:
        is_zero = np.zeros(x.shape[0], dtype=bool)

    n_zero = int(is_zero.sum())
    nz = np.flatnonzero(~is_zero)
    offset = 0
    if n_zero > 0:
        codes[is_zero] = 0
        offset = 1

    if nz.size == 0:
        # Constant-zero coordinate. Everything is in the atom; there is nothing
        # to test a relation against, and the caller will drop it for having
        # fewer than 2 usable bins.
        return codes

    xv = x[nz]
    # Quantile edges on the non-atom part. `np.unique` collapses duplicate
    # edges, which is what happens when the coordinate has further point masses
    # beyond zero -- the effective bin count drops rather than producing empty
    # bins, which is the behaviour we want.
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(xv, qs))
    if edges.size < 2:
        # Constant on its support.
        codes[nz] = offset
        return codes

    # right=False with the last edge nudged so the maximum lands in the last bin
    inner = edges[1:-1]
    codes[nz] = np.searchsorted(inner, xv, side="right") + offset
    return codes


# --------------------------------------------------------------------------
# 2x2 tables per bin
# --------------------------------------------------------------------------


def stratified_tables(
    codes: np.ndarray,
    y_pos: np.ndarray,
    g: np.ndarray,
    min_cell: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one 2x2 table per bin and drop the uninformative ones.

    Layout of each table, in the variable names used throughout this module:

                     y = +1     y = -1     row total
        g = 0          a          b           n0
        g = 1          c          d           n1
        col total     m1         m0            N

    Parameters
    ----------
    codes : (N,) int
        Bin assignment from `bin_coordinate`; -1 means unusable.
    y_pos : (N,) bool
        True where y == +1.
    g : (N,) int in {0, 1}
    min_cell : int
        A bin is kept only if ALL FOUR cells hold at least this many samples.

        Guarding the row totals alone is not enough, and getting this wrong cost
        a full debugging cycle, so the reason is recorded here. In the separable
        regime the manuscript studies, the extreme bins of a coordinate are
        SATURATED: every sample in the top bin has y = +1, giving tables like

            a = 2294, b = 0,   c = 206, d = 0.

        The odds ratio of such a table is 0/0. The Haldane correction below makes
        it finite by adding 0.5 to each cell, but the value it produces,
        log(a*d / b*c) ~ log(4 a c), then GROWS LIKE log(n) -- a number
        manufactured entirely by the correction, carrying no information about
        the group difference. Two things follow, both observed:

          * `het` is inflated for every coordinate whose tails saturate, which
            is most of the relevant ones;
          * the Breslow-Day statistic becomes erratic -- 6.3 to 57.6 across
            coordinates that are identical by construction -- because the
            saturated bins contribute large, arbitrary residuals.

        The symptom at the top level was F1 that did not increase with n. If that
        ever reappears, look here first.

    Returns
    -------
    a, b, c, d : (K_used,) float64
        Counts for the retained bins, in the layout above.
    """
    keep = codes >= 0
    cc, yy, gg = codes[keep], y_pos[keep], g[keep]
    if cc.size == 0:
        z = np.zeros(0)
        return z, z, z, z

    K = int(cc.max()) + 1
    idx = cc + K * (gg.astype(np.int64) * 2 + yy.astype(np.int64))
    cnt = np.bincount(idx, minlength=4 * K).astype(np.float64)
    # ordering above: (g, y) -> block index  0:(0,-1) 1:(0,+1) 2:(1,-1) 3:(1,+1)
    b_ = cnt[0 * K:1 * K]     # g=0, y=-1
    a_ = cnt[1 * K:2 * K]     # g=0, y=+1
    d_ = cnt[2 * K:3 * K]     # g=1, y=-1
    c_ = cnt[3 * K:4 * K]     # g=1, y=+1

    ok = ((a_ >= min_cell) & (b_ >= min_cell)
          & (c_ >= min_cell) & (d_ >= min_cell))
    return a_[ok], b_[ok], c_[ok], d_[ok]


# --------------------------------------------------------------------------
# Mantel-Haenszel common odds ratio and the Breslow-Day statistic
# --------------------------------------------------------------------------


def mantel_haenszel_or(a, b, c, d) -> float:
    """Common odds ratio under H_0 (delta_b constant across bins).

    psi_MH = sum_b (a_b d_b / N_b) / sum_b (b_b c_b / N_b)

    This is the standard stratified estimator. It is preferred over pooling the
    tables (which would reintroduce exactly the confounding the stratification
    removes) and over averaging the per-bin log odds ratios (which is undefined
    whenever a cell is empty).
    """
    N = a + b + c + d
    num = float(np.sum(a * d / N))
    den = float(np.sum(b * c / N))
    if den <= 0.0:
        return np.inf if num > 0 else 1.0
    return num / den


def _expected_a(psi: float, n0, m1, N):
    """Expected a_b given the margins and a common odds ratio psi.

    Solves  psi = [A (N - n0 - m1 + A)] / [(n0 - A)(m1 - A)]  for A, which
    rearranges to the quadratic

        (psi - 1) A^2  -  [psi (n0 + m1) + (N - n0 - m1)] A  +  psi n0 m1  =  0.

    The admissible root is the one inside [max(0, n0 + m1 - N), min(n0, m1)],
    the range in which all four cells of the table stay non-negative. Both roots
    are computed and the admissible one selected, rather than relying on a sign
    convention that is only correct for psi > 1.
    """
    n0 = np.asarray(n0, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)

    lo = np.maximum(0.0, n0 + m1 - N)
    hi = np.minimum(n0, m1)

    if not np.isfinite(psi) or psi <= 0:
        # Degenerate common odds ratio: fall back to the independence expectation.
        return np.clip(n0 * m1 / N, lo, hi)

    A_ind = n0 * m1 / N
    if abs(psi - 1.0) < 1e-10:
        return np.clip(A_ind, lo, hi)

    qa = psi - 1.0
    qb = -(psi * (n0 + m1) + (N - n0 - m1))
    qc = psi * n0 * m1
    disc = np.maximum(qb * qb - 4.0 * qa * qc, 0.0)
    sq = np.sqrt(disc)
    r1 = (-qb + sq) / (2.0 * qa)
    r2 = (-qb - sq) / (2.0 * qa)

    eps = 1e-9
    ok1 = (r1 >= lo - eps) & (r1 <= hi + eps)
    A = np.where(ok1, r1, r2)
    return np.clip(A, lo, hi)


def breslow_day(a, b, c, d, psi: float | None = None) -> tuple[float, int]:
    """Breslow-Day statistic for homogeneity of the odds ratio across bins.

        BD = sum_b (a_b - A_b)^2 / V_b

    with A_b the expected count under the common odds ratio psi and

        V_b = [ 1/A_b + 1/(n0_b - A_b) + 1/(m1_b - A_b) + 1/(N_b - n0_b - m1_b + A_b) ]^{-1}

    the corresponding hypergeometric variance. Under H_0 (delta_b constant) this
    is asymptotically chi-square on (K - 1) degrees of freedom, one lost to
    estimating psi.

    Returns (statistic, degrees_of_freedom).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)
    K = a.size
    if K < 2:
        return float("nan"), 0

    n0 = a + b
    m1 = a + c
    N = a + b + c + d
    if psi is None:
        psi = mantel_haenszel_or(a, b, c, d)

    A = _expected_a(psi, n0, m1, N)
    # The four cells implied by A and the margins.
    cell = np.stack([A, n0 - A, m1 - A, N - n0 - m1 + A])
    cell = np.maximum(cell, 1e-9)
    V = 1.0 / np.sum(1.0 / cell, axis=0)
    stat = float(np.sum((a - A) ** 2 / np.maximum(V, 1e-12)))
    return stat, K - 1


# --------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------


def _log_odds(p_num, p_den):
    """log(p_num / p_den), guarded. Inputs are already Haldane-corrected counts."""
    return np.log(np.maximum(p_num, 1e-12)) - np.log(np.maximum(p_den, 1e-12))


def effect_sizes(a, b, c, d) -> tuple[float, float, np.ndarray]:
    """Return (strength, het, delta) -- both effect sizes and the per-bin log ORs.

    Haldane-Anscombe: 0.5 is added to every cell first, so that a single empty
    cell produces a finite, shrunken value rather than an infinity that would
    dominate whatever statistic it enters.

    strength -- weighted standard deviation, across bins, of the POOLED log-odds
                of y. Answers "how much does the log-odds of y move as this
                coordinate moves", i.e. is the coordinate relevant to y at all.

    het      -- weighted standard deviation, across bins, of delta_b, the
                per-bin log odds ratio between the two groups. Answers "how much
                does the GROUP EFFECT move as this coordinate moves", which is
                the alternative hypothesis H_1.

    Weights are bin sample sizes, so a bin holding 2% of the data cannot drive
    either number on its own.
    """
    a, b = a + 0.5, b + 0.5
    c, d = c + 0.5, d + 0.5
    N = a + b + c + d
    w = N / N.sum()

    # pooled log-odds of y=+1 within each bin
    lam = _log_odds(a + c, b + d)
    lam_bar = float(np.sum(w * lam))
    strength = float(np.sqrt(np.sum(w * (lam - lam_bar) ** 2)))

    # per-bin log odds ratio between groups
    delta = _log_odds(a * d, b * c)
    delta_bar = float(np.sum(w * delta))
    het = float(np.sqrt(np.sum(w * (delta - delta_bar) ** 2)))

    return strength, het, delta


# --------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns a boolean rejection mask at level q.

    With d in the thousands, an uncorrected 0.05 threshold rejects hundreds of
    coordinates by chance alone. BH controls the expected proportion of false
    discoveries among the rejected set, which is the relevant guarantee when the
    rejected set IS the deliverable (it becomes Phi_s).

    NaN p-values -- coordinates with too few usable bins to test -- are never
    rejected and are excluded from the ranking rather than being treated as 1.0,
    which would inflate the denominator and cost power.
    """
    p = np.asarray(p, dtype=np.float64)
    rej = np.zeros(p.shape, dtype=bool)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return rej
    pv = p[ok]
    order = np.argsort(pv)
    m = pv.size
    thresh = q * (np.arange(1, m + 1) / m)
    passed = pv[order] <= thresh
    if not passed.any():
        return rej
    kmax = int(np.flatnonzero(passed).max())
    rej[ok[order[: kmax + 1]]] = True
    return rej


# --------------------------------------------------------------------------
# The test, vectorised over coordinates
# --------------------------------------------------------------------------


def invariance_test(
    phi: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_bins: int = 8,
    min_cell: int = 10,
    zero_atom: bool = True,
) -> dict:
    """Run the Breslow-Day heterogeneity test on every coordinate of `phi`.

    Parameters
    ----------
    phi : (N, d)
        The representation, raw or SAE. Standardisation is irrelevant here --
        every statistic is computed from quantile bins and from counts, both
        invariant under any monotone rescaling of a coordinate. This is a real
        advantage over the weight-based alternatives, whose output depends on
        the scaling convention.
    y : (N,)
        Label in the +/-1 convention.
    g : (N,)
        Group index in {0, 1}.

    Returns a dict of per-coordinate arrays: `stat`, `dof`, `p`, `strength`,
    `het`, `psi`, `n_bins_used`.
    """
    from scipy.stats import chi2

    phi = np.asarray(phi, dtype=np.float64)
    N, d = phi.shape
    y_pos = np.asarray(y).ravel() > 0
    g = np.asarray(g).ravel().astype(int)

    stat = np.full(d, np.nan)
    dof = np.zeros(d, dtype=int)
    strength = np.zeros(d)
    het = np.zeros(d)
    psi = np.full(d, np.nan)
    nb_used = np.zeros(d, dtype=int)

    for k in range(d):
        codes = bin_coordinate(phi[:, k], n_bins=n_bins, zero_atom=zero_atom)
        a, b, c, dd = stratified_tables(codes, y_pos, g, min_cell=min_cell)
        nb_used[k] = a.size
        if a.size < 2:
            # Fewer than two usable bins: there is nothing to test constancy
            # ACROSS. Left as NaN, never rejected, and filed `weak` downstream.
            continue
        p_mh = mantel_haenszel_or(a, b, c, dd)
        psi[k] = p_mh
        s, df = breslow_day(a, b, c, dd, psi=p_mh)
        stat[k], dof[k] = s, df
        strength[k], het[k], _ = effect_sizes(a, b, c, dd)

    p = np.full(d, np.nan)
    ok = np.isfinite(stat) & (dof > 0)
    p[ok] = chi2.sf(stat[ok], dof[ok])

    return {
        "stat": stat, "dof": dof, "p": p,
        "strength": strength, "het": het, "psi": psi,
        "n_bins_used": nb_used,
        "n_bins": n_bins, "min_cell": min_cell, "zero_atom": bool(zero_atom),
    }


def bootstrap_pvalues(
    phi: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    cols: np.ndarray,
    n_boot: int = 2000,
    n_bins: int = 8,
    min_cell: int = 10,
    zero_atom: bool = True,
    seed: int = 0,
) -> dict:
    """Exact conditional p-values for the coordinates in `cols`. Slow, for auditing.

    The asymptotic chi-square null in `invariance_test` is standard, but standard
    is not the same as calibrated on our tables -- quantile bins keep the row
    margins even, yet a coordinate can still put most of its mass in the zero
    atom and leave thin cells behind. This re-derives the p-value without the
    asymptotics.

    Method: hold every 2x2 table's margins fixed and resample its top-left cell
    from Fisher's noncentral hypergeometric distribution at the fitted common
    odds ratio psi_MH. Fixed margins plus a common odds ratio IS H_0, so the
    resulting statistics are draws from the exact conditional null. The p-value
    is the fraction of draws at least as extreme as the observed statistic, with
    the usual +1/+1 correction so it can never be exactly zero.

    Cost is roughly `n_boot * K` noncentral hypergeometric draws per coordinate,
    which is why `cols` is a subset. Compare the output against the asymptotic
    p-values on the same coordinates; if they agree, the asymptotic numbers for
    the other coordinates can be trusted.
    """
    from scipy.stats import nchypergeom_fisher

    rng = np.random.default_rng(seed)
    y_pos = np.asarray(y).ravel() > 0
    g = np.asarray(g).ravel().astype(int)

    out_p, out_obs, out_cols = [], [], []
    for k in np.asarray(cols, dtype=int):
        codes = bin_coordinate(phi[:, k], n_bins=n_bins, zero_atom=zero_atom)
        a, b, c, d = stratified_tables(codes, y_pos, g, min_cell=min_cell)
        if a.size < 2:
            continue
        psi = mantel_haenszel_or(a, b, c, d)
        obs, _ = breslow_day(a, b, c, d, psi=psi)
        if not np.isfinite(obs):
            continue

        n0 = a + b
        m1 = a + c
        Nb = a + b + c + d
        odds = psi if np.isfinite(psi) and psi > 0 else 1.0

        # (n_boot, K) resampled top-left cells, margins fixed.
        a_star = np.empty((n_boot, a.size))
        for j in range(a.size):
            a_star[:, j] = nchypergeom_fisher.rvs(
                M=int(Nb[j]), n=int(m1[j]), N=int(n0[j]),
                odds=float(odds), size=n_boot, random_state=rng,
            )
        b_star = n0[None, :] - a_star
        c_star = m1[None, :] - a_star
        d_star = Nb[None, :] - n0[None, :] - m1[None, :] + a_star

        null = np.empty(n_boot)
        for i in range(n_boot):
            null[i], _ = breslow_day(
                a_star[i], b_star[i], c_star[i], d_star[i], psi=psi
            )
        pb = (1.0 + np.sum(null >= obs)) / (1.0 + n_boot)
        out_cols.append(int(k)); out_obs.append(float(obs)); out_p.append(float(pb))

    return {
        "cols": np.asarray(out_cols, dtype=int),
        "stat": np.asarray(out_obs),
        "p_boot": np.asarray(out_p),
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def invariance_identify(
    phi: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    tau_rel: float = 0.20,
    tau_het: float = 0.20,
    q: float = 0.05,
    n_bins: int = 8,
    min_cell: int = 10,
    zero_atom: bool = True,
    precomputed: dict | None = None,
) -> dict:
    """Split the columns of phi into r-type, s-type and weak.

        r-type  if  strength >= tau_rel  and not (het >= tau_het and BH-rejected)
        s-type  if  strength >= tau_rel  and      het >= tau_het and BH-rejected
        weak    otherwise

    BOTH conditions are required for s-type, and the effect-size one is doing
    real work: at CelebA's ~163k samples a significance-only rule rejects nearly
    every coordinate and returns an empty r-block. Both thresholds are in
    log-odds, so tau = 0.20 means "the quantity swings by at least 0.2 log-odds
    (about a factor 1.22 in odds) across the range of the coordinate".

    Returns the same contract as `two_concept_identify` and `sign_flip_identify`
    in identify_rs.py -- idx_r, idx_s, n_r, n_s, n_weak -- so `agreement()` can
    be pointed at it with no changes.
    """
    res = precomputed if precomputed is not None else invariance_test(
        phi, y, g, n_bins=n_bins, min_cell=min_cell, zero_atom=zero_atom
    )
    rejected = bh_fdr(res["p"], q=q)
    relevant = res["strength"] >= tau_rel
    hetero = (res["het"] >= tau_het) & rejected

    idx_s = np.flatnonzero(relevant & hetero)
    idx_r = np.flatnonzero(relevant & ~hetero)
    d = phi.shape[1]

    out = dict(res)
    out.update({
        "idx_r": idx_r, "idx_s": idx_s,
        "n_r": int(idx_r.size), "n_s": int(idx_s.size),
        "n_weak": int(d - idx_r.size - idx_s.size),
        "rejected": rejected, "n_rejected": int(rejected.sum()),
        "tau_rel": tau_rel, "tau_het": tau_het, "q": q,
        "rule": "invariance",
    })
    return out


def tau_sensitivity(
    res: dict, d: int,
    taus=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
) -> list[dict]:
    """How n_r and n_s move with the two thresholds. Report this, always.

    Both thresholds are conventions, and a split that only exists at one setting
    of a convention is not a finding. The table this produces is what lets a
    reader see whether the split is a plateau or a knife edge.
    """
    rows = []
    for t in taus:
        relevant = res["strength"] >= t
        hetero = (res["het"] >= t) & res["rejected"]
        n_s = int(np.sum(relevant & hetero))
        n_r = int(np.sum(relevant & ~hetero))
        rows.append({"tau": float(t), "n_r": n_r, "n_s": n_s,
                     "n_weak": int(d - n_r - n_s)})
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def to_markdown(res: dict, name: str) -> str:
    d = res["strength"].size
    L = [f"\n### Invariance rule (Breslow-Day)  {name}",
         "",
         f"d = {d}, bins = {res['n_bins']} (+ zero atom = {res['zero_atom']}), "
         f"min_cell = {res['min_cell']}, FDR q = {res['q']}",
         f"tau_rel = {res['tau_rel']}, tau_het = {res['tau_het']}",
         "",
         f"- BH-rejected (heterogeneous relation): **{res['n_rejected']}** of {d}",
         f"- n_r = **{res['n_r']}**, n_s = **{res['n_s']}**, n_weak = {res['n_weak']}",
         f"- coordinates with < 2 usable bins (untestable): "
         f"{int(np.sum(res['n_bins_used'] < 2))}",
         ""]

    st, he = res["strength"], res["het"]
    fin = np.isfinite(st) & np.isfinite(he)
    if fin.any():
        L += ["| quantile | strength (log-odds) | het (log-odds) |",
              "|---|---|---|"]
        for qq in (0.5, 0.75, 0.9, 0.99, 1.0):
            L.append(f"| {qq:.2f} | {np.quantile(st[fin], qq):.3f} "
                     f"| {np.quantile(he[fin], qq):.3f} |")
        L.append("")

    rows = tau_sensitivity(res, d)
    L += ["Threshold sensitivity (tau_rel = tau_het = tau):", "",
          "| tau | n_r | n_s | n_weak |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['tau']:.2f} | {r['n_r']} | {r['n_s']} | {r['n_weak']} |")
    L.append("")
    return "\n".join(L)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def main() -> None:
    from common import FeatureBundle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True,
                    help="cached FeatureBundle .npz from extract_features.py")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--tau-rel", type=float, default=0.20)
    ap.add_argument("--tau-het", type=float, default=0.20)
    ap.add_argument("--q", type=float, default=0.05, help="BH-FDR level")
    ap.add_argument("--no-zero-atom", action="store_true",
                    help="disable the ReLU zero-atom bin (dense bases only)")
    ap.add_argument("--null", choices=["asymptotic", "bootstrap"],
                    default="asymptotic")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-subset", type=int, default=200,
                    help="how many coordinates to audit with the exact null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default=None,
                    help="output name; defaults to the bundle stem")
    args = ap.parse_args()

    fb = FeatureBundle.load(args.bundle)
    tag = args.tag or os.path.splitext(os.path.basename(args.bundle))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    res = invariance_identify(
        fb.phi, fb.y, fb.g,
        tau_rel=args.tau_rel, tau_het=args.tau_het, q=args.q,
        n_bins=args.bins, min_cell=args.min_cell,
        zero_atom=not args.no_zero_atom,
    )

    md = to_markdown(res, tag)

    if args.null == "bootstrap":
        rng = np.random.default_rng(args.seed)
        testable = np.flatnonzero(np.isfinite(res["p"]))
        pick = rng.choice(testable, size=min(args.boot_subset, testable.size),
                          replace=False)
        bs = bootstrap_pvalues(
            fb.phi, fb.y, fb.g, pick, n_boot=args.n_boot,
            n_bins=args.bins, min_cell=args.min_cell,
            zero_atom=not args.no_zero_atom, seed=args.seed,
        )
        pa = res["p"][bs["cols"]]
        agree = float(np.mean((pa < args.q) == (bs["p_boot"] < args.q)))
        md += (f"\nExact-null audit on {bs['cols'].size} random coordinates "
               f"({args.n_boot} conditional resamples each):\n\n"
               f"- decisions agreeing with the asymptotic null at q={args.q}: "
               f"**{agree:.3f}**\n"
               f"- median |p_asymptotic - p_exact| = "
               f"{float(np.median(np.abs(pa - bs['p_boot']))):.4f}\n")
        res["bootstrap"] = bs

    with open(os.path.join(args.out_dir, f"invariance_{tag}.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out_dir, f"invariance_{tag}.json"), "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    print(md)


if __name__ == "__main__":
    main()
