"""Session 6: screen pre-existing image properties for a group that avoids the margin.

THE FACT THIS SCREEN IS BUILT ON (session 5, 23 Sept 2026)
==========================================================
`w_hat`, the minimum-norm separator, is fitted on `(phi, y)` ONLY -- the group
variable never enters the fit (`group_margins._svc_ladder` takes `X, y`). So when
only the group variable changes, the separator does not move, and the per-group
margin ratio is pure bookkeeping over ONE fixed vector of point margins

    u_i = y_i (w_hat . x_i) / ||w_hat||.

Call S the MARGIN SET: the points whose `u_i` is within a relative `TIE_TOL`
(1e-3, the same tolerance `group_margins.py` uses to call a tie) of the global
minimum `u*`. Then for a candidate group G:

    gamma(G) / gamma(rest) > 1 + TIE_TOL    <=>    G contains no point of S.

So there is no optimisation to run per candidate: one SVM fit, then every
candidate is a lookup. On the dinov2 train bundle |S| = 421 of n = 4,795.

WHAT IS SCREENED -- THE FAMILY, FIXED BEFORE THE RUN
===================================================
Every candidate is a property of the PHOTOGRAPH that exists before any training,
taken from CUB-200-2011's own annotations (joined by `cub_meta.py`). Nothing here
looks at the margins to define a group -- that would be session 5's oracle, which
is circular. The family, in full:

  A. the 312 CUB attributes, each on BOTH sides:
       G = {images annotated is_present = 1}   and   G = {is_present = 0}
     (as annotated, any certainty level)                        624 candidates
  B. four continuous difficulty proxies, each tail tested separately, at
     q in {0.05, 0.10, 0.25}:                                    24 candidates
       bird_frac              bird pixels / image pixels (CUB masks, session 5)
       bbox_area_frac         bounding-box area / image area
       n_visible_parts        how many of CUB's 15 parts are marked visible
       frac_attr_not_visible  share of the 312 attributes marked "not visible"
     "top q" is G = {value >= the (1-q) quantile}; "bottom q" is G = {value <=
     the q quantile}. Ties at the cut go in, so a tail can exceed q.
  C. bird_in_frame (bounding box at least 2 px from every edge), both sides.
                                                                  2 candidates
Species is NOT in the family: Waterbirds' label is a function of species, so any
species group is a label proxy. ~650 candidates in all; the exact count is in the
report.

WHICH CANDIDATES ARE ELIGIBLE (filters that use only y and G, never the margins,
so applying them before the test does not bias it)
-------------------------------------------------------------------------------
  * size: at least `--min-frac` of n in G AND in the rest (default 1%);
  * both labels present in G, at least 10 of each;
  * not a label proxy: |AUC(1_G -> y) - 0.5| <= 0.10 (session 5's rule) AND
    |phi(1_G, y)| <= 0.10. AUC alone is blind to a SMALL label-pure group (100
    waterbirds and nothing else has AUC 0.545); the phi coefficient is not.
Leakage against `place` and the old group is REPORTED, not filtered.

THE TEST, IN WORDS
==================
Null hypothesis for a candidate G of size m, with m1 waterbirds and m0 landbirds:
G is a random subset of the images, drawn with that label composition, and
unrelated to margin. Statistic: t = min over G of u_i (G's own hard margin). The
p-value is the exact probability that a random such subset has a minimum at
least as large:

    p = [C(N1(t), m1) / C(n1, m1)] * [C(N0(t), m0) / C(n0, m0)]

where n_c is the number of label-c points and N_c(t) how many of them have
u >= t. No simulation is needed; `--self-test` checks the formula against one.
Stratifying by label matters because the two labels need not sit at the margin
equally often, and a group's label mix would otherwise masquerade as an effect.
A group containing a margin point has t = u* and p = 1.

Multiple testing: Holm's step-down over all eligible candidates, family-wise
alpha 0.05. A HIT is a candidate with a non-tie ratio (G the larger-margin
group) whose Holm-adjusted p is <= 0.05.

The report also gives, for every eligible candidate, a GRADED statistic that is
informative even when nothing is a hit: k = |G ∩ S| against its expectation, and
the exact probability of seeing k or fewer margin points (label-stratified
hypergeometric). It answers "how close did anything get".

And it gives the DETECTABILITY FLOOR: the smallest m at which a margin-free group
survives Holm by the UNSTRATIFIED bound C(n-|S|, m)/C(n, m). It is exact for a group
whose label mix matches the data's, and approximate otherwise (a group made mostly
of the label that rarely sits in S gets a weaker stratified p). Below it, a
margin-free group can arise by chance and cannot, in general, be told from luck.

CONFIRMATION WITH THE UNMODIFIED INSTRUMENT
===========================================
For every hit, and always for the three best-ranked eligible candidates (so the
check runs even when there is no hit), a bundle with g = 1[G] is written with
`regroup.write_bundle` (no row dropped, phi and y bit-identical). The runner then
measures those bundles with the UNMODIFIED `group_margins.py`, and
`--compare` checks that its ratios equal the screen's. They must, because the
separator is group-blind; a disagreement means the screen is wrong.

REPLICATION ON AN INDEPENDENT SAMPLE
====================================
A train-split hit is a lead, not a result: with ~650 candidates, Holm holds the
family-wise false-positive rate at 5%, not at zero. The runner therefore re-runs
the same screen on the TEST split (a disjoint set of photographs, separator
re-fitted there) and `--replicate` asks, for each train hit and nothing else,
whether the same property avoids that sample's margin set too. Because the hits
are fixed in advance of looking at the test split, only they are tested, with
Holm over their number. The test split's background composition differs from
train's (it is group-balanced), which is irrelevant to the question asked: does
this property of the photograph keep its images off the boundary.

Usage
-----
    python sv_screen.py --self-test
    python sv_screen.py --bundle features_v4_waterbirds_dinov2_train.npz \\
                        --meta results/v6_cub_meta.npz
    python sv_screen.py --compare results/v6_confirm_margins.json
    python sv_screen.py --bundle <test bundle> --split test --tag v6_sv_screen_test \
                        --no-confirm
    python sv_screen.py --replicate results/v6_sv_screen_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

from progress import pbar

ALPHA = 0.05
QS = (0.05, 0.10, 0.25)
PROXIES = ("bird_frac", "bbox_area_frac", "n_visible_parts", "frac_attr_not_visible")
MIN_PER_LABEL = 10
MAX_AUC_DEV = 0.10
MAX_ABS_PHI = 0.10
N_CONFIRM_ALWAYS = 3
MAX_CONFIRM_HITS = 5


# ----------------------------------------------------------------------------
# Exact combinatorics
# ----------------------------------------------------------------------------
def _lcomb(a, b):
    from scipy.special import gammaln
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def p_min_stratified(u, y01, members) -> float:
    """P(a random label-matched subset has min u >= the group's min u). Exact."""
    t = float(u[members].min())
    logp = 0.0
    for c in (0, 1):
        in_c = y01 == c
        m_c = int((members & in_c).sum())
        if m_c == 0:
            continue
        n_c = int(in_c.sum())
        N_c = int((u[in_c] >= t).sum())
        if N_c < m_c:
            return 0.0
        logp += float(_lcomb(N_c, m_c) - _lcomb(n_c, m_c))
    return float(np.exp(logp))


def p_depletion_stratified(in_S, y01, members) -> tuple[int, float, float]:
    """(k, expected k, P(K <= k)) with K the margin-point count of a random
    label-matched subset. Sum of two independent hypergeometrics, exact."""
    from scipy.stats import hypergeom
    pmfs, exp_k = [], 0.0
    for c in (0, 1):
        in_c = y01 == c
        n_c = int(in_c.sum())
        s_c = int((in_S & in_c).sum())
        m_c = int((members & in_c).sum())
        ks = np.arange(0, min(s_c, m_c) + 1)
        pmfs.append(hypergeom.pmf(ks, n_c, s_c, m_c))
        exp_k += m_c * s_c / max(n_c, 1)
    k = int((members & in_S).sum())
    conv = np.convolve(pmfs[0], pmfs[1])
    return k, float(exp_k), float(min(1.0, conv[: k + 1].sum()))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    F = p.size
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(F)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (F - rank) * p[i]))
        adj[i] = running
    return adj


def detectability_floor(n: int, s: int, n_tests: int, alpha: float = ALPHA) -> int:
    """Smallest m with C(n-s, m)/C(n, m) <= alpha / n_tests (unstratified bound).

    Unstratified: exact for a group with the data's label mix, approximate for a
    group whose mix is skewed toward the label that rarely sits in S.
    """
    thr = np.log(alpha / max(n_tests, 1))
    for m in range(1, n - s + 1):
        if float(_lcomb(n - s, m) - _lcomb(n, m)) <= thr:
            return m
    return -1


# ----------------------------------------------------------------------------
# Leakage of a binary group indicator into a binary variable
# ----------------------------------------------------------------------------
def binary_leakage(ind: np.ndarray, lab: np.ndarray) -> dict:
    ind = np.asarray(ind, dtype=float)
    lab = np.asarray(lab).astype(int)
    tpr = float(ind[lab == 1].mean()) if (lab == 1).any() else float("nan")
    fpr = float(ind[lab == 0].mean()) if (lab == 0).any() else float("nan")
    # For a 0/1 score, AUC = 0.5 + (TPR - FPR)/2 exactly (ties count one half).
    auc = 0.5 + 0.5 * (tpr - fpr)
    sd = ind.std() * lab.std()
    phi = float(np.mean((ind - ind.mean()) * (lab - lab.mean())) / sd) if sd > 0 else 0.0
    return {"auc": auc, "phi": phi}


# ----------------------------------------------------------------------------
# The family
# ----------------------------------------------------------------------------
def build_family(meta: dict, rows: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """[(family, name, member mask)] over the selected rows. See module docstring."""
    fam = []
    A = meta["attr_present"][rows]
    names = [str(s) for s in meta["attr_names"]]
    for j in range(A.shape[1]):
        fam.append(("A_attribute", f"{names[j]} = 1", A[:, j] == 1))
        fam.append(("A_attribute", f"{names[j]} = 0", A[:, j] == 0))
    for prox in PROXIES:
        v = np.asarray(meta[prox], dtype=float)[rows]
        for q in QS:
            hi = float(np.quantile(v, 1 - q))
            lo = float(np.quantile(v, q))
            fam.append(("B_proxy_tail", f"{prox} top {q:g} (>= {hi:.4g})", v >= hi))
            fam.append(("B_proxy_tail", f"{prox} bottom {q:g} (<= {lo:.4g})", v <= lo))
    b = np.asarray(meta["bird_in_frame"], dtype=int)[rows]
    fam.append(("C_binary", "bird_in_frame = 1", b == 1))
    fam.append(("C_binary", "bird_in_frame = 0", b == 0))
    return fam


# ----------------------------------------------------------------------------
# The separator and the margins
# ----------------------------------------------------------------------------
def fit_margins(X, y, ladder, max_iter, tol) -> dict:
    """Same solver, same ladder, same selection rule as group_margins.py."""
    from group_margins import _geometric_margins, _svc_ladder
    zero = np.zeros(len(y), dtype=int)
    clean = []
    for r in _svc_ladder(X, y, ladder, max_iter, tol):
        if "w" in r and _geometric_margins(X, y, zero, r["w"])["n_violations"] == 0:
            clean.append(r)
    if not clean:
        raise SystemExit("no C in the ladder separated this bundle with zero violations")
    w = clean[-1]["w"]
    u = (y * (X @ w)) / float(np.linalg.norm(w))
    out = {"u": u, "C_used": float(clean[-1]["C"]), "margin": float(u.min()),
           "converged": bool(clean[-1].get("converged"))}
    if len(clean) >= 2:
        a = float(((y * (X @ clean[-2]["w"])) / np.linalg.norm(clean[-2]["w"])).min())
        out["plateau_rel"] = float(abs(out["margin"] - a) / max(abs(out["margin"]), 1e-300))
    return out


def screen(u, y01, place, g_orig, family, min_frac, tie_tol) -> tuple[list, dict]:
    n = u.size
    ustar = float(u.min())
    in_S = u <= ustar * (1.0 + tie_tol)
    min_n = max(1, int(np.ceil(min_frac * n)))
    recs = []
    for fam, name, mem in pbar(family, desc="candidates", unit="cand"):
        mem = np.asarray(mem, dtype=bool)
        m = int(mem.sum())
        r = {"family": fam, "name": name, "m": m, "eps": m / n,
             "m_y1": int((mem & (y01 == 1)).sum()), "m_y0": int((mem & (y01 == 0)).sum())}
        lk = binary_leakage(mem, y01)
        r.update(auc_y=lk["auc"], phi_y=lk["phi"])
        if place is not None:
            r["auc_place"] = binary_leakage(mem, place)["auc"]
        r["auc_old_g"] = binary_leakage(mem, g_orig)["auc"]
        why = []
        if m < min_n or n - m < min_n:
            why.append(f"size {m} outside [{min_n}, {n - min_n}]")
        if min(r["m_y1"], r["m_y0"]) < MIN_PER_LABEL:
            why.append(f"fewer than {MIN_PER_LABEL} of one label")
        if abs(lk["auc"] - 0.5) > MAX_AUC_DEV:
            why.append(f"label proxy (AUC {lk['auc']:.3f})")
        if abs(lk["phi"]) > MAX_ABS_PHI:
            why.append(f"label proxy (phi {lk['phi']:+.3f})")
        r["eligible"] = not why
        r["why_ineligible"] = "; ".join(why)
        if m == 0 or m == n:
            recs.append(r)
            continue
        mg, mr = float(u[mem].min()), float(u[~mem].min())
        lo = min(mg, mr)
        rel = abs(mg - mr) / max(lo, 1e-300)
        r.update(margin_G=mg, margin_rest=mr, ratio=max(mg, mr) / lo,
                 larger=("tie" if rel <= tie_tol else ("G" if mg > mr else "rest")))
        k, ek, pdep = p_depletion_stratified(in_S, y01, mem)
        r.update(k_S=k, expected_k_S=ek, depletion=(k / ek if ek > 0 else float("nan")),
                 p_depletion=pdep)
        r["p_min"] = p_min_stratified(u, y01, mem)
        qg, qr = float(np.quantile(u[mem], 0.05)), float(np.quantile(u[~mem], 0.05))
        r["q05_ratio_G_over_rest"] = qg / qr if qr > 0 else float("nan")
        recs.append(r)

    elig = [r for r in recs if r["eligible"] and "p_min" in r]
    if elig:
        adj = holm(np.array([r["p_min"] for r in elig]))
        adj_d = holm(np.array([r["p_depletion"] for r in elig]))
        for r, a, ad in zip(elig, adj, adj_d):
            r["p_min_holm"] = float(a)
            r["p_depletion_holm"] = float(ad)
            r["hit"] = bool(r["larger"] == "G" and a <= ALPHA)
    summ = {"n": int(n), "margin": ustar, "tie_tol": tie_tol,
            "S_size": int(in_S.sum()), "S_y1": int((in_S & (y01 == 1)).sum()),
            "S_y0": int((in_S & (y01 == 0)).sum()),
            "family_size": len(recs), "eligible": len(elig), "min_group": min_n,
            "hits": sum(1 for r in elig if r.get("hit")),
            "floor_m": detectability_floor(n, int(in_S.sum()), max(len(elig), 1))}
    return recs, summ


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _row(r) -> str:
    f = lambda v, fmt: (fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "-")
    return (f"| {r['name']} | {r['m']} | {r['m_y1']}/{r['m_y0']} | {f(r.get('ratio'), '{:.4f}')} | "
            f"{r.get('larger', '-')} | {r.get('k_S', '-')} | {f(r.get('expected_k_S'), '{:.1f}')} | "
            f"{f(r.get('p_min'), '{:.2e}')} | {f(r.get('p_min_holm'), '{:.2e}')} | "
            f"{f(r.get('p_depletion'), '{:.2e}')} | {f(r.get('p_depletion_holm'), '{:.2e}')} | "
            f"{f(r.get('auc_y'), '{:.3f}')} | {f(r.get('phi_y'), '{:+.3f}')} | "
            f"{f(r.get('auc_place'), '{:.3f}')} | {f(r.get('auc_old_g'), '{:.3f}')} |")


HEAD = ("| candidate G | m | waterbird/landbird | ratio | larger | margin pts in G | "
        "expected | p (min) | Holm p (min) | p (depletion) | Holm p (depl.) | "
        "AUC y | phi y | AUC place | AUC old g |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def to_markdown(recs, summ, fit, bundle, confirm) -> str:
    elig = [r for r in recs if r["eligible"] and "p_min" in r]
    hits = sorted([r for r in elig if r.get("hit")], key=lambda r: r["p_min"])
    by_dep = sorted(elig, key=lambda r: (r["p_depletion"], -r["m"]))
    fam_counts = {}
    for r in recs:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    L = ["# Session 6 -- does any image property avoid the margin set?", "",
         f"- bundle: `{bundle}`   n = {summ['n']}",
         f"- separator: LinearSVC hinge, no intercept, C = {fit['C_used']:g}, "
         f"margin u* = {summ['margin']:.4f}, plateau {fit.get('plateau_rel', float('nan')):.1e}",
         f"- **margin set S** (u within {summ['tie_tol']:g} relative of u*): "
         f"**{summ['S_size']}** points ({summ['S_y1']} waterbirds, {summ['S_y0']} landbirds)",
         f"- family: {summ['family_size']} candidates "
         f"({', '.join(f'{k} {v}' for k, v in sorted(fam_counts.items()))}); "
         f"**{summ['eligible']} eligible** after the size / label-balance / label-proxy "
         f"filters (minimum group size {summ['min_group']})",
         f"- **detectability floor: m >= {summ['floor_m']}** (unstratified bound). A "
         f"margin-free group of at least this size, with the data's label mix, survives "
         f"Holm. Below it, a margin-free group can arise by chance.", "",
         "## Answer", ""]
    if hits:
        L += [f"**{len(hits)} HIT(S).** These candidates contain no margin point, more "
              "than chance allows after Holm correction over the whole eligible family. "
              "Each is a real, pre-existing image property whose group has the larger "
              "hard margin. The confirmation below must agree before anything is "
              "claimed.", "", HEAD] + [_row(r) for r in hits] + [""]
    else:
        L += ["**NO HIT.** No eligible candidate contains zero margin points beyond what "
              "chance allows. See the graded table below for how close the best came.", ""]
    L += ["## The closest candidates, by depletion of margin points", "",
          "`margin pts in G` against `expected` is the graded statistic: how many of "
          "the S points fall in G, against the label-matched expectation. "
          "`p (depletion)` is the exact probability of that few or fewer under the "
          "null. `ratio` exceeds 1 only when the count is 0.", "", HEAD]
    L += [_row(r) for r in by_dep[:25]] + [""]
    if confirm:
        L += ["## Confirmation bundles written", "",
              "Measured next by the unmodified `group_margins.py`; `--compare` checks "
              "agreement.", ""]
        L += [f"- `{c['bundle']}` -- {c['name']} (screen ratio {c['ratio']:.6f}, "
              f"larger {c['larger']}, reason: {c['reason']})" for c in confirm]
        L += [""]
    L += ["Every candidate, eligible or not, is in the `.json` next to this file.", ""]
    return "\n".join(L)


def to_markdown_full(recs) -> str:
    L = ["# Session 6 -- every eligible candidate, sorted by p (depletion)", "", HEAD]
    elig = sorted([r for r in recs if r["eligible"] and "p_min" in r],
                  key=lambda r: (r["p_depletion"], -r["m"]))
    L += [_row(r) for r in elig]
    L += ["", "## Ineligible", "", "| candidate G | m | reason |", "|---|---|---|"]
    L += [f"| {r['name']} | {r['m']} | {r['why_ineligible'] or 'degenerate'} |"
          for r in recs if not (r["eligible"] and "p_min" in r)]
    return "\n".join(L) + "\n"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:60]


# ----------------------------------------------------------------------------
# Compare mode: the unmodified instrument must agree with the screen
# ----------------------------------------------------------------------------
def compare(screen_json: str, margins_json: str, out_md: str, tie_tol: float) -> int:
    scr = json.load(open(screen_json))
    gm = {r["bundle"]: r for r in json.load(open(margins_json))}
    L = ["# Session 6 -- screen vs the unmodified group_margins.py", "",
         "The separator does not depend on g, so the two must agree to solver "
         "precision (LinearSVC shuffles its coordinates, so ~1e-9, checked at 1e-4).",
         "", "| bundle | candidate | screen ratio | group_margins ratio | screen larger | "
         "group_margins larger | agree |", "|---|---|---|---|---|---|---|"]
    bad = 0
    for c in scr.get("confirm", []):
        r = gm.get(c["bundle"])
        if r is None or not r.get("separable_by_svc"):
            L.append(f"| {c['bundle']} | {c['name']} | {c['ratio']:.6f} | MISSING | "
                     f"{c['larger']} | - | **NO** |")
            bad += 1
            continue
        lg = r.get("larger_margin_group")
        gm_larger = "tie" if lg is None else ("G" if lg == 1 else "rest")
        ok = (abs(r["gamma_min_theorem"] - c["ratio"]) <= 1e-4 * c["ratio"]
              and gm_larger == c["larger"])
        bad += (not ok)
        L.append(f"| {c['bundle']} | {c['name']} | {c['ratio']:.6f} | "
                 f"{r['gamma_min_theorem']:.6f} | {c['larger']} | {gm_larger} | "
                 f"{'yes' if ok else '**NO**'} |")
    L += ["", "**AGREEMENT OK**" if bad == 0 else f"**{bad} DISAGREEMENT(S) -- the screen "
          "is not measuring what group_margins.py measures; do not read the screen.**", ""]
    open(out_md, "w").write("\n".join(L))
    print("\n".join(L))
    return 0 if bad == 0 else 1


def replicate(train_json: str, test_json: str, out_md: str) -> int:
    tr = json.load(open(train_json))
    te = {r["name"]: r for r in json.load(open(test_json))["candidates"]}
    hits = [r for r in tr["candidates"] if r.get("hit")]
    L = ["# Session 6 -- do the train-split hits replicate on the test split?", "",
         f"- train: `{tr['bundle']}`, |S| = {tr['summary']['S_size']}",
         "- test: separator re-fitted on the test photographs; same candidate "
         "definitions (proxy cuts re-computed on the test split's own values).", ""]
    if not hits:
        L += ["**Nothing to replicate: the train screen had no hit.**", "",
              "For information only, the three best-ranked train candidates on test:", ""]
        show = sorted([r for r in tr["candidates"] if r.get("eligible") and "p_min" in r],
                      key=lambda r: (r["p_min"], r["p_depletion"], -r["m"]))[:3]
    else:
        show = hits
    rows, ps = [], []
    for r in show:
        t = te.get(r["name"])
        if t is None or "p_min" not in t:
            rows.append((r, None))
            continue
        rows.append((r, t))
        ps.append(t["p_min"])
    adj = list(holm(np.array(ps))) if ps else []
    L += ["| candidate G | train: m, margin pts, ratio | test: m | test margin pts / expected | "
          "test ratio | test larger | test p (min) | Holm over hits | replicates |",
          "|---|---|---|---|---|---|---|---|---|"]
    n_rep = 0
    for r, t in rows:
        if t is None:
            L.append(f"| {r['name']} | {r['m']}, {r['k_S']}, {r['ratio']:.4f} | "
                     "not measurable on test | - | - | - | - | - | no |")
            continue
        a = adj.pop(0)
        ok = bool(hits) and t["larger"] == "G" and a <= ALPHA
        n_rep += ok
        L.append(f"| {r['name']} | {r['m']}, {r['k_S']}, {r['ratio']:.4f} | {t['m']} | "
                 f"{t['k_S']} / {t['expected_k_S']:.1f} | {t['ratio']:.4f} | {t['larger']} | "
                 f"{t['p_min']:.2e} | {a:.2e} | {'**yes**' if ok else 'no'} |")
    if hits:
        L += ["", f"**{n_rep} of {len(hits)} train hit(s) replicate on the test split.**"]
    L.append("")
    open(out_md, "w").write("\n".join(L))
    print("\n".join(L))
    return 0


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _toy(n=600, d=20, seed=0):
    rng = np.random.default_rng(seed)
    y01 = (rng.random(n) < 0.3).astype(int)
    y = 2.0 * y01 - 1
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d)) + 2.5 * y[:, None] * (w / np.linalg.norm(w))[None, :]
    return X, y, y01


def _self_test() -> int:
    from group_margins import C_LADDER, TIE_TOL
    print("sv_screen self-test")
    ok = True
    rng = np.random.default_rng(1)

    # 1. exact p (min) against Monte Carlo
    n = 200
    u = rng.random(n)
    y01 = (rng.random(n) < 0.35).astype(int)
    mem = np.zeros(n, bool)
    top = np.argsort(-u)[:160]          # p lands near 0.1: an informative check
    mem[rng.choice(top, 12, replace=False)] = True
    p = p_min_stratified(u, y01, mem)
    t = u[mem].min()
    m1, m0 = int((mem & (y01 == 1)).sum()), int((mem & (y01 == 0)).sum())
    i1, i0 = np.where(y01 == 1)[0], np.where(y01 == 0)[0]
    R = 40000
    hits = 0
    for _ in range(R):
        s = np.concatenate([rng.choice(i1, m1, replace=False), rng.choice(i0, m0, replace=False)])
        hits += u[s].min() >= t
    mc = hits / R
    se = np.sqrt(max(mc * (1 - mc), 1e-12) / R)
    if abs(mc - p) > 4 * se + 1e-4:
        print(f"  FAIL exact p {p:.5f} vs Monte Carlo {mc:.5f} (se {se:.5f})"); ok = False
    else:
        print(f"  ok   exact label-stratified p (min) {p:.4f} matches Monte Carlo {mc:.4f}")

    # 2. exact p (depletion) against Monte Carlo
    in_S = u <= np.quantile(u, 0.15)
    mem2 = np.zeros(n, bool)
    mem2[rng.choice(np.where(~in_S)[0], 25, replace=False)] = True
    mem2[rng.choice(np.where(in_S)[0], 1, replace=False)] = True
    k, ek, pdep = p_depletion_stratified(in_S, y01, mem2)
    m1, m0 = int((mem2 & (y01 == 1)).sum()), int((mem2 & (y01 == 0)).sum())
    cnt = 0
    for _ in range(R):
        s = np.concatenate([rng.choice(i1, m1, replace=False), rng.choice(i0, m0, replace=False)])
        cnt += in_S[s].sum() <= k
    mc = cnt / R
    se = np.sqrt(max(mc * (1 - mc), 1e-12) / R)
    if abs(mc - pdep) > 4 * se + 1e-4:
        print(f"  FAIL exact depletion p {pdep:.5f} vs Monte Carlo {mc:.5f}"); ok = False
    else:
        print(f"  ok   exact depletion p {pdep:.4f} matches Monte Carlo {mc:.4f} (k={k}, "
              f"expected {ek:.2f})")

    # 3. Holm on a known vector
    adj = holm(np.array([0.01, 0.04, 0.03, 0.005]))
    if not np.allclose(adj, [0.03, 0.06, 0.06, 0.02]):
        print(f"  FAIL Holm {adj}"); ok = False
    else:
        print("  ok   Holm step-down on a known vector")

    # 4. leakage: a small label-pure group passes AUC but is caught by phi
    lab = np.zeros(4795, int); lab[:1113] = 1
    ind = np.zeros(4795, bool); ind[:100] = True
    lk = binary_leakage(ind, lab)
    if not (abs(lk["auc"] - 0.5) <= MAX_AUC_DEV and abs(lk["phi"]) > MAX_ABS_PHI):
        print(f"  FAIL label-pure small group: {lk}"); ok = False
    else:
        print(f"  ok   small label-pure group: AUC {lk['auc']:.3f} passes, phi "
              f"{lk['phi']:+.3f} catches it")

    # 5. planted margin-avoiding group is found, random decoys are not, and the
    #    screen's ratio equals group_margins' own on a written bundle.
    X, y, y01 = _toy()
    fit = fit_margins(X, y, C_LADDER, 200000, 1e-8)
    u = fit["u"]
    n = u.size
    in_S = u <= u.min() * (1 + TIE_TOL)
    easy = np.where(u >= np.quantile(u, 0.5))[0]
    fam = []
    planted = np.zeros(n, bool)
    for c in (0, 1):                                 # label-proportional, so no leakage
        pool = easy[y01[easy] == c]
        planted[rng.choice(pool, int(round(80 * (y01 == c).mean())), replace=False)] = True
    fam.append(("test", "planted", planted))
    for j in range(40):
        dm = np.zeros(n, bool)
        dm[rng.choice(n, 80, replace=False)] = True
        fam.append(("test", f"decoy {j}", dm))
    recs, summ = screen(u, y01, None, np.zeros(n, int), fam, 0.01, TIE_TOL)
    got = {r["name"]: r for r in recs}
    pl = got["planted"]
    false_hits = [r["name"] for r in recs if r.get("hit") and r["name"] != "planted"]
    if not (pl.get("hit") and pl["larger"] == "G" and pl["k_S"] == 0):
        print(f"  FAIL planted group not found: {pl}"); ok = False
    elif false_hits:
        print(f"  FAIL random decoys called hits: {false_hits}"); ok = False
    else:
        print(f"  ok   planted margin-free group found (ratio {pl['ratio']:.3f}, Holm p "
              f"{pl['p_min_holm']:.1e}); 0 of 40 random decoys called (|S| = "
              f"{summ['S_size']})")

    import tempfile
    from common import FeatureBundle
    from group_margins import analyse_bundle
    from regroup import write_bundle
    with tempfile.TemporaryDirectory() as td:
        fb = FeatureBundle(phi=X, y=y.astype(int), g=np.zeros(n, int),
                           idx_r=np.array([], int), idx_s=np.array([], int),
                           place=None, meta={"standardized": True})
        pth = os.path.join(td, "c.npz")
        write_bundle(fb, np.ones(n, bool), planted.astype(int), pth, {})
        res = analyse_bundle(pth, C_LADDER, 200000, 1e-8, 60.0, run_lp=False)
        if abs(res["gamma_min_theorem"] - pl["ratio"]) > 1e-4 * pl["ratio"] or \
                res["larger_margin_group"] != 1:
            print(f"  FAIL group_margins disagrees: {res['gamma_min_theorem']} vs "
                  f"{pl['ratio']}"); ok = False
        else:
            print(f"  ok   unmodified group_margins.py reports the same ratio "
                  f"({res['gamma_min_theorem']:.6f})")

    # 6. the floor is where the bound crosses
    fl = detectability_floor(4795, 421, 650)
    b_at = float(np.exp(_lcomb(4795 - 421, fl) - _lcomb(4795, fl)))
    b_before = float(np.exp(_lcomb(4795 - 421, fl - 1) - _lcomb(4795, fl - 1)))
    if not (b_at <= ALPHA / 650 < b_before):
        print(f"  FAIL detectability floor {fl}"); ok = False
    else:
        print(f"  ok   detectability floor at n=4795, |S|=421, 650 tests: m >= {fl}")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    from group_margins import C_LADDER, TIE_TOL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bundle")
    ap.add_argument("--meta", default="results/v6_cub_meta.npz")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--C", nargs="+", type=float, default=list(C_LADDER))
    ap.add_argument("--max-iter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="v6_sv_screen")
    ap.add_argument("--compare", metavar="MARGINS_JSON",
                    help="check group_margins.py's output against the screen")
    ap.add_argument("--no-confirm", action="store_true",
                    help="write no confirmation bundles (used for the replication run)")
    ap.add_argument("--replicate", metavar="TEST_SCREEN_JSON",
                    help="check the train hits (results/<tag>.json) on a test screen")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.replicate:
        return replicate(os.path.join(args.out_dir, f"{args.tag}.json"), args.replicate,
                         os.path.join(args.out_dir, "v6_replication.md"))
    if args.compare:
        return compare(os.path.join(args.out_dir, f"{args.tag}.json"), args.compare,
                       os.path.join(args.out_dir, "v6_confirm_check.md"), TIE_TOL)
    if not args.bundle:
        raise SystemExit("--bundle is required (or --self-test / --compare)")

    from common import FeatureBundle
    from datasets import DECLARATIONS
    from regroup import check_alignment, write_bundle

    z = np.load(args.meta, allow_pickle=True)
    meta = {k: z[k] for k in z.files}
    code = DECLARATIONS["waterbirds"]["splits"][args.split]
    rows = np.where(meta["split"] == code)[0]
    fb = FeatureBundle.load(args.bundle)
    align = check_alignment(fb, meta["y"][rows], meta["place"][rows])
    if not align["ok"]:
        raise SystemExit(f"ROW ORDER CHECK FAILED: {align.get('why')}")
    print(f"row order check: ok ({align['n_bundle']} rows, y and place both match)")

    X = np.asarray(fb.phi, dtype=np.float64)
    y = np.asarray(fb.y, dtype=np.float64)
    y01 = (y > 0).astype(int)
    fit = fit_margins(X, y, args.C, args.max_iter, args.tol)
    print(f"separator: margin {fit['margin']:.4f} at C = {fit['C_used']:g}, plateau "
          f"{fit.get('plateau_rel', float('nan')):.1e}")

    family = build_family(meta, rows)
    recs, summ = screen(fit["u"], y01, meta["place"][rows], meta["g_orig"][rows],
                        family, args.min_frac, TIE_TOL)
    print(f"|S| = {summ['S_size']}; {summ['eligible']} of {summ['family_size']} "
          f"candidates eligible; floor m >= {summ['floor_m']}; hits: {summ['hits']}")

    # Confirmation bundles: every hit (capped), plus the best-ranked few always.
    elig = [r for r in recs if r["eligible"] and "p_min" in r]
    hits = sorted([r for r in elig if r.get("hit")], key=lambda r: r["p_min"])
    ranked = sorted(elig, key=lambda r: (r["p_min"], r["p_depletion"], -r["m"]))
    chosen, seen = [], set()
    for r, why in ([] if args.no_confirm else
                   [(r, "hit") for r in hits[:MAX_CONFIRM_HITS]] +
                   [(r, "best-ranked") for r in ranked[:N_CONFIRM_ALWAYS]]):
        if r["name"] not in seen:
            seen.add(r["name"]); chosen.append((r, why))
    stem = os.path.basename(args.bundle).replace(".npz", "")
    base = stem.split("_", 2)[-1]
    lookup = {name: mem for _, name, mem in family}
    confirm = []
    os.makedirs(args.out_dir, exist_ok=True)
    for r, why in pbar(chosen, desc="confirm bundles", unit="bundle"):
        fname = f"features_v6_{base}_{_slug(r['name'])}.npz"
        write_bundle(fb, np.ones(fb.y.size, bool), lookup[r["name"]].astype(int),
                     os.path.join(args.out_dir, fname),
                     {"regroup_rule": "v6_screen", "regroup_candidate": r["name"],
                      "regroup_source": os.path.basename(args.bundle),
                      "regroup_split": args.split})
        confirm.append({"bundle": fname, "name": r["name"], "ratio": r["ratio"],
                        "larger": r["larger"], "reason": why})
    if not args.no_confirm:
        with open(os.path.join(args.out_dir, "v6_confirm_list.txt"), "w") as fh:
            fh.write("\n".join(os.path.join(args.out_dir, c["bundle"])
                               for c in confirm) + "\n")

    fit_out = {k: v for k, v in fit.items() if k != "u"}
    out = {"bundle": args.bundle, "split": args.split, "fit": fit_out, "summary": summ,
           "alignment": align, "confirm": confirm, "candidates": recs}
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    md = to_markdown(recs, summ, fit_out, os.path.basename(args.bundle), confirm)
    with open(os.path.join(args.out_dir, f"{args.tag}.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.out_dir, f"{args.tag}_full.md"), "w") as fh:
        fh.write(to_markdown_full(recs))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
