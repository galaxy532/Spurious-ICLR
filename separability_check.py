"""Is each (backbone, eps) cell of the arm-A sweep linearly separable, and with what margin?

WHY THIS EXISTS
===============

`eps_backbone_sweep.py` measures a decay exponent `beta` in  err ~ z^(-beta)
and reads a regime off it. That reading is only meaningful inside the
implicit-bias phase, which requires the training subsample to be LINEARLY
SEPARABLE through the origin (there is no intercept in `common.logistic_gd`).

  * separable      -> ||w|| grows without bound, margins grow, err -> 0 like a
                      power of z. beta is a real exponent.
  * not separable  -> w converges to a finite minimiser, err levels off at a
                      positive constant. beta -> 0, and that 0 means "the
                      question does not apply here", NOT "alpha is small".

The Waterbirds run produced both behaviours INSIDE ONE eps GRID. CLIP at
eps=0.05 is frozen in the fourth digit across a 4x increase in z
(0.32968 / 0.32951 / 0.32946) while CLIP at eps=0.005 decays like z^-0.93. If
separability flips partway along the grid, the kappa fit is regressing "how hard
is this subsample to fit" against log eps rather than the eps law.

This script answers that directly, on CPU, in seconds per cell.


WHAT IT REPORTS, AND WHY THE MARGIN MATTERS AS MUCH AS THE YES/NO
=================================================================

Separability alone does not explain a slow curve. Two of the four Waterbirds
backbones (`erm_rn50`, `under_rn50`) never froze -- they decayed slowly and got
slower as eps rose. That is equally consistent with "separable, but the margin
shrank so convergence is slower", which is a DIFFERENT diagnosis with a
different remedy (run longer) from "not separable" (no remedy; the regime is
absent). So the margin is reported for every separable cell.

A useful logical handle, used in the verdict line:

    ANY SUBSET OF A SEPARABLE SET IS SEPARABLE.

The largest eps point uses the whole training split. So if the FULL bundle is
separable, every smaller-eps cell is separable too, and a separability FLIP
cannot be the explanation for that backbone -- look at the margin column
instead. Only when the full bundle is non-separable can the grid straddle the
boundary.


HOW SEPARABILITY IS DECIDED
===========================

Two stages, so the expensive one runs only when it has to:

  1. CONSTRUCTIVE. Fit logistic regression with almost no regularisation and no
     intercept. If it classifies every training point correctly, that w IS an
     explicit separator -- a proof, no further work needed. Weakly-regularised
     logistic regression also converges in direction to the max-margin
     separator (Soudry et al. 2018), so its margin is a good approximation to
     the max margin and is reported as a lower bound on it.

  2. LP. If stage 1 leaves violations, that could be the optimiser rather than
     the geometry, so settle it exactly:

         maximise  rho   s.t.  y_i (w . x_i) >= rho  for all i,
                               -1 <= w_j <= 1,  0 <= rho <= 1.

     w = 0, rho = 0 is always feasible, so the optimum is 0 when no separator
     exists and strictly positive when one does. HiGHS solves it exactly.

Reported `margin` is geometric: min_i y_i (w.x_i) / ||w||_2. Features are
standardised, so ||x_i||_2 ~ sqrt(d) and d differs across backbones; `margin_n`
divides by mean_i ||x_i||_2 so the column is comparable across backbones too.
Within one backbone, either column shows the trend.


REPRODUCING THE SWEEP'S EXACT SUBSAMPLES
========================================

`sweep_one` creates ONE rng at the top and calls `subsample_to_eps` once per eps
IN ORDER, so the draws are correlated with the position of eps in the list. This
script reproduces that exactly: same seed, same eps order, same call sequence.
Pass the SAME --eps and --seed you passed to the sweep or you are checking
different subsamples than the ones that produced the curves.


Examples
--------
    # the Waterbirds cells that produced results/waterbirds.md
    python separability_check.py --bundles 'features_waterbirds_*_train.npz' \
        --eps 0.005,0.01,0.02,0.035,0.05 --tag waterbirds

    # CelebA later; --lp-max-n keeps the LP from being attempted at 162k x 2048
    python separability_check.py --bundles 'features_celeba_*_train.npz' \
        --eps 0.02,0.05,0.10,0.20,0.40 --tag celeba --lp-max-n 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from common import FeatureBundle, subsample_to_eps
from progress import pbar

LP_TOL = 1e-9


def _margin(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, int]:
    """Geometric margin of the separator w, and how many points it gets wrong."""
    nw = float(np.linalg.norm(w))
    if nw == 0.0:
        return 0.0, int(y.size)
    s = y * (X @ w)
    return float(s.min() / nw), int((s <= 0).sum())


def _try_logistic(X: np.ndarray, y: np.ndarray, C: float, max_iter: int):
    """Stage 1. Returns (w, margin, n_violations). n_violations == 0 proves separable."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=max_iter,
                             tol=1e-10, solver="lbfgs")
    # sklearn emits a convergence warning at huge C on separable data; that is
    # expected (||w|| -> infinity) and harmless, only the sign pattern matters.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    w = clf.coef_.ravel().astype(np.float64)
    m, nv = _margin(X, y, w)
    return w, m, nv


def _try_lp(X: np.ndarray, y: np.ndarray, time_limit: float):
    """Stage 2. Exact. Returns (separable, rho_star, w, note).

    separable is None when the LP could not decide (unavailable, failed, or hit
    `time_limit`), which is reported as such rather than guessed either way.
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return None, float("nan"), None, "scipy unavailable"

    n, d = X.shape
    Z = y[:, None].astype(np.float64) * X
    A_ub = np.empty((n, d + 1), dtype=np.float64)
    A_ub[:, :d] = -Z
    A_ub[:, d] = 1.0
    b_ub = np.zeros(n)
    c = np.zeros(d + 1)
    c[d] = -1.0
    bounds = [(-1.0, 1.0)] * d + [(0.0, 1.0)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                  options={"time_limit": float(time_limit)})
    if (not res.success) or res.x is None:
        # status 1 is the iteration/time limit; anything else is a solver failure.
        why = "LP hit time limit" if res.status == 1 else f"LP failed (status {res.status})"
        return None, float("nan"), None, why
    rho = float(res.x[d])
    return (rho > LP_TOL), rho, res.x[:d].copy(), "ok"


def quick_separable(X: np.ndarray, y: np.ndarray, C: float = 1e6,
                    max_iter: int = 5000):
    """Constructive-only separability test, for callers that need a fast gate.

    Returns (True, margin) when a separator is FOUND -- which is a proof -- and
    (None, nan) when none was found, which is NOT a proof of the negative: the
    optimiser may simply have missed a tiny margin. Callers that need the
    negative settled must run this module's LP (i.e. the full script).
    """
    w, m, nv = _try_logistic(X, y, C, max_iter)
    if nv == 0:
        return True, m
    w2, m2, nv2 = _try_logistic(X, y, C * 1e3, max_iter * 4)
    if nv2 == 0:
        return True, m2
    return None, float("nan")


def check_cell(X: np.ndarray, y: np.ndarray, C: float, max_iter: int,
               lp_max_n: int, lp_time_limit: float) -> dict:
    """Decide separability for one cell and measure its margin.

    Stage 1 twice before stage 2: a cell whose max margin is tiny is exactly the
    case where lbfgs stops early AND the LP is slowest, and raising C costs a
    second where the LP costs minutes.
    """
    t0 = time.time()
    scale = float(np.mean(np.linalg.norm(X, axis=1)))

    def done(**kw):
        kw.setdefault("rho_lp", None)
        kw["secs"] = round(time.time() - t0, 1)
        mg = kw.get("margin", float("nan"))
        kw["margin_n"] = mg / scale if (scale and np.isfinite(mg)) else float("nan")
        return kw

    w, m, nv = _try_logistic(X, y, C, max_iter)
    if nv == 0:
        return done(separable=True, how="logistic (constructive)",
                    margin=m, n_viol_logistic=0)

    w2, m2, nv2 = _try_logistic(X, y, C * 1e3, max_iter * 4)
    if nv2 == 0:
        return done(separable=True, how="logistic at higher C (constructive)",
                    margin=m2, n_viol_logistic=0)
    nv = min(nv, nv2)

    if lp_max_n <= 0 or X.shape[0] > lp_max_n:
        return done(separable=None, how=f"{nv} violations left; LP skipped",
                    margin=float("nan"), n_viol_logistic=nv)

    sep, rho, w_lp, note = _try_lp(X, y, lp_time_limit)
    if sep is None:
        return done(separable=None, how=f"{nv} violations left; {note}",
                    margin=float("nan"), n_viol_logistic=nv)
    if sep:
        mlp, _ = _margin(X, y, w_lp)
        return done(separable=True, how="LP (logistic missed it)",
                    margin=mlp, n_viol_logistic=nv, rho_lp=rho)
    return done(separable=False, how="LP proof (max margin is 0)",
                margin=0.0, n_viol_logistic=nv, rho_lp=rho)


def check_bundle(fb: FeatureBundle, eps_list: list[float], seed: int,
                 C: float, max_iter: int, lp_max_n: int,
                 lp_time_limit: float) -> dict:
    """Full bundle first, then every eps cell, reproducing the sweep's draws."""
    # Draw every subsample FIRST, in the sweep's order, so the rng consumption
    # is identical to sweep_one's -- the full split does not touch the rng. Then
    # the cells are a known-length list, which is what the bar needs.
    rng = np.random.default_rng(seed)
    cells = [(None, "full", None)]
    for eps in eps_list:
        cells.append((eps, f"{eps:g}", subsample_to_eps(fb.y, fb.g, eps, rng)))

    rows = []
    bar = pbar(total=len(cells), unit="cell", desc="  cells")
    for eps, label, idx in cells:
        bar.set_postfix_str(f"{label}")
        X = fb.phi if idx is None else fb.phi[idx]
        yy = fb.y if idx is None else fb.y[idx]
        gg = fb.g if idx is None else fb.g[idx]
        r = check_cell(X, yy, C, max_iter, lp_max_n, lp_time_limit)
        r.update({"eps": eps, "label": label, "n": int(yy.size),
                  "n_min": int(np.sum(gg == 1))})
        rows.append(r)
        print(f"  {label:<13} n={r['n']:>7}  sep={r['separable']}  "
              f"margin={r['margin']:.4g}  [{r['how']}, {r['secs']}s]")
        bar.update(1)
    bar.close()

    return {"rows": rows, "meta": fb.meta, "eps_list": eps_list, "seed": seed}


def verdict(rows: list[dict]) -> str:
    """One line saying which diagnosis this backbone's curves need."""
    full = rows[0]
    cells = rows[1:]
    if full["separable"] is True:
        ms = [c["margin"] for c in cells if np.isfinite(c["margin"])]
        if len(ms) >= 2 and min(ms) > 0:
            ratio = max(ms) / min(ms)
            return (f"FULL SPLIT IS SEPARABLE, so every eps cell is too (subsets of a "
                    f"separable set are separable). A separability flip CANNOT explain "
                    f"this backbone's eps-dependence. Margin varies {ratio:.1f}x across "
                    f"the grid -- if that is large, the curves differ in CONVERGENCE "
                    f"SPEED, not in regime, and the remedy is a longer run.")
        return "FULL SPLIT IS SEPARABLE, so every eps cell is too."
    if full["separable"] is False:
        flips = [c["label"] for c in cells if c["separable"] is True]
        if flips:
            return (f"FULL SPLIT IS NOT SEPARABLE but cells {flips} ARE. The grid "
                    f"STRADDLES the boundary: beta is a real exponent at those eps and "
                    f"is identically 0 at the others, so the kappa fit mixes two "
                    f"different quantities and cannot be interpreted.")
        return ("NO CELL IS SEPARABLE. There is no implicit-bias phase anywhere on this "
                "backbone, so every beta is ~0 by construction and kappa is undefined "
                "rather than small.")
    return ("UNDETERMINED for the full split -- raise --lp-max-n and/or "
            "--lp-time-limit and re-run just this bundle.")


def to_markdown(all_res: dict) -> str:
    L = ["\n## linear separability of each arm-A cell", "",
         "Separability is through the ORIGIN (`common.logistic_gd` has no intercept).",
         "`margin` is geometric, min_i y_i(w.x_i)/||w||; it is a LOWER BOUND on the max",
         "margin (the separator comes from weakly-regularised logistic regression, which",
         "converges in direction to max-margin but is not run to convergence).",
         "`margin_n` divides by mean ||x_i|| so it compares across backbones.", ""]
    for key, r in all_res.items():
        L += [f"### {key}", "",
              "| cell | n | n_min | separable | margin | margin_n | decided by |",
              "|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            sep = {True: "yes", False: "**no**", None: "?"}[row["separable"]]
            mg = "--" if not np.isfinite(row["margin"]) else f"{row['margin']:.4g}"
            mn = "--" if not np.isfinite(row["margin_n"]) else f"{row['margin_n']:.4g}"
            L.append(f"| {row['label']} | {row['n']} | {row['n_min']} | {sep} "
                     f"| {mg} | {mn} | {row['how']} |")
        L += ["", f"**Verdict.** {verdict(r['rows'])}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--eps", default="0.005,0.01,0.02,0.035,0.05",
                    help="MUST match the --eps you gave eps_backbone_sweep.py, "
                         "in the same order, or you are checking different draws.")
    ap.add_argument("--seed", type=int, default=0, help="must match the sweep's --seed")
    ap.add_argument("--C", type=float, default=1e6,
                    help="inverse regularisation for the constructive stage")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--lp-max-n", type=int, default=20000,
                    help="skip the exact LP above this many rows (it is dense, "
                         "n x (d+1) float64). 0 disables the LP entirely.")
    ap.add_argument("--lp-time-limit", type=float, default=600.0,
                    help="seconds HiGHS may spend on one cell. A cell that hits "
                         "this is reported UNDETERMINED, never guessed.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="separability")
    args = ap.parse_args()

    paths = []
    for p in args.bundles:
        paths.extend(sorted(glob.glob(p)) or [p])
    eps_list = [float(x) for x in args.eps.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    all_res = {}
    for p in pbar(paths, unit="bundle", desc="bundles"):
        key = os.path.splitext(os.path.basename(p))[0].replace("features_", "")
        print(f"[{key}] loading {p}")
        fb = FeatureBundle.load(p)
        all_res[key] = check_bundle(fb, eps_list, args.seed, args.C,
                                    args.max_iter, args.lp_max_n,
                                    args.lp_time_limit)
        print(f"  -> {verdict(all_res[key]['rows'])}\n")
        # Written after EVERY bundle, not at the end: a killed run keeps what it
        # already earned. This is the thing eps_backbone_sweep.py gets wrong.
        with open(os.path.join(args.out_dir, f"{args.tag}_separability.md"), "w") as f:
            f.write(to_markdown(all_res))
        with open(os.path.join(args.out_dir, f"{args.tag}_separability.json"), "w") as f:
            json.dump(all_res, f, indent=1, default=float)

    print(to_markdown(all_res))


if __name__ == "__main__":
    main()
