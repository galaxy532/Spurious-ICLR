"""Per-group hard margins gamma_maj, gamma_min from a feature bundle.

WHAT THIS MEASURES AND WHY IT IS THE POINT OF SESSION 4
=======================================================
The manuscript defines the group hard-margins (supplementary Eq. eq:margins) as

    gamma_g = ess inf over group g of  y * (w_hat . x),      min(gamma_maj, gamma_min) = 1

where `w_hat` is the minimum-norm hard-margin separator -- the thing population
GD on a separable problem converges to in direction (Soudry et al. 2018, and
Proposition prop:soudry-continuous-main in the supplementary). So gamma_g is a
property of the FROZEN REPRESENTATION alone: no gradient descent, no z, no eps.

It matters because of the alignment identity the manuscript derives
(neurips_2026.tex, "Interpretation of the main results"):

    gamma_min = max(1, alpha_xi(1 + delta_maj) - delta_min)  ~=  max(1, alpha)

Read from right to left, that turns a measurement into a PREDICTION. Measure the
ratio of per-group margins here; the theory then says the larger-margin group's
error exponent beta should be that same number, and the other group's should be
exactly 1. `long_horizon.py` measures both betas independently. Agreement or
disagreement between the two is a parameter-free test of Theorem 5.3 -- there is
nothing to tune, and in particular nothing requiring the (r, s) identification
that the Waterbirds arm in `Rebuttals/` could not settle.

It also answers, in one number, the question sessions 1-3 could not: whether a
setting is anywhere near the alpha = 1 phase transition. On the session-3
Waterbirds bundles the iterate's own per-group margins were still spread from
0.87 to 1.30 across the eps grid at z = 1e6 and shrinking by only ~11% per
half-decade, so the LONG GD RUN CANNOT SETTLE THIS. This script can, in minutes,
because it solves the margin problem directly instead of waiting for GD to walk
there at O(1/log t).

HOW w_hat IS OBTAINED, AND THE HONEST LIMITATION
================================================
`w_hat` has NO INTERCEPT -- the GD in `long_horizon.py` has none either, so the
two must match or the comparison is meaningless.

There is no exact hard-margin solver here. Instead `LinearSVC(loss="hinge",
fit_intercept=False)` is run along a ladder of C, and the whole ladder is
reported. As C grows the soft-margin solution approaches the hard-margin one, so
the diagnostic to look at is whether the margin has PLATEAUED across the top of
the ladder: `plateau_rel` is the relative change over the last two C values.
Below ~1e-3 the number is settled; above ~1e-2 it is not, and the run should be
repeated with a longer ladder rather than quoted.

Two independent cross-checks are printed next to it:
  * `margin_logistic` -- logistic regression at huge C, via
    `separability_check.quick_separable`. A different loss and a different
    optimiser reaching the same geometric margin is real evidence.
  * `rho_lp` -- the exact linear program of `separability_check`. That LP uses a
    BOX constraint (|w_j| <= 1), so its optimum is an L-infinity-flavoured margin
    and is NOT comparable in value to the L2 margins here. It is reported only
    because it PROVES separability when positive, which the SVC cannot.

Usage
-----
    python group_margins.py --bundles 'features_v4_waterbirds_dinov2_*_train.npz'
    python group_margins.py --bundles a.npz b.npz --tag v4_margins --out-dir results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from common import FeatureBundle
from progress import pbar

C_LADDER = (1e2, 1e3, 1e4, 1e5, 1e6)

# Relative gap below which the two per-group margins are called EQUAL. This is
# not cosmetic. Every real bundle measured so far is a tie, and a script that
# always names a "larger" group would report a branch asymmetry built out of
# floating-point noise. A tie means the setting sits ON the alpha = 1 phase
# transition, which is a finding, not a missing value.
TIE_TOL = 1e-3


def _geometric_margins(X, y, g, w):
    """min_i y_i (w.x_i)/||w||, overall and within each group."""
    nw = float(np.linalg.norm(w))
    if not np.isfinite(nw) or nw == 0.0:
        return dict(margin=float("nan"), margin_g0=float("nan"),
                    margin_g1=float("nan"), n_violations=int(len(y)))
    u = (y * (X @ w)) / nw
    out = dict(
        margin=float(u.min()),
        margin_g0=float(u[g == 0].min()) if np.any(g == 0) else float("nan"),
        margin_g1=float(u[g == 1].min()) if np.any(g == 1) else float("nan"),
        n_violations=int((u <= 0).sum()),
    )
    return out


def _svc_ladder(X, y, ladder, max_iter, tol):
    """LinearSVC hinge, no intercept, along a ladder of C. Returns per-C rows."""
    from sklearn.svm import LinearSVC
    rows = []
    for C in ladder:
        t0 = time.time()
        clf = LinearSVC(loss="hinge", fit_intercept=False, C=float(C),
                        max_iter=int(max_iter), tol=float(tol), dual=True)
        converged = True
        try:
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf.fit(X, y)
                converged = not any(issubclass(c.category, ConvergenceWarning)
                                    for c in caught)
        except Exception as exc:                       # noqa: BLE001
            rows.append(dict(C=float(C), error=str(exc)[:200], secs=round(time.time() - t0, 1)))
            continue
        w = clf.coef_.ravel().astype(np.float64)
        rows.append(dict(C=float(C), converged=bool(converged),
                         secs=round(time.time() - t0, 1), w=w))
    return rows


def analyse_bundle(path, ladder, max_iter, tol, lp_time_limit, run_lp) -> dict:
    fb = FeatureBundle.load(path)
    X = np.asarray(fb.phi, dtype=np.float64)
    y = np.asarray(fb.y, dtype=np.float64)          # already +/-1
    g = np.asarray(fb.g, dtype=int)
    meta = dict(fb.meta or {})

    res = {
        "bundle": os.path.basename(path),
        "n": int(X.shape[0]), "d": int(X.shape[1]),
        "n_g0": int((g == 0).sum()), "n_g1": int((g == 1).sum()),
        "eps_natural": float(np.mean(g == 1)),
        "standardized": bool(meta.get("standardized", False)),
        "backbone": meta.get("backbone"), "split": meta.get("split"),
        "degraded": bool(meta.get("degraded", False)),
        "degrade_kind": meta.get("degrade_kind"),
        "degrade_level": meta.get("degrade_level"),
        "degrade_group": meta.get("degrade_group"),
        "degrade_at": meta.get("degrade_at"),
        "ladder": [],
    }
    if not res["standardized"]:
        res["WARNING"] = "bundle is not marked standardised; margins are not comparable"

    rows = _svc_ladder(X, y, ladder, max_iter, tol)
    best = None
    for r in pbar(rows, desc=f"  {os.path.basename(path)[:34]:34s}", unit="C"):
        if "w" not in r:
            res["ladder"].append({k: v for k, v in r.items() if k != "w"})
            continue
        m = _geometric_margins(X, y, g, r["w"])
        row = {k: v for k, v in r.items() if k != "w"}
        row.update(m)
        res["ladder"].append(row)
        if m["n_violations"] == 0:
            best = (r, m)

    if best is None:
        res["separable_by_svc"] = False
        res["note"] = ("no C in the ladder produced a separator with zero violations; "
                       "either the bundle is not separable or the ladder is too short")
    else:
        r, m = best
        res["separable_by_svc"] = True
        res["C_used"] = r["C"]
        res["converged"] = r.get("converged")
        res.update({k: m[k] for k in ("margin", "margin_g0", "margin_g1")})
        # Has the margin stopped moving across the top of the ladder?
        clean = [row for row in res["ladder"]
                 if row.get("n_violations") == 0 and np.isfinite(row.get("margin", np.nan))]
        if len(clean) >= 2:
            a, b = clean[-2]["margin"], clean[-1]["margin"]
            res["plateau_rel"] = float(abs(b - a) / max(abs(b), 1e-300))
            res["plateau_verdict"] = ("settled" if res["plateau_rel"] < 1e-3
                                      else "moving" if res["plateau_rel"] < 1e-2
                                      else "NOT SETTLED -- extend the ladder")
        else:
            res["plateau_rel"] = None
            res["plateau_verdict"] = "only one clean C; cannot judge"

        # The manuscript's normalisation: min(gamma_maj, gamma_min) = 1.
        m0, m1 = m["margin_g0"], m["margin_g1"]
        lo = min(m0, m1)
        res["gamma_g0"] = float(m0 / lo)
        res["gamma_g1"] = float(m1 / lo)
        res["gamma_ratio_g1_over_g0"] = float(m1 / m0)
        rel = abs(m1 - m0) / max(lo, 1e-300)
        res["margin_asymmetry_rel"] = float(rel)
        if rel <= TIE_TOL:
            res["larger_margin_group"] = None
            res["asymmetry_verdict"] = "tie -- the setting sits on the alpha = 1 transition"
        else:
            res["larger_margin_group"] = int(1 if m1 > m0 else 0)
            res["asymmetry_verdict"] = f"g={res['larger_margin_group']} has the larger margin"
        res["gamma_min_theorem"] = float(max(m0, m1) / lo)   # >= 1 by construction
        # Alignment identity gamma_min ~= max(1, alpha): the predicted exponent
        # for the LARGER-margin group. The other group is predicted to be 1.
        res["beta_pred_larger_margin_group"] = res["gamma_min_theorem"]
        res["alpha_gt_1_predicted"] = bool(res["gamma_min_theorem"] > 1.0)
        # How many points sit essentially ON the margin, per group: a ratio driven
        # by a single point is fragile and this is how that shows up.
        nw = float(np.linalg.norm(r["w"]))
        u = (y * (X @ r["w"])) / nw
        for gg in (0, 1):
            sel = (g == gg)
            if sel.any():
                res[f"n_at_margin_g{gg}"] = int((u[sel] <= u[sel].min() * 1.01).sum())

    if run_lp:
        from separability_check import _try_lp, quick_separable
        ok, mlog = quick_separable(X, y)
        res["margin_logistic"] = float(mlog) if ok else None
        sep, rho, _, why = _try_lp(X, y, lp_time_limit)
        res["rho_lp"] = float(rho) if np.isfinite(rho) else None
        res["lp_separable"] = sep
        res["lp_status"] = why
    return res


def to_markdown(rows: list[dict]) -> str:
    L = ["# Per-group hard margins",
         "",
         "`gamma_g` are the manuscript's group hard-margins (supplementary Eq eq:margins):",
         "the minimum of `y * w_hat . x` within each group, under the global minimum-norm",
         "separator `w_hat`, normalised so the SMALLER of the two is 1. `w_hat` has no",
         "intercept, matching the GD in `long_horizon.py`.",
         "",
         "The alignment identity `gamma_min ~= max(1, alpha)` makes the last column a",
         "prediction: the LARGER-margin group's error exponent `beta` should equal it, and",
         "the other group's should be 1. `long_horizon.py` measures both independently.",
         "",
         "Read `plateau` first. It is the relative change of the margin across the top two",
         "C values of the solver ladder. Below 1e-3 the margin is settled; above 1e-2 the",
         "row is not a measurement yet and must not be quoted.",
         "",
         "| bundle | degradation | n | d | margin | gamma(g=0) | gamma(g=1) | larger | beta predicted | plateau |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("separable_by_svc"):
            L.append(f"| {r['bundle']} | {_degstr(r)} | {r['n']} | {r['d']} | "
                     f"NOT SEPARABLE (svc) | - | - | - | - | - |")
            continue
        L.append(
            f"| {r['bundle']} | {_degstr(r)} | {r['n']} | {r['d']} | "
            f"{r['margin']:.4g} | {r['gamma_g0']:.4f} | {r['gamma_g1']:.4f} | "
            f"{_larger(r)} | {r['beta_pred_larger_margin_group']:.4f} | "
            f"{r['plateau_rel']:.1e} |" if r.get("plateau_rel") is not None else
            f"| {r['bundle']} | {_degstr(r)} | {r['n']} | {r['d']} | "
            f"{r['margin']:.4g} | {r['gamma_g0']:.4f} | {r['gamma_g1']:.4f} | "
            f"{_larger(r)} | {r['beta_pred_larger_margin_group']:.4f} | - |")
    L += ["", "## Cross-checks and fragility", "",
          "`margin_logistic` is the same geometric margin from logistic regression at huge C",
          "(a different loss and optimiser). `rho_lp` is the exact LP of `separability_check`,",
          "which uses a BOX constraint and is therefore NOT comparable in value -- it is here",
          "only because a positive value PROVES separability. `n at margin` counts the points",
          "within 1% of their group's minimum: a `gamma` ratio resting on one point is fragile.",
          "",
          "| bundle | margin (svc) | margin (logistic) | rho_lp | lp separable | n at margin g=0 | g=1 | C used | converged |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k, fmt="{:.4g}": (fmt.format(r[k]) if isinstance(r.get(k), (int, float))
                                     and np.isfinite(r.get(k)) else "-")
        L.append(f"| {r['bundle']} | {f('margin')} | {f('margin_logistic')} | {f('rho_lp')} | "
                 f"{r.get('lp_separable')} | {r.get('n_at_margin_g0', '-')} | "
                 f"{r.get('n_at_margin_g1', '-')} | {f('C_used')} | {r.get('converged')} |")
    return "\n".join(L) + "\n"


def _larger(r) -> str:
    lg = r.get("larger_margin_group")
    return "tie" if lg is None else f"g={lg}"


def _degstr(r) -> str:
    if not r.get("degraded"):
        return "none"
    return (f"{r.get('degrade_kind')} {r.get('degrade_level')} "
            f"on g={r.get('degrade_group')} ({r.get('degrade_at')})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", nargs="+", required=True,
                    help="bundle .npz paths or globs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="group_margins")
    ap.add_argument("--C", nargs="+", type=float, default=list(C_LADDER),
                    help="the LinearSVC ladder (default 1e2..1e6)")
    ap.add_argument("--max-iter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--no-lp", action="store_true",
                    help="skip the LP and logistic cross-checks (faster)")
    ap.add_argument("--lp-time-limit", type=float, default=600.0)
    args = ap.parse_args()

    paths = []
    for b in args.bundles:
        hits = sorted(glob.glob(b))
        if not hits and os.path.exists(b):
            hits = [b]
        if not hits:
            print(f"  WARNING: no bundle matched {b!r}")
        paths.extend(hits)
    if not paths:
        raise SystemExit("no bundles matched")

    rows = []
    for p in pbar(paths, desc="bundles", unit="bundle"):
        print(f"\n{p}")
        r = analyse_bundle(p, args.C, args.max_iter, args.tol,
                           args.lp_time_limit, run_lp=not args.no_lp)
        rows.append(r)
        if r.get("separable_by_svc"):
            print(f"  margin {r['margin']:.4g}   gamma(g=0) {r['gamma_g0']:.4f}   "
                  f"gamma(g=1) {r['gamma_g1']:.4f}   {r['asymmetry_verdict']}   "
                  f"beta predicted {r['beta_pred_larger_margin_group']:.4f}   "
                  f"plateau {r.get('plateau_verdict')}")
        else:
            print(f"  {r.get('note')}")

    os.makedirs(args.out_dir, exist_ok=True)
    js = os.path.join(args.out_dir, f"{args.tag}.json")
    md = os.path.join(args.out_dir, f"{args.tag}.md")
    with open(js, "w") as f:
        json.dump(rows, f, indent=1, default=float)
    with open(md, "w") as f:
        f.write(to_markdown(rows))
    print(f"\nwrote {js}\nwrote {md}")


if __name__ == "__main__":
    main()
