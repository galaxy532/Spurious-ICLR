"""Can the per-group margin ratio be anything other than 1 on real features?

WHY THIS EXISTS
===============
Every margin ratio this project has measured on real data is a tie: seven
session-4 bundles at 1.0000 to within 2e-9, and the five representations of
sessions 1-3 before them. Those nulls only mean something if the instrument
could have reported otherwise. Nothing currently establishes that.

`validate_group_margins.py` recovers a ratio of 1.6000 on its `asymmetric` case,
but read how that case is built (`make_bundle`): every point is `x[0] = y * m`
plus a perturbation that is orthogonal to `e_1` AND mirrored (`sgn = +/-1`). The
mirroring punishes any tilt out of `e_1` symmetrically, so the max-margin
direction is PINNED to `e_1` and the group margins are exactly `m0` and `m1` by
construction. It is a correctness check -- the arithmetic is right -- and it
deliberately removes the one thing that matters here: the solver's freedom to
rotate.

On a real bundle (n = 4795, d = 768) the solver has that freedom in abundance.
`gamma_g` is measured under the GLOBAL max-margin separator, and the ratio
exceeds 1 only if one group contributes NO support vector. When the optimiser can
rotate, it will trade margin between the groups until both bind -- which is
exactly what a tie is. Session 4's numbers are consistent with this: `g=1` kept
49-71 of its 240 points on the margin at every degradation level, and the sign of
`ratio - 1` flipped randomly across levels at the 1e-9 level.

So the question this script answers is not "is Waterbirds asymmetric". It is the
prior one:

    On these features, does ANY partition into two groups produce a ratio
    other than 1 after the separator is re-solved?

If the answer is no, then the bird-size re-partition cannot work either, and we
know that for minutes of CPU instead of a day of work. If the answer is yes, this
also says how large an asymmetry has to be before it survives -- which is the
power curve the project has for the invariance rule (`validate_invariance.py`
check C2b, the F1-vs-n curve that got Waterbirds dropped from that rule) and has
never had for the margin ratio.

This is an instrument calibration, NOT a result. Nothing here is evidence for or
against Theorem 5.3, and it is not the semi-synthetic `Phi` with a tunable alpha
rejected in August 2026 -- that one manufactured an exponent and reported it.

THE FOUR SWEEPS
===============
1. `oracle` -- THE DECISIVE ONE. Under the baseline separator, rank every point
   by its own margin and put the top `k` into group 1. That is the most
   margin-asymmetric partition this dataset admits: group 1 is made of the points
   furthest from the boundary, the ones most likely to stay off the margin under
   any separator. Sweeping `k` traces the best case available to ANY group
   variable, bird size included. If the oracle partition ties after re-solving,
   no natural partition will do better.

   It is an upper bound in spirit, not a theorem: the ranking is optimal under
   the BASELINE separator, and re-solving changes the separator.

2. `pinned` -- a positive control at the real scale. Rebuild the repo's own
   `make_bundle` geometry at this bundle's n and d, with a known ratio. If the
   ratio is recovered here but nowhere else, the difference is the solver's
   freedom to rotate and not a bug.

3. `translate` -- shift a random half of the points by `delta` along
   `y * w / ||w||`, which adds exactly `delta` to each of their margins under
   `w`. This is the naive way to plant an asymmetry, and it plants it along the
   one direction the optimiser can most easily undo. Reported to show, with
   numbers, how completely re-solving cancels it.

4. `heavy_tail` -- the oracle partition with a fraction `q` of group 1 replaced
   by randomly chosen (i.e. possibly hard) points. `gamma_g` is an ess inf, set
   by the single worst point in the group, so one hard member is enough to pull
   the whole group's margin back down. Real difficulty asymmetries are
   heterogeneous; this says whether the estimand can see them at all.

ALONGSIDE THE ESS INF, A QUANTILE DIAGNOSTIC
============================================
Every row also reports the same comparison at the 1st and 5th percentile of each
group. That costs nothing and separates two very different situations: "the
groups really are equally hard" (ess inf and both quantile ratios at 1) from "an
asymmetry is there and the ess inf cannot see it" (quantile ratios move, ess inf
does not).

This is a diagnostic, not a proposed replacement for the manuscript's `gamma_g`.
The error exponent is a tail quantity and the ess inf may well be the right
object; whether a quantile could stand in for it is a theory question this script
does not touch.

Usage
-----
    python margin_power.py --self-test
    python margin_power.py --bundle results/features_v4_waterbirds_dinov2_train.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import FeatureBundle
from group_margins import C_LADDER, TIE_TOL, _geometric_margins, _svc_ladder
from progress import pbar

DEFAULT_ORACLE_FRACS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_DELTAS = (0.05, 0.20, 0.80, 2.00)
DEFAULT_QS = (1.0, 0.999, 0.99, 0.95)
QUANTILES = (0.01, 0.05)


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------
def fit_w(X, y, ladder, max_iter, tol):
    """The separator at the top of the ladder that separates with no violations."""
    g0 = np.zeros(len(y), dtype=int)
    best = None
    for r in _svc_ladder(X, y, ladder, max_iter, tol):
        if "w" not in r:
            continue
        if _geometric_margins(X, y, g0, r["w"])["n_violations"] == 0:
            best = r["w"]
    if best is None:
        raise SystemExit("no C in the ladder separated this bundle cleanly; "
                         "extend --C before calibrating anything on it")
    return np.asarray(best, dtype=np.float64)


def margins_under(X, y, w):
    """y_i <w, x_i> / ||w||, the geometric margin of every point."""
    return (np.asarray(y, dtype=np.float64) * (X @ w)) / float(np.linalg.norm(w))


def plant_translation(X, y, w, rows, delta):
    """Add exactly `delta` to the margin of every point in `rows`, under `w`.

    margin_i = y_i <w, x_i> / ||w||. With x_i -> x_i + delta * y_i * w / ||w||,
    the change is delta * y_i^2 * ||w||^2 / ||w||^2 = delta, since y_i^2 = 1.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    u = np.asarray(w, dtype=np.float64) / float(np.linalg.norm(w))
    X[rows] += float(delta) * np.asarray(y, dtype=np.float64)[rows, None] * u[None, :]
    return X


def pinned_bundle(n, d, m0, m1, frac_g1, seed):
    """`validate_group_margins.make_bundle`'s geometry, at an arbitrary n and d.

    All signal on e_1; the orthogonal perturbation is mirrored so any tilt out of
    e_1 is punished symmetrically and the optimum is pinned there. Group margins
    are then exactly m0 and m1.
    """
    rng = np.random.default_rng(seed)
    n1 = max(4, int(round(frac_g1 * n)))
    n0 = max(4, n - n1)
    X, y, g = [], [], []
    for gg, (m, nn) in enumerate([(m0, n0), (m1, n1)]):
        k = max(1, nn // 4)
        for _ in range(k):
            v = rng.normal(size=d)
            v[0] = 0.0
            v *= 0.5 / max(np.linalg.norm(v), 1e-12)
            for sgn in (+1.0, -1.0):
                for yy in (+1.0, -1.0):
                    x = np.zeros(d)
                    x[0] = yy * m
                    X.append(x + sgn * v)
                    y.append(yy)
                    g.append(gg)
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64),
            np.asarray(g, dtype=int))


def measure(X, y, g, ladder, max_iter, tol) -> dict:
    """Exactly what `group_margins.py` would report for this (X, y, g)."""
    best = None
    for r in _svc_ladder(X, y, ladder, max_iter, tol):
        if "w" not in r:
            continue
        m = _geometric_margins(X, y, g, r["w"])
        if m["n_violations"] == 0:
            best = (r, m)
    if best is None:
        return {"separable": False}
    r, m = best
    m0, m1 = m["margin_g0"], m["margin_g1"]
    lo = min(m0, m1)
    rel = abs(m1 - m0) / max(lo, 1e-300)
    out = {"separable": True, "C_used": float(r["C"]), "margin": float(m["margin"]),
           "margin_g0": float(m0), "margin_g1": float(m1),
           "ratio": float(max(m0, m1) / lo), "asymmetry_rel": float(rel),
           "tie": bool(rel <= TIE_TOL),
           "larger": None if rel <= TIE_TOL else int(1 if m1 > m0 else 0)}
    u = margins_under(X, y, r["w"])
    for qq in QUANTILES:
        a = float(np.quantile(u[g == 0], qq))
        b = float(np.quantile(u[g == 1], qq))
        lo_q = min(a, b)
        out[f"q{int(qq * 100):02d}_ratio"] = (float(max(a, b) / lo_q)
                                              if lo_q > 0 else float("nan"))
    for gg in (0, 1):
        sel = g == gg
        if sel.any():
            out[f"n_at_margin_g{gg}"] = int((u[sel] <= u[sel].min() * 1.01).sum())
    return out


def target_ratio(u, g) -> float:
    """The ratio under the BASELINE separator, before re-solving."""
    a, b = u[g == 0].min(), u[g == 1].min()
    lo = min(a, b)
    return float(max(a, b) / lo) if lo > 0 else float("nan")


# ----------------------------------------------------------------------------
# The sweeps
# ----------------------------------------------------------------------------
def run(X, y, w, args) -> list[dict]:
    n = X.shape[0]
    u0 = margins_under(X, y, w)
    order = np.argsort(-u0, kind="mergesort")          # largest margin first
    ladder, mi, tol = args.C, args.max_iter, args.tol
    base = float(u0.min())
    rows = []

    jobs = []
    for f in args.oracle_fracs:
        jobs.append(("oracle", dict(frac=float(f))))
    for m1 in args.pinned_ratios:
        jobs.append(("pinned", dict(m1=float(m1))))
    for seed in args.seeds:
        for d_ in args.deltas:
            jobs.append(("translate", dict(delta=float(d_), seed=int(seed))))
        for q in args.qs:
            jobs.append(("heavy_tail", dict(q=float(q), seed=int(seed),
                                            frac=float(args.heavy_tail_frac))))

    for sweep, p in pbar(jobs, desc="configs", unit="cfg"):
        rec = {"sweep": sweep, **p}
        if sweep == "oracle":
            k = max(1, int(round(p["frac"] * n)))
            g = np.zeros(n, dtype=int)
            g[order[:k]] = 1                            # the k easiest points
            rec["n_g1"] = int(k)
            rec["target_ratio"] = target_ratio(u0, g)
            rec.update(measure(X, y, g, ladder, mi, tol))

        elif sweep == "pinned":
            Xp, yp, gp = pinned_bundle(n, X.shape[1], 1.0, p["m1"], 0.25, seed=0)
            rec["n_g1"] = int((gp == 1).sum())
            rec["target_ratio"] = float(p["m1"])
            rec.update(measure(Xp, yp, gp, ladder, mi, tol))

        elif sweep == "translate":
            rng = np.random.default_rng(p["seed"])
            sel = rng.permutation(n)[: n // 2]
            g = np.zeros(n, dtype=int)
            g[sel] = 1
            Xp = plant_translation(X, y, w, sel, p["delta"] * base)
            rec["n_g1"] = int(sel.size)
            rec["target_ratio"] = target_ratio(margins_under(Xp, y, w), g)
            rec.update(measure(Xp, y, g, ladder, mi, tol))

        else:  # heavy_tail
            k = max(1, int(round(p["frac"] * n)))
            rng = np.random.default_rng(p["seed"])
            easy = order[:k]
            n_keep = int(round(p["q"] * k))
            pool = np.setdiff1d(np.arange(n), easy, assume_unique=False)
            swap = rng.choice(pool, size=max(0, k - n_keep), replace=False)
            members = np.concatenate([easy[:n_keep], swap])
            g = np.zeros(n, dtype=int)
            g[members] = 1
            rec["n_g1"] = int(members.size)
            rec["n_swapped"] = int(swap.size)
            rec["target_ratio"] = target_ratio(u0, g)
            rec.update(measure(X, y, g, ladder, mi, tol))

        rows.append(rec)
    return rows


def verdict(rows) -> dict:
    """The one question the run exists to answer."""
    orc = [r for r in rows if r["sweep"] == "oracle" and r.get("separable")]
    hits = [r for r in orc if not r["tie"]]
    out = {"oracle_n": len(orc), "oracle_non_ties": len(hits)}
    if hits:
        best = max(hits, key=lambda r: r["ratio"])
        out["answer"] = "YES"
        out["best_ratio"] = best["ratio"]
        out["best_frac"] = best["frac"]
        out["headline"] = (
            f"The oracle partition reaches a ratio of {best['ratio']:.4f} at "
            f"frac={best['frac']:g}. A non-tie is attainable on these features, so a "
            "real partition is worth trying; compare its ratio against this ceiling.")
    else:
        q = max((r.get("q05_ratio", 1.0) for r in orc), default=float("nan"))
        out["answer"] = "NO"
        out["best_q05_ratio"] = float(q)
        out["headline"] = (
            "NO partition of these features produced a ratio other than 1, including "
            "the oracle partition built from the easiest points available. The "
            "bird-size re-partition cannot do better than a partition chosen with "
            "full knowledge of the margins, so it will tie too. The ceiling on the "
            f"5th-percentile ratio was {q:.4f}: if that is well above 1, an asymmetry "
            "IS present in the features and the ess inf is what hides it.")
    return out


def to_markdown(rows, ver, head) -> str:
    L = ["# Can the per-group margin ratio be anything but 1 on these features?", "",
         "An instrument calibration, not a result. See the module docstring for why",
         "`validate_group_margins.py`'s 1.6000 does not answer this.", "",
         f"- bundle: `{head['bundle']}`   n = {head['n']}, d = {head['d']}",
         f"- baseline margin: **{head['base_margin']:.4f}**",
         f"- tie tolerance `{TIE_TOL:g}`, C ladder {head['ladder']}", "",
         "## Answer", "", f"**{ver['answer']}.** {ver['headline']}", "",
         "## Every configuration", "",
         "`target` is the ratio under the BASELINE separator, before re-solving.",
         "`ratio` is what `group_margins.py` reports after re-solving. The gap",
         "between them is how much the optimiser undoes.", "",
         "| sweep | knob | n_g1 | target | ratio | verdict | q01 ratio | q05 ratio | "
         "SV g0 | SV g1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        knob = (f"frac={r['frac']:g}" if r["sweep"] == "oracle" else
                f"m1={r['m1']:g}" if r["sweep"] == "pinned" else
                f"delta={r['delta']:g} seed={r['seed']}" if r["sweep"] == "translate" else
                f"q={r['q']:g} seed={r['seed']}")
        if not r.get("separable"):
            L.append(f"| {r['sweep']} | {knob} | {r.get('n_g1', '-')} | "
                     f"{r.get('target_ratio', float('nan')):.4f} | NOT SEPARABLE "
                     f"| - | - | - | - | - |")
            continue
        L.append(
            f"| {r['sweep']} | {knob} | {r['n_g1']} | {r['target_ratio']:.4f} | "
            f"{r['ratio']:.6f} | {'tie' if r['tie'] else 'g=' + str(r['larger'])} | "
            f"{r.get('q01_ratio', float('nan')):.4f} | "
            f"{r.get('q05_ratio', float('nan')):.4f} | "
            f"{r.get('n_at_margin_g0', -1)} | {r.get('n_at_margin_g1', -1)} |")
    L += ["", "## How to read this", "",
          "- `pinned` must recover its `m1`. If it does not, the instrument is broken",
          "  at this n and d and nothing else on the page can be read.",
          "- `oracle` is the ceiling. Every row a tie means no group variable can",
          "  produce a non-tie on this representation, bird size included.",
          "- `translate` shows how much a naive planted gap is undone by re-solving.",
          "  A large `target` next to a `ratio` of 1 is the optimiser rotating away",
          "  from the asymmetry -- the same mechanism that makes real partitions tie.",
          "- `heavy_tail` shows what one hard member does to a group's ess inf.",
          "- If the `q01`/`q05` ratios move while `ratio` stays at 1, the asymmetry is",
          "  real and the ess inf is hiding it. That is a question for the theory, not",
          "  for this code.", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test() -> int:
    print("margin_power self-test")
    ok = True
    rng = np.random.default_rng(0)
    n, d = 400, 16
    X = rng.normal(size=(n, d))
    y = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    w = rng.normal(size=d)
    # Push far enough that w itself separates: the noise projected on w is roughly
    # standard normal, and min over n=400 draws reaches about -3.2.
    X += 6.0 * y[:, None] * (w / np.linalg.norm(w))[None, :]

    u0 = margins_under(X, y, w)
    if u0.min() <= 0:
        print(f"  FAIL self-test fixture is not separable by w (min {u0.min():.3f})")
        return 1
    rows_ = np.arange(0, n, 3)
    u1 = margins_under(plant_translation(X, y, w, rows_, 0.7), y, w)
    if np.abs(u1[rows_] - u0[rows_] - 0.7).max() > 1e-9:
        print("  FAIL translation does not add exactly delta"); ok = False
    else:
        print("  ok   translation adds exactly delta to the chosen margins")
    other = np.setdiff1d(np.arange(n), rows_)
    if np.abs(u1[other] - u0[other]).max() > 1e-12:
        print("  FAIL translation moved points it should not have"); ok = False
    else:
        print("  ok   every other point is untouched")
    if np.abs(plant_translation(X, y, w, rows_, 0.0) - X).max() != 0.0:
        print("  FAIL delta = 0 is not the identity"); ok = False
    else:
        print("  ok   delta = 0 is exactly the identity")

    # The pinned construction MUST be recovered -- this is the positive control
    # the whole script leans on.
    Xp, yp, gp = pinned_bundle(600, 32, 1.0, 1.6, 0.25, seed=3)
    res = measure(Xp, yp, gp, (1e4, 1e5, 1e6), 200000, 1e-8)
    if not res.get("separable"):
        print("  FAIL pinned control is not separable"); ok = False
    elif abs(res["ratio"] - 1.6) > 0.05:
        print(f"  FAIL pinned control ratio {res['ratio']:.4f} != 1.6"); ok = False
    elif res["larger"] != 1:
        print(f"  FAIL pinned control larger group is {res['larger']}, not 1"); ok = False
    else:
        print(f"  ok   pinned control recovers its planted ratio ({res['ratio']:.4f})")

    # An unplanted random split must be a tie.
    g = np.zeros(n, dtype=int); g[: n // 2] = 1
    r0 = measure(X, y, g, (1e4, 1e5, 1e6), 200000, 1e-8)
    if r0.get("separable") and not r0["tie"]:
        print(f"  FAIL random split not a tie (ratio {r0['ratio']:.6f})"); ok = False
    else:
        print("  ok   an unplanted random split is a tie")

    # Oracle partition: target must exceed 1 by construction (it is built from the
    # largest margins). What re-solving does to it is the measurement, not a test.
    order = np.argsort(-u0, kind="mergesort")
    g_or = np.zeros(n, dtype=int); g_or[order[: n // 4]] = 1
    t = target_ratio(u0, g_or)
    if not (t > 1.0):
        print(f"  FAIL oracle target ratio {t:.4f} is not above 1"); ok = False
    else:
        r_or = measure(X, y, g_or, (1e4, 1e5, 1e6), 200000, 1e-8)
        print(f"  ok   oracle target {t:.3f} before re-solving; after re-solving "
              f"{r_or.get('ratio', float('nan')):.6f} "
              f"({'tie' if r_or.get('tie') else 'non-tie'}) -- informational")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bundle")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="v5_margin_power")
    ap.add_argument("--C", nargs="+", type=float, default=list(C_LADDER))
    ap.add_argument("--max-iter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--oracle-fracs", nargs="+", type=float,
                    default=list(DEFAULT_ORACLE_FRACS))
    ap.add_argument("--pinned-ratios", nargs="+", type=float, default=[1.05, 1.60])
    ap.add_argument("--deltas", nargs="+", type=float, default=list(DEFAULT_DELTAS))
    ap.add_argument("--qs", nargs="+", type=float, default=list(DEFAULT_QS))
    ap.add_argument("--heavy-tail-frac", type=float, default=0.25)
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.bundle:
        raise SystemExit("--bundle is required (or use --self-test)")

    fb = FeatureBundle.load(args.bundle)
    X = np.asarray(fb.phi, dtype=np.float64)
    y = np.asarray(fb.y, dtype=np.float64)
    n, d = X.shape
    print(f"{os.path.basename(args.bundle)}  n={n}  d={d}")

    print("fitting the baseline separator ...")
    w = fit_w(X, y, args.C, args.max_iter, args.tol)
    base = float(margins_under(X, y, w).min())
    print(f"baseline margin {base:.4f}")

    rows = run(X, y, w, args)
    ver = verdict(rows)
    head = {"bundle": os.path.basename(args.bundle), "n": n, "d": d,
            "base_margin": base, "ladder": args.C, "tie_tol": TIE_TOL}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as fh:
        json.dump({"head": head, "verdict": ver, "rows": rows}, fh, indent=1)
    txt = to_markdown(rows, ver, head)
    with open(os.path.join(args.out_dir, f"{args.tag}.md"), "w") as fh:
        fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
