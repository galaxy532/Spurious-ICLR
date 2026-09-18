"""Validation for `group_margins.py` against constructions with KNOWN margins.

Nothing in `group_margins.py` produces a number that can be sanity-checked by
eye: a per-group margin of 1.37 is not obviously right or obviously wrong. So it
is checked here the same way `validate_screen.py` checks the Rebuttals_2 screen
and `Estimator_Validation/` checks `common.py` -- on data where the answer is
known by construction.

THE CONSTRUCTION
================
Pick `w* = e_1`. For a group with intended margin `m`, place every point at

    x = y * m * e_1  +  v,        v orthogonal to e_1,

so that `y (w* . x) = m` exactly and `||w*|| = 1`. The orthogonal parts are
MIRRORED: every `v` appears together with `-v`, at both labels. That symmetry is
what pins the maximum-margin direction to `e_1` -- without it the solver is free
to tilt `w` into the orthogonal block and buy margin there, and the "known"
answer would not be known at all.

Two groups with different `m` then give a set whose per-group hard margins are
exactly `m_0` and `m_1`, and whose ratio is exactly `m_1 / m_0`.

FOUR CASES
==========
1. `asymmetric`  m_0 = 1.0, m_1 = 1.6  -- the alpha > 1 shape. Must recover both
   margins and the ratio 1.6, and must report g=1 as the larger-margin group.
2. `symmetric`   m_0 = m_1 = 1.0 -- the shape every real bundle has shown so far.
   Must return a ratio of 1.000 AND must report a TIE rather than ranking the two
   groups, since at that point the ranking is floating-point noise. A method that
   manufactures asymmetry out of unequal group SIZES also fails here, because the
   two groups have different n.
3. `reversed`    m_0 = 1.5, m_1 = 1.0 -- the mirror case. Must report g=0.
4. `nonseparable` -- overlapping labels. Must NOT report a margin.

Run (a few seconds, no data, no GPU):

    python validate_group_margins.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from common import FeatureBundle
from group_margins import analyse_bundle
from progress import pbar

REL_TOL = 0.02          # 2% on a soft-margin solver approaching a hard margin


def make_bundle(m0: float, m1: float, n0: int, n1: int, d: int, seed: int,
                separable: bool = True) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for gg, (m, n) in enumerate([(m0, n0), (m1, n1)]):
        k = max(1, n // 4)                      # 4 points per orthogonal direction
        for _ in range(k):
            v = rng.normal(size=d)
            v[0] = 0.0                          # orthogonal to e_1 by construction
            v *= 0.5 / max(np.linalg.norm(v), 1e-12)
            for sgn in (+1.0, -1.0):            # mirror the orthogonal part
                for yy in (+1.0, -1.0):
                    x = np.zeros(d)
                    x[0] = yy * m
                    X.append(x + sgn * v)
                    y.append(yy)
                    g.append(gg)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    g = np.asarray(g, dtype=int)
    if not separable:
        flip = rng.choice(len(y), size=max(2, len(y) // 20), replace=False)
        y[flip] *= -1
    return FeatureBundle(phi=X, y=y, g=g,
                         idx_r=np.array([], int), idx_s=np.array([], int),
                         meta={"standardized": True, "split": "synthetic",
                               "backbone": "synthetic", "degraded": False})


def _check(name, res, want_m0, want_m1, want_larger, fails):
    if not res.get("separable_by_svc"):
        fails.append(f"{name}: solver found no separator, but the data is separable")
        return
    for key, want in (("margin_g0", want_m0), ("margin_g1", want_m1)):
        got = res[key]
        if not np.isfinite(got) or abs(got - want) / want > REL_TOL:
            fails.append(f"{name}: {key} = {got:.4f}, expected {want:.4f} "
                         f"(rel {abs(got - want) / want:.3f} > {REL_TOL})")
    want_ratio = want_m1 / want_m0
    got_ratio = res["gamma_ratio_g1_over_g0"]
    if abs(got_ratio - want_ratio) / want_ratio > REL_TOL:
        fails.append(f"{name}: ratio = {got_ratio:.4f}, expected {want_ratio:.4f}")
    if res["larger_margin_group"] != want_larger:
        fails.append(f"{name}: larger group reported {res['larger_margin_group']}, "
                     f"expected {want_larger} (None means the two margins must be "
                     f"called equal, not ranked by floating-point noise)")
    if res.get("plateau_rel") is not None and res["plateau_rel"] > 1e-2:
        fails.append(f"{name}: margin had not plateaued (plateau_rel "
                     f"{res['plateau_rel']:.1e}) -- the ladder is too short")


def main() -> int:
    cases = [
        # name,         m0,  m1,   n0,   n1,  separable, want_larger
        ("asymmetric", 1.0, 1.6,  800,  200, True,  1),
        ("symmetric",  1.0, 1.0,  800,  200, True,  None),  # tie -> must report a TIE
        ("reversed",   1.5, 1.0,  400,  600, True,  0),
        ("nonseparable", 1.0, 1.2, 600, 400, False, None),
    ]
    fails, table = [], []
    tmp = tempfile.mkdtemp(prefix="gm_validate_")
    for name, m0, m1, n0, n1, sep, want_larger in pbar(cases, desc="cases", unit="case"):
        fb = make_bundle(m0, m1, n0, n1, d=64, seed=abs(hash(name)) % (2**31), separable=sep)
        p = os.path.join(tmp, f"{name}.npz")
        fb.save(p)
        res = analyse_bundle(p, (1e2, 1e3, 1e4, 1e5, 1e6),
                             max_iter=200000, tol=1e-8, lp_time_limit=60.0, run_lp=False)
        if sep:
            _check(name, res, m0, m1, want_larger, fails)
            table.append((name, m0, m1, res.get("margin_g0"), res.get("margin_g1"),
                          res.get("gamma_ratio_g1_over_g0"), res.get("plateau_rel"),
                          res.get("larger_margin_group")))
        else:
            if res.get("separable_by_svc"):
                fails.append(f"{name}: reported a margin on data with flipped labels")
            table.append((name, m0, m1, None, None, None, None, None))

    print("\n| case | m0 | m1 | margin g0 | margin g1 | ratio | plateau | larger |")
    print("|---|---|---|---|---|---|---|---|")
    for row in table:
        f = lambda v, fmt="{:.4f}": (fmt.format(v) if isinstance(v, float) else "-")
        print(f"| {row[0]} | {row[1]} | {row[2]} | {f(row[3])} | {f(row[4])} | "
              f"{f(row[5])} | {f(row[6], '{:.1e}')} | "
              f"{('g=' + str(row[7])) if row[7] is not None else '-'} |")

    if fails:
        print("\nVALIDATION FAILED")
        for x in fails:
            print("  -", x)
        return 1
    print("\nVALIDATION OK -- per-group margins and their ratio are recovered to "
          f"within {REL_TOL:.0%}, a tie is reported as a tie, and non-separable data "
          "is refused.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
