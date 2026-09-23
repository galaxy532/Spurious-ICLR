"""Integration check for session 5: does a re-partitioned bundle flow through?

The three new modules each have their own self-test (`python cub_masks.py
--self-test`, and the same for `regroup.py` and `margin_power.py`). Those check
the pieces. This checks the JOIN between them, which is where a silent error
would actually live:

1. A bundle written by `regroup.py` LOADS in `group_margins.py` unchanged, and
   the group counts it reports are the ones `regroup.py` intended. If
   `group_margins.py` needed editing to read these bundles, the instrument that
   measured the session-4 ties would no longer be the same instrument.

2. `regroup.py` replaces ONLY `g`. `phi` and `y` must come through bit-identical
   whenever no row is dropped, because the whole argument for re-partitioning
   over re-degrading is that the features are untouched.

3. The row-order guard fires on a shuffled bundle. `extract_features.py` shuffles
   for the rwg and gdro recipes (`shuffle = mode != "erm"`), and a shuffled
   bundle joined against metadata order gives a group vector that is simply
   wrong, with no symptom except a plausible-looking number.

4. A planted asymmetry reaches `group_margins.py` intact: build a bundle whose
   two groups have known different margins, re-partition nothing, and confirm the
   reported ratio is the planted one. This ties the session-5 path back to
   `validate_group_margins.py`'s own guarantee.

No data, no download, no GPU. A few seconds.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from common import FeatureBundle
from group_margins import C_LADDER, analyse_bundle
from margin_power import pinned_bundle
from progress import pbar
from regroup import check_alignment, group_from_fraction, write_bundle

LADDER = (1e4, 1e5, 1e6)


def _mk_source(n=400, d=24, seed=0):
    """A small stand-in for a real Waterbirds bundle, in metadata order."""
    rng = np.random.default_rng(seed)
    y01 = rng.integers(0, 2, n)
    place = rng.integers(0, 2, n)
    y = (2 * y01 - 1).astype(int)
    w = rng.normal(size=d)
    phi = rng.normal(size=(n, d))
    phi += 6.0 * y[:, None] * (w / np.linalg.norm(w))[None, :]
    fb = FeatureBundle(phi=phi, y=y, g=(place != y01).astype(int),
                       idx_r=np.array([], int), idx_s=np.array([], int),
                       place=place, meta={"standardized": True, "split": "train",
                                          "backbone": "synthetic"})
    return fb, y01, place


def main() -> int:
    print("session 5 integration check")
    fails = []
    checks = [
        "regrouped bundle loads in group_margins unchanged",
        "phi and y pass through untouched when no row is dropped",
        "row-order guard fires on a shuffled bundle",
        "a planted asymmetry survives the session-5 path",
    ]

    with tempfile.TemporaryDirectory() as td:
        fb, y01, place = _mk_source()
        n = fb.y.size
        rng = np.random.default_rng(1)
        frac = rng.random(n)

        bar = pbar(checks, desc="checks", unit="check")
        it = iter(bar)

        # ---- 1 ----------------------------------------------------------
        next(it)
        g_new, keep, _ = group_from_fraction(frac, "median")
        p = os.path.join(td, "regrouped.npz")
        info = write_bundle(fb, keep, g_new, p, {"regroup_rule": "median",
                                                 "regroup_role": "treatment"})
        res = analyse_bundle(p, LADDER, 200000, 1e-8, 60.0, run_lp=False)
        if not res.get("separable_by_svc"):
            fails.append("regrouped bundle came back non-separable")
        elif (res["n_g0"], res["n_g1"]) != (info["n_g0"], info["n_g1"]):
            fails.append(f"group counts disagree: group_margins {res['n_g0']}/"
                         f"{res['n_g1']} vs regroup {info['n_g0']}/{info['n_g1']}")
        elif not res["standardized"]:
            fails.append("group_margins does not see the bundle as standardised")

        # ---- 2 ----------------------------------------------------------
        next(it)
        back = FeatureBundle.load(p)
        if not np.array_equal(back.phi, fb.phi):
            fails.append("phi was modified by a no-drop re-partition")
        if not np.array_equal(back.y, fb.y):
            fails.append("y was modified by a no-drop re-partition")
        if np.array_equal(back.g, fb.g):
            fails.append("g was NOT changed by the re-partition")

        # ---- 3 ----------------------------------------------------------
        next(it)
        perm = np.random.default_rng(2).permutation(n)
        shuf = FeatureBundle(phi=fb.phi[perm], y=fb.y[perm], g=fb.g[perm],
                             idx_r=fb.idx_r, idx_s=fb.idx_s, place=fb.place[perm],
                             meta=dict(fb.meta))
        if check_alignment(shuf, y01, place)["ok"]:
            fails.append("row-order guard did NOT fire on a shuffled bundle")
        if not check_alignment(fb, y01, place)["ok"]:
            fails.append("row-order guard fired on a correctly ordered bundle")

        # ---- 4 ----------------------------------------------------------
        next(it)
        Xp, yp, gp = pinned_bundle(600, 32, 1.0, 1.6, 0.25, seed=5)
        pb = os.path.join(td, "planted.npz")
        FeatureBundle(phi=Xp, y=yp.astype(int), g=gp,
                      idx_r=np.array([], int), idx_s=np.array([], int),
                      place=None, meta={"standardized": True, "split": "synthetic",
                                        "backbone": "synthetic"}).save(pb)
        rp = analyse_bundle(pb, LADDER, 200000, 1e-8, 60.0, run_lp=False)
        if not rp.get("separable_by_svc"):
            fails.append("planted bundle came back non-separable")
        elif abs(rp["gamma_min_theorem"] - 1.6) > 0.05:
            fails.append(f"planted ratio 1.6 came back as "
                         f"{rp['gamma_min_theorem']:.4f}")
        elif rp["larger_margin_group"] != 1:
            fails.append(f"planted larger group is g={rp['larger_margin_group']}, not 1")

        for _ in it:
            pass

    print()
    for c in checks:
        print(f"  {c}")
    if fails:
        print("\nINTEGRATION FAILED")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nINTEGRATION OK -- a re-partitioned bundle is read by the unmodified "
          "group_margins.py, the features are untouched, a shuffled bundle is "
          "refused, and a planted ratio survives the path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
