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

5. THE REAL COMMAND LINE runs, on a miniature dataset: `cub_masks.py` then
   `regroup.py` then `group_margins.py`, as subprocesses, with the flags
   `run_session5.sh` actually passes. Both `cub_masks.py` and `regroup.py` do
   some of their imports inside `main()`, so an import that does not exist is
   invisible to every in-process check -- it surfaces only when the step runs for
   real, forty minutes into a job. One did: the Waterbirds loader is
   `load_metadata`, not `load_raw`. This check is what closes that hole.

No data, no download, no GPU. A few seconds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

from common import FeatureBundle
from group_margins import analyse_bundle
from margin_power import pinned_bundle
from progress import pbar
from regroup import check_alignment, group_from_fraction, write_bundle

LADDER = (1e4, 1e5, 1e6)
HERE = os.path.dirname(os.path.abspath(__file__))


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


def _run(args, cwd, env):
    """Run one of this repo's scripts from `cwd`, so relative results/ paths work."""
    script, rest = os.path.join(HERE, args[0]), list(args[1:])
    p = subprocess.run([sys.executable, script, *rest], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def _cli_chain() -> list[str]:
    """cub_masks -> regroup -> group_margins, as subprocesses with real flags."""
    from cub_masks import _mini_dataset

    bad = []
    with tempfile.TemporaryDirectory() as td:
        n_per_class = 40
        seg = _mini_dataset(td, n_per_class=n_per_class)
        env = dict(os.environ, SPURIOUS_DATA_ROOT=td, NO_PROGRESS="1",
                   PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
        work = os.path.join(td, "work")
        os.makedirs(os.path.join(work, "results"), exist_ok=True)

        rc, out = _run(["cub_masks.py", "--no-download", "--seg-root", seg,
                        "--tag", "v5_bird_fraction"], work, env)
        if rc != 0:
            return [f"cub_masks.py CLI exited {rc}: {out}"]

        # A bundle matching split 0 (the mini 'train'), in metadata order.
        import pandas as pd
        d = pd.read_csv(os.path.join(work, "results", "v5_bird_fraction.csv"))
        d = d[d["split"] == 0].reset_index(drop=True)
        n, dim = len(d), 16
        rng = np.random.default_rng(7)
        y = (2 * d["y"].to_numpy().astype(int) - 1)
        w = rng.normal(size=dim)
        phi = rng.normal(size=(n, dim))
        phi += 6.0 * y[:, None] * (w / np.linalg.norm(w))[None, :]
        FeatureBundle(phi=phi, y=y, g=d["g_orig"].to_numpy().astype(int),
                      idx_r=np.array([], int), idx_s=np.array([], int),
                      place=d["place"].to_numpy().astype(int),
                      meta={"standardized": True, "split": "train",
                            "backbone": "mini"}
                      ).save(os.path.join(work, "results",
                                          "features_v4_mini_dinov2_train.npz"))

        for rule in ("median", "tercile"):
            rc, out = _run(["regroup.py",
                            "--bundle", "results/features_v4_mini_dinov2_train.npz",
                            "--fractions", "results/v5_bird_fraction.csv",
                            "--rule", rule,
                            # The mini labels are random, so the leakage gate is
                            # meaningless here; it is exercised by its own unit check.
                            "--min-auc-dev", "0.5"], work, env)
            if rc != 0:
                bad.append(f"regroup.py --rule {rule} CLI exited {rc}: {out}")
                continue
            rc, out = _run(["group_margins.py", "--no-lp",
                            "--tag", f"v5_group_margins_{rule}",
                            "--bundles", f"results/features_v5_*_bf{rule}*.npz"],
                           work, env)
            if rc != 0:
                bad.append(f"group_margins.py after {rule} exited {rc}: {out}")
                continue
            md = os.path.join(work, "results", f"v5_group_margins_{rule}.md")
            if not os.path.exists(md):
                bad.append(f"group_margins.py wrote no table for {rule}")
            elif rule == "tercile":
                txt = open(md).read()
                if "_control" not in txt:
                    bad.append("the tercile run produced no matched control row")
    return bad


def main() -> int:
    print("session 5 integration check")
    fails = []
    checks = [
        "regrouped bundle loads in group_margins unchanged",
        "phi and y pass through untouched when no row is dropped",
        "row-order guard fires on a shuffled bundle",
        "a planted asymmetry survives the session-5 path",
        "the real command line runs end to end on a mini dataset",
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

        # ---- 5 ----------------------------------------------------------
        next(it)
        fails.extend(_cli_chain())

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
