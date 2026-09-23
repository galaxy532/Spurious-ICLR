"""Re-partition a Waterbirds feature bundle by bird size instead of background.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO
====================================================
Reads an existing feature bundle, replaces its group vector `g`, and writes a NEW
bundle. `phi` and `y` are copied through untouched. No images are read, no
backbone is run, no feature is modified -- session 4 showed that modifying images
introduces confounds (the degradation destroyed the bird but preserved the
background, so the spurious-aligned group stayed separable through the shortcut).
Re-partitioning has no such failure mode.

The existing `group_margins.py` is then run on the new bundle, unchanged. That is
the point of writing a new bundle rather than adding a flag: the instrument that
measured the session-4 ties is the same instrument, byte for byte.

THE GROUP CONVENTION, AND WHY LARGE BIRDS ARE g == 1
====================================================
`FeatureBundle` documents `g == 0` as G_maj and `g == 1` as G_min, and Theorem
5.3's WLOG (`gamma~_maj = 1`, `gamma~_min >= 1`) labels G_min as the group with
the LARGER margin. Group size is irrelevant to which branch you are in -- it only
decides which group's error carries the `1/eps`.

A large bird in frame is easier to classify from the core feature, so it should
have the larger margin. Assigning `g == 1` to large birds therefore makes the
script's "min" the theorem's "min", and the reports read straight:
`gamma(g=1) > gamma(g=0)` is the direction the branch predicts. `--invert` flips
it if you want the check run the other way round.

THE MATCHED CONTROL, AND WHY THE TERCILE RULE NEEDS ONE
=======================================================
`median` keeps every row: the features are bit-identical to the bundle already
measured, only `g` differs. Its control is therefore the original bundle, whose
tie is already on record.

`tercile` DROPS the middle third to widen the gap. That changes the row set, and
`common.standardize` centres each coordinate -- with no intercept in the margin
problem (`fit_intercept=False`, matching `long_horizon.py`), a change in the mean
moves the solution. So a tercile bundle is NOT comparable to the original, and a
ratio measured on it would be uninterpretable on its own.

This script therefore emits a MATCHED CONTROL alongside every row-dropping rule:
the same surviving rows, re-standardised identically, carrying the ORIGINAL
`g = 1[place != y]`. If the control ties and the treatment does not, the
difference is the partition and nothing else.

Usage
-----
    python regroup.py --self-test
    python regroup.py --bundle results/features_v4_waterbirds_dinov2_train.npz \\
                      --fractions results/v5_bird_fraction.csv --rule median
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import FeatureBundle, standardize
from progress import pbar

RULES = ("median", "tercile")


# ----------------------------------------------------------------------------
# Building the new group vector
# ----------------------------------------------------------------------------
def group_from_fraction(frac: np.ndarray, rule: str, invert: bool = False):
    """Return (g, keep) -- the new group vector and which rows survive.

    g == 1 is the LARGE-bird (easier, larger-margin) group unless `invert`.
    `keep` is all-True except for `tercile`, which drops the middle third.
    """
    frac = np.asarray(frac, dtype=float)
    if rule == "median":
        cut = float(np.median(frac))
        big = frac > cut
        keep = np.ones(frac.size, dtype=bool)
        cuts = {"median": cut}
    elif rule == "tercile":
        lo = float(np.quantile(frac, 1 / 3))
        hi = float(np.quantile(frac, 2 / 3))
        big = frac >= hi
        small = frac <= lo
        keep = big | small
        cuts = {"q33": lo, "q67": hi}
    else:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    g = np.where(big, 1, 0).astype(int)
    if invert:
        g = 1 - g
    return g, keep, cuts


# ----------------------------------------------------------------------------
# The order check. Load-bearing: without it the join is silent nonsense.
# ----------------------------------------------------------------------------
def check_alignment(fb: FeatureBundle, y_meta: np.ndarray, place_meta: np.ndarray) -> dict:
    """Verify the bundle's rows are in metadata order, element by element.

    `extract_features.py` shuffles for the rwg and gdro recipes (`shuffle = mode
    != "erm"`), and a shuffled bundle joined against metadata order would produce
    a group vector that is simply wrong -- with no symptom except a number that
    looks plausible. The bundle stores `y` and `place`, so this is checkable
    exactly rather than assumed.
    """
    y_b = (np.asarray(fb.y) > 0).astype(int)          # bundle y is +/-1
    out = {"n_bundle": int(y_b.size), "n_meta": int(np.asarray(y_meta).size)}
    if out["n_bundle"] != out["n_meta"]:
        out["ok"] = False
        out["why"] = (f"row counts differ: bundle {out['n_bundle']} vs metadata "
                      f"{out['n_meta']}. Wrong split, or a subsampled bundle.")
        return out
    if fb.place is None:
        out["ok"] = False
        out["why"] = "bundle carries no `place`; cannot verify row order"
        return out
    y_ok = bool(np.array_equal(y_b, np.asarray(y_meta).astype(int)))
    p_ok = bool(np.array_equal(np.asarray(fb.place).astype(int),
                               np.asarray(place_meta).astype(int)))
    out["y_matches"] = y_ok
    out["place_matches"] = p_ok
    out["ok"] = bool(y_ok and p_ok)
    if not out["ok"]:
        out["why"] = ("the bundle's (y, place) sequence does not match metadata order, "
                      "so its rows were shuffled at extraction. Re-extract this bundle "
                      "with `--shuffle no` before re-partitioning it.")
    return out


def write_bundle(fb: FeatureBundle, keep: np.ndarray, g_new: np.ndarray,
                 out_path: str, extra_meta: dict) -> dict:
    """Write a bundle with the surviving rows and the new group vector."""
    keep = np.asarray(keep, dtype=bool)
    phi = np.asarray(fb.phi, dtype=np.float64)[keep]
    dropped = int((~keep).sum())
    meta = dict(fb.meta or {})
    if dropped:
        # Rows were removed, so the standardisation of the parent no longer holds.
        # Re-standardise here; both the treatment and its matched control get the
        # identical treatment, so the comparison between them stays clean.
        phi = standardize(phi)
        meta["restandardized_after_row_drop"] = True
    meta.update(extra_meta)
    meta["standardized"] = True

    out = FeatureBundle(
        phi=phi,
        y=np.asarray(fb.y)[keep],
        g=np.asarray(g_new, dtype=int)[keep],
        idx_r=np.asarray(fb.idx_r, dtype=int),
        idx_s=np.asarray(fb.idx_s, dtype=int),
        place=None if fb.place is None else np.asarray(fb.place)[keep],
        meta=meta,
    )
    out.save(out_path)
    g_k = out.g
    return {"path": out_path, "n": int(g_k.size), "n_dropped": dropped,
            "n_g0": int((g_k == 0).sum()), "n_g1": int((g_k == 1).sum()),
            "eps": float(np.mean(g_k == 1))}


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test() -> int:
    import tempfile
    print("regroup self-test")
    ok = True

    frac = np.array([0.01, 0.02, 0.03, 0.10, 0.20, 0.30])

    g, keep, _ = group_from_fraction(frac, "median")
    if not (keep.all() and np.array_equal(g, [0, 0, 0, 1, 1, 1])):
        print(f"  FAIL median split: g={g} keep={keep}"); ok = False
    else:
        print("  ok   median keeps every row and puts the large birds in g=1")

    g, keep, _ = group_from_fraction(frac, "tercile")
    if not (np.array_equal(keep, [1, 1, 0, 0, 1, 1]) and
            np.array_equal(g[keep], [0, 0, 1, 1])):
        print(f"  FAIL tercile split: g={g} keep={keep}"); ok = False
    else:
        print("  ok   tercile drops the middle third, large birds in g=1")

    g_i, _, _ = group_from_fraction(frac, "median", invert=True)
    g_n, _, _ = group_from_fraction(frac, "median", invert=False)
    if not np.array_equal(g_i, 1 - g_n):
        print("  FAIL --invert did not flip the assignment"); ok = False
    else:
        print("  ok   --invert flips the assignment")

    # Order check: catches a permuted bundle.
    rng = np.random.default_rng(0)
    n = 40
    y01 = rng.integers(0, 2, n)
    place = rng.integers(0, 2, n)
    fb = FeatureBundle(phi=rng.normal(size=(n, 5)), y=(2 * y01 - 1),
                       g=(place != y01).astype(int),
                       idx_r=np.array([], int), idx_s=np.array([], int),
                       place=place, meta={"standardized": True})
    if not check_alignment(fb, y01, place)["ok"]:
        print("  FAIL aligned bundle rejected"); ok = False
    else:
        print("  ok   aligned bundle accepted")

    perm = rng.permutation(n)
    fb_shuf = FeatureBundle(phi=fb.phi[perm], y=fb.y[perm], g=fb.g[perm],
                            idx_r=fb.idx_r, idx_s=fb.idx_s, place=fb.place[perm],
                            meta={"standardized": True})
    if check_alignment(fb_shuf, y01, place)["ok"]:
        print("  FAIL shuffled bundle NOT caught"); ok = False
    else:
        print("  ok   shuffled bundle caught")

    # Round-trip: a written bundle reloads with the right g and rows.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.npz")
        keep = np.zeros(n, bool); keep[:20] = True
        gnew = np.zeros(n, int); gnew[:10] = 1
        info = write_bundle(fb, keep, gnew, p, {"regroup_rule": "test"})
        back = FeatureBundle.load(p)
        if back.g.size != 20 or int((back.g == 1).sum()) != 10:
            print(f"  FAIL round-trip: {info}"); ok = False
        elif not back.meta.get("restandardized_after_row_drop"):
            print("  FAIL row drop did not trigger re-standardisation"); ok = False
        elif abs(float(back.phi.mean())) > 1e-10:
            print("  FAIL re-standardised features are not centred"); ok = False
        else:
            print("  ok   round-trip keeps the right rows, g, and re-standardises")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bundle", help="feature bundle .npz to re-partition")
    ap.add_argument("--fractions", default="results/v5_bird_fraction.csv")
    ap.add_argument("--rule", default="median", choices=list(RULES))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--invert", action="store_true",
                    help="put the SMALL-bird group in g=1 instead")
    ap.add_argument("--out-prefix", default="features_v5")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--min-auc-dev", type=float, default=0.10,
                    help="refuse if |AUC(bird fraction -> y) - 0.5| exceeds this")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.bundle:
        raise SystemExit("--bundle is required (or use --self-test)")

    import pandas as pd
    from cub_masks import _leakage
    from datasets import DECLARATIONS

    df = pd.read_csv(args.fractions)
    code = DECLARATIONS["waterbirds"]["splits"][args.split]
    sub = df[df["split"] == code].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"no rows for split {args.split!r} in {args.fractions}")

    bad = int((~sub["size_match"].astype(bool)).sum() +
              (~np.isfinite(sub["bird_frac"].to_numpy(dtype=float))).sum())
    if bad:
        raise SystemExit(
            f"{bad} of {len(sub)} rows in this split have no usable mask. "
            "cub_masks.py must report alignment OK before re-partitioning; "
            "read results/v5_bird_fraction.md.")

    fb = FeatureBundle.load(args.bundle)
    align = check_alignment(fb, sub["y"].to_numpy(), sub["place"].to_numpy())
    if not align["ok"]:
        raise SystemExit(f"ROW ORDER CHECK FAILED: {align.get('why')}")
    print(f"row order check: ok ({align['n_bundle']} rows, y and place both match)")

    frac = sub["bird_frac"].to_numpy(dtype=float)
    lk = _leakage(frac, sub["y"].to_numpy().astype(int))
    dev = abs(lk["auc"] - 0.5)
    print(f"leakage of bird fraction into y: AUC {lk['auc']:.4f} (deviation {dev:.4f})")
    if dev > args.min_auc_dev:
        raise SystemExit(
            f"REFUSING: bird size predicts the label (AUC {lk['auc']:.4f}). The new "
            "group variable must track difficulty, not y. Rerun with a larger "
            "--min-auc-dev only if you intend to report that confound.")

    g_new, keep, cuts = group_from_fraction(frac, args.rule, args.invert)
    stem = os.path.basename(args.bundle).replace(".npz", "")
    tagged = f"{args.out_prefix}_{stem.split('_', 2)[-1]}_bf{args.rule}"
    os.makedirs(args.out_dir, exist_ok=True)

    written = []
    jobs = [("treatment", g_new, f"{tagged}.npz")]
    if not keep.all():
        # Matched control: identical rows and standardisation, ORIGINAL group.
        g_orig = sub["g_orig"].to_numpy().astype(int)
        jobs.append(("matched_control", g_orig, f"{tagged}_control.npz"))

    for role, gv, fname in pbar(jobs, desc="bundles", unit="bundle"):
        info = write_bundle(
            fb, keep, gv, os.path.join(args.out_dir, fname),
            {"regroup_rule": args.rule, "regroup_role": role,
             "regroup_cuts": cuts, "regroup_invert": bool(args.invert),
             "regroup_source": os.path.basename(args.bundle),
             "regroup_split": args.split,
             "regroup_auc_frac_vs_y": float(lk["auc"])},
        )
        info["role"] = role
        written.append(info)
        print(f"  {role:16s} {fname}  n={info['n']}  "
              f"g0={info['n_g0']} g1={info['n_g1']}  eps={info['eps']:.4f}"
              + (f"  (dropped {info['n_dropped']})" if info["n_dropped"] else ""))

    rep = {"rule": args.rule, "cuts": cuts, "split": args.split,
           "invert": bool(args.invert), "source": args.bundle,
           "alignment": align, "leakage_frac_vs_y": lk, "bundles": written}
    with open(os.path.join(args.out_dir, f"{tagged}_regroup.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"wrote {os.path.join(args.out_dir, tagged + '_regroup.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
