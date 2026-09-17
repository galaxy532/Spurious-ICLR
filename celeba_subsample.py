"""celeba_subsample.py -- a CelebA version of the speed experiments that fits in ~3 GPU hours.

WHY
===

The full CelebA train split (n = 162,770) costs ~26 GPU-hours for the eps
sweep, and with n far above d (512 to 2,048) it is probably not linearly
separable at all -- in which case the implicit-bias regime the theorem is about
does not exist there and the run would measure nothing.

The step cost of GD is proportional to n * d, so a fixed subsample of the train
split fixes both problems at once: it is ~16x to ~65x cheaper, and a smaller
point set is more likely to be separable.


WHY A SUBSAMPLE DOES NOT BRING BACK THE MARGIN CONFOUND
======================================================

The confound (PREREGISTRATION.md, Amendment 2) was that the point set CHANGED
WITH eps: deleting minority rows moved the maximum margin. Here the subsample is
drawn ONCE, before any eps is chosen, and every eps is then applied to that same
subsample by loss weights (long_horizon.py). The margin is therefore constant
across the eps grid, exactly as on Waterbirds. The subsample is simply a smaller
training set.


HOW THE SUBSAMPLE IS DRAWN
==========================

Stratified by the four (label, group) cells of CelebA -- (not blond, female),
(blond, female), (not blond, male), (blond, male) -- in their natural
proportions, so the subsample has the same cell proportions as the full train
split (and so the same natural eps). Rounding is corrected so the total is
exactly n.

The sizes are NESTED: each cell is shuffled once with a fixed seed and every
size takes a prefix of it, so the 5,000 subsample is contained in the 10,000
one, and so on. Because any subset of a separable set is separable, the
separability verdicts are then monotone in n, and "the largest separable size"
is well defined.

Features are NOT re-standardised: the subsample keeps exactly the Phi of the full
split, so the numbers remain comparable with anything computed on the full split.


WHAT IT WRITES
==============

    features_celeba_<backbone>_n<size>_train.npz          one bundle per size
    results/<tag>_separability.{md,json}                  if --check (default)

The separability check is separability_check.check_cell: a constructive
logistic fit first, and the exact LP when that leaves violations and
n <= --lp-max-n (default 20,000). So for n <= 20,000 both "separable" and
"not separable" are PROVEN; above it, only "separable" can be.

The json is in the format long_horizon.py reads with --margins-json.


TYPICAL USE ON PAPERSPACE
=========================

    python celeba_subsample.py --bundles 'features_celeba_*_train.npz' \\
        --sizes 5000,10000,20000
    # read results/celeba_sub_separability.md, pick the size, then
    python long_horizon.py --benchmark --device cuda \\
        --bundles 'features_celeba_*_n10000_train.npz'
    python long_horizon.py --device cuda --bundles 'features_celeba_*_n10000_train.npz' \\
        --margins-json results/celeba_sub_separability.json --tag celeba_n10000_speed
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from common import FeatureBundle
from progress import pbar


def nested_stratified_indices(y, g, sizes, seed=0):
    """Return {size: index array}, nested, stratified over the (y, g) cells."""
    rng = np.random.default_rng(seed)
    n_all = y.size
    cells = []
    for yy in np.unique(y):
        for gg in np.unique(g):
            idx = np.flatnonzero((y == yy) & (g == gg))
            if idx.size:
                cells.append(rng.permutation(idx))
    props = np.array([c.size for c in cells], float) / n_all
    out = {}
    for n in sorted(sizes):
        if n > n_all:
            raise ValueError(f"size {n} exceeds the split size {n_all}")
        k = np.floor(props * n).astype(int)
        # largest-remainder rounding so the total is exactly n
        rem = props * n - k
        for j in np.argsort(-rem)[: n - k.sum()]:
            k[j] += 1
        k = np.minimum(k, [c.size for c in cells])
        out[n] = np.sort(np.concatenate([c[:kk] for c, kk in zip(cells, k)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", nargs="+", default=["features_celeba_*_train.npz"])
    ap.add_argument("--sizes", default="5000,10000,20000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true",
                    help="write the subsample bundles only, skip separability")
    ap.add_argument("--C", type=float, default=1e6)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--lp-max-n", type=int, default=20000)
    ap.add_argument("--lp-time-limit", type=float, default=600.0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="celeba_sub")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    paths = []
    for p in args.bundles:
        # never re-subsample an existing subsample
        paths.extend(q for q in (sorted(glob.glob(p)) or [p])
                     if not re.search(r"_n\d+_train\.npz$", os.path.basename(q)))
    os.makedirs(args.out_dir, exist_ok=True)
    res = {}

    for p in pbar(paths, desc="bundles", unit="bundle"):
        base = os.path.basename(p)
        key0 = os.path.splitext(base)[0].replace("features_", "").replace("_train", "")
        print(f"[{key0}] loading {p}")
        fb = FeatureBundle.load(p)
        idxs = nested_stratified_indices(fb.y, fb.g, sizes, args.seed)
        for n in pbar(sorted(sizes), desc=f"  sizes {key0}", unit="size"):
            idx = idxs[n]
            sub = FeatureBundle(
                phi=fb.phi[idx], y=fb.y[idx], g=fb.g[idx],
                idx_r=fb.idx_r, idx_s=fb.idx_s,
                place=None if fb.place is None else fb.place[idx],
                meta={**fb.meta, "subsample_of": base, "subsample_n": int(n),
                      "subsample_seed": args.seed, "subsample": "nested, stratified by (y, g)",
                      "eps": float(np.mean(fb.g[idx] == 1))})
            out = os.path.join(os.path.dirname(p) or ".",
                               f"features_{key0}_n{n}_train.npz")
            sub.save(out)
            key = f"{key0}_n{n}_train"
            cells = {f"y={yy},g={gg}": int(np.sum((sub.y == yy) & (sub.g == gg)))
                     for yy in np.unique(sub.y) for gg in np.unique(sub.g)}
            print(f"  wrote {out}  cells {cells}")
            if args.no_check:
                continue
            from separability_check import check_cell
            r = check_cell(sub.phi, sub.y, args.C, args.max_iter,
                           args.lp_max_n, args.lp_time_limit)
            r.update({"label": "full", "n": int(n), "n_min": int(np.sum(sub.g == 1)),
                      "cells": cells})
            print(f"  separable={r['separable']}  margin={r['margin']:.4g}  "
                  f"[{r['how']}, {r['secs']}s]")
            res[key] = {"rows": [r], "full_split_margin":
                        float(r["margin"]) if r["separable"] is True else None,
                        "separable": r["separable"]}
            _write(args, res)
    if not args.no_check:
        _write(args, res)
        print(open(os.path.join(args.out_dir, f"{args.tag}_separability.md")).read())


def _write(args, res):
    L = ["## CelebA nested subsamples -- separability of the full subsample", "",
         "Through the origin (no intercept). margin = min_i y_i w.x_i / ||w|| for the "
         "separator found; a lower bound on the maximum margin. 'LP proof' = proven.",
         "", "| bundle | n | n_min (g=1) | separable | margin | decided by | cells |",
         "|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        r = v["rows"][0]
        sep = {True: "yes", False: "**no**", None: "?"}[r["separable"]]
        mg = "--" if not np.isfinite(r["margin"]) else f"{r['margin']:.4g}"
        L.append(f"| {k} | {r['n']} | {r['n_min']} | {sep} | {mg} | {r['how']} | "
                 f"{r['cells']} |")
    with open(os.path.join(args.out_dir, f"{args.tag}_separability.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(args.out_dir, f"{args.tag}_separability.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
