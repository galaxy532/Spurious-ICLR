"""dfr_bridge.py -- does balancing by REWEIGHTING change where GD ends up, and does
DFR-style SUBSAMPLING change it? The experiment behind THEORY_AND_DFR.md.

THE CLAIM BEING TESTED (THEORY_AND_DFR.md, sections 3-4)
=======================================================

Corollary 7.1 says the long-run test error depends on the training data only
through the max-margin direction w_hat. w_hat is the same for every group
proportion eps in (0, 1) when the training POINTS are the same. So:

  (P1) Reweighting the groups (same points, eps moved by loss weights) changes how
       fast the last layer gets somewhere, not where it ends up: the weight
       DIRECTIONS for different eps converge to each other, and so do their test
       errors.
  (P2) Deleting points (DFR's group-balanced subsample: keep the smallest (y, g)
       cell whole, cut the other cells to its size) can change w_hat -- but only
       if the deleted points include points that DEFINE the margin (the support
       vectors). Deleting points that are not on the margin leaves the hard-margin
       SVM unchanged.

WHAT IT RUNS, per backbone
==========================

One batched GD run (long_horizon.Engine: plain GD on the logistic loss, no
intercept, no regularisation) on the retraining set (--retrain-on train|val), with
one weight column per arm:

  A_eps   reweighting: every point of the retraining set, weights
          (1-eps)/n_0 on g = 0 and eps/n_1 on g = 1, for eps on --eps.
  B_k     DFR-style subsample number k (--n-sub draws): weight 1/m on the m kept
          points, 0 on the deleted ones (a zero-weight point does not enter the
          gradient, which is exactly deleting it).

At every recorded training time z = h * t, from the weights after t steps:
  test accuracy and test soft error in each of the four (y, g) cells, worst-cell
  accuracy, mean accuracy; and the cosine between weight directions.

At the end:
  * P1 check: smallest pairwise cosine among the A columns, over z (should -> 1).
  * P2 check: cosine of each B column with the mean A direction (can stay < 1),
    and, for each B draw, the fraction of the retraining set's TIGHTEST points
    (the --n-tight smallest normalised margins under the final A direction, an
    approximation of the support vectors, since GD's direction is only within
    O(1/log t) of w_hat) that the draw deleted, next to the fraction of ALL points
    it deleted. If deleting margin points is what moves w_hat, the draws that
    delete more tight points should have lower cosine to A.
  * C reference, DFR itself: l1-regularised logistic regression (sklearn,
    liblinear, with intercept, as DFR) fitted on each B subsample and the weights
    averaged, for each C in --dfr-C. If a val bundle exists and --retrain-on train,
    C is selected by worst-cell accuracy on val; otherwise all C are reported and
    none is selected.

WHY THE STANDARDISATION CHECK
=============================

Before 17 Sept 2026 extract_features.py standardised each split with its own
statistics, i.e. a different affine map on train and test. A last layer fitted on
train and evaluated on such test features is not evaluated on the same Phi. This
script refuses bundles without meta["standardized_with"] == "train statistics"
unless --allow-per-split-standardization (then the results are not interpretable).

USAGE (Paperspace)
==================

    python dfr_bridge.py --validate-only --device cuda
    python dfr_bridge.py --device cuda --dataset waterbirds \\
        --backbones dinov2,erm_rn50,rwg_rn50,gdro_rn50 --prefix features_v2
    python dfr_bridge.py --device cuda --dataset waterbirds --retrain-on val \\
        --backbones dinov2,erm_rn50,rwg_rn50,gdro_rn50 --prefix features_v2

Outputs, per backbone: results/dfr_bridge_<dataset>_<backbone>_<retrain>.{md,json}
and ..._curves.npz. Stops after --max-hours and resumes when rerun (state in
results/state_dfr_bridge/).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from long_horizon import Engine, checkpoints, eps_weights, validate as lh_validate
from progress import pbar

CELLS = [(-1, 0), (-1, 1), (1, 0), (1, 1)]


def dfr_subsample(y, g, rng):
    """Footnote 3 of Kirichenko et al. (2023): keep all of the smallest (y, g) cell,
    subsample every other cell to the same size. Returns a boolean keep-mask."""
    cells = [np.flatnonzero((y == yy) & (g == gg)) for yy, gg in CELLS]
    cells = [c for c in cells if c.size]
    k = min(c.size for c in cells)
    keep = np.zeros(y.size, bool)
    for c in cells:
        keep[rng.choice(c, k, replace=False)] = True
    return keep


def sneg(u):
    """1 - sigmoid(u), numerically stable (same formula as common.logistic_gd)."""
    e = np.exp(-np.abs(u))
    return np.where(u >= 0, e / (1.0 + e), 1.0 / (1.0 + e))


def test_metrics(W, Xt, yt, gt):
    """Per column of W: accuracy and soft error in each (y, g) cell."""
    U = (yt[:, None] * Xt) @ W                     # y * w.x, (n_test, K)
    acc, soft = [], []
    for yy, gg in CELLS:
        m = (yt == yy) & (gt == gg)
        if m.sum() == 0:
            acc.append(np.full(W.shape[1], np.nan))
            soft.append(np.full(W.shape[1], np.nan))
            continue
        acc.append((U[m] > 0).mean(0))
        soft.append(sneg(U[m]).mean(0))
    acc, soft = np.array(acc), np.array(soft)      # (4, K)
    return {"cell_acc": acc, "cell_soft": soft,
            "worst_acc": np.nanmin(acc, 0), "mean_acc": (U > 0).mean(0)}


def unit(W):
    n = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.where(n > 0, n, 1.0)


def load_bundle(path, allow_bad_std):
    from common import FeatureBundle
    fb = FeatureBundle.load(path)
    if fb.meta.get("standardized_with") != "train statistics" and not allow_bad_std:
        raise SystemExit(
            f"{path} was not standardised with train statistics (made before the "
            f"17 Sept fix to extract_features.py). Re-extract with the current "
            f"extract_features.py (use --out-prefix to avoid overwriting), or pass "
            f"--allow-per-split-standardization knowing results are not interpretable.")
    return fb


def run_backbone(args, bk, dtype):
    tag = f"dfr_bridge_{args.dataset}_{bk}_{args.retrain_on}"
    pre = f"{args.prefix}_{args.dataset}_{bk}"
    fit = load_bundle(f"{pre}_{args.retrain_on}.npz", args.allow_per_split_standardization)
    tst = load_bundle(f"{pre}_test.npz", args.allow_per_split_standardization)
    val = None
    if args.retrain_on == "train" and os.path.exists(f"{pre}_val.npz"):
        val = load_bundle(f"{pre}_val.npz", args.allow_per_split_standardization)
    X, y, g = fit.phi, fit.y, fit.g
    print(f"[{bk}] retrain on {args.retrain_on}: n={X.shape[0]} d={X.shape[1]}  "
          f"test n={tst.phi.shape[0]}  val={'yes' if val is not None else 'no'}")

    # separability of the retraining set (needed for P1 to hold at all: on
    # non-separable data GD converges to a finite minimiser that DOES depend on eps)
    from separability_check import quick_separable
    sep, marg = quick_separable(X, y)
    print(f"  retraining set separable: {sep}  margin: {marg}")
    if sep is not True and not args.allow_nonseparable:
        print("  no separator found; P1 does not apply. Skipping "
              "(--allow-nonseparable to run anyway).")
        return None

    eps = [float(e) for e in args.eps.split(",")]
    rng = np.random.default_rng(args.seed)
    keeps = [dfr_subsample(y, g, rng) for _ in range(args.n_sub)]
    colsA = [eps_weights(g, e) for e in eps]
    colsB = [k / k.sum() for k in keeps]
    C = np.stack(colsA + colsB, axis=1)
    KA, KB = len(colsA), len(colsB)
    names = [f"A_eps{e:g}" for e in eps] + [f"B_{k}" for k in range(KB)]
    sub_sep = [quick_separable(X[k], y[k])[0] for k in keeps]
    print(f"  DFR subsamples: {int(keeps[0].sum())} points each; separable: {sub_sep}")

    # ---- GD with resume -------------------------------------------------------
    sdir = os.path.join(args.out_dir, "state_dfr_bridge")
    os.makedirs(sdir, exist_ok=True)
    spath = os.path.join(sdir, f"{tag}.pkl")
    cfg = [bk, args.dataset, args.retrain_on, eps, args.n_sub, args.seed, args.h,
           dtype, args.per_decade, list(X.shape)]
    state = None
    if os.path.exists(spath):
        with open(spath, "rb") as f:
            state = pickle.load(f)
        if state["cfg"] != cfg:
            raise SystemExit(f"{spath} is from a different configuration; use another "
                             f"--out-dir or delete it deliberately.")
        print(f"  resuming from step {state['t']}")
    else:
        state = {"cfg": cfg, "t": 0, "W": None, "rec": []}

    eng = Engine(X, y, g, C, args.h, device=args.device, dtype=dtype, W0=state["W"])
    ck = checkpoints(args.T, args.per_decade)
    ck = ck[ck > state["t"]]
    deadline = time.time() + 3600 * args.max_hours
    last_save = time.time()
    bar = pbar(total=args.T - state["t"], desc=f"  GD {bk}", unit="step")
    stopped = False
    for tck in ck:
        t_new, _ = eng.run_to(state["t"], int(tck), [], bar=bar)
        state["t"] = t_new
        W = eng.weights_numpy()
        m = test_metrics(W, tst.phi, tst.y, tst.g)
        U = unit(W)
        cosA = U[:, :KA].T @ U[:, :KA]
        uA = unit(U[:, :KA].mean(1, keepdims=True))
        rec = {"t": t_new, "z": args.h * t_new, **m,
               "cosA_min": float(cosA.min()),
               "cosB_toA": (U[:, KA:].T @ uA).ravel()}
        state["rec"].append(rec)
        bar.set_postfix_str(f"z={rec['z']:.3g} worstA={np.nanmean(m['worst_acc'][:KA]):.3f} "
                            f"worstB={np.nanmean(m['worst_acc'][KA:]):.3f}")
        now = time.time()
        if now - last_save > 60 * args.save_every_min or now > deadline:
            state["W"] = W
            with open(spath, "wb") as f:
                pickle.dump(state, f)
            last_save = now
        if now > deadline:
            stopped = True
            break
    bar.close()
    state["W"] = eng.weights_numpy()
    with open(spath, "wb") as f:
        pickle.dump(state, f)
    if stopped:
        print(f"  stopped at --max-hours (z = {args.h * state['t']:.3g}); rerun to resume")
        return {"status": "partial"}

    # ---- final diagnostics ----------------------------------------------------
    W = state["W"]
    U = unit(W)
    uA = unit(U[:, :KA].mean(1, keepdims=True)).ravel()
    marg_n = (y * (X @ uA))                         # normalised margins under A
    tight = np.argsort(marg_n)[: args.n_tight]
    p2 = []
    for k, keep in enumerate(keeps):
        p2.append({"draw": k, "frac_all_deleted": float(1 - keep.mean()),
                   "frac_tight_deleted": float(1 - keep[tight].mean()),
                   "cos_to_A": float(U[:, KA + k] @ uA),
                   "subsample_separable": sub_sep[k]})

    # ---- C: DFR itself --------------------------------------------------------
    dfr = []
    from sklearn.linear_model import LogisticRegression
    for cval in pbar([float(c) for c in args.dfr_C.split(",")], desc="  DFR l1 fits",
                     unit="C"):
        coefs, ints = [], []
        for keep in keeps:
            lr = LogisticRegression(penalty="l1", C=cval, solver="liblinear",
                                    max_iter=5000).fit(X[keep], y[keep])
            coefs.append(lr.coef_.ravel())
            ints.append(float(lr.intercept_[0]))
        w_avg, b_avg = np.mean(coefs, 0), float(np.mean(ints))

        def wacc(fb):
            pred = np.where(fb.phi @ w_avg + b_avg >= 0, 1, -1)
            accs = [np.mean(pred[(fb.y == yy) & (fb.g == gg)] == yy)
                    for yy, gg in CELLS if np.any((fb.y == yy) & (fb.g == gg))]
            return float(min(accs)), float(np.mean(pred == fb.y))
        te_w, te_m = wacc(tst)
        row = {"C": cval, "test_worst_acc": te_w, "test_mean_acc": te_m,
               "cos_to_A": float(w_avg @ uA / max(np.linalg.norm(w_avg), 1e-12))}
        if val is not None:
            row["val_worst_acc"] = wacc(val)[0]
        dfr.append(row)
    sel = max(dfr, key=lambda r: r["val_worst_acc"]) if val is not None else None

    out = {"backbone": bk, "dataset": args.dataset, "retrain_on": args.retrain_on,
           "n_fit": int(X.shape[0]), "separable": sep, "margin": marg,
           "eps": eps, "n_sub": KB, "subsample_size": int(keeps[0].sum()),
           "columns": names, "h": args.h, "T": args.T,
           "p2": p2, "dfr": dfr, "dfr_selected_on_val": sel}
    _write(args, tag, out, state["rec"], KA, KB)
    return out


def _write(args, tag, out, rec, KA, KB):
    zs = np.array([r["z"] for r in rec])
    picks = sorted({int(np.argmin(np.abs(zs - v))) for v in
                    [1, 10, 100, 1e3, 1e4, 1e5, 1e6, zs[-1]] if v <= zs[-1]})
    eps = out["eps"]
    L = [f"# DFR bridge: {out['backbone']} on {out['dataset']}, last layer retrained "
         f"on {out['retrain_on']}", "",
         "Definitions and predictions: docstring of `dfr_bridge.py` and "
         "`THEORY_AND_DFR.md`. A = reweighting (same points), B = DFR-style "
         "group-balanced subsample (points deleted). Plain GD, logistic loss, no "
         "intercept, no regularisation. z = h * steps.", "",
         f"retraining set: n = {out['n_fit']}, separable = {out['separable']}, "
         f"margin (lower bound) = {out['margin']}; subsample size = "
         f"{out['subsample_size']}, {out['n_sub']} draws", "",
         "## P1 -- do the reweighted runs end in the same place?", "",
         "| z | min cosine among A | " + " | ".join(f"worst-cell test acc, eps={e:g}" for e in eps)
         + " |", "|---|---|" + "---|" * len(eps)]
    for i in picks:
        r = rec[i]
        L.append(f"| {r['z']:.3g} | {r['cosA_min']:.5f} | " +
                 " | ".join(f"{v:.3f}" for v in r["worst_acc"][:KA]) + " |")
    L += ["", "## P2 -- do the DFR-style subsamples end somewhere else?", "",
          "| z | cosine of B to mean A direction (mean, min) | worst-cell test acc, "
          "B (mean +- sd) | worst-cell test acc, A (mean) |", "|---|---|---|---|"]
    for i in picks:
        r = rec[i]
        cb = r["cosB_toA"]
        wb = r["worst_acc"][KA:]
        L.append(f"| {r['z']:.3g} | {cb.mean():.4f}, {cb.min():.4f} | "
                 f"{np.nanmean(wb):.3f} +- {np.nanstd(wb):.3f} | "
                 f"{np.nanmean(r['worst_acc'][:KA]):.3f} |")
    L += ["", f"Tightest {args.n_tight} retraining points under the final A direction "
          "(approximate support vectors): how many did each draw delete?", "",
          "| draw | fraction of ALL points deleted | fraction of TIGHT points deleted "
          "| cosine to A | subsample separable |", "|---|---|---|---|---|"]
    for p in out["p2"]:
        L.append(f"| {p['draw']} | {p['frac_all_deleted']:.3f} | "
                 f"{p['frac_tight_deleted']:.3f} | {p['cos_to_A']:.4f} | "
                 f"{p['subsample_separable']} |")
    L += ["", "## C -- DFR itself (l1 logistic regression with intercept, weights "
          "averaged over the same subsamples)", "",
          "| C | test worst-cell acc | test mean acc | cosine to A | val worst-cell acc |",
          "|---|---|---|---|---|"]
    for d in out["dfr"]:
        L.append(f"| {d['C']:g} | {d['test_worst_acc']:.3f} | {d['test_mean_acc']:.3f} | "
                 f"{d['cos_to_A']:.4f} | {d.get('val_worst_acc', float('nan')):.3f} |")
    if out["dfr_selected_on_val"]:
        L.append(f"\nC selected on val: {out['dfr_selected_on_val']['C']:g}")
    else:
        L.append("\nNo val bundle: no C selected (reporting all is not a tuned DFR).")
    base = os.path.join(args.out_dir, tag)
    with open(base + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    with open(base + ".json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    np.savez_compressed(base + "_curves.npz",
                        z=zs, columns=np.array(out["columns"]),
                        worst_acc=np.array([r["worst_acc"] for r in rec]),
                        mean_acc=np.array([r["mean_acc"] for r in rec]),
                        cell_acc=np.array([r["cell_acc"] for r in rec]),
                        cell_soft=np.array([r["cell_soft"] for r in rec]),
                        cosA_min=np.array([r["cosA_min"] for r in rec]),
                        cosB_toA=np.array([r["cosB_toA"] for r in rec]))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--backbones", default="dinov2,erm_rn50,rwg_rn50,gdro_rn50")
    ap.add_argument("--prefix", default="features")
    ap.add_argument("--retrain-on", default="train", choices=["train", "val"])
    ap.add_argument("--eps", default="0.01,0.05,0.2,0.5")
    ap.add_argument("--n-sub", type=int, default=5)
    ap.add_argument("--n-tight", type=int, default=50)
    ap.add_argument("--dfr-C", default="0.1,1,10,100,1000")
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=2_000_000)
    ap.add_argument("--per-decade", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None,
                    choices=[None, "float32", "float64", "mixed"],
                    help="default 'mixed' on cuda (float64 weights, float32 matrix "
                         "products); see long_horizon.Engine")
    ap.add_argument("--max-hours", type=float, default=5.5)
    ap.add_argument("--save-every-min", type=float, default=15.0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--allow-nonseparable", action="store_true")
    ap.add_argument("--allow-per-split-standardization", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    dtype = args.dtype or ("float64" if args.device == "cpu" else "mixed")

    print(f"validating the batched GD engine on {args.device} ...")
    if not lh_validate(device=args.device):
        raise SystemExit("validation failed; refusing to run")
    if args.validate_only:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    for bk in pbar(args.backbones.split(","), desc="backbones", unit="backbone"):
        pre = f"{args.prefix}_{args.dataset}_{bk}"
        if not os.path.exists(f"{pre}_{args.retrain_on}.npz") or not os.path.exists(f"{pre}_test.npz"):
            print(f"[{bk}] missing {pre}_{args.retrain_on}.npz or {pre}_test.npz; skipped")
            continue
        r = run_backbone(args, bk, dtype)
        if r is not None and r.get("status") == "partial":
            break


if __name__ == "__main__":
    np.seterr(over="ignore", under="ignore")
    main()
