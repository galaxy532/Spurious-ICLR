"""The epsilon x backbone sweep -- the arm that turns the theory into a prediction.

WHAT IS BEING TESTED
====================

The rebuttal to reviewer oDsv claimed the theory is usable WITHOUT identifying
Phi_r and Phi_s, because its predictions relate quantities that are observable
without any decomposition: per-group error curves, the group proportion eps, and
training time. Specifically, holding the representation fixed and sweeping eps
by subsampling:

    worst-group error scales as 1/eps at fixed z_t   =>  alpha < 1,
                                                          balancing is the lever
    worst-group error is insensitive to eps          =>  alpha > 1, the majority
                                                          is binding, and
                                                          balancing cannot help

That claim is a REGIME DIAGNOSTIC. On its own it describes rather than predicts,
and a reviewer is entitled to say: I measured the sensitivity, I learned the
system is sensitive, and I was going to run balancing anyway.

This script makes it a prediction by adding the second axis. `alpha` is a
property of Phi, not of the dataset (see `backbones.py`). So sweep the backbone
with the dataset held fixed, and the theory says the REGIME SHOULD MOVE -- and
therefore that balancing should help for some backbones and not others, on the
same data. The script then checks that against ground truth by actually running
group-balanced last-layer retraining and measuring the gain.

The deliverable is one table:

    backbone | kappa | regime | predicted: balancing helps? | measured DFR gain

If the prediction column tracks the measurement column, the theory has earned a
practical use. If it does not, that is a real negative result about the theory's
usefulness and should be reported as one.


THE THREE THINGS THAT WILL GO WRONG IF THEY ARE NOT CONTROLLED
==============================================================

1. MATCHED z_t, NOT MATCHED EPOCHS. The prediction holds at matched normalised
   training time z_t = sum of step sizes, which with a constant step size is
   h * t. Comparing runs at equal wall-clock, equal epochs, or equal loss will
   produce a clean-looking wrong answer. `common.logistic_gd` records `z`
   directly, and every cross-eps and cross-backbone comparison here is made by
   interpolating onto a shared z grid.

2. SCALE. z_t is only comparable across backbones if the features are on the
   same scale, because a rescaling of Phi is exactly a rescaling of the
   effective step size. CLIP embeddings are near unit norm; ResNet-50
   penultimate activations are not. Features are standardised
   (`common.standardize`) at extraction and the bundle records it. This is not a
   cosmetic choice and it cannot be skipped.

3. THE REGIME IS ASYMPTOTIC, SO THE RUN MUST BE LONG. `Rebuttals/README.md`
   records the minority exponent converging to max(alpha, 1) from BELOW and
   slowly: 0.716 at z_T = 1,500, 0.752 at z_T = 6,000, against a target of 1.0,
   with the majority control at 0.94-0.98 throughout. A short run looks like a
   refutation. Exponents are reported here with their drift across two windows,
   never as a single converged number, and `--T` defaults high for that reason.

Because the regime lives in the late implicit-bias phase and needs separable
data, the venue is last-layer training on frozen features -- which is also,
conveniently, exactly the setting practitioners use for DFR.


Examples
--------
    # one backbone
    python eps_backbone_sweep.py --bundles features_waterbirds_erm_rn50_train.npz

    # the actual deliverable: all four backbones on one dataset
    python eps_backbone_sweep.py --bundles features_waterbirds_*_train.npz --dfr

    # CelebA, wider eps range because the natural eps is ~0.42
    python eps_backbone_sweep.py --bundles features_celeba_*_train.npz \
        --eps 0.02,0.05,0.10,0.20,0.40 --dfr
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from common import FeatureBundle, local_slope, logistic_gd, subsample_to_eps

# The two z-windows the exponents are reported over. Two numbers with their
# drift, never one number pretending to have converged.
Z_WINDOWS = ((0.30, 0.60), (0.60, 1.00))


def _err_at_z(rec: dict, z_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group soft error interpolated onto a shared z grid.

    Log-log interpolation, because the predicted decay is polynomial in z and
    linear interpolation of a power law on a linear scale is badly biased when
    the checkpoints are log-spaced (which they are).
    """
    z = np.asarray(rec["z"], float)
    out = []
    for key in ("err_maj", "err_min"):
        e = np.asarray(rec[key], float)
        m = np.isfinite(z) & np.isfinite(e) & (z > 0) & (e > 0)
        out.append(np.exp(np.interp(np.log(z_targets), np.log(z[m]),
                                    np.log(e[m]))))
    return out[0], out[1]


def _window_slope(rec: dict, key: str, lo: float, hi: float) -> float:
    """Decay exponent beta in err ~ z^{-beta}, over a fractional window of the run."""
    z = np.asarray(rec["z"], float)
    e = np.asarray(rec[key], float)
    m = np.isfinite(z) & np.isfinite(e) & (z > 0) & (e > 0)
    z, e = z[m], e[m]
    if z.size < 6:
        return float("nan")
    a, b = int(lo * z.size), int(hi * z.size)
    if b - a < 4:
        return float("nan")
    return local_slope(z[a:b], e[a:b], lo_frac=0.0)


def sweep_one(
    fb: FeatureBundle,
    eps_list: list[float],
    h: float = 0.05,
    T: int = 2_000_000,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
) -> dict:
    """Run the eps sweep for one (dataset, backbone) bundle.

    The majority group is held FIXED across the sweep and only the minority is
    subsampled (`common.subsample_to_eps`), so the majority curve is a control:
    if it moves with eps, something other than the group proportion changed and
    the run is not measuring what it claims to.

    `device="cuda"` routes the recurrence through `gd_gpu.logistic_gd_torch`,
    which is required to agree numerically with the certified `common` version
    -- run `python gd_gpu.py --validate` once before trusting it. At CelebA scale
    the CPU path takes 78 h per eps and the GPU path about 2 h, so this is a
    feasibility switch rather than a convenience.
    """
    rng = np.random.default_rng(seed)
    X = fb.phi
    rows, curves = [], {}

    if device != "cpu":
        from gd_gpu import logistic_gd_torch

        def _run(Xi, yi, gi):
            return logistic_gd_torch(Xi, yi, gi, h=h, T=T,
                                     device=device, dtype=dtype)
    else:
        def _run(Xi, yi, gi):
            return logistic_gd(Xi, yi, gi, h=h, T=T)

    for eps in eps_list:
        idx = subsample_to_eps(fb.y, fb.g, eps, rng)
        rec = _run(X[idx], fb.y[idx], fb.g[idx])
        curves[f"{eps:g}"] = {k: np.asarray(v).tolist() for k, v in rec.items()}
        row = {"eps": eps, "n": int(idx.size),
               "n_min": int(np.sum(fb.g[idx] == 1))}
        for wi, (lo, hi) in enumerate(Z_WINDOWS):
            row[f"beta_min_w{wi}"] = _window_slope(rec, "err_min", lo, hi)
            row[f"beta_maj_w{wi}"] = _window_slope(rec, "err_maj", lo, hi)
        row["z_max"] = float(rec["z"][-1])
        rows.append(row)

    # Worst-group error at matched z_t, across eps. This is the quantity the
    # oDsv claim is about.
    z_max = min(r["z_max"] for r in rows)
    z_targets = np.array([0.25, 0.5, 1.0]) * z_max
    wg = np.zeros((len(eps_list), z_targets.size))
    for i, eps in enumerate(eps_list):
        rec = {k: np.asarray(v) for k, v in curves[f"{eps:g}"].items()}
        e_maj, e_min = _err_at_z(rec, z_targets)
        wg[i] = np.maximum(e_maj, e_min)

    # kappa: worst-group error ~ eps^{-kappa} at fixed z_t.
    #   kappa ~ 1  ->  the 1/eps law  ->  alpha < 1, balancing is the lever
    #   kappa ~ 0  ->  insensitive    ->  alpha > 1, the majority is binding
    le = np.log(np.asarray(eps_list, float))
    kappas = []
    for j in range(z_targets.size):
        good = np.isfinite(wg[:, j]) & (wg[:, j] > 0)
        if good.sum() >= 3:
            A = np.vstack([le[good], np.ones(good.sum())]).T
            sol, *_ = np.linalg.lstsq(A, np.log(wg[good, j]), rcond=None)
            kappas.append(-float(sol[0]))
        else:
            kappas.append(float("nan"))

    k_med = float(np.nanmedian(kappas)) if np.any(np.isfinite(kappas)) else float("nan")
    return {
        "rows": rows, "curves": curves,
        "z_targets": z_targets.tolist(),
        "worst_group_at_z": wg.tolist(),
        "kappa_per_z": kappas, "kappa": k_med,
        "regime": classify(k_med),
        "eps_list": eps_list, "h": h, "T": T,
        "meta": fb.meta,
    }


def classify(kappa: float) -> str:
    """Map the measured eps-sensitivity onto the predicted regime.

    The cut points are conventions and are reported alongside kappa so a reader
    can move them. `ambiguous` is a real and expected outcome, not a failure --
    the theory's transition is continuous (part iii of the main theorem), so
    systems near alpha = 1 SHOULD land here, and forcing them into a bin would
    be misrepresenting the prediction.
    """
    if not np.isfinite(kappa):
        return "undetermined"
    if kappa >= 0.60:
        return "alpha<1 (balancing is the lever)"
    if kappa <= 0.20:
        return "alpha>1 (majority binding; balancing cannot accelerate)"
    return "ambiguous (near the transition)"


def dfr_gain(fb: FeatureBundle, seed: int = 0, n_boot: int = 20) -> dict:
    """Measured worst-group accuracy of ERM vs group-balanced last-layer retraining.

    This is the GROUND TRUTH the kappa prediction is checked against, so it is
    deliberately the plain, standard thing: logistic regression on frozen
    features, once on the natural sample and once on a group-balanced
    resample, scored by worst-group accuracy on a held-out half.

    Averaged over `n_boot` resamples because the balanced subsample is small by
    construction -- it is capped by the smallest group -- and a single draw of it
    is noisy enough to invert the sign of the gain.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n = fb.y.size
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    Xtr, ytr, gtr = fb.phi[tr], fb.y[tr], fb.g[tr]
    Xte, yte, gte = fb.phi[te], fb.y[te], fb.g[te]

    def worst_group_acc(clf):
        pred = clf.predict(Xte)
        accs = []
        for yy in (-1, 1):
            for gg in (0, 1):
                m = (yte == yy) & (gte == gg)
                if m.sum() >= 20:
                    accs.append(float(np.mean(pred[m] == yte[m])))
        return min(accs) if accs else float("nan")

    erm = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    a_erm = worst_group_acc(erm)

    cells = [np.flatnonzero((ytr == yy) & (gtr == gg))
             for yy in (-1, 1) for gg in (0, 1)]
    cells = [c for c in cells if c.size > 0]
    k = min(c.size for c in cells)

    gains = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(c, k, replace=False) for c in cells])
        bal = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr[idx], ytr[idx])
        gains.append(worst_group_acc(bal) - a_erm)

    return {"wg_acc_erm": a_erm,
            "dfr_gain_mean": float(np.mean(gains)),
            "dfr_gain_std": float(np.std(gains)),
            "balanced_cell_size": int(k), "n_boot": n_boot}


def to_markdown(all_res: dict) -> str:
    L = ["\n## epsilon x backbone sweep", "",
         "`kappa` is the exponent in  worst-group error ~ eps^(-kappa)  at "
         "matched z_t.", "",
         "| backbone | n | kappa | regime | predicted | DFR gain (measured) |",
         "|---|---|---|---|---|---|"]
    for key, r in all_res.items():
        pred = ("helps" if r["kappa"] >= 0.60
                else "no help" if r["kappa"] <= 0.20 else "unclear")
        d = r.get("dfr")
        dfr = (f"{d['dfr_gain_mean']:+.3f} ± {d['dfr_gain_std']:.3f}"
               if d else "not run")
        n = r["rows"][0]["n"] if r["rows"] else 0
        L.append(f"| {key} | {n} | {r['kappa']:.3f} | {r['regime']} "
                 f"| {pred} | {dfr} |")
    L.append("")

    for key, r in all_res.items():
        L += [f"### {key}", "",
              "| eps | n | n_min | beta_min w0 | beta_min w1 | "
              "beta_maj w0 | beta_maj w1 |", "|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            L.append(
                f"| {row['eps']:g} | {row['n']} | {row['n_min']} "
                f"| {row['beta_min_w0']:.3f} | {row['beta_min_w1']:.3f} "
                f"| {row['beta_maj_w0']:.3f} | {row['beta_maj_w1']:.3f} |")
        L += ["",
              "beta is reported over two z-windows. It converges to "
              "max(alpha,1) from BELOW and slowly, so read the DRIFT between "
              "w0 and w1, never w1 alone. The majority columns are the control: "
              "they should sit near 1 and should NOT move with eps.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", nargs="+", required=True,
                    help="one .npz per (dataset, backbone); globs are expanded")
    ap.add_argument("--eps", default="0.01,0.02,0.05,0.10,0.25")
    ap.add_argument("--h", type=float, default=0.05, help="constant step size")
    ap.add_argument("--T", type=int, default=2_000_000,
                    help="GD steps. High on purpose: the regime is asymptotic "
                         "and a short run reads as a refutation.")
    ap.add_argument("--dfr", action="store_true",
                    help="also measure the balanced last-layer retraining gain, "
                         "which is what the kappa prediction is checked against")
    ap.add_argument("--device", default="cpu",
                    help="'cuda' routes GD through gd_gpu.py. Required for "
                         "CelebA: the CPU path is 78 h per eps there.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="GPU only. float32 unless gd_gpu.py --validate says "
                         "the precision drift is too large.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="eps_backbone")
    args = ap.parse_args()

    paths = []
    for p in args.bundles:
        paths.extend(sorted(glob.glob(p)) or [p])
    eps_list = [float(x) for x in args.eps.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device != "cpu":
        # Refuse to burn GPU hours on an unverified implementation. gd_gpu.py
        # has never been executed by its author -- see its module docstring.
        from gd_gpu import validate as _gd_validate
        print("verifying the GPU path against common.logistic_gd ...")
        if not _gd_validate(device=args.device):
            raise SystemExit(
                "gd_gpu.py does not reproduce common.logistic_gd. Refusing to "
                "run. Investigate before spending GPU time."
            )

    all_res = {}
    for p in paths:
        key = os.path.splitext(os.path.basename(p))[0].replace("features_", "")
        print(f"[{key}] loading {p}")
        fb = FeatureBundle.load(p)
        if not fb.meta.get("standardized", False):
            print(f"  WARNING: {key} is not marked standardised. z_t is not "
                  f"comparable across backbones without it -- see the module "
                  f"docstring, point 2.")
        res = sweep_one(fb, eps_list, h=args.h, T=args.T, seed=args.seed,
                        device=args.device, dtype=args.dtype)
        if args.dfr:
            res["dfr"] = dfr_gain(fb, seed=args.seed)
        all_res[key] = res
        print(f"  kappa = {res['kappa']:.3f}  ->  {res['regime']}")

    md = to_markdown(all_res)
    with open(os.path.join(args.out_dir, f"{args.tag}.md"), "w") as f:
        f.write(md)
    # Curves are dropped from the json: they are large and every number that
    # goes in the paper is already in the tables above.
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "curves"}
            for k, v in all_res.items()}
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(slim, f, indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()
