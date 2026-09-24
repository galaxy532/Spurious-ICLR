"""The eps-law as a finite-sample implicit-bias prediction: the "sum of alphas" check.

WHAT THIS TESTS, IN WORDS
=========================
Session 3 measured the eps-law on real frozen features: each group's soft error
behaves like const / (eps_g z), with kappa ~ 0.95-0.98 at z = 1e6 on five
Waterbirds backbones. That the law HOLDS is established. The question here is
WHY: is it already implied by finite-sample implicit bias, and does implicit bias
predict the CONSTANT, not only the exponent?

The prediction (derived 24 Sept 2026 from Soudry et al., JMLR 2018; NOT a
published result, see the derivation below). For GD with per-sample weights c_n
and step h on the logistic loss over a separable training set,

    eps * z * err_min  ->  A_min := sum of alpha_n over the MINORITY support vectors
    (1-eps) * z * err_maj  ->  A_maj := sum of alpha_n over the MAJORITY support vectors

where z = h t, err_g is the group's mean soft error over its training points
(exactly what long_horizon.py records), and alpha_n are the dual variables of the
hard-margin SVM  min ||w||^2 s.t. y_n w.x_n >= 1,  i.e.  w_hat = sum alpha_n y_n x_n.
Both limits are independent of eps and of z: that is the eps-law with kappa = 1,
AND a parameter-free value for the constant. A_min + A_maj = ||w_hat||^2 =
1/margin^2 exactly, so the prediction is really about how the total splits
between the two groups.

Per point, the same argument predicts  c_n * z * (soft error of n) -> alpha_n  for
every support vector n: a point-by-point check, made on the final GD iterate when
long_horizon.py's state file is available.

The alpha > 1 branch is the case A_min = 0 (the minority holds no support vector):
then eps * z * err_min -> 0, which is what the synthetic alpha = 1.6 run shows.

DERIVATION (so it can be audited)
---------------------------------
Soudry et al. Theorem 3 (checked in 18-188.pdf): w(t) = w_hat log t + rho(t) with
rho bounded for almost all datasets; Theorem 4 (when the support vectors span the
data): rho(t) -> w_tilde with  eta exp(-x_n . w_tilde) = alpha_n  on the support
vectors (their eq. 7, unit weights). With weights c_n the same matching argument
gives  h c_n exp(-y_n x_n . w_tilde) = alpha_n : the GD increment
h sum_n c_n exp(-y_n x_n . w(t)) y_n x_n must equal w_hat (log(t+1) - log t) ~ w_hat/t.
Then a support vector's soft error is ~ exp(-log t - y_n x_n . w_tilde)
= alpha_n / (h c_n t) = alpha_n / (c_n z). Averaging over group g with
c_n = eps_g / n_g gives the group limits above.

TWO THINGS TO KNOW BEFORE READING THE OUTPUT
--------------------------------------------
1. Convergence is SLOW and slowest at small eps. The residual rho(t) - w_tilde is
   driven by points just above the margin, whose error decays like t^-(theta) with
   theta slightly above 1, so the limit is approached roughly logarithmically.
   On a toy where everything is known (--self-test) the minority support vector
   reads 0.34 of its limit at eps = 0.05 and z = 2e5, still climbing. So the test
   is the TREND of R = measured / predicted along z: toward 1, and the spread
   across eps shrinking. A single endpoint is not the test.
2. Theorem 4 assumes the support vectors span the data. On Waterbirds they do not
   (hundreds of support vectors in d = 768-2048). The support-vector errors only
   depend on the component of rho in the span of the support vectors, so the
   prediction is still expected to hold, but this is the step the data checks
   rather than a theorem.

HOW alpha IS OBTAINED
=====================
Exactly, not from the soft-margin SVC. The hard-margin dual
    max  sum alpha_n - 1/2 || sum alpha_n y_n x_n ||^2,   alpha >= 0
is solved with L-BFGS-B on a candidate set (points whose margin under the
LinearSVC separator is within `--band` of the minimum), then CERTIFIED on the
whole training set: primal feasibility min_n y_n w_hat.x_n >= 1 - 1e-6 (points
violating it are added and the dual re-solved), duality gap
|sum alpha - ||w_hat||^2| / sum alpha, complementary slackness, and agreement of
the margin and direction with LinearSVC. All certificates are in the report.

Usage
-----
    python dual_mass_check.py --self-test
    python dual_mass_check.py                       # Waterbirds v3 + synthetic
    python dual_mass_check.py --no-synthetic        # Waterbirds only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from progress import pbar

SV_REL_TOL = 1e-8          # alpha_n > SV_REL_TOL * max(alpha) counts as a support vector
FEAS_TOL = 1e-6            # primal feasibility tolerance on y w_hat.x >= 1
Z_REPORT = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


# ----------------------------------------------------------------------------
# The exact hard-margin dual
# ----------------------------------------------------------------------------
def _solve_dual(Gc: np.ndarray, a0: np.ndarray | None, maxiter: int) -> np.ndarray:
    """max sum a - 1/2 ||Gc a||^2, a >= 0, via L-BFGS-B on the negated objective."""
    from scipy.optimize import minimize
    m = Gc.shape[1]

    def f(a):
        v = Gc @ a
        return 0.5 * float(v @ v) - float(a.sum()), Gc.T @ v - 1.0

    x0 = np.zeros(m) if a0 is None else a0
    bar = pbar(total=None, desc="    dual iterations", unit="it")
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * m,
                   callback=lambda xk: bar.update(1),
                   options=dict(maxiter=maxiter, maxfun=10 * maxiter, ftol=1e-20,
                                gtol=1e-13, maxcor=50))
    bar.close()
    return np.maximum(res.x, 0.0)


def hard_margin_dual(X, y, band: float = 0.5, max_rounds: int = 6,
                     maxiter: int = 200000, w_init=None) -> dict:
    """Exact alpha on (X, y), certified on every point. y is +/-1."""
    Xy = y[:, None] * X
    if w_init is None:
        from group_margins import C_LADDER, _geometric_margins, _svc_ladder
        zero = np.zeros(len(y), dtype=int)
        w_init = None
        for r in _svc_ladder(X, y, C_LADDER, 200000, 1e-8):
            if "w" in r and _geometric_margins(X, y, zero, r["w"])["n_violations"] == 0:
                w_init = r["w"]
        if w_init is None:
            raise SystemExit("LinearSVC found no separator; is this bundle separable?")
    u = (Xy @ w_init) / np.linalg.norm(w_init)
    ustar = float(u.min())
    # Candidates: every point within `band` (relative) of the smallest margin, and in
    # any case the smallest-margin points up to a floor, so a poor starting separator
    # (even one with a negative minimum) still yields a non-empty set. Whatever it
    # misses is caught by the feasibility loop below.
    floor = min(len(y), max(50, X.shape[1] // 2))
    cand = np.union1d(np.where(u <= ustar + band * abs(ustar))[0],
                      np.argsort(u, kind="mergesort")[:floor])
    a_full = np.zeros(len(y))
    rounds = []
    for rd in range(max_rounds):
        a_c = _solve_dual(Xy[cand].T, a_full[cand] if rd else None, maxiter)
        a_full[:] = 0.0
        a_full[cand] = a_c
        w_hat = Xy[cand].T @ a_c
        fm = Xy @ w_hat                               # functional margins
        viol = np.where(fm < 1.0 - FEAS_TOL)[0]
        rounds.append({"round": rd, "n_candidates": int(cand.size),
                       "min_functional_margin": float(fm.min()),
                       "n_violations": int(viol.size)})
        if viol.size == 0:
            break
        cand = np.union1d(cand, viol)
    sv = a_full > SV_REL_TOL * a_full.max()
    sum_a, wn2 = float(a_full.sum()), float(w_hat @ w_hat)
    cos = float(w_hat @ w_init / (np.linalg.norm(w_hat) * np.linalg.norm(w_init)))
    return {
        "alpha": a_full, "w_hat": w_hat, "sv": sv,
        "certificate": {
            "feasible": bool(fm.min() >= 1.0 - FEAS_TOL),
            "min_functional_margin": float(fm.min()),
            "duality_gap_rel": abs(sum_a - wn2) / max(sum_a, 1e-300),
            "max_slack_on_SVs": float(np.abs(fm[sv] - 1.0).max()) if sv.any() else float("nan"),
            "geometric_margin": 1.0 / np.sqrt(wn2),
            "linearsvc_margin": ustar,
            "cos_to_linearsvc": cos,
            "n_sv": int(sv.sum()),
            "rank_sv": int(np.linalg.matrix_rank(X[sv])) if sv.any() else 0,
            "d": int(X.shape[1]),
            "rounds": rounds,
        },
    }


# ----------------------------------------------------------------------------
# The comparison against long_horizon's curves
# ----------------------------------------------------------------------------
def group_constants(alpha, sv, g) -> dict:
    return {"A_min": float(alpha[sv & (g == 1)].sum()),
            "A_maj": float(alpha[sv & (g == 0)].sum()),
            "n_sv_min": int((sv & (g == 1)).sum()),
            "n_sv_maj": int((sv & (g == 0)).sum())}


def ratios_along_z(t, err_min, err_maj, eps, h, A_min, A_maj) -> dict:
    """R(z, eps) = measured scaled error / predicted constant, per group."""
    t = np.asarray(t, dtype=float)
    z = h * t
    em, eM = np.asarray(err_min, float), np.asarray(err_maj, float)
    e = np.asarray(eps, float)[None, :]
    out = {"z": z.tolist(),
           "scaled_min": (e * z[:, None] * em).tolist(),
           "scaled_maj": ((1 - e) * z[:, None] * eM).tolist()}
    out["R_min"] = ((e * z[:, None] * em) / A_min).tolist() if A_min > 0 else None
    out["R_maj"] = (((1 - e) * z[:, None] * eM) / A_maj).tolist() if A_maj > 0 else None
    return out


def per_sv_check(X, y, g, W, eps, h, z, alpha, sv) -> list[dict]:
    """At the final iterate: c_n z Q_n / alpha_n over support vectors, per eps."""
    from long_horizon import eps_weights
    Xy = y[:, None] * X
    rows = []
    for k, e in enumerate(eps):
        c = eps_weights(g, e)
        U = Xy @ W[:, k]
        Q = 1.0 / (1.0 + np.exp(np.clip(U, -700, 700)))
        rat = (c * z * Q)[sv] / alpha[sv]
        row = {"eps": e}
        for grp, name in ((1, "min"), (0, "maj")):
            sel = (g[sv] == grp)
            if sel.any():
                r = rat[sel]
                row[f"median_{name}"] = float(np.median(r))
                row[f"q25_{name}"] = float(np.quantile(r, 0.25))
                row[f"q75_{name}"] = float(np.quantile(r, 0.75))
            gm = g == grp
            row[f"nonsv_share_{name}"] = float(Q[gm & ~sv].sum() / Q[gm].sum())
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_curves(path_json: str) -> dict:
    if os.path.exists(path_json):
        return json.load(open(path_json))
    npz = path_json.replace("_curves.json", "_curves.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return {k: z[k].tolist() for k in z.files}
    raise SystemExit(f"neither {path_json} nor {npz} exists")


def locate_bundles(patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or sorted(glob.glob(os.path.join("results", p))) \
            or sorted(glob.glob(os.path.join("..", p)))
        out.extend(hits)
    return out


def run_synthetic(alpha_target: float, T: int, per_decade: int) -> dict:
    """The exact synthetic specs long_horizon.py uses, GD run here in float64."""
    sys.path.insert(0, os.path.join("..", "Estimator_Validation"))
    from synthetic import SyntheticSpec, generate
    from long_horizon import Engine, checkpoints, eps_weights
    gam_min = max(1.05, alpha_target * 1.25)
    b, truth = generate(SyntheticSpec(alpha=alpha_target, gam_min=gam_min, n=9000,
                                      d_r=5, d_s=8, eps=0.1, seed=7))
    X, y, g = b.phi.astype(float), b.y.astype(float), b.g.astype(int)
    eps = [0.01, 0.03, 0.08, 0.2, 0.5]
    h = 0.05
    C = np.stack([eps_weights(g, e) for e in eps], axis=1)
    eng = Engine(X, y, g, C, h, device="cpu", dtype="float64")
    ck = checkpoints(T, per_decade)
    bar = pbar(total=T, desc=f"  GD synthetic alpha={alpha_target:g}", unit="step")
    _, rec = eng.run_to(0, T, ck, bar=bar)
    bar.close()
    t = [r[0] for r in rec]
    return {"X": X, "y": y, "g": g, "eps": eps, "h": h, "t": t,
            "err_min": [r[1]["err_min"].tolist() for r in rec],
            "err_maj": [r[1]["err_maj"].tolist() for r in rec],
            "W": eng.weights_numpy(), "truth": truth}


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _at(z_arr, z0):
    z_arr = np.asarray(z_arr)
    i = int(np.argmin(np.abs(np.log(z_arr) - np.log(z0))))
    return i if abs(np.log(z_arr[i] / z0)) < 0.35 else None


def section(name, dual, consts, rz, eps, per_sv, extra="") -> list[str]:
    c = dual["certificate"]
    L = [f"## {name}", ""]
    if extra:
        L += [extra, ""]
    L += [f"- hard-margin dual CERTIFICATE: feasible **{c['feasible']}** (min functional "
          f"margin {c['min_functional_margin']:.8f}), duality gap {c['duality_gap_rel']:.1e}, "
          f"max slack on support vectors {c['max_slack_on_SVs']:.1e}, margin "
          f"{c['geometric_margin']:.6f} vs LinearSVC {c['linearsvc_margin']:.6f}, cosine to "
          f"LinearSVC {c['cos_to_linearsvc']:.8f}",
          f"- support vectors: {c['n_sv']} (minority {consts['n_sv_min']}, majority "
          f"{consts['n_sv_maj']}), rank {c['rank_sv']} in d = {c['d']}"
          + (" -- they do NOT span the data (Theorem 4's condition fails)"
             if c['rank_sv'] < c['d'] else ""),
          f"- **predicted constants: A_min = {consts['A_min']:.6g}, A_maj = "
          f"{consts['A_maj']:.6g}** (sum = {consts['A_min'] + consts['A_maj']:.6g} = "
          f"1/margin^2 = {1 / c['geometric_margin'] ** 2:.6g})", ""]
    for grp, key, A in (("minority", "R_min", consts["A_min"]),
                        ("majority", "R_maj", consts["A_maj"])):
        L += [f"### {grp}: R = measured scaled error / predicted constant (target 1)", ""]
        if rz[key] is None:
            sk = "scaled_min" if key == "R_min" else "scaled_maj"
            L += [f"The {grp} holds NO support vector, so the predicted constant is 0: "
                  f"the scaled error must go to 0 (the alpha > 1 branch). Scaled error:", "",
                  "| z | " + " | ".join(f"eps={e:g}" for e in eps) + " |",
                  "|---|" + "---|" * len(eps)]
            for z0 in Z_REPORT:
                i = _at(rz["z"], z0)
                if i is not None:
                    L.append(f"| {rz['z'][i]:.3g} | " +
                             " | ".join(f"{v:.3g}" for v in rz[sk][i]) + " |")
            L.append("")
            continue
        L += ["| z | " + " | ".join(f"eps={e:g}" for e in eps) + " | spread (max/min) |",
              "|---|" + "---|" * (len(eps) + 1)]
        for z0 in Z_REPORT:
            i = _at(rz["z"], z0)
            if i is None:
                continue
            row = np.asarray(rz[key][i])
            L.append(f"| {rz['z'][i]:.3g} | " + " | ".join(f"{v:.3f}" for v in row) +
                     f" | {row.max() / row.min():.3f} |")
        L.append("")
    if per_sv:
        L += ["### Point by point at the final iterate: c_n z Q_n / alpha_n over support "
              "vectors (target 1)", "",
              "| eps | minority median [q25, q75] | majority median [q25, q75] | "
              "non-SV share of err_min | non-SV share of err_maj |",
              "|---|---|---|---|---|"]
        for r in per_sv:
            mn = (f"{r['median_min']:.3f} [{r['q25_min']:.3f}, {r['q75_min']:.3f}]"
                  if "median_min" in r else "no minority SV")
            mj = (f"{r['median_maj']:.3f} [{r['q25_maj']:.3f}, {r['q75_maj']:.3f}]"
                  if "median_maj" in r else "no majority SV")
            L.append(f"| {r['eps']:g} | {mn} | {mj} | {r['nonsv_share_min']:.3f} | "
                     f"{r['nonsv_share_maj']:.3f} |")
        L.append("")
    return L


HEADER = """# The eps-law as a finite-sample implicit-bias prediction (sum-of-alphas check)

Prediction (derivation in `dual_mass_check.py`): for GD on the training set with
per-sample weights, `eps * z * err_min -> A_min` and `(1-eps) * z * err_maj -> A_maj`,
where A_g is the total hard-margin dual weight (sum of alpha_n) of group g's support
vectors. Both limits are independent of eps and z: the eps-law with kappa = 1, plus
a parameter-free VALUE for the constant. `R` below is measured / predicted.

**How to read it.** Convergence is slow (roughly logarithmic in z) and slowest at
small eps; on the self-test toy the minority support vector reads 0.34 of its limit
at eps = 0.05 and z = 2e5. So the test is the TREND down each column -- R moving
toward 1 -- and the eps-spread shrinking, not the value at the last z. First check
that every dual certificate is clean.
"""


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test() -> int:
    print("dual_mass_check self-test")
    ok = True
    # 1. exact dual on a case solvable by hand: y x in {e1, e2, (2,2)}.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    y = np.ones(3)
    d = hard_margin_dual(X, y, w_init=np.array([1.0, 1.0]), band=10.0)
    if not (np.allclose(d["alpha"], [1, 1, 0], atol=1e-6) and np.allclose(d["w_hat"], [1, 1], atol=1e-6)):
        print(f"  FAIL hand case: alpha {d['alpha']}, w {d['w_hat']}"); ok = False
    else:
        print("  ok   exact dual on a hand-solvable case (alpha = 1, 1, 0; w_hat = (1, 1))")

    # 2. certificates on a random separable problem, and the candidate-set loop:
    #    a band so narrow that the first solve MUST miss support vectors.
    rng = np.random.default_rng(0)
    n, dd = 600, 30
    yy = rng.choice([-1.0, 1.0], n)
    w0 = rng.normal(size=dd); w0 /= np.linalg.norm(w0)
    XX = rng.normal(size=(n, dd)) + 4.0 * yy[:, None] * w0
    XX = (XX - XX.mean(0)) / XX.std(0)
    d = hard_margin_dual(XX, yy)
    c = d["certificate"]
    if not (c["feasible"] and c["duality_gap_rel"] < 1e-6 and c["cos_to_linearsvc"] > 0.9999
            and abs(c["geometric_margin"] / c["linearsvc_margin"] - 1) < 1e-3):
        print(f"  FAIL certificates: {c}"); ok = False
    else:
        print(f"  ok   certified dual on n={n}, d={dd}: gap {c['duality_gap_rel']:.1e}, "
              f"{c['n_sv']} SVs, cosine to LinearSVC {c['cos_to_linearsvc']:.8f}")
    # A deliberately WRONG starting separator makes the first candidate set miss
    # support vectors; the feasibility loop must add them and land on the same alpha.
    w_bad = w0 + 0.8 * rng.normal(size=dd)
    d2 = hard_margin_dual(XX, yy, band=0.02, w_init=w_bad)
    c2 = d2["certificate"]
    if len(c2["rounds"]) < 2:
        print("  FAIL a wrong starting separator did not trigger a re-solve"); ok = False
    elif not (c2["feasible"] and np.allclose(d2["alpha"], d["alpha"], atol=1e-5 * d["alpha"].max())):
        print(f"  FAIL re-solve did not recover the same alpha: {c2}"); ok = False
    else:
        print(f"  ok   wrong starting separator: missed support vectors added over "
              f"{len(c2['rounds'])} rounds, same alpha recovered")

    # 3. the prediction itself on a toy where it is expected: trend toward 1.
    from long_horizon import Engine, eps_weights
    rng = np.random.default_rng(1)
    n, dd = 150, 3
    y3 = rng.choice([-1.0, 1.0], n)
    X3 = rng.normal(size=(n, dd)) + 3.0 * y3[:, None] * np.array([1.0, 0, 0])
    g3 = (rng.random(n) < 0.3).astype(int)
    d3 = hard_margin_dual(X3, y3)
    k3 = group_constants(d3["alpha"], d3["sv"], g3)
    eps = [0.05, 0.2, 0.5]
    C = np.stack([eps_weights(g3, e) for e in eps], axis=1)
    eng = Engine(X3, y3, g3, C, 0.05, device="cpu", dtype="float64")
    tprev, R = 0, []
    for T in (10 ** 4, 10 ** 5, 10 ** 6):
        _, rec = eng.run_to(tprev, T, [T]); tprev = T
        s = rec[-1][1]
        z = 0.05 * T
        R.append([e * z * s["err_min"][k] / k3["A_min"] for k, e in enumerate(eps)])
    R = np.array(R)
    toward = np.all(np.abs(np.log(R[-1])) < np.abs(np.log(R[0])))
    if not (k3["A_min"] > 0 and toward):
        print(f"  FAIL toy: R_min did not move toward 1: {R}"); ok = False
    else:
        print("  ok   toy: eps*z*err_min / A_min moves toward 1 at every eps "
              f"({np.round(R[0], 3).tolist()} -> {np.round(R[-1], 3).tolist()})")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bundles", nargs="+", default=["features_v3_waterbirds_*_train.npz"])
    ap.add_argument("--speed-tag", default="waterbirds_speed_v3",
                    help="long_horizon.py tag whose curves are compared")
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--synthetic-T", type=int, default=2_000_000,
                    help="GD steps for the synthetic runs (z = 0.05 T)")
    ap.add_argument("--per-decade", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="dual_mass_check")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()

    from common import FeatureBundle
    L = [HEADER]
    out = {}
    t0 = time.time()

    # ---- Waterbirds: the session-3 runs ------------------------------------
    js = os.path.join(args.out_dir, f"{args.speed_tag}.json")
    speed = json.load(open(js))
    curves = load_curves(os.path.join(args.out_dir, f"{args.speed_tag}_curves.json"))
    state_dir = os.path.join(args.out_dir, f"state_{args.speed_tag}")
    paths = locate_bundles(args.bundles)
    if not paths:
        print(f"WARNING: no bundle matched {args.bundles}; Waterbirds part skipped",
              file=sys.stderr)
    for p in pbar(paths, desc="bundles", unit="bundle"):
        key = os.path.splitext(os.path.basename(p))[0].replace("features_", "")
        if key not in speed or f"{key}__t" not in curves:
            print(f"  {key}: no long_horizon curves under tag {args.speed_tag}; skipped")
            continue
        print(f"\n[{key}]")
        fb = FeatureBundle.load(p)
        X, y, g = fb.phi.astype(np.float64), fb.y.astype(np.float64), fb.g.astype(int)
        dual = hard_margin_dual(X, y, band=args.band)
        consts = group_constants(dual["alpha"], dual["sv"], g)
        a = speed[key]["analysis"]
        eps, h = a["eps"], a["h"]
        rz = ratios_along_z(curves[f"{key}__t"], curves[f"{key}__err_min"],
                            curves[f"{key}__err_maj"], eps, h, consts["A_min"],
                            consts["A_maj"])
        per_sv = None
        sp = os.path.join(state_dir, f"{key}.npz")
        if os.path.exists(sp):
            st = np.load(sp)
            z_fin = h * float(st["t_done"])
            per_sv = per_sv_check(X, y, g, st["W"].astype(np.float64), eps, h, z_fin,
                                  dual["alpha"], dual["sv"])
        extra = ("state file not found; the point-by-point check is skipped"
                 if per_sv is None else f"point-by-point check at the final iterate, z = {z_fin:.3g}")
        L += section(key, dual, consts, rz, eps, per_sv, extra)
        out[key] = {"certificate": dual["certificate"], **consts, "eps": eps, "h": h,
                    "ratios": rz, "per_sv": per_sv}
        print(f"  A_min {consts['A_min']:.5g}  A_maj {consts['A_maj']:.5g}  "
              f"certificate feasible={dual['certificate']['feasible']} "
              f"gap={dual['certificate']['duality_gap_rel']:.1e}")

    # ---- Synthetic: where the theory's parameters are known ----------------
    if not args.no_synthetic:
        for at in pbar([0.6, 1.6], desc="synthetic", unit="run"):
            print(f"\n[synthetic alpha={at}]")
            s = run_synthetic(at, args.synthetic_T, args.per_decade)
            dual = hard_margin_dual(s["X"], s["y"], band=args.band)
            consts = group_constants(dual["alpha"], dual["sv"], s["g"])
            rz = ratios_along_z(s["t"], s["err_min"], s["err_maj"], s["eps"], s["h"],
                                consts["A_min"], consts["A_maj"])
            z_fin = s["h"] * s["t"][-1]
            per_sv = per_sv_check(s["X"], s["y"], s["g"], s["W"], s["eps"], s["h"],
                                  z_fin, dual["alpha"], dual["sv"])
            name = f"synthetic_alpha{at:g}"
            L += section(name, dual, consts, rz, s["eps"], per_sv,
                         f"Estimator_Validation/synthetic.py with long_horizon.py's specs "
                         f"(n=9000, d_r=5, d_s=8, seed=7); alpha from parameters = {at}; "
                         f"GD run here in float64 to z = {z_fin:.3g}")
            out[name] = {"certificate": dual["certificate"], **consts, "eps": s["eps"],
                         "h": s["h"], "ratios": rz, "per_sv": per_sv, "truth": s["truth"]}

    L += [f"Total wall clock: {(time.time() - t0) / 60:.1f} min.", ""]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.tag}.md"), "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
