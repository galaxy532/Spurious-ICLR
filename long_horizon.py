"""long_horizon.py -- is the run long enough? One long reweighted GD run per backbone.

WHAT THIS SCRIPT IS FOR
=======================

Two direct tests of the manuscript's isotropic theorem (thm:main-iso) on real
frozen features, both about SPEED of learning:

  1. eps-dependence: at a fixed training time z, does each group's error scale
     like 1/eps_g (the eps_g-law), or not depend on eps at all?
  2. z-dependence: does each group's error decay like 1/z, or faster, like
     z^(-alpha) with alpha > 1?

`eps_backbone_sweep.py --mode reweight` ran the same GD to z = 1e5 but kept only
the worst of the two groups, which cannot answer either question.

It trains the same model as the sweep (full-batch GD on the logistic loss, no
intercept, frozen standardised features, eps set by per-sample loss weights),
for all eps values AT ONCE, for as long as you ask, and records everything
needed to judge convergence. It can stop before the Paperspace session cap and
resume from where it stopped.

It does not replace the sweep and does not change any preregistered analysis.
It is a diagnostic.


EVERY QUANTITY IT REPORTS, IN WORDS
===================================

Training time.  z = h * t, where t is the number of GD steps and h the step
    size. With h = 0.05, T = 2e6 steps is z = 1e5 and T = 2e7 is z = 1e6.
    The loss weights sum to 1 at every eps, so z means the same thing at every
    eps and every eps is read at exactly the same z (they are one batched run).

Soft error of a group.  err_maj(z), err_min(z): over the training points of
    that group, the average of (1 - probability the model gives to the true
    label). This is the quantity in the manuscript's Theorem (thm:main-iso-easier).

Decay exponent beta.  If err(z) behaves like C * z^(-beta), then on a log-log
    plot of err against z it is a straight line of slope -beta. At a reported z
    this script takes every recorded point whose z lies between z/2 and 2z
    (a window four times wide, centred on z in log scale), fits a straight line
    to log err against log z by least squares, and reports minus the slope.
    So "beta_min at z = 1e6 is 0.93" means: between z = 5e5 and z = 2e6 the
    minority error fell like z^(-0.93).
    Theorem (thm:main-iso and its mirror case, neurips_2026.tex l.391-423):
      alpha < 1: beta -> 1 for BOTH groups.
      alpha > 1: the group with the LARGER r-margin has beta -> alpha > 1; the
                 other group has beta -> 1. Which group that is depends on the
                 geometry of Phi, not on which group is small.
    So "is alpha > 1?" is answered by beta, group by group. The theorem gives
    the limits only, not how fast they are reached.

eps-exponent of the minority, kappa_min.  At one fixed z, take the minority
    error from each eps run, fit a straight line to log err_min against log eps,
    and report minus the slope. So err_min ~ eps^(-kappa_min) across the grid.
    Theorem: a group whose beta -> 1 has error ~ const / (eps_g z) with the
    constant independent of eps. For the eps-weighted group (g = 1) that gives
    kappa_min -> 1. This holds for alpha < 1, AND for alpha > 1 when g = 1 is
    the smaller-margin group (mirror case) -- so kappa_min -> 1 alone does NOT
    show alpha < 1; read it together with beta.
    For the group that escapes (beta -> alpha > 1) the theorem's rate is
    Theta((ln z)^(alpha-1) / z^alpha) and its eps-dependence sits inside the
    Theta, unspecified. On synthetic data with alpha = 1.6 it went to 0
    (results/synthetic_long_horizon_alpha1.6.md) -- an observation, not a result
    of the theorem.

eps-exponent of the majority, kappa_maj.  Same fit with err_maj (group g = 0,
    weight 1-eps). If g = 0 is not the escaping group, err_maj ~ const /
    ((1 - eps) z), so kappa_maj -> the slope of log(1/(1-eps)) against log eps
    on the grid, printed as kappa_maj_theory (-0.157 for the default grid).

Scaled errors.  eps * z * err_min and (1 - eps) * z * err_maj. For a group with
    beta -> 1 the theorem says its scaled error tends to a constant that depends
    neither on z nor on eps.

Iterate margin.  For the current weights w: min over training points of
    y * (w . x) / ||w||. Negative means at least one training point is still
    misclassified. Soudry et al. (JMLR 2018, Theorem 5) show that on separable
    data it rises to the maximum margin at rate O(1/log t) -- so multiplying T
    by 10 moves it only a little. It is reported per group (tightest majority
    point, tightest minority point) and next to the full-split margin measured
    by separability_check (which is itself a lower bound on the true maximum).

Training accuracy per group.  Fraction of that group's points with y w.x > 0.

Separability.  The theorem needs linearly separable data (Assumption ass:sep).
    A bundle is run only if a separator is known: its full_split_margin in
    --margins-json, or, failing that, one found here by
    separability_check.quick_separable (can take minutes on CelebA). A bundle
    with no separator found is skipped unless --allow-nonseparable.


WHAT TO RUN ON PAPERSPACE
=========================

    python long_horizon.py --validate-only --device cuda          # ~1 min, must say OK
    python long_horizon.py --benchmark --device cuda \\
        --bundles 'features_waterbirds_*_train.npz'                # prints hours needed
    python long_horizon.py --device cuda --T 2000000 \\
        --bundles 'features_waterbirds_*_train.npz' --tag waterbirds_speed

The last command stops cleanly after --max-hours (default 5.5), saving its
state under results/state_<tag>/. Run the SAME command again in a new session
and it continues. Results for each bundle are written as soon as that bundle
finishes, and a partial report is written every time the state is saved.
Increasing --T on a later invocation extends a finished run instead of
restarting it.

PRECISION. On cuda the default dtype is "mixed": the weights are accumulated in
float64 and the two matrix products run in float32. Plain float32 loses part of
every late update (one step moves W by only a few float32 ulps by then) and biases
beta upward by 0.03-0.05 at the last z; kappa is unaffected at the 0.004 level.
Run `--validate-late` once on a new box to check the dtype end to end.

Outputs: results/<tag>.md (tables), results/<tag>.json (all reported numbers),
results/<tag>_curves.npz (every recorded point, so nothing needs re-running to
make a plot). Default T = 2e6 (z = 1e5), the same horizon as the 16 Sept sweep.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

from progress import pbar

DEFAULT_EPS = "0.01,0.03,0.08,0.2,0.5"
STAT_KEYS = ("err_maj", "err_min", "acc_maj", "acc_min",
             "margin", "margin_maj", "margin_min", "wnorm")


# ---------------------------------------------------------------------------
# weights and checkpoints
# ---------------------------------------------------------------------------

def eps_weights(g: np.ndarray, eps: float) -> np.ndarray:
    """Same weights as eps_backbone_sweep.eps_weights (copied, not imported, so
    this script does not pull in the sweep's sklearn/separability imports)."""
    n_maj, n_min = int(np.sum(g == 0)), int(np.sum(g == 1))
    if n_maj == 0 or n_min == 0:
        raise ValueError("both groups must be present")
    c = np.empty(g.size, dtype=np.float64)
    c[g == 0] = (1.0 - eps) / n_maj
    c[g == 1] = eps / n_min
    return c


def checkpoints(T: int, per_decade: int) -> np.ndarray:
    """Log-spaced step indices that do NOT depend on T (so a run can be extended):
    t_k = round(10^(k / per_decade)), deduplicated, up to T, and T itself."""
    kmax = int(np.ceil(np.log10(max(T, 1)) * per_decade))
    t = np.unique(np.round(10.0 ** (np.arange(kmax + 1) / per_decade)).astype(np.int64))
    t = t[(t >= 1) & (t <= T)]
    if t.size == 0 or t[-1] != T:
        t = np.append(t, T)
    return t


# ---------------------------------------------------------------------------
# the batched GD engine (numpy or torch, identical recurrence)
# ---------------------------------------------------------------------------

class Engine:
    """Runs K weighted GD recurrences in one batch.

    W has shape (d, K), one column per eps. One step is

        U = Xy @ W                 (N, K)   margins y_i w_k . x_i
        Q = 1 - sigmoid(U)         (N, K)   computed as in common.logistic_gd
        W = W + h * Xy.T @ (C * Q)          C[i, k] = weight of point i at eps_k

    Column k is exactly gd_gpu.logistic_gd_weighted with c = C[:, k]; only the
    order of floating-point additions inside the matrix products can differ,
    which --validate measures.

    As in common.logistic_gd, the statistics recorded at step t are computed
    from the weights ENTERING step t, before the update.
    """

    def __init__(self, X, y, g, C, h, device="cpu", dtype="float64", W0=None):
        self.h = float(h)
        self.device = device
        self.N, self.d = X.shape
        self.K = C.shape[1]
        self.m_maj = (g == 0)
        self.m_min = (g == 1)
        # dtype "mixed": the weights are accumulated in float64 while the two
        # matrix products stay in float32. Late in training one GD step changes W
        # by only a few float32 ulps (measured: ~20 ulps at z = 1e5 on Waterbirds
        # dinov2, fewer later), so a pure float32 run loses part of every update.
        # That is invisible in kappa (<0.004) but biases beta upward by 0.03-0.05
        # at the last z, measured against a float64 reference. The products
        # themselves only need ~1e-7 relative accuracy, hence this split.
        self.mixed = (dtype == "mixed")
        Xy = (y[:, None] * X).astype(np.float64)
        W0 = np.zeros((self.d, self.K)) if W0 is None else np.asarray(W0, np.float64)

        if device == "cpu":
            npd = {"float32": np.float32, "float64": np.float64,
                   "mixed": np.float32}[dtype]
            self.torch = None
            self.Xy = np.ascontiguousarray(Xy.astype(npd))
            self.XyT = np.ascontiguousarray(self.Xy.T)
            self.C = C.astype(npd)
            self.W = W0.astype(np.float64 if self.mixed else npd)
            # compute-precision copy of W; an alias for W unless we are mixed
            self.Wc = np.empty((self.d, self.K), npd) if self.mixed else self.W
            self.U = np.empty((self.N, self.K), npd)
            self.E = np.empty_like(self.U)
            self.D = np.empty_like(self.U)
            self.Q = np.empty_like(self.U)
            self.P = np.empty((self.N, self.K), bool)
            self.G = np.empty((self.d, self.K), npd)
        else:
            import torch
            self.torch = torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("device is cuda but torch.cuda.is_available() "
                                   "is False; refusing to fall back to CPU silently")
            td = {"float32": torch.float32, "float64": torch.float64,
                  "mixed": torch.float32}[dtype]
            self.Xy = torch.as_tensor(Xy, dtype=td, device=device)
            self.C = torch.as_tensor(C, dtype=td, device=device)
            self.W = torch.as_tensor(
                W0, dtype=torch.float64 if self.mixed else td, device=device).clone()
            self.mmaj_t = torch.as_tensor(self.m_maj, device=device)
            self.mmin_t = torch.as_tensor(self.m_min, device=device)

    # -- one step; returns U and Q of the iterate entering the step ------------
    def _step_numpy(self):
        if self.mixed:
            np.copyto(self.Wc, self.W)      # float64 weights -> float32 products
        np.matmul(self.Xy, self.Wc, out=self.U)
        np.abs(self.U, out=self.E)
        np.negative(self.E, out=self.E)
        np.exp(self.E, out=self.E)                       # e = exp(-|u|)
        np.add(1.0, self.E, out=self.D)                  # 1 + e
        np.greater_equal(self.U, 0.0, out=self.P)
        # u >= 0: e/(1+e) = exp(-u)/(1+exp(-u));  u < 0: 1/(1+e) = 1/(1+exp(u))
        np.divide(self.E, self.D, out=self.Q, where=self.P)
        np.divide(1.0, self.D, out=self.Q, where=~self.P)
        np.multiply(self.Q, self.C, out=self.D)          # reuse D as C*Q
        np.matmul(self.XyT, self.D, out=self.G)
        self.G *= self.h
        self.W += self.G                    # in mixed mode: float64 += float32

    def _step_torch(self, record):
        torch = self.torch
        Wc = self.W.to(self.Xy.dtype) if self.mixed else self.W
        U = self.Xy @ Wc
        # same expression as gd_gpu.logistic_gd_torch (validated on the box);
        # torch.sigmoid is numerically stable for large |U|
        Q = torch.sigmoid(-U)
        if record:
            stats = self._stats_torch(U, Q)
        G = self.Xy.T @ (Q * self.C)
        self.W.add_(G.double() if self.mixed else G, alpha=self.h)
        return stats if record else None

    def _stats_torch(self, U, Q):
        wn = self.torch.linalg.norm(self.W, dim=0)
        Umaj, Umin = U[self.mmaj_t], U[self.mmin_t]
        out = {
            "err_maj": Q[self.mmaj_t].mean(0), "err_min": Q[self.mmin_t].mean(0),
            "acc_maj": (Umaj > 0).double().mean(0), "acc_min": (Umin > 0).double().mean(0),
            "margin": U.min(0).values / wn,
            "margin_maj": Umaj.min(0).values / wn,
            "margin_min": Umin.min(0).values / wn,
            "wnorm": wn,
        }
        return {k: v.double().cpu().numpy() for k, v in out.items()}

    def run_to(self, t_now, t_target, record_at, bar=None):
        """Advance from step t_now (already done) to t_target. Returns
        (t_reached, list of (t, stats)) with stats for every t in record_at."""
        rec = []
        record_at = set(int(x) for x in record_at)
        chunk = 1000
        t = t_now
        while t < t_target:
            t += 1
            do_rec = t in record_at
            if self.torch is None:
                if do_rec:
                    # _step_numpy leaves U and Q of the iterate ENTERING the step
                    self._step_numpy()
                    rec.append((t, self._stats_numpy_pre()))
                else:
                    self._step_numpy()
            else:
                s = self._step_torch(do_rec)
                if do_rec:
                    rec.append((t, s))
            if bar is not None and t % chunk == 0:
                bar.update(chunk)
        return t, rec

    def _stats_numpy_pre(self):
        # U and Q still hold the pre-update values; W is post-update. The norm
        # and margin must use the PRE-update W, which is W - G.
        W_pre = self.W - self.G
        wn = np.linalg.norm(W_pre, axis=0)
        safe = np.where(wn > 0, wn, np.nan)
        U, Q = self.U, self.Q
        return {
            "err_maj": Q[self.m_maj].mean(0).astype(np.float64),
            "err_min": Q[self.m_min].mean(0).astype(np.float64),
            "acc_maj": (U[self.m_maj] > 0).mean(0), "acc_min": (U[self.m_min] > 0).mean(0),
            "margin": (U.min(0) / safe).astype(np.float64),
            "margin_maj": (U[self.m_maj].min(0) / safe).astype(np.float64),
            "margin_min": (U[self.m_min].min(0) / safe).astype(np.float64),
            "wnorm": wn.astype(np.float64),
        }

    def weights_numpy(self):
        if self.torch is None:
            return self.W.astype(np.float64).copy()
        return self.W.double().cpu().numpy()

    def sync(self):
        if self.torch is not None and str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _lstsq_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    A = np.vstack([x[m], np.ones(m.sum())]).T
    return float(np.linalg.lstsq(A, y[m], rcond=None)[0][0])


def _slope_resid(x, y):
    """(-slope, max |residual|) of a least-squares line fit. The residual says
    whether the points ARE a straight line: a kappa fitted through a visibly bent
    curve is a number, not an estimate of an exponent."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    A = np.vstack([x[m], np.ones(m.sum())]).T
    coef = np.linalg.lstsq(A, y[m], rcond=None)[0]
    resid = y[m] - A @ coef
    return float(-coef[0]), float(np.max(np.abs(resid)))


def report_z_values(z: np.ndarray) -> np.ndarray:
    """1, 3, 10, 30, ... up to the last z, each snapped to the nearest recorded z,
    keeping only those with a full [z/2, 2z] window inside the run (the last one
    is allowed a half window)."""
    cands = []
    k = 0
    while True:
        for m in (1.0, 3.0):
            v = m * 10.0 ** k
            if v > z[-1]:
                break
            cands.append(v)
        if 10.0 ** (k + 1) > z[-1]:
            break
        k += 1
    cands.append(z[-1])
    out = []
    for v in cands:
        i = int(np.argmin(np.abs(np.log(z) - np.log(v))))
        if z[i] / 2 >= z[0]:
            out.append(z[i])
    return np.unique(np.array(out))


def analyze(t, stats, eps, h, ref_margin=None) -> dict:
    t = np.asarray(t, float)
    z = h * t
    eps = np.asarray(eps, float)
    le = np.log(eps)
    if np.any(eps >= 1.0) or np.any(eps <= 0.0):
        raise ValueError("eps must lie strictly between 0 and 1")
    l1me = np.log(1.0 - eps)             # the majority's own weight share
    kmaj_theory = -_lstsq_slope(le, -np.log(1.0 - eps))
    zr = report_z_values(z)
    rows = []
    for zv in zr:
        i = int(np.argmin(np.abs(z - zv)))
        win = (z >= zv / 2) & (z <= zv * 2) & (z > 0)
        row = {"z": float(z[i]), "t": int(t[i])}
        row["half_window"] = bool(z[-1] < zv * 2)
        for grp in ("min", "maj"):
            e = stats[f"err_{grp}"]                      # (n_ckpt, K)
            with np.errstate(divide="ignore"):
                row[f"beta_{grp}"] = [
                    -_lstsq_slope(np.log(z[win]), np.log(e[win, k]))
                    if win.sum() >= 3 else float("nan") for k in range(len(eps))]
        # Each group is fitted against ITS OWN weight share, which is how the
        # theorem writes the rates: err_min ~ c/(eps z) and err_maj ~ c/((1-eps) z).
        # So kappa_min -> 1 and kappa_maj -> 1, both grid-independent. Fitting the
        # majority against log(eps) instead (what this script did before 18 Sept
        # 2026) gives a target that depends on the eps grid -- -0.157 for
        # 0.01,0.03,0.08,0.2,0.5 -- because log(1-eps) is not a straight line in
        # log(eps). That legacy number is kept as kappa_maj_eps so the earlier
        # reports stay comparable; kappa_maj is the one to read.
        with np.errstate(divide="ignore"):
            row["kappa_min"], row["kappa_min_resid"] = _slope_resid(
                le, np.log(stats["err_min"][i]))
            row["kappa_maj"], row["kappa_maj_resid"] = _slope_resid(
                l1me, np.log(stats["err_maj"][i]))
            row["kappa_maj_eps"] = -_lstsq_slope(le, np.log(stats["err_maj"][i]))
        row["eps_z_err_min"] = (eps * z[i] * stats["err_min"][i]).tolist()
        row["one_minus_eps_z_err_maj"] = ((1 - eps) * z[i] * stats["err_maj"][i]).tolist()
        for key in ("margin", "margin_maj", "margin_min", "acc_maj", "acc_min"):
            row[key] = stats[key][i].tolist()
        rows.append(row)
    return {"eps": eps.tolist(), "h": h, "z_max": float(z[-1]),
            "kappa_maj_theory": kmaj_theory, "ref_margin": ref_margin,
            "rows": rows}


def _f(x, p=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def _g(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3g}"


def to_markdown(results: dict, partial: bool) -> str:
    L = ["# Long-horizon diagnostics", ""]
    if partial:
        L += ["**PARTIAL** -- written at a save point; the run is not finished.", ""]
    L += ["All quantities are defined in words in the docstring of "
          "`long_horizon.py`. Short version: z = h * (number of GD steps); "
          "beta = decay exponent of a group's soft error in z, fitted on the "
          "window [z/2, 2z]; kappa = exponent of a group's soft error in eps at "
          "that single z, fitted across the eps grid.", ""]
    for key, r in results.items():
        a = r["analysis"]
        eps = a["eps"]
        L += [f"## {key}", "",
              f"status: {r['status']}, z reached: {a['z_max']:.4g}, "
              f"full-split margin (lower bound, separability_check): "
              f"{_g(a['ref_margin'])}", ""]
        L += ["### eps-exponents at each z", "",
              "Each group is fitted against its OWN weight share, as the theorem "
              "writes it: err_min ~ c/(eps z) gives kappa_min -> 1 (regression of "
              "log err_min on log eps), err_maj ~ c/((1-eps) z) gives kappa_maj -> 1 "
              "(regression of log err_maj on log(1-eps)). Both targets are 1 and "
              "neither depends on the eps grid. Each fit uses the "
              f"{len(eps)} eps values at that single z.", "",
              "'max resid' is the largest residual of that 5-point fit: it says "
              "whether the points are a straight line at all. Below ~0.02 the "
              "exponent means something; at 0.1 or more the curve is bent and the "
              "slope is just a number. A group whose beta has not reached its limit "
              "has no reason to show a clean power law in eps.", "",
              "kappa_maj_eps is the superseded definition (majority fitted against "
              f"log eps), whose grid-dependent target is {a['kappa_maj_theory']:.3f} "
              "for this grid; it is kept only to compare with reports written "
              "before 18 Sept 2026. For a group whose beta -> alpha > 1 the theorem "
              "gives no eps-dependence at all.", "",
              "| z | kappa_min (-> 1) | max resid | kappa_maj (-> 1) | max resid | "
              "kappa_maj_eps (legacy) |", "|---|---|---|---|---|---|"]
        for row in a["rows"]:
            L.append(f"| {row['z']:.3g} | {_f(row['kappa_min'])} | "
                     f"{_f(row.get('kappa_min_resid'))} | {_f(row['kappa_maj'])} | "
                     f"{_f(row.get('kappa_maj_resid'))} | "
                     f"{_f(row.get('kappa_maj_eps'))} |")
        L += ["", "### decay exponents in z (one column per eps)", "",
              "Theory: alpha < 1: both groups -> 1. alpha > 1: the larger-r-margin "
              "group -> alpha, the other -> 1.", ""]
        head = "| z | " + " | ".join(f"eps={e:g}" for e in eps) + " |"
        sep = "|---|" + "---|" * len(eps)
        for grp in ("min", "maj"):
            L += [f"beta_{grp}:", "", head, sep]
            for row in a["rows"]:
                mark = " (half window)" if row.get("half_window") else ""
                L.append(f"| {row['z']:.3g}{mark} | " +
                         " | ".join(_f(v) for v in row[f"beta_{grp}"]) + " |")
            L += ["", "A row marked (half window) is fitted on [z/2, z] only, "
                  "because the run stops at that z; it is biased and should not be "
                  "read as the limit.", ""]
        L += ["### scaled errors (theory: constant in z and eps for a group whose beta -> 1)", ""]
        for name in ("eps_z_err_min", "one_minus_eps_z_err_maj"):
            L += [f"{name}:", "", head, sep]
            for row in a["rows"]:
                L.append(f"| {row['z']:.3g} | " +
                         " | ".join(_g(v) for v in row[name]) + " |")
            L.append("")
        L += ["### iterate margin and training accuracy", "",
              "margin = min over training points of y w.x / ||w|| "
              "(negative: some point misclassified). Rises to the maximum margin "
              "at rate O(1/log t) (Soudry et al. 2018, Thm 5).", "",
              head, sep]
        for row in a["rows"]:
            L.append(f"| {row['z']:.3g} | " +
                     " | ".join(_g(v) for v in row["margin"]) + " |")
        L += ["", "minority training accuracy:", "", head, sep]
        for row in a["rows"]:
            L.append(f"| {row['z']:.3g} | " +
                     " | ".join(_f(v) for v in row["acc_min"]) + " |")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# state (resume across sessions)
# ---------------------------------------------------------------------------

def _config_id(key, X, y, g, eps, h, dtype, per_decade):
    hsh = hashlib.sha256()
    hsh.update(np.ascontiguousarray(X[:50]).tobytes())
    hsh.update(np.ascontiguousarray(y).tobytes())
    hsh.update(np.ascontiguousarray(g).tobytes())
    hsh.update(json.dumps([key, list(map(float, eps)), float(h), dtype,
                           int(per_decade), list(X.shape)]).encode())
    return hsh.hexdigest()[:16]


def save_state(path, cfg_id, t_done, W, t_list, stats_list):
    tmp = path + ".tmp.npz"
    arrays = {k: np.array([s[k] for s in stats_list]) if stats_list else np.zeros((0,))
              for k in STAT_KEYS}
    np.savez(tmp, cfg_id=cfg_id, t_done=t_done, W=W,
             t_list=np.array(t_list, dtype=np.int64), **arrays)
    os.replace(tmp, path)


def load_state(path, cfg_id):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["cfg_id"]) != cfg_id:
        raise SystemExit(
            f"state file {path} belongs to a different configuration (data, eps, "
            f"h, dtype or checkpoint density changed). Refusing to mix runs. "
            f"Use a different --tag, or delete that file deliberately.")
    t_list = z["t_list"].tolist()
    stats = [{k: z[k][i] for k in STAT_KEYS} for i in range(len(t_list))]
    return int(z["t_done"]), z["W"], t_list, stats


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(device="cpu", T=3000, h=0.05, eps=(0.02, 0.1, 0.5)) -> bool:
    """Batched engine vs gd_gpu.logistic_gd_weighted run one eps at a time.

    1. float64 on the chosen device must agree with the float64 numpy reference
       (tight tolerance): this checks the implementation.
    2. float32 is reported against the same reference: this is the precision
       budget, not a pass/fail on correctness.
    3. The iterate margin is recomputed from the returned weights in plain numpy.
    """
    from gd_gpu import _make_problem, logistic_gd_weighted, _max_rel

    X, y, g = _make_problem(n=3000, d=40, eps=0.2, seed=3)
    C = np.stack([eps_weights(g, e) for e in eps], axis=1)
    ref_ck = np.unique(np.round(np.logspace(0, np.log10(T), 40)).astype(int))
    ref = [logistic_gd_weighted(X, y, g, c=C[:, k], h=h, T=T, n_ckpt=40)
           for k in range(len(eps))]
    ok = True
    for dtype, tol in (("float64", 1e-9), ("mixed", None), ("float32", None)):
        eng = Engine(X, y, g, C, h, device=device, dtype=dtype)
        _, rec = eng.run_to(0, T, ref_ck)
        worst = 0.0
        for k in range(len(eps)):
            for key in ("err_maj", "err_min"):
                got = np.array([s[key][k] for _, s in rec])
                worst = max(worst, _max_rel(got, ref[k][key]))
        if tol is not None:
            flag = "OK" if worst <= tol else "FAIL"
            ok &= worst <= tol
            print(f"  [{device} {dtype}] batched vs per-eps reference: "
                  f"max rel diff {worst:.2e}  (tol {tol:.0e})  {flag}")
        else:
            print(f"  [{device} {dtype}] precision budget: max rel diff {worst:.2e}")
        if dtype == "float64":
            W = eng.weights_numpy()
            # stats at the final checkpoint were taken from the iterate entering
            # step T; recompute them from scratch for that iterate
            eng2 = Engine(X, y, g, C, h, device="cpu", dtype="float64")
            eng2.run_to(0, T - 1, [])
            Wp = eng2.weights_numpy()
            U = (y[:, None] * X) @ Wp
            m_direct = U.min(0) / np.linalg.norm(Wp, axis=0)
            m_rec = rec[-1][1]["margin"]
            dm = float(np.max(np.abs(m_direct - m_rec)))
            ok &= dm < 1e-9
            print(f"  [{device} float64] iterate margin recomputed directly: "
                  f"max abs diff {dm:.2e}  {'OK' if dm < 1e-9 else 'FAIL'}")
            del W
    print("VALIDATION OK" if ok else "VALIDATION FAILED")
    return ok


def validate_late(device="cpu", dtype="mixed", T=2_000_000, h=0.05,
                  eps=(0.01, 0.1, 0.5), tol_err=5e-3, tol_beta=0.02) -> bool:
    """LATE-TIME precision check: the dtype a long run will actually use, against a
    float64 numpy reference, on a small problem taken all the way to T.

    `validate` cannot see this problem. It compares dtypes at T = 3000, where they
    agree to 5e-7. The error appears only once a single GD step changes the weights
    by a few float32 ulps, which happens late: on a synthetic problem run to
    z = 1e5 (T = 2e6), float32 against float64 was 1.6e-2 off in the soft error and
    +0.04 in beta at the last z, while kappa moved by 0.004. This check refits both
    and fails if the dtype under test is not within tol_err / tol_beta of the
    reference. Small problem, so it costs a couple of minutes, not hours."""
    from gd_gpu import _make_problem

    X, y, g = _make_problem(n=400, d=30, eps=0.2, seed=5)
    C = np.stack([eps_weights(g, e) for e in eps], axis=1)
    ck = checkpoints(T, 20)
    runs = {}
    for label, dt, dev in (("test", dtype, device), ("ref", "float64", "cpu")):
        eng = Engine(X, y, g, C, h, device=dev, dtype=dt)
        bar = pbar(total=T, desc=f"  late check {label} ({dt} on {dev})", unit="step")
        _, rec = eng.run_to(0, T, ck, bar=bar)
        bar.close()
        runs[label] = ([r[0] for r in rec],
                       {k: np.array([r[1][k] for r in rec]) for k in STAT_KEYS})
    t, st = runs["test"]
    _, rf = runs["ref"]
    worst_err = 0.0
    for key in ("err_maj", "err_min"):
        worst_err = max(worst_err, float(np.max(np.abs(st[key] / rf[key] - 1.0))))
    a_t = analyze(t, st, eps, h)
    a_r = analyze(t, rf, eps, h)
    db = dk = 0.0
    for rt, rr in zip(a_t["rows"], a_r["rows"]):
        for grp in ("min", "maj"):
            db = max(db, float(np.max(np.abs(np.array(rt[f"beta_{grp}"])
                                             - np.array(rr[f"beta_{grp}"])))))
            dk = max(dk, abs(rt[f"kappa_{grp}"] - rr[f"kappa_{grp}"]))
    ok = (worst_err <= tol_err) and (db <= tol_beta)
    print(f"  [{dtype} on {device}] vs float64 reference at T={T:.3g} "
          f"(z={h * T:.3g}): worst relative error in the soft error "
          f"{worst_err:.2e} (tol {tol_err:.0e}), worst beta difference {db:.3f} "
          f"(tol {tol_beta:.2f}), worst kappa difference {dk:.3f}")
    print("LATE VALIDATION OK" if ok else "LATE VALIDATION FAILED")
    return ok


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_inputs(args):
    """Yield (key, X, y, g, ref_margin)."""
    ref_margins = {}
    if args.margins_json and os.path.exists(args.margins_json):
        with open(args.margins_json) as f:
            for k, v in json.load(f).items():
                if v.get("full_split_margin") is not None:
                    ref_margins[k] = float(v["full_split_margin"])
                elif v.get("separable") is False:
                    ref_margins[k] = False      # proven non-separable

    if args.synthetic_alpha:
        sys.path.insert(0, args.synthetic_dir)
        from synthetic import SyntheticSpec, generate
        for a in [float(x) for x in args.synthetic_alpha.split(",")]:
            gam_min = max(1.05, a * 1.25)
            b, _ = generate(SyntheticSpec(alpha=a, gam_min=gam_min, n=args.synthetic_n,
                                          d_r=5, d_s=8, eps=0.1, seed=7))
            yield f"synthetic_alpha{a:g}", b.phi, b.y, b.g, None
        return

    from common import FeatureBundle
    paths = []
    for p in args.bundles:
        paths.extend(sorted(glob.glob(p)) or [p])
    for p in paths:
        key = os.path.splitext(os.path.basename(p))[0].replace("features_", "")
        if key in args.skip:
            print(f"[{key}] skipped (--skip)")
            continue
        fb = FeatureBundle.load(p)
        if not fb.meta.get("standardized", False):
            print(f"  WARNING: {key} is not marked standardised")
        yield key, fb.phi, fb.y, fb.g, ref_margins.get(key)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", nargs="*", default=["features_waterbirds_*_train.npz"])
    ap.add_argument("--skip", nargs="*", default=["waterbirds_clip_train"],
                    help="bundle keys to skip; CLIP is not separable on Waterbirds")
    ap.add_argument("--eps", default=DEFAULT_EPS)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=2_000_000,
                    help="total GD steps (z = h*T). Can be raised on a later run.")
    ap.add_argument("--per-decade", type=int, default=20,
                    help="recorded points per decade of t")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None,
                    choices=[None, "float32", "float64", "mixed"],
                    help="default 'mixed' on cuda (float64 weights, float32 matrix "
                         "products) and float64 on cpu. Plain float32 biases beta "
                         "at large z; see Engine and validate_late.")
    ap.add_argument("--max-hours", type=float, default=5.5,
                    help="stop cleanly (state saved) after this much wall-clock")
    ap.add_argument("--save-every-min", type=float, default=15.0)
    ap.add_argument("--allow-nonseparable", action="store_true")
    ap.add_argument("--margins-json", default="results/waterbirds_rw.json",
                    help="where to read the full-split margins from, if present")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="long_horizon")
    ap.add_argument("--benchmark", action="store_true",
                    help="time 300 steps per bundle, print projected hours, exit")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--validate-late", action="store_true",
                    help="also run the late-time precision check (a couple of "
                         "minutes) before the real run, and refuse to continue if "
                         "it fails")
    ap.add_argument("--late-T", type=int, default=2_000_000,
                    help="horizon of the late-time check")
    ap.add_argument("--synthetic-alpha", default=None,
                    help="e.g. 0.6,1.6: run on Estimator_Validation synthetic data "
                         "with known alpha instead of bundles")
    ap.add_argument("--synthetic-dir", default="../Estimator_Validation")
    ap.add_argument("--synthetic-n", type=int, default=9000)
    args = ap.parse_args()

    dtype = args.dtype or ("float64" if args.device == "cpu" else "mixed")
    eps = [float(x) for x in args.eps.split(",")]

    print(f"validating the batched engine on {args.device} (dtype {dtype}) ...")
    if not validate(device=args.device):
        raise SystemExit("validation failed; refusing to run")
    if args.validate_late:
        if not validate_late(device=args.device, dtype=dtype, T=args.late_T,
                             h=args.h):
            raise SystemExit("late-time validation failed; refusing to run")
    if args.validate_only:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    state_dir = os.path.join(args.out_dir, f"state_{args.tag}")
    os.makedirs(state_dir, exist_ok=True)
    t_start = time.time()
    deadline = t_start + 3600.0 * args.max_hours
    results = {}
    curves = {}
    stopped_early = False

    inputs = list(load_inputs(args))
    for key, X, y, g, ref_margin in pbar(inputs, desc="bundles", unit="bundle"):
        C = np.stack([eps_weights(g, e) for e in eps], axis=1)
        print(f"[{key}] n={X.shape[0]} d={X.shape[1]} n_min={int((g == 1).sum())} "
              f"ref margin={_g(ref_margin) if ref_margin is not False else 'not separable'}")
        if ref_margin is False:
            if not args.allow_nonseparable:
                print("  PROVEN NOT SEPARABLE (margins json): the regime the theorem "
                      "describes does not exist here. Skipping.")
                continue
            ref_margin = None
        elif ref_margin is None and not args.synthetic_alpha:
            from separability_check import quick_separable
            print("  no recorded margin; searching for a separator (constructive only)")
            sep, marg = quick_separable(X, y)
            if sep is True:
                ref_margin = float(marg)
                print(f"  separable, margin {marg:.4g}")
            elif not args.allow_nonseparable:
                print("  NO SEPARATOR FOUND: the implicit-bias regime the theorem "
                      "describes does not exist here. Skipping (run "
                      "separability_check.py for a proof, or pass "
                      "--allow-nonseparable).")
                continue

        if args.benchmark:
            eng = Engine(X, y, g, C, args.h, device=args.device, dtype=dtype)
            eng.run_to(0, 50, [])
            eng.sync()
            t0 = time.time()
            eng.run_to(50, 350, [])
            eng.sync()
            ms = (time.time() - t0) / 300 * 1e3
            print(f"  {ms:.3f} ms/step for all {len(eps)} eps together -> "
                  f"T={args.T:.3g} needs {ms * args.T / 3.6e6:.2f} h")
            continue

        cfg = _config_id(key, X, y, g, eps, args.h, dtype, args.per_decade)
        spath = os.path.join(state_dir, f"{key}.npz")
        st = load_state(spath, cfg)
        if st is None:
            t_done, W0, t_list, stats_list = 0, None, [], []
        else:
            t_done, W0, t_list, stats_list = st
            print(f"  resuming from step {t_done} (z = {args.h * t_done:.4g})")
        eng = Engine(X, y, g, C, args.h, device=args.device, dtype=dtype, W0=W0)
        ck = checkpoints(args.T, args.per_decade)
        ck = ck[ck > t_done]

        bar = pbar(total=args.T - t_done, desc=f"  GD {key}", unit="step")
        last_save = time.time()
        status = "finished"
        while t_done < args.T:
            # advance in segments so the clock and the save timer are checked
            seg_end = min(args.T, t_done + 20_000)
            t_done, rec = eng.run_to(t_done, seg_end, ck[ck <= seg_end], bar=bar)
            for tt, s in rec:
                t_list.append(tt)
                stats_list.append(s)
            if rec:
                s = rec[-1][1]
                bar.set_postfix_str(f"z={args.h * t_done:.3g} "
                                    f"err_min[eps0]={s['err_min'][0]:.2e}")
            now = time.time()
            if now - last_save > 60 * args.save_every_min or now > deadline:
                save_state(spath, cfg, t_done, eng.weights_numpy(), t_list, stats_list)
                last_save = now
                _write(args, results | {key: _pack(key, t_list, stats_list, eps,
                                                   args.h, ref_margin, "running")},
                       curves, partial=True)
            if now > deadline:
                status = "stopped at --max-hours; rerun the same command to resume"
                stopped_early = True
                break
        bar.close()
        save_state(spath, cfg, t_done, eng.weights_numpy(), t_list, stats_list)
        results[key] = _pack(key, t_list, stats_list, eps, args.h, ref_margin, status)
        curves[key] = (t_list, stats_list)
        _write(args, results, curves, partial=stopped_early)
        if stopped_early:
            print(f"  {status}")
            break

    if not args.benchmark:
        _write(args, results, curves, partial=stopped_early)
        print(to_markdown(results, stopped_early))


def _pack(key, t_list, stats_list, eps, h, ref_margin, status):
    stats = {k: np.array([s[k] for s in stats_list]) for k in STAT_KEYS}
    return {"status": status,
            "analysis": analyze(t_list, stats, eps, h, ref_margin)
            if len(t_list) >= 6 else {"eps": eps, "h": h, "z_max": h * (t_list[-1] if t_list else 0),
                                      "kappa_maj_theory": float("nan"),
                                      "ref_margin": ref_margin, "rows": []}}


def _write(args, results, curves, partial):
    base = os.path.join(args.out_dir, args.tag)
    with open(base + ".md", "w") as f:
        f.write(to_markdown(results, partial))
    with open(base + ".json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    if curves:
        arrs = {}
        for key, (t_list, stats_list) in curves.items():
            arrs[f"{key}__t"] = np.array(t_list)
            for k in STAT_KEYS:
                arrs[f"{key}__{k}"] = np.array([s[k] for s in stats_list])
        np.savez_compressed(base + "_curves.npz", **arrs)


if __name__ == "__main__":
    np.seterr(over="ignore", under="ignore")
    main()
