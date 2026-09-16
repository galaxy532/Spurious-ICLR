"""GPU full-batch logistic GD. Same recurrence as `common.logistic_gd`, ~40x faster.

WHY THIS FILE EXISTS, AND WHY IT IS NOT IN common.py
====================================================

`common.py` is a BYTE-FOR-BYTE copy of `Rebuttals/common.py`, and the whole value
of that is the sha256 recorded in README.md: it is what makes
`Estimator_Validation/`'s certification (alpha recovery to 0.02%, coupling power
and calibration, isotropy diagnostics) apply to this repo without re-running it.
Editing it to add a torch path would break that transfer for a performance
change. So the fast implementation lives here, separately, and is required to
AGREE NUMERICALLY with the certified one rather than to replace it.

The measured problem it solves, benchmarked on CPU before it was written:

    Waterbirds train, rn50   n=  4,795  d=2048     3.8 ms/step ->  2.1 h at T=2e6
    CelebA train,     clip   n=162,770  d= 512    51.2 ms/step -> 28.4 h at T=2e6
    CelebA train,     rn50   n=162,770  d=2048   140.5 ms/step -> 78.0 h at T=2e6

CelebA at 5 eps x 4 backbones is ~1,560 CPU-hours, about 65 days. Not a tuning
problem, a feasibility one.

On an A6000 the step is memory-bandwidth bound, not compute bound: each step
streams the design matrix twice (once for `X @ w`, once for `X.T @ p`), so at
~768 GB/s and 1.33 GB for CelebA/rn50 in float32 the floor is ~3.5 ms/step, i.e.
about 2 h per eps at T = 2e6 instead of 78 h. Expect roughly 26 h for all of
CelebA across four backbones and five eps, and about an hour for Waterbirds.


!! THIS FILE HAS NEVER BEEN EXECUTED !!
=======================================

It was written in a sandbox where the PyTorch package host is blocked, so the
torch path could not be run even once. Every line is therefore written to be
checked by the machine that first runs it rather than trusted:

    python gd_gpu.py --validate            # DO THIS BEFORE ANY REAL RUN

`--validate` runs both implementations on the same small problem and asserts
that the recorded per-group error curves agree to a tight tolerance. If it does
not print AGREEMENT OK, the GPU numbers are wrong and nothing built on them
means anything. It takes under a minute.


PRECISION -- THE ONE REAL RISK
==============================

float32 is the default because the step is memory-bound, so float64 costs a
factor of two in wall-clock for free precision we probably do not need. But
"probably" is doing work in that sentence, and this is a long asymptotic run:
`err_min` decays roughly like `z^-1`, so at z_T = 1e5 the tracked quantity is
around 1e-5. float32 carries ~1e-7 relative precision, which leaves adequate
headroom; accumulated drift over two million steps is the thing that could still
bite, and it is not something to reason about from first principles.

So `--validate` measures it: it runs float32 and float64 side by side and reports
the divergence in the recorded curves. Read that number, then decide. If it is
large, use `--dtype float64` and accept 2x.

TF32 is available via `--tf32` and is NOT the default. It carries a 10-bit
mantissa, about 1e-3 relative, which is the same order as the quantities being
measured late in the run. Do not enable it without re-running `--validate`.
"""

from __future__ import annotations

import argparse

import numpy as np

from progress import pbar


def logistic_gd_torch(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    h: float = 0.01,
    T: int = 100_000,
    n_ckpt: int = 60,
    w0: np.ndarray | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    tf32: bool = False,
    progress: bool = True,
    c: np.ndarray | None = None,
) -> dict:
    """Drop-in replacement for `common.logistic_gd`, identical contract.

    Replicates the reference recurrence exactly, including two details that are
    easy to get subtly wrong and would produce plausible-looking wrong curves:

      * The gradient uses `one_minus_p` evaluated at the CURRENT w, BEFORE the
        update, and the errors recorded at step t are that same quantity. So the
        recorded error at checkpoint t is the error at the ITERATE ENTERING step
        t, not the one leaving it. Recording after the update shifts every curve
        by one step, which is invisible at large t and corrupts the early
        checkpoints -- exactly the ones that anchor a log-log slope.
      * Checkpoints are the same log-spaced integers the reference computes, and
        they are computed here with the same numpy expression rather than
        re-derived, so the two implementations cannot drift apart.

    Returns a dict of numpy arrays: t, z, err_maj, err_min, train_loss.
    """
    import torch
    import torch.nn.functional as F

    if not tf32:
        # Off by default and set explicitly: torch has changed the default for
        # this across versions, so leaving it implicit makes the numerics depend
        # on which torch happens to be installed.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' but torch.cuda.is_available() is False. Refusing to "
            "silently fall back to CPU: at these sizes that is the difference "
            "between two hours and three days, and it should be a decision, "
            "not an accident."
        )

    td = {"float32": torch.float32, "float64": torch.float64}[dtype]
    N, d = X.shape

    # Xy = y * X, formed once. Same memory as X; the alternative is multiplying
    # by y inside the loop, which costs an extra pass over the matrix per step
    # on a kernel that is already bandwidth bound.
    Xy = torch.as_tensor(np.asarray(y, dtype=np.float64)[:, None] * X,
                         dtype=td, device=device)
    w = (torch.zeros(d, dtype=td, device=device) if w0 is None
         else torch.as_tensor(w0, dtype=td, device=device).clone())

    # Identical to the reference, deliberately via the same numpy call.
    ckpts = np.unique(np.round(np.logspace(0, np.log10(T), n_ckpt)).astype(int))
    ckpts = ckpts[(ckpts >= 1) & (ckpts <= T)]
    ck = set(int(c) for c in ckpts)

    m_maj = torch.as_tensor(np.asarray(g) == 0, device=device)
    m_min = torch.as_tensor(np.asarray(g) == 1, device=device)
    has_maj, has_min = bool(m_maj.any()), bool(m_min.any())

    rec = {"t": [], "z": [], "err_maj": [], "err_min": [], "train_loss": []}
    scale = h / N

    # The bar is driven in CHUNKS and its postfix is refreshed only at the
    # checkpoints, where a host sync already happens anyway. Reading any GPU
    # tensor on the other steps to feed a progress bar would add a sync per
    # step and dominate the runtime -- exactly the trap the checkpoint design
    # exists to avoid. Bar bookkeeping itself is pure Python and never touches
    # the device.
    chunk = max(1, T // 2000)
    bar = pbar(total=T, unit="step", leave=False,
               desc=f"GD n={N} d={d} {dtype}{'' if c is None else ' weighted'}"
               ) if progress else None

    # Per-sample weights c_i realise the population mixture
    #     L(w) = (1-eps) E_maj[.] + eps E_min[.]
    # with the SAMPLE HELD FIXED, which is the whole point of --mode reweight:
    # the point set, the margin and separability no longer move with eps.
    #
    # The c is None branch keeps the ORIGINAL expression verbatim rather than
    # routing uniform weights through the weighted one. Measured in numpy, the
    # two forms agree only to ~1.1e-16 (about 2 ulp) because the scalar
    # multiply moves relative to the matmul. That is harmless arithmetic, but
    # the unweighted path is the one --validate compares against the certified
    # common.logistic_gd, and that comparison is worth keeping bit-exact.
    ct = None if c is None else torch.as_tensor(
        np.asarray(c, dtype=np.float64), dtype=td, device=device)

    for t in range(1, T + 1):
        u = Xy @ w
        one_minus_p = torch.sigmoid(-u)          # == 1 - sigmoid(u), stable
        if ct is None:
            w += scale * (Xy.T @ one_minus_p)
        else:
            w += h * (Xy.T @ (ct * one_minus_p))

        # .item() forces a host sync, so it happens only at the ~60 checkpoints.
        # Syncing every step would dominate the runtime entirely.
        if t in ck:
            rec["t"].append(t)
            rec["z"].append(h * t)
            rec["err_maj"].append(
                float(one_minus_p[m_maj].mean()) if has_maj else np.nan)
            rec["err_min"].append(
                float(one_minus_p[m_min].mean()) if has_min else np.nan)
            # logaddexp(0, -u) == softplus(-u), the logistic loss per sample.
            rec["train_loss"].append(float(F.softplus(-u).mean()))
            if bar is not None:
                bar.set_postfix_str(
                    f"z={h * t:.4g} err_min={rec['err_min'][-1]:.3e}")

        if bar is not None and t % chunk == 0:
            bar.update(chunk)

    if bar is not None:
        bar.close()
    return {k: np.asarray(v) for k, v in rec.items()}


# --------------------------------------------------------------------------
# Validation -- run before trusting anything this file produces
# --------------------------------------------------------------------------


def logistic_gd_weighted(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    c: np.ndarray,
    h: float = 0.01,
    T: int = 100_000,
    n_ckpt: int = 60,
    w0: np.ndarray | None = None,
) -> dict:
    """numpy weighted GD -- the CPU path for --mode reweight, and the reference
    the torch weighted path is validated against.

    This lives here and NOT in common.py for the same reason logistic_gd_torch
    does: common.py is a byte-for-byte copy whose sha256 is what transfers
    Estimator_Validation's certification, and adding a weights argument to it
    would break that transfer for a feature change.

    `c` are per-sample weights that should sum to 1, so that the step is the
    empirical version of the population gradient. At c_i = 1/N this reduces to
    common.logistic_gd up to floating-point reassociation (~2 ulp).
    """
    N, d = X.shape
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N,):
        raise ValueError(f"c must have shape ({N},), got {c.shape}")
    Xy = (y[:, None] * X).astype(np.float64)
    w = np.zeros(d) if w0 is None else w0.astype(np.float64).copy()

    ckpts = np.unique(np.round(np.logspace(0, np.log10(T), n_ckpt)).astype(int))
    ckpts = ckpts[(ckpts >= 1) & (ckpts <= T)]
    ck = set(int(t) for t in ckpts)

    m_maj, m_min = (g == 0), (g == 1)
    rec = {"t": [], "z": [], "err_maj": [], "err_min": [], "train_loss": []}

    chunk = max(1, T // 500)
    bar = pbar(total=T, unit="step", leave=False, desc=f"GD-cpu n={N} d={d}")
    for t in range(1, T + 1):
        u = Xy @ w
        one_minus_p = np.where(u >= 0, np.exp(-u) / (1.0 + np.exp(-u)),
                               1.0 / (1.0 + np.exp(u)))
        w += h * (Xy.T @ (c * one_minus_p))
        if t in ck:
            rec["t"].append(t)
            rec["z"].append(h * t)
            rec["err_maj"].append(float(one_minus_p[m_maj].mean()) if m_maj.any() else np.nan)
            rec["err_min"].append(float(one_minus_p[m_min].mean()) if m_min.any() else np.nan)
            rec["train_loss"].append(float(np.mean(np.logaddexp(0.0, -u))))
        if t % chunk == 0:
            bar.update(chunk)
    bar.close()
    return {k: np.asarray(v) for k, v in rec.items()}


def _make_problem(n=4000, d=64, eps=0.25, seed=0):
    """A small separable-ish problem, shaped like the real ones."""
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < eps).astype(int)
    r = rng.standard_normal((n, d))
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    y = np.sign(r @ v)
    y[y == 0] = 1
    return r, y.astype(int), g


def _max_rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m]))) if m.any() else np.nan


def validate(T=3000, h=0.05, device="cuda", tol=2e-3) -> bool:
    """Assert the GPU path reproduces the certified numpy path, and measure fp32 drift.

    Two separate questions, reported separately because they have different
    remedies:

      1. IS THE IMPLEMENTATION RIGHT?  float64 on device against float64 on CPU.
         Any disagreement here is a bug in this file. Tolerance is tight.
      2. IS float32 GOOD ENOUGH?       float32 on device against float64 on CPU.
         Disagreement here is not a bug, it is a precision budget. If it is
         large, run with --dtype float64 and pay 2x.
    """
    from common import logistic_gd

    X, y, g = _make_problem()
    ref = logistic_gd(X, y, g, h=h, T=T)

    print(f"reference (numpy, float64): {len(ref['t'])} checkpoints, "
          f"z_T = {ref['z'][-1]:.1f}")
    print(f"  final err_maj = {ref['err_maj'][-1]:.6e}  "
          f"err_min = {ref['err_min'][-1]:.6e}")

    ok = True
    for dt, label, this_tol in (("float64", "correctness", tol),
                                ("float32", "precision budget", None)):
        got = logistic_gd_torch(X, y, g, h=h, T=T, device=device, dtype=dt)
        if not np.array_equal(got["t"], ref["t"]):
            print(f"  [{dt}] FAIL: checkpoint grids differ")
            ok = False
            continue
        rmaj = _max_rel(got["err_maj"], ref["err_maj"])
        rmin = _max_rel(got["err_min"], ref["err_min"])
        rl = _max_rel(got["train_loss"], ref["train_loss"])
        worst = max(rmaj, rmin, rl)
        tag = ""
        if this_tol is not None:
            passed = worst <= this_tol
            ok = ok and passed
            tag = "  <= tol" if passed else f"  !! EXCEEDS tol {this_tol:g}"
        print(f"  [{dt:<7}] {label:<17} max rel diff: "
              f"err_maj {rmaj:.2e}  err_min {rmin:.2e}  loss {rl:.2e}"
              f"   worst {worst:.2e}{tag}")

    # The weighted path has no certified reference to check against, so it is
    # checked the same way the unweighted one was: an independent numpy
    # implementation of the same recurrence, run side by side. A bug here would
    # silently rescale the gradient and produce plausible wrong curves.
    cw = np.where(g == 1, 0.6 / max(1, int((g == 1).sum())),
                  0.4 / max(1, int((g == 0).sum())))
    ref_w = logistic_gd_weighted(X, y, g, c=cw, h=h, T=T)
    got_w = logistic_gd_torch(X, y, g, c=cw, h=h, T=T, device=device,
                              dtype="float64")
    rw = max(_max_rel(got_w["err_maj"], ref_w["err_maj"]),
             _max_rel(got_w["err_min"], ref_w["err_min"]))
    passed_w = rw <= tol
    ok = ok and passed_w
    print(f"  [weighted] float64 vs numpy weighted reference: max rel diff "
          f"{rw:.2e}" + ("  <= tol" if passed_w else f"  !! EXCEEDS tol {tol:g}"))

    # And the reduction property: uniform weights must reproduce the unweighted
    # run. If this drifts, --mode reweight and --mode subsample are not running
    # the same recurrence and nothing can be compared across them.
    uni = np.full(y.size, 1.0 / y.size)
    got_u = logistic_gd_torch(X, y, g, c=uni, h=h, T=T, device=device,
                              dtype="float64")
    ru = max(_max_rel(got_u["err_maj"], ref["err_maj"]),
             _max_rel(got_u["err_min"], ref["err_min"]))
    passed_u = ru <= tol
    ok = ok and passed_u
    print(f"  [weighted] uniform c reduces to the unweighted run: max rel diff "
          f"{ru:.2e}" + ("  <= tol" if passed_u else f"  !! EXCEEDS tol {tol:g}"))

    print("\nAGREEMENT OK" if ok else "\nAGREEMENT FAILED -- do not use the GPU path")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true",
                    help="check against common.logistic_gd. Run this first.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--bench", action="store_true",
                    help="time one configuration at CelebA/rn50 scale")
    args = ap.parse_args()

    if args.bench:
        import time
        X, y, g = _make_problem(n=162_770, d=2048)
        t0 = time.time()
        logistic_gd_torch(X, y, g, h=0.05, T=200, n_ckpt=5, device=args.device)
        dt = (time.time() - t0) / 200
        print(f"{dt * 1000:.3f} ms/step  ->  T=2e6 takes {dt * 2e6 / 3600:.2f} h per eps")
        return

    ok = validate(T=args.T, device=args.device, tol=args.tol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
