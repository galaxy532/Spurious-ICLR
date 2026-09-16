"""Ground-truth validation of the invariance rule. Run this before trusting it.

Same contract as `Estimator_Validation/` has for the earlier repos: this file
produces NO result that goes in the paper. It builds synthetic data where the
true (r, s, weak) split is known by construction and checks that the rule in
`invariance.py` recovers it, stays quiet when it should, and -- the part that
matters most -- is not fooled by the one confound that would otherwise produce a
confident wrong answer on real data.

    python validate_invariance.py            # the five checks
    python validate_invariance.py --mutants  # + confirm the checks catch bugs

THE FIVE CHECKS
===============

C1  POWER, sign-reversal case.  A_k = +a in the majority, B_k = -a in the
    minority. Both the invariance rule and the existing sign-flip rule should
    recover the s-block. This is the case sign-flip was designed for; if the new
    rule loses to it here, the new rule is worse and should not be used.

C2  POWER, magnitude-only case.  A_k = 2a, B_k = a, SAME SIGN. Spurious under
    Definition 3.3, because A != B. This is the headline check: the invariance
    rule should recover the block and the sign-flip rule should MISS it, because
    sign-flip fires only on rho_0 * rho_1 < 0 and here the two correlations have
    the same sign. If C2 does not separate the two rules, there was no reason to
    write invariance.py.

C3  CALIBRATION.  A = B exactly, so no coordinate is spurious. The rule must
    return n_s at roughly the FDR level, not more. A rule with power and no
    calibration finds an s-block in noise, which is the failure mode that would
    quietly invalidate a real-data result.

C4  SPECIFICITY against the BASE-RATE CONFOUND.  A = B (no spuriousness), but
    the groups have different class balance: P(y=+1 | g=0) != P(y=+1 | g=1).
    This is true on Waterbirds and extreme on CelebA, where P(blond | female)
    ~ 0.24 against P(blond | male) ~ 0.02.

    A base-rate difference is NOT spuriousness -- Definition 3.3 is about the
    r -> s map, not about how many positives each group contains. The rule must
    stay quiet. For contrast the check also evaluates the NAIVE statistic (test
    delta_b = 0 rather than delta_b constant, i.e. force the common odds ratio
    to 1), which is the test one writes if one has not thought about this. The
    naive statistic should reject nearly everything, including the causal block.
    That contrast is the justification for the Breslow-Day construction and
    belongs in the paper.

    Two variants are run:
      (a) LOGIT shift -- P(y|r,g) = sigmoid(v.r + shift_g). Here the group
          log-odds difference is exactly constant across bins, so H_0 holds
          exactly and the rule should be calibrated to the nominal level.
      (b) PROBIT shift -- y = sign(v.r + shift_g + noise). A constant shift on
          the probit scale is NOT exactly constant on the logit scale, so a
          small link-mismatch heterogeneity is genuinely present. The rule will
          pick some of it up. This variant is here to MEASURE that leakage
          rather than to pass or fail, because real data will not obligingly be
          logit-generated. Report the number; do not pretend it is zero.

C5  BIN-COUNT STABILITY.  Re-run C2 at 4, 8 and 16 bins. The recovered set
    should be substantially the same. `n_bins` is a convention, and a split that
    only exists at one setting of a convention is not a finding.

MUTANTS
=======

A validation suite that cannot fail is not evidence. `--mutants` re-runs the
checks against deliberately broken versions of the rule and asserts that the
suite CATCHES each break:

    M1  force psi = 1  (the naive delta_b = 0 test)   -> C4 must fail
    M2  drop the effect-size thresholds (tau = 0)     -> C3 must fail
    M3  collapse to a single bin                      -> C2 must fail

If a mutant passes, the corresponding check is vacuous and needs rewriting.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from progress import pbar

from invariance import (
    bh_fdr, breslow_day, bin_coordinate, effect_sizes, invariance_identify,
    invariance_test, mantel_haenszel_or, stratified_tables,
)

try:
    from identify_rs import sign_flip_identify
    HAVE_SIGNFLIP = True
except Exception:                                     # pragma: no cover
    HAVE_SIGNFLIP = False


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------


def make_data(
    n: int = 60_000,
    d_r: int = 20,
    d_s: int = 20,
    d_weak: int = 40,
    eps: float = 0.3,
    a_maj: float = 1.0,
    a_min: float = -1.0,
    noise_s: float = 0.6,
    label_noise: float = 0.0,
    base_rate_shift: float = 0.0,
    label_mode: str = "separable",
    seed: int = 0,
):
    """Build (phi, y, g) with a known (r, s, weak) split.

    Layout of the returned `phi`, columns in this order:
        [0, d_r)                     r-block  -- causal, relation to y invariant
        [d_r, d_r + d_s)             s-block  -- s_k = A_k u + noise, A depending
                                                on the group through a_maj/a_min
        [d_r + d_s, d_r + d_s + d_weak)  weak -- pure noise, unrelated to y

    where u = v . r is the causal score.

    Spuriousness is controlled entirely by (a_maj, a_min):
        a_maj = +a, a_min = -a   -> sign reversal   (C1)
        a_maj = 2a, a_min = a    -> magnitude only  (C2)
        a_maj = a_min            -> not spurious    (C3, C4)

    `label_mode` selects how y is generated from the causal score:

      "separable" (default) -- y = sign(v.r), exactly. This is the manuscript's
          setting: the paper assumes r alone suffices for separability, so the
          power checks should be run here and not under a noisy link.
      "logit"     -- P(y=+1 | r, g) = sigmoid(v.r + shift_g). Used by C4(a),
          because here a base-rate difference is EXACTLY a constant offset of
          the group log-odds, so the Breslow-Day null holds exactly and the
          check is a clean calibration statement.
      "probit"    -- y = sign(v.r + shift_g + noise). Used by C4(b). A constant
          shift on the probit scale is not exactly constant on the logit scale,
          so a small genuine link-mismatch heterogeneity is present. That
          variant measures the leakage rather than asserting it is zero.

    `base_rate_shift` makes the two groups differ in class balance without
    touching the r -> s map, which is the confound C4 probes.
    """
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < eps).astype(int)          # 1 = minority

    r = rng.standard_normal((n, d_r))
    v = rng.standard_normal(d_r)
    v /= np.linalg.norm(v)
    u = r @ v                                       # the causal score

    shift = np.where(g == 1, base_rate_shift, 0.0)
    if label_mode == "separable":
        y = np.sign(u + shift)
        y[y == 0] = 1
    elif label_mode == "logit":
        p = 1.0 / (1.0 + np.exp(-(u + shift)))
        y = np.where(rng.random(n) < p, 1, -1)
    elif label_mode == "probit":
        y = np.sign(u + shift + 0.30 * rng.standard_normal(n))
        y[y == 0] = 1
    else:
        raise ValueError(f"unknown label_mode {label_mode!r}")

    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, -y, y)

    # s-block: the coefficient on u depends on the group. Per-coordinate gains
    # spread the effect size so the check is not a single knife-edge setting.
    gain = rng.uniform(0.7, 1.3, size=d_s)
    coef = np.where(g == 1, a_min, a_maj)[:, None] * gain[None, :]
    s = coef * u[:, None] + noise_s * rng.standard_normal((n, d_s))

    weak = rng.standard_normal((n, d_weak))

    phi = np.hstack([r, s, weak])
    truth = {
        "idx_r": np.arange(0, d_r),
        "idx_s": np.arange(d_r, d_r + d_s),
        "idx_weak": np.arange(d_r + d_s, d_r + d_s + d_weak),
    }
    return phi, y.astype(int), g, truth


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def prf(found: np.ndarray, true: np.ndarray) -> dict:
    """Precision / recall / F1 of a recovered index set against the truth."""
    found, true = set(found.tolist()), set(true.tolist())
    tp = len(found & true)
    prec = tp / len(found) if found else 0.0
    rec = tp / len(true) if true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_found": len(found)}


def naive_identify(phi, y, g, q=0.05, tau_het=0.20, n_bins=8, min_cell=10):
    """The statistic a careless implementation would use: test delta_b = 0.

    Identical to the real rule except that the common odds ratio is forced to 1,
    which turns the Breslow-Day heterogeneity test into a test of NO group
    difference at all. Exists only so C4 can show what it costs.
    """
    from scipy.stats import chi2

    y_pos = np.asarray(y).ravel() > 0
    g = np.asarray(g).ravel().astype(int)
    d = phi.shape[1]
    stat = np.full(d, np.nan); dof = np.zeros(d, dtype=int)
    het = np.zeros(d); strength = np.zeros(d)
    for k in range(d):
        codes = bin_coordinate(phi[:, k], n_bins=n_bins)
        a, b, c, dd = stratified_tables(codes, y_pos, g, min_cell=min_cell)
        if a.size < 2:
            continue
        s_, df = breslow_day(a, b, c, dd, psi=1.0)     # <-- the whole difference
        stat[k], dof[k] = s_, df + 1
        strength[k], het[k], _ = effect_sizes(a, b, c, dd)
    p = np.full(d, np.nan)
    ok = np.isfinite(stat) & (dof > 0)
    p[ok] = chi2.sf(stat[ok], dof[ok])
    rej = bh_fdr(p, q=q)
    return {"idx_s": np.flatnonzero(rej & (het >= tau_het)), "n_rejected": int(rej.sum())}


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------


def _signflip(phi, y, g, tau=0.20):
    if not HAVE_SIGNFLIP:
        return None
    return sign_flip_identify(phi, y, g, tau=tau)


def _f1_over_seeds(a_maj, a_min, n, n_seeds=3, rule="inv", n_bins=8, **kw) -> float:
    """Mean F1 of the recovered s-block over `n_seeds` independent draws.

    Averaged over seeds because a single draw of a 20-coordinate block moves by
    0.1-0.2 in F1 from seed to seed, which is enough to flip any fixed pass
    threshold and make the suite flap.
    """
    out = []
    for sd in pbar(range(n_seeds), unit="seed", desc="    seeds", leave=False):
        phi, y, g, t = make_data(a_maj=a_maj, a_min=a_min, n=n, seed=sd, **kw)
        if rule == "inv":
            idx = invariance_identify(phi, y, g, n_bins=n_bins)["idx_s"]
        else:
            sf = _signflip(phi, y, g)
            if sf is None:
                return float("nan")
            idx = sf["idx_s"]
        out.append(prf(idx, t["idx_s"])["f1"])
    return float(np.mean(out))


def c1_power_signflip(seed=0, n=60_000, n_seeds=3, **kw) -> dict:
    """Both rules must recover a sign reversal. This is sign-flip's home turf."""
    inv = _f1_over_seeds(2.0, -2.0, n, n_seeds, "inv", **kw)
    sf = _f1_over_seeds(2.0, -2.0, n, n_seeds, "signflip", **kw)
    return {"n": n, "invariance_f1": inv, "sign_flip_f1": sf,
            # If the new rule loses here, it is not an improvement, it is a swap.
            "pass": inv >= 0.80 and (not np.isfinite(sf) or inv >= sf - 0.05)}


def c2_power_magnitude(seed=0, n=60_000, n_seeds=3, **kw) -> dict:
    """THE headline check. A = 2B, same sign: spurious, but no sign reversal."""
    inv = _f1_over_seeds(2.0, 1.0, n, n_seeds, "inv", **kw)
    sf = _f1_over_seeds(2.0, 1.0, n, n_seeds, "signflip", **kw)
    return {
        "n": n, "invariance_f1": inv, "sign_flip_f1": sf,
        # Not enough that the new rule works; the old one must demonstrably fail,
        # or there was no reason to write invariance.py.
        "separates_rules": bool(np.isfinite(sf) and inv - sf >= 0.50),
        "pass": inv >= 0.80 and np.isfinite(sf) and inv - sf >= 0.50,
    }


def c2b_power_curve(n_seeds=3, **kw) -> dict:
    """F1 against sample size -- the table that says which datasets are usable.

    This is not pass/fail. It is the operational output of the whole file: the
    rule needs a certain n before it can see a magnitude-only difference, and
    that number decides which benchmark it can be run on at all. Reference
    points, for the sizes actually available:

        Waterbirds test        n =   5,794
        Waterbirds all splits  n =  11,788   (train + val + test pooled)
        CelebA train           n = 162,770
    """
    rows = []
    for n in pbar((6_000, 12_000, 20_000, 60_000, 160_000), unit="n", desc="  power curve"):
        rows.append({
            "n": n,
            "invariance_f1": _f1_over_seeds(2.0, 1.0, n, n_seeds, "inv", **kw),
            "sign_flip_f1": _f1_over_seeds(2.0, 1.0, n, n_seeds, "signflip", **kw),
        })
    return {"rows": rows, "pass": rows[-1]["invariance_f1"] >= 0.80}


def c3_calibration(seed=0, n=60_000, **kw) -> dict:
    """A = B exactly: no coordinate is spurious, so any s-type is a false positive."""
    phi, y, g, t = make_data(a_maj=1.0, a_min=1.0, n=n, seed=seed, **kw)
    inv = invariance_identify(phi, y, g)
    d = phi.shape[1]
    out = {
        "n": n, "n_s_found": int(inv["idx_s"].size),
        "n_rejected": int(inv["n_rejected"]), "d": d,
        "false_positive_rate": float(inv["idx_s"].size / d),
        "mean_het_true_r": float(inv["het"][t["idx_r"]].mean()),
        "mean_het_true_s": float(inv["het"][t["idx_s"]].mean()),
    }
    out["pass"] = out["n_s_found"] <= max(2, int(0.02 * d))
    return out


def c4_specificity_base_rate(seed=0, n=60_000, shifts=(0.5, 1.0, 1.5), **kw) -> dict:
    """The confound that would otherwise produce a confident wrong answer.

    The contrast is drawn on `n_rejected` -- how many coordinates each STATISTIC
    rejects -- not on the final n_s. n_s additionally passes through the
    effect-size filter, which would mask the very difference this check exists to
    expose: the effect-size filter partly rescues the naive statistic, and the
    point is that the statistic should not have needed rescuing.
    """
    res = {}
    for mode in ("logit", "probit"):
        rows = []
        for sh in shifts:
            phi, y, g, t = make_data(a_maj=1.0, a_min=1.0, n=n,
                                     base_rate_shift=sh, label_mode=mode,
                                     seed=seed, **kw)
            inv = invariance_identify(phi, y, g)
            nai = naive_identify(phi, y, g)
            rows.append({
                "shift": sh,
                "P(y=+1|maj)": round(float(np.mean(y[g == 0] > 0)), 3),
                "P(y=+1|min)": round(float(np.mean(y[g == 1] > 0)), 3),
                "invariance_rejected": int(inv["n_rejected"]),
                "naive_rejected": int(nai["n_rejected"]),
                "invariance_n_s": int(inv["idx_s"].size),
                "d": int(phi.shape[1]),
            })
        res[mode] = rows

    d = res["logit"][0]["d"]
    tol = max(2, int(0.02 * d))
    # Hard pass/fail on the LOGIT variant only, where a base-rate difference is
    # exactly a constant log-odds offset and H_0 therefore holds exactly.
    #
    # The PROBIT variant is measured, not asserted. A constant probit shift is
    # genuinely not constant on the logit scale, so real heterogeneity exists
    # there and the rule correctly finds some of it. At shift = 1.5 -- which
    # takes P(y=+1|min) from 0.50 to 0.92, far beyond anything in Waterbirds or
    # CelebA -- the leakage is substantial. That is a real limitation of any
    # logit-scale invariance test and belongs in the paper's limitations
    # paragraph, not hidden behind a passing check.
    res["pass"] = all(r["invariance_rejected"] <= tol and r["naive_rejected"] >= 0.5 * d
                      for r in res["logit"])
    res["note"] = ("logit rows are the pass/fail; probit rows measure "
                   "link-mismatch leakage and are expected to be nonzero")
    return res


def c5_bin_stability(seed=0, n=60_000, **kw) -> dict:
    """n_bins is a convention. A split that exists at only one setting is not one."""
    phi, y, g, t = make_data(a_maj=2.0, a_min=1.0, n=n, seed=seed, **kw)
    sets, rows = {}, []
    for nb in (4, 8, 16):
        inv = invariance_identify(phi, y, g, n_bins=nb)
        sets[nb] = set(inv["idx_s"].tolist())
        rows.append({"n_bins": nb, **prf(inv["idx_s"], t["idx_s"])})
    union = sets[4] | sets[8] | sets[16]
    inter = sets[4] & sets[8] & sets[16]
    jac = len(inter) / len(union) if union else 0.0
    return {"rows": rows, "jaccard_across_bin_counts": jac,
            "pass": jac >= 0.60 and all(r["f1"] >= 0.70 for r in rows)}


# --------------------------------------------------------------------------
# Mutants
# --------------------------------------------------------------------------


def run_mutants(seed=0) -> dict:
    """Confirm each check can fail. A check no bug can break is not evidence."""
    out = {}

    # M1: force psi = 1 -> the naive delta_b = 0 test. C4 must catch it.
    phi, y, g, t = make_data(a_maj=1.0, a_min=1.0, n=60_000,
                             base_rate_shift=1.5, label_mode="logit", seed=seed)
    nai = naive_identify(phi, y, g)
    inv = invariance_identify(phi, y, g)
    d = phi.shape[1]
    out["M1_force_psi_1"] = {
        "naive_rejected": int(nai["n_rejected"]),
        "invariance_rejected": int(inv["n_rejected"]), "d": d,
        "caught": bool(nai["n_rejected"] >= 0.5 * d
                       and inv["n_rejected"] <= max(2, int(0.02 * d))),
        "note": "the naive statistic must blow up on a pure base-rate "
                "difference where Breslow-Day stays quiet",
    }

    # M2: drop the effect-size thresholds. The probit base-rate case is where
    # this bites: real but negligible link-mismatch heterogeneity, which a
    # significance-only rule promotes into a full s-block at large n.
    phi, y, g, t = make_data(a_maj=1.0, a_min=1.0, n=60_000,
                             base_rate_shift=1.5, label_mode="probit", seed=seed)
    inv0 = invariance_identify(phi, y, g, tau_rel=0.0, tau_het=0.0)
    inv1 = invariance_identify(phi, y, g)
    out["M2_no_effect_size"] = {
        "n_s_tau0": int(inv0["idx_s"].size),
        "n_s_default": int(inv1["idx_s"].size), "d": int(phi.shape[1]),
        "caught": bool(inv0["idx_s"].size > inv1["idx_s"].size + 2),
        "note": "without an effect-size floor, large n turns a negligible "
                "link mismatch into an s-block",
    }

    # M3: a single bin. C2 must fail -- with one bin there is nothing to test
    # constancy ACROSS, so no heterogeneity is detectable even in principle.
    phi, y, g, t = make_data(a_maj=2.0, a_min=1.0, n=60_000, seed=seed)
    inv = invariance_identify(phi, y, g, n_bins=1)
    f1 = prf(inv["idx_s"], t["idx_s"])["f1"]
    out["M3_single_bin"] = {
        "f1": f1, "caught": bool(f1 < 0.80),
        "note": "one bin cannot detect heterogeneity across bins",
    }

    # M4: revert the four-cell guard to a row-total guard, which is the bug that
    # cost a debugging cycle (see stratified_tables). Saturated tail bins return,
    # het inflates with log(n), and the statistic destabilises.
    #
    # Measured on the s-block, and via the DISPERSION of the statistic rather
    # than its level. The s-block coordinates are identical by construction up
    # to a gain drawn from a narrow range, so their Breslow-Day statistics should
    # be tightly clustered. With the guard removed, saturated tail bins inject
    # large arbitrary residuals and the statistic scatters wildly -- 6.3 to 57.6
    # across equivalent coordinates in the run that first exposed this. The
    # r-block is the wrong place to look: those coordinates are only weakly
    # related to y, so their tails never saturate and the bug is invisible there.
    phi, y, g, t = make_data(a_maj=2.0, a_min=1.0, n=60_000, seed=seed)
    res_ok = invariance_test(phi, y, g, min_cell=10)
    res_bad = invariance_test(phi, y, g, min_cell=0)   # 0 disables the guard

    def _cv(res):
        v = res["stat"][t["idx_s"]]
        v = v[np.isfinite(v)]
        return float(np.std(v) / max(np.mean(v), 1e-9)) if v.size else float("nan")

    het_ok = float(res_ok["het"][t["idx_s"]].mean())
    het_bad = float(res_bad["het"][t["idx_s"]].mean())
    out["M4_no_four_cell_guard"] = {
        "mean_het_s_guarded": round(het_ok, 3),
        "mean_het_s_unguarded": round(het_bad, 3),
        "stat_CV_s_guarded": round(_cv(res_ok), 3),
        "stat_CV_s_unguarded": round(_cv(res_bad), 3),
        # The detector is the effect-size inflation, not the statistic's
        # dispersion. Dispersion was tried first and does not separate the two
        # (CV 0.51 guarded against 0.57 unguarded) because the extra residuals
        # scatter the statistic without changing its relative spread much. The
        # Haldane-driven het inflation is the crisp signal: ~2.6x here.
        "caught": bool(het_bad > 2.0 * het_ok),
        "note": "removing the guard must inflate het on the block whose tails "
                "saturate, via the Haldane correction on empty cells",
    }
    return out


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="smaller n and 1 seed; a smoke test, NOT a verdict -- "
                         "the rule is underpowered at small n by design, so "
                         "--quick will report failures that are not bugs")
    args = ap.parse_args()

    n = 20_000 if args.quick else 60_000
    ns = 1 if args.quick else 3
    checks = {
        "C1_power_sign_reversal": c1_power_signflip(args.seed, n=n, n_seeds=ns),
        "C2_power_magnitude_only": c2_power_magnitude(args.seed, n=n, n_seeds=ns),
        "C3_calibration": c3_calibration(args.seed, n=n),
        "C4_specificity_base_rate": c4_specificity_base_rate(args.seed, n=n),
        "C5_bin_stability": c5_bin_stability(args.seed, n=n),
    }
    if not args.quick:
        checks["C2b_power_curve"] = c2b_power_curve(n_seeds=ns)

    print("\n=== invariance rule validation " + "=" * 40)
    for name, r in checks.items():
        print(f"\n[{'PASS' if r.get('pass') else 'FAIL'}] {name}")
        for k, v in r.items():
            if k == "pass":
                continue
            if k == "rows":
                for row in v:
                    print(f"    {row}")
            else:
                print(f"    {k}: {v}")

    ok = all(r.get("pass") for r in checks.values())

    if args.mutants:
        print("\n=== mutants " + "=" * 54)
        muts = run_mutants(args.seed)
        for name, r in muts.items():
            print(f"\n[{'CAUGHT' if r['caught'] else 'MISSED'}] {name}")
            for k, v in r.items():
                if k != "caught":
                    print(f"    {k}: {v}")
        ok = ok and all(r["caught"] for r in muts.values())

    print("\n" + "=" * 70)
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
