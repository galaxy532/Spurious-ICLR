"""Read the stage-A margin ladder and print the exact stage-B command.

Stage A measures the per-group hard margins at every degradation level. Choosing
which levels to spend six GPU-hours on is a mechanical decision, so it is made
here rather than left to whoever reads the table.

THE RULES, in order
===================
1. DROP rows that are not measurements: not separable, or `plateau_rel >= 1e-2`
   (the solver ladder had not converged, so the margin is not a number yet).
2. The CONTROL, level 1.0, is always kept. It is the row that says whether the
   pipeline still reproduces session 3's tie; without it nothing else is
   interpretable.
3. DROP rows whose OVERALL margin is below `--margin-floor` (default 0.05). A
   level can have a beautiful margin ratio and still be useless: the margin sets
   how long until the asymptotics appear, and a run that never reaches the regime
   within z = 1e6 measures nothing. The audit's ladder is the calibration --
   dinov2 at 0.551 was in the regime by z ~ 3e3, gdro at 0.043 was only entering
   at z = 1e5, CelebA at 0.0065 never got there.
4. Among what survives, a level is a CANDIDATE if its margin ratio exceeds
   1 + `--tie-tol` (default 1e-3), i.e. it actually moved the branch condition.
5. Pick two candidates: the SMALLEST ratio above the threshold (the interesting
   near-transition case, where the theory predicts beta only just above 1) and
   the LARGEST ratio (the strongest signal). If only one candidate survives, pick
   it. If none does, say so and recommend NOT running stage B.

Usage
-----
    python pick_levels.py                           # reads results/v4_group_margins.json
    python pick_levels.py --json results/v4_group_margins.json --margin-floor 0.05
"""

from __future__ import annotations

import argparse
import json
import os


def fmt_level(v) -> str:
    """0.2 -> '0.20', 1.0 -> '1.0' (the form run_session4.sh treats as the control)."""
    v = float(v)
    return "1.0" if v >= 1.0 else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="results/v4_group_margins.json")
    ap.add_argument("--margin-floor", type=float, default=0.05,
                    help="reject a level whose OVERALL margin is below this (default 0.05)")
    ap.add_argument("--tie-tol", type=float, default=1e-3,
                    help="a ratio within this of 1 is a tie, not a candidate")
    ap.add_argument("--plateau-max", type=float, default=1e-2,
                    help="reject a level whose solver ladder had not converged")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"no such file: {args.json}\nRun stage A first:  bash run_session4.sh A")
        return 2
    rows = json.load(open(args.json))

    kept, dropped = [], []
    for r in rows:
        lv = r.get("degrade_level")
        lv = 1.0 if lv is None else float(lv)
        if not r.get("separable_by_svc"):
            dropped.append((lv, "not separable")); continue
        pl = r.get("plateau_rel")
        if pl is not None and pl >= args.plateau_max:
            dropped.append((lv, f"plateau {pl:.1e} -- solver had not converged")); continue
        ratio = r.get("gamma_min_theorem")
        margin = r.get("margin")
        if ratio is None or margin is None:
            dropped.append((lv, "incomplete row")); continue
        kept.append(dict(level=lv, ratio=float(ratio), margin=float(margin),
                         larger=r.get("larger_margin_group")))

    kept.sort(key=lambda d: -d["level"])
    print(f"{'level':>7} {'margin':>9} {'ratio':>8} {'larger':>7}   status")
    control = None
    candidates = []
    for d in kept:
        why = ""
        if d["level"] >= 1.0:
            control = d
            why = "CONTROL"
            if d["ratio"] > 1 + args.tie_tol:
                why += "  <-- WARNING: the undegraded control is NOT a tie"
        elif d["margin"] < args.margin_floor:
            why = f"rejected: margin below the floor {args.margin_floor}"
        elif d["ratio"] <= 1 + args.tie_tol:
            why = "tie -- the knob did not move the branch condition here"
        else:
            why = "CANDIDATE"
            candidates.append(d)
        print(f"{d['level']:>7.2f} {d['margin']:>9.4g} {d['ratio']:>8.4f} "
              f"{str(d['larger']):>7}   {why}")
    for lv, why in sorted(dropped, reverse=True):
        print(f"{lv:>7.2f} {'-':>9} {'-':>8} {'-':>7}   dropped: {why}")

    print()
    if control is None:
        print("The level = 1.0 control is missing or unusable. Fix that before stage B: "
              "without it there is no baseline to compare against.")
        return 1
    if not candidates:
        print("NO CANDIDATE LEVEL. Every degraded level either left the margin ratio at 1 "
              "or collapsed the overall margin below the floor.")
        print()
        print("Do NOT run stage B -- it would spend six GPU-hours confirming the tie that")
        print("sessions 1-3 already measured. This is a reportable result: image degradation")
        print("does not move gamma_min/gamma_maj on this backbone, so the alpha > 1 branch")
        print("needs a group asymmetry these benchmarks do not have and that image quality")
        print("cannot manufacture.")
        print()
        print("Before accepting it, one cheap retry is worth it: rerun stage A with a")
        print("larger-margin backbone, which has more room before the floor bites:")
        print('    BACKBONE=dinov2_l nohup setsid bash run_session4.sh A > session4A_L.out 2>&1 &')
        return 0

    picks = [control]
    picks.append(min(candidates, key=lambda d: d["ratio"]))       # just above 1
    strongest = max(candidates, key=lambda d: d["ratio"])
    if strongest["level"] != picks[-1]["level"]:
        picks.append(strongest)

    levels = " ".join(fmt_level(d["level"]) for d in sorted(picks, key=lambda d: -d["level"]))
    print(f"{len(candidates)} candidate level(s). Chosen: the control, the smallest ratio "
          f"above 1, and the largest ratio that clears the margin floor.")
    print()
    for d in sorted(picks, key=lambda d: -d["level"]):
        role = ("control" if d["level"] >= 1.0 else
                "near-transition" if d is picks[1] else "strongest signal")
        print(f"  level {fmt_level(d['level']):>4}  margin {d['margin']:.4g}  "
              f"ratio {d['ratio']:.4f}  ->  beta predicted {d['ratio']:.4f} for group "
              f"g={d['larger']}, 1.0000 for the other   ({role})")
    print()
    print("RUN THIS:")
    print()
    print(f'    LEVELS="{levels}" MAX_HOURS=5.0 nohup setsid bash run_session4.sh B '
          f'> session4B_nohup.out 2>&1 &')
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
