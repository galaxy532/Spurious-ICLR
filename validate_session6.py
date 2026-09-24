"""Integration check for session 6: the real command line, on a miniature dataset.

`cub_meta.py --self-test` and `sv_screen.py --self-test` check the pieces. This
checks the chain `run_session6.sh` actually executes, as subprocesses with its
real flags, over a miniature Waterbirds + CUB tree built so that the answer is
known in advance:

  * one CUB attribute is PLANTED to contain only images whose margin is above the
    median under the bundle's own separator (label-proportional, so it is not a
    label proxy). The screen must call it a hit.
  * every other attribute is a fair coin over ~210 images, which cannot avoid the
    margin set by chance. None may be called a hit. (The small random proxy tails
    can, at the 5% family-wise rate Holm allows; that rate is checked separately
    in `sv_screen.py --self-test`.)

Then the chain continues exactly as the runner does: the confirmation bundles the
screen writes are measured by the UNMODIFIED `group_margins.py`, and
`sv_screen.py --compare` must report agreement. Finally, the features in a
confirmation bundle must be bit-identical to the source's, and `cub_meta.py` must
REFUSE a tree in which one Waterbirds image is missing from `images.txt`.

Why subprocesses: session 5 showed that an import done inside `main()` is
invisible to every in-process check (`load_raw` vs `load_metadata`). Only running
the scripts for real catches that class of bug.

No data, no download, no GPU. Well under a minute.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from progress import pbar

HERE = os.path.dirname(os.path.abspath(__file__))
PLANTED = "has_thing::value_1 = 1"


def _run(args, cwd, env):
    p = subprocess.run([sys.executable, os.path.join(HERE, args[0]), *args[1:]],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def _build(td: str, drop_one_image: bool = False):
    """Mini Waterbirds metadata + v5 fractions csv + source bundle + CUB tree."""
    import pandas as pd
    from common import FeatureBundle
    from cub_meta import write_mini_cub
    from datasets import DECLARATIONS
    from group_margins import C_LADDER
    from sv_screen import fit_margins

    dec = DECLARATIONS["waterbirds"]
    rng = np.random.default_rng(3)
    n_tr, n_va, n_te = 420, 50, 300
    n = n_tr + n_va + n_te
    rels = [f"{i % 4 + 1:03d}.Cls{i % 4 + 1}/Cls{i % 4 + 1}_{i:04d}.jpg" for i in range(n)]
    split = np.array([0] * n_tr + [1] * n_va + [2] * n_te)
    y01 = (rng.random(n) < 0.3).astype(int)
    place = np.where(rng.random(n) < 0.85, y01, 1 - y01)
    base = os.path.join(td, dec["dir"])
    os.makedirs(base, exist_ok=True)
    pd.DataFrame({dec["image_col"]: rels, dec["y_col"]: y01, dec["attr_col"]: place,
                  dec["split_col"]: split}).to_csv(os.path.join(base, dec["metadata"]),
                                                   index=False)
    work = os.path.join(td, "work")
    os.makedirs(os.path.join(work, "results"), exist_ok=True)
    W, H = 500.0, 400.0
    pd.DataFrame({"img_filename": rels, "mask_found": True,
                  "bird_frac": rng.uniform(0.01, 0.4, n), "size_match": True,
                  "img_w": W, "img_h": H, "mask_w": W, "mask_h": H, "split": split,
                  "y": y01, "place": place, "g_orig": (place != y01).astype(int)}
                 ).to_csv(os.path.join(work, "results", "v5_bird_fraction.csv"), index=False)

    # Source bundles for the train and test splits, in metadata order. Attribute 1
    # is planted, on each split separately, on easy points under that split's own
    # separator, label-proportionally.
    d = 16
    w = rng.normal(size=d)
    n_attr, n_parts = 5, 15
    present = rng.integers(0, 2, (n, n_attr))
    present[:, 0] = 0
    for code, name, k_plant in ((0, "train", 90), (2, "test", 70)):
        sp = split == code
        y = 2 * y01[sp] - 1
        X = rng.normal(size=(int(sp.sum()), d)) + 5.0 * y[:, None] * (w / np.linalg.norm(w))[None, :]
        X = (X - X.mean(0)) / X.std(0)
        FeatureBundle(phi=X, y=y, g=(place[sp] != y01[sp]).astype(int),
                      idx_r=np.array([], int), idx_s=np.array([], int), place=place[sp],
                      meta={"standardized": True, "split": name, "backbone": "mini"}
                      ).save(os.path.join(work, f"features_v4_mini_dinov2_{name}.npz"))
        u = fit_margins(X, y.astype(float), C_LADDER, 200000, 1e-8)["u"]
        easy = np.where(u >= np.median(u))[0]
        idx = np.where(sp)[0]
        for c in (0, 1):
            pool = easy[y01[sp][easy] == c]
            k = int(round(k_plant * (y01[sp] == c).mean()))
            present[idx[rng.choice(pool, k, replace=False)], 0] = 1
    cert = rng.integers(1, 5, (n, n_attr))
    vis = rng.integers(0, 2, (n, n_parts))
    box = np.column_stack([rng.uniform(0, 60, n), rng.uniform(0, 60, n),
                           rng.uniform(50, 400, n), rng.uniform(50, 300, n)])
    cub_rels = rels[1:] if drop_one_image else rels
    keep = slice(1, None) if drop_one_image else slice(None)
    write_mini_cub(td, cub_rels, n_attr, present[keep], cert[keep], vis[keep], box[keep])
    return work


def main() -> int:
    print("session 6 integration check")
    checks = ["cub_meta.py CLI joins a mini tree",
              "sv_screen.py CLI finds the planted attribute, no coin-flip one",
              "unmodified group_margins.py measures the confirmation bundles",
              "sv_screen.py --compare reports agreement",
              "confirmation bundle features are bit-identical to the source",
              "the replication run on the test split confirms the planted hit",
              "cub_meta.py refuses a tree with a Waterbirds image missing"]
    fails = []
    bar = pbar(checks, desc="checks", unit="check")
    it = iter(bar)
    with tempfile.TemporaryDirectory() as td:
        work = _build(td)
        env = dict(os.environ, SPURIOUS_DATA_ROOT=td, NO_PROGRESS="1",
                   PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))

        next(it)
        rc, out = _run(["cub_meta.py", "--no-download", "--fractions",
                        "results/v5_bird_fraction.csv", "--tag", "v6_cub_meta"], work, env)
        if rc != 0 or not os.path.exists(os.path.join(work, "results", "v6_cub_meta.npz")):
            fails.append(f"cub_meta.py exited {rc}: {out}")

        next(it)
        rc, out = _run(["sv_screen.py", "--bundle", "features_v4_mini_dinov2_train.npz",
                        "--meta", "results/v6_cub_meta.npz"], work, env)
        if rc != 0:
            fails.append(f"sv_screen.py exited {rc}: {out}")
        else:
            js = json.load(open(os.path.join(work, "results", "v6_sv_screen.json")))
            hits = [r["name"] for r in js["candidates"] if r.get("hit")]
            # The coin-flip attributes (~210 images each) cannot avoid the margin
            # set by chance, so any of them called a hit is a bug. The small random
            # proxy tails CAN, at the family-wise rate Holm allows (5%): that is
            # the test working, not failing, so they are not held to zero here --
            # the false-positive rate itself is checked in `sv_screen.py --self-test`.
            coin = [h for h in hits if h.startswith("has_thing::") and h != PLANTED]
            if PLANTED not in hits:
                fails.append(f"the planted attribute was not called a hit; hits: {hits}")
            if coin:
                fails.append(f"coin-flip attributes called hits: {coin}")

        next(it)
        lst = open(os.path.join(work, "results", "v6_confirm_list.txt")).read().split()
        rc, out = _run(["group_margins.py", "--no-lp", "--tag", "v6_confirm_margins",
                        "--bundles", *lst], work, env)
        if rc != 0:
            fails.append(f"group_margins.py exited {rc}: {out}")

        next(it)
        rc, out = _run(["sv_screen.py", "--compare", "results/v6_confirm_margins.json"],
                       work, env)
        if rc != 0 or "AGREEMENT OK" not in out:
            fails.append(f"--compare did not report agreement ({rc}): {out}")

        next(it)
        from common import FeatureBundle
        src = FeatureBundle.load(os.path.join(work, "features_v4_mini_dinov2_train.npz"))
        for p in lst:
            b = FeatureBundle.load(os.path.join(work, p))
            if not (np.array_equal(b.phi, src.phi) and np.array_equal(b.y, src.y)):
                fails.append(f"{p}: phi or y differs from the source bundle")

        next(it)
        before = open(os.path.join(work, "results", "v6_confirm_list.txt")).read()
        rc, out = _run(["sv_screen.py", "--bundle", "features_v4_mini_dinov2_test.npz",
                        "--split", "test", "--tag", "v6_sv_screen_test", "--no-confirm",
                        "--meta", "results/v6_cub_meta.npz"], work, env)
        if rc != 0:
            fails.append(f"test-split screen exited {rc}: {out}")
        else:
            rc, out = _run(["sv_screen.py", "--replicate",
                            "results/v6_sv_screen_test.json"], work, env)
            md = open(os.path.join(work, "results", "v6_replication.md")).read()
            line = [l for l in md.splitlines() if l.startswith(f"| {PLANTED} |")]
            if rc != 0 or not line or "**yes**" not in line[0]:
                fails.append(f"planted hit did not replicate ({rc}): {md[-800:]}")
            if open(os.path.join(work, "results", "v6_confirm_list.txt")).read() != before:
                fails.append("--no-confirm run overwrote the confirmation list")

    with tempfile.TemporaryDirectory() as td:
        next(it)
        work = _build(td, drop_one_image=True)
        env = dict(os.environ, SPURIOUS_DATA_ROOT=td, NO_PROGRESS="1",
                   PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
        rc, out = _run(["cub_meta.py", "--no-download", "--fractions",
                        "results/v5_bird_fraction.csv"], work, env)
        if rc == 0:
            fails.append("cub_meta.py accepted a tree with a missing image")
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
    print("\nINTEGRATION OK -- the real command line joins the CUB metadata, finds a "
          "planted margin-free attribute and no coin-flip one, and the unmodified "
          "group_margins.py agrees with the screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
