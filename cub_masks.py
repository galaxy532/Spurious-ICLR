"""Bird-pixel fraction for every Waterbirds image, from the CUB-200-2011 masks.

WHY THIS EXISTS
===============
Sessions 1-4 never found a case of the `alpha > 1` branch of Theorem 5.3. The
branch needs the two groups to have genuinely DIFFERENT core-feature margins:

    alpha > 1   <=>   gamma~_min / gamma~_maj  >  gamma_crit = (1 + mu_A)/(1 + mu)

Waterbirds' group variable is `g = 1[place != y]` -- derived from the background.
The background is a NUISANCE: it is chosen to be uncorrelated with how hard the
bird itself is to recognise. So the two groups are equally hard by construction,
the margin ratio comes out at 1.0000 (session 4 measured it at seven degradation
levels, all ties to within 2e-9), and the setting sits ON the phase transition.

Session 5 changes the PARTITION instead of the images. Bird size in frame is a
real difficulty axis -- a bird occupying 30% of the pixels carries far more core
signal than one occupying 1% -- and it is a property of the photograph, measured
before any training happens. This module measures it.

Nothing here modifies any image. Session 4's degradation operators created a
confound (they destroy the bird but preserve the background, so the
spurious-aligned group stays separable through the shortcut). Re-partitioning
has no such problem: the pixels are untouched and the features already extracted
are reused verbatim.

WHERE THE MASKS COME FROM
=========================
NOT from Waterbirds. No Waterbirds release ships segmentation masks; the tarball
is `metadata.csv` plus composited JPEGs. The masks are CUB-200-2011's own, a
separate 39.3 MB download from CaltechDATA:

    https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1
    md5 4d47ba1228eae64f2fa547c47bc65255

Waterbirds preserves CUB's `<class folder>/<image name>` paths, so the join is by
filename -- exact, no segmentation model involved. That matters: a learned
segmenter's errors would be largest exactly where the bird is small or
low-contrast, i.e. correlated with the quantity being measured. That confound
would be built into the instrument.

THE ALIGNMENT CHECK, AND WHY IT IS NOT OPTIONAL
===============================================
The CUB mask describes the ORIGINAL CUB photograph. Waterbirds pastes the
segmented bird onto a Places background. If that generation step cropped or
resized anything, the mask does not describe the composite and every number
downstream is meaningless.

The check is cheap and decisive: a crop or a resize changes the image
dimensions, so the mask's pixel dimensions must equal the composite's, for every
image. This script reads both headers (PIL does not decode the pixels for
`.size`) and refuses to write a usable output if the match rate is not 100%.

Usage
-----
    python cub_masks.py --self-test        # no data, no download, a few seconds
    python cub_masks.py                    # fetch if needed, then measure
    python cub_masks.py --no-download      # fail instead of fetching
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request

import numpy as np

from progress import pbar

SEG_URL = ("https://data.caltech.edu/records/w9d68-gec53/files/"
           "segmentations.tgz?download=1")
SEG_MD5 = "4d47ba1228eae64f2fa547c47bc65255"
SEG_TGZ = "segmentations.tgz"

# The archive's own top-level directory, plus the places a hand-extraction
# plausibly puts it. Checked in order; the first that contains class folders wins.
SEG_CANDIDATES = (
    "segmentations",
    os.path.join("CUB_200_2011", "segmentations"),
    os.path.join("cub_200_2011", "segmentations"),
)

# CUB masks are 8-bit greyscale with antialiased edges, not hard binary. Anything
# above mid-grey is bird. The exact cut barely matters (the boundary is a few
# pixels wide) and `--mask-threshold` exposes it for a sensitivity check.
DEFAULT_THRESHOLD = 127


# ----------------------------------------------------------------------------
# Locating / fetching the archive
# ----------------------------------------------------------------------------
def _looks_like_segmentations(p: str) -> bool:
    """True if `p` holds class subfolders with PNGs in them.

    Deliberately not keyed on CUB's `001.Name` convention: the check is
    structural (a subfolder containing at least one .png), so a re-release or a
    hand-extraction under a different naming scheme still passes.
    """
    try:
        entries = sorted(os.listdir(p))
    except OSError:
        return False
    for e in entries[:50]:
        sub = os.path.join(p, e)
        if not os.path.isdir(sub):
            continue
        try:
            if any(f.lower().endswith(".png") for f in os.listdir(sub)[:50]):
                return True
        except OSError:
            continue
    return False


def find_segmentations(root: str) -> str | None:
    """Return the segmentations directory under `root`, or None."""
    for cand in SEG_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isdir(p) and _looks_like_segmentations(p):
            return p
    return None


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    total = os.path.getsize(path)
    with open(path, "rb") as fh, pbar(total=total, desc="md5", unit="B") as bar:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            bar.update(len(b))
    return h.hexdigest()


def download_segmentations(root: str) -> str:
    """Fetch, verify and extract segmentations.tgz into `root`. Returns the dir."""
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, SEG_TGZ)

    if not os.path.exists(tgz):
        print(f"downloading CUB segmentations (39.3 MB) into {root} ...")
        with pbar(total=None, desc="download", unit="B") as bar:
            def hook(count, block, total):
                bar.total = total if total and total > 0 else None
                bar.update(block)
            try:
                urllib.request.urlretrieve(SEG_URL, tgz, reporthook=hook)
            except Exception as exc:                       # noqa: BLE001
                if os.path.exists(tgz):
                    os.remove(tgz)
                raise RuntimeError(
                    f"could not download the CUB segmentations: {exc}\n"
                    f"Fetch it by hand instead -- it is one public file:\n"
                    f"    cd {root}\n"
                    f'    curl -L -o {SEG_TGZ} "{SEG_URL}"\n'
                    f"    md5sum {SEG_TGZ}   # {SEG_MD5}\n"
                    f"    tar xzf {SEG_TGZ}\n"
                ) from exc

    got = _md5(tgz)
    if got != SEG_MD5:
        raise RuntimeError(
            f"md5 mismatch on {tgz}\n  expected {SEG_MD5}\n  got      {got}\n"
            "Delete the file and try again; a truncated download is the usual cause."
        )
    print(f"md5 ok ({SEG_MD5})")

    print("extracting ...")
    with tarfile.open(tgz, "r:gz") as tf:
        try:
            tf.extractall(root, filter="data")     # py>=3.12: blocks traversal
        except TypeError:
            tf.extractall(root)

    seg = find_segmentations(root)
    if seg is None:
        raise RuntimeError(
            f"extracted {SEG_TGZ} into {root} but found no segmentations directory.\n"
            f"Looked for: {', '.join(SEG_CANDIDATES)}\n"
            f"Run `ls {root}` and tell me the top-level name."
        )
    return seg


# ----------------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------------
def mask_path_for(seg_root: str, img_filename: str) -> str:
    """Waterbirds `001.Foo/Foo_0001_123.jpg` -> CUB `001.Foo/Foo_0001_123.png`."""
    stem, _ = os.path.splitext(img_filename)
    return os.path.join(seg_root, stem + ".png")


def measure(paths, img_filenames, seg_root: str, threshold: int) -> list[dict]:
    """Per-image bird-pixel fraction plus the mask/composite dimension check."""
    from PIL import Image

    rows = []
    for p, rel in pbar(list(zip(paths, img_filenames)), desc="masks", unit="img"):
        mp = mask_path_for(seg_root, rel)
        row = {"img_filename": rel, "mask_found": os.path.exists(mp)}
        if not row["mask_found"]:
            row.update(bird_frac=float("nan"), size_match=False,
                       img_w=None, img_h=None, mask_w=None, mask_h=None)
            rows.append(row)
            continue
        try:
            with Image.open(p) as im:               # header only, no decode
                iw, ih = im.size
            with Image.open(mp) as mk:
                mw, mh = mk.size
                arr = np.asarray(mk.convert("L"))
        except Exception as exc:                     # noqa: BLE001
            row.update(bird_frac=float("nan"), size_match=False, error=str(exc)[:160],
                       img_w=None, img_h=None, mask_w=None, mask_h=None)
            rows.append(row)
            continue
        row.update(
            img_w=int(iw), img_h=int(ih), mask_w=int(mw), mask_h=int(mh),
            size_match=bool((iw, ih) == (mw, mh)),
            bird_frac=float((arr > threshold).mean()),
        )
        rows.append(row)
    return rows


def read_waterbirds_metadata(root: str):
    """(relative image paths, absolute paths, y, place, g, split), in file order.

    Read straight from `metadata.csv` rather than through `datasets.load_metadata`
    so there is exactly one pass over the file and the row order used here is the
    row order `regroup.py` joins against. The column names and the group rule
    come from `datasets.DECLARATIONS`, so the single source of truth for
    `g = 1[place != y]` is still that file.
    """
    import pandas as pd
    from datasets import DECLARATIONS

    dec = DECLARATIONS["waterbirds"]
    base = os.path.join(root, dec["dir"])
    md = pd.read_csv(os.path.join(base, dec["metadata"]))
    need = {dec["image_col"], dec["y_col"], dec["attr_col"], dec["split_col"]}
    missing = need - set(md.columns)
    if missing:
        raise SystemExit(f"{dec['metadata']} is missing columns {sorted(missing)}")

    rels = [str(p) for p in md[dec["image_col"]]]
    y = md[dec["y_col"]].to_numpy().astype(int)
    place = md[dec["attr_col"]].to_numpy().astype(int)
    g = (place != y).astype(int)                     # dec["group_rule"]
    split = md[dec["split_col"]].to_numpy().astype(int)
    paths = [os.path.join(base, p) for p in rels]
    return rels, paths, y, place, g, split


def report(rows, y, place, g, split_code, threshold) -> dict:
    """Aggregate, and decide whether the join is trustworthy."""
    n = len(rows)
    found = np.array([r["mask_found"] for r in rows])
    match = np.array([bool(r["size_match"]) for r in rows])
    frac = np.array([r["bird_frac"] for r in rows], dtype=float)
    usable = found & match & np.isfinite(frac)

    rep = {
        "n": int(n),
        "n_mask_found": int(found.sum()),
        "n_size_match": int(match.sum()),
        "n_usable": int(usable.sum()),
        "mask_threshold": int(threshold),
        "size_match_rate": float(match.mean()) if n else float("nan"),
    }
    rep["alignment_ok"] = bool(rep["n_usable"] == n)
    rep["alignment_verdict"] = (
        "OK -- every mask has the same pixel dimensions as its composite, so the "
        "CUB segmentation describes the Waterbirds image"
        if rep["alignment_ok"] else
        "FAILED -- some masks do not match their composite's dimensions. The "
        "Waterbirds generation must have cropped or resized; the bird fraction "
        "does NOT describe these images and nothing downstream should use it."
    )

    if usable.any():
        f = frac[usable]
        rep["bird_frac"] = {
            "min": float(f.min()), "max": float(f.max()),
            "mean": float(f.mean()), "median": float(np.median(f)),
            "q10": float(np.quantile(f, 0.10)), "q33": float(np.quantile(f, 1 / 3)),
            "q50": float(np.quantile(f, 0.50)), "q67": float(np.quantile(f, 2 / 3)),
            "q90": float(np.quantile(f, 0.90)),
        }
        # LEAKAGE CHECK. The group variable must track DIFFICULTY, not the label.
        # If waterbirds are systematically photographed further away than
        # landbirds, bird size is a proxy for y and the whole partition is void.
        yy = np.asarray(y)[usable]
        rep["leakage"] = _leakage(f, yy)
        pl = np.asarray(place)[usable]
        rep["leakage_vs_place"] = _leakage(f, pl)
        gg = np.asarray(g)[usable]
        rep["leakage_vs_old_group"] = _leakage(f, gg)
    return rep


def _leakage(frac: np.ndarray, lab: np.ndarray) -> dict:
    """How well does bird fraction predict a binary label? AUC + group means.

    AUC is the rank statistic, so it is invariant to any monotone rescaling of
    the fraction -- which is what a threshold split uses. 0.5 = no information.
    """
    lab = np.asarray(lab).astype(int)
    a, b = frac[lab == 1], frac[lab == 0]
    if a.size == 0 or b.size == 0:
        return {"auc": float("nan"), "mean_1": float("nan"), "mean_0": float("nan")}
    # Mann-Whitney U / (n1 n0), ties counted as half.
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    srt = np.concatenate([a, b])[order]
    i = 0
    r = np.empty(order.size, dtype=float)
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = r
    auc = (ranks[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size)
    return {"auc": float(auc), "mean_1": float(a.mean()), "mean_0": float(b.mean()),
            "n_1": int(a.size), "n_0": int(b.size)}


def to_markdown(rep: dict, out_csv: str) -> str:
    L = ["# Bird-pixel fraction from the CUB-200-2011 segmentations", "",
         "The fraction of each Waterbirds image's pixels that belong to the bird,",
         "measured from CUB's own segmentation mask joined by filename. Nothing is",
         "modified; this is a property of the photograph, used in `regroup.py` to",
         "build a group variable that tracks core-task DIFFICULTY instead of the",
         "background nuisance.", "",
         "## Alignment", "",
         f"- images: **{rep['n']}**",
         f"- masks found: **{rep['n_mask_found']}**",
         f"- mask dimensions == composite dimensions: **{rep['n_size_match']}** "
         f"({100 * rep['size_match_rate']:.2f}%)",
         f"- usable: **{rep['n_usable']}**", "",
         f"**{rep['alignment_verdict']}**", ""]

    if "bird_frac" in rep:
        b = rep["bird_frac"]
        L += ["## Distribution of the bird fraction", "",
              "| min | q10 | q33 | median | q67 | q90 | max | mean |",
              "|---|---|---|---|---|---|---|---|",
              f"| {b['min']:.4f} | {b['q10']:.4f} | {b['q33']:.4f} | {b['q50']:.4f} | "
              f"{b['q67']:.4f} | {b['q90']:.4f} | {b['max']:.4f} | {b['mean']:.4f} |", "",
              "## Leakage checks -- READ THESE BEFORE USING THE PARTITION", "",
              "The new group variable must track how hard the bird is to see, NOT the",
              "label and NOT the old group. An AUC near 0.5 means no information; far",
              "from 0.5 means bird size is a proxy for that variable and the partition",
              "would be confounded.", "",
              "| against | AUC | mean if 1 | mean if 0 | verdict |",
              "|---|---|---|---|---|"]
        for key, name in (("leakage", "y (label)"),
                          ("leakage_vs_place", "place (background)"),
                          ("leakage_vs_old_group", "g = 1[place != y]")):
            lk = rep.get(key, {})
            auc = lk.get("auc", float("nan"))
            dev = abs(auc - 0.5)
            verdict = ("clean" if dev < 0.05 else
                       "mild -- report it" if dev < 0.10 else
                       "CONFOUNDED -- do not use this partition")
            L.append(f"| {name} | {auc:.4f} | {lk.get('mean_1', float('nan')):.4f} | "
                     f"{lk.get('mean_0', float('nan')):.4f} | {verdict} |")
        L.append("")
    L += [f"Per-image values: `{out_csv}`", ""]
    return "\n".join(L)


def run_on(root: str, seg_root: str, threshold: int, out_dir: str, tag: str):
    """The whole measurement, from a data root to the written files.

    Shared by `main()` and by the end-to-end self-test, so the path the self-test
    exercises is the path the real run takes -- imports, column names, join and
    all. An earlier draft had the self-test cover only the helpers, and a wrong
    import inside `main()` sailed through every check before failing on the real
    data.
    """
    import pandas as pd

    rels, paths, y, place, g, split = read_waterbirds_metadata(root)
    rows = measure(paths, rels, seg_root, threshold)
    rep = report(rows, y, place, g, split, threshold)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{tag}.csv")
    df = pd.DataFrame(rows)
    df["split"] = split
    df["y"] = y
    df["place"] = place
    df["g_orig"] = g
    df.to_csv(out_csv, index=False)

    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(os.path.join(out_dir, f"{tag}.md"), "w") as fh:
        fh.write(to_markdown(rep, out_csv))
    return rep, rows, out_csv


# ----------------------------------------------------------------------------
# Self-test: no data, no download, no GPU
# ----------------------------------------------------------------------------
def mini_rows_on(k: int, n_per_class: int, h: int) -> int:
    """Lit rows of the k-th mini mask: strictly increasing, never the full image."""
    return max(1, int(round(h * (k + 1) / (n_per_class + 1))))


def _mini_dataset(td: str, n_per_class: int = 6):
    """A miniature Waterbirds + CUB-segmentations tree, for the end-to-end check."""
    import pandas as pd
    from PIL import Image
    from datasets import DECLARATIONS

    dec = DECLARATIONS["waterbirds"]
    base = os.path.join(td, dec["dir"])
    seg = os.path.join(td, "segmentations")
    rng = np.random.default_rng(0)
    recs = []
    for ci, cls in enumerate(("001.Alpha", "002.Beta")):
        os.makedirs(os.path.join(base, cls), exist_ok=True)
        os.makedirs(os.path.join(seg, cls), exist_ok=True)
        for k in range(n_per_class):
            name = f"{cls.split('.')[1]}_{k:04d}"
            w, h = 40, 50
            Image.fromarray(np.zeros((h, w, 3), np.uint8), "RGB").save(
                os.path.join(base, cls, name + ".jpg"))
            m = np.zeros((h, w), np.uint8)
            # A distinct, strictly increasing fraction per image, so a median or
            # tercile split over them is well defined however many there are.
            m[: mini_rows_on(k, n_per_class, h), :] = 255
            Image.fromarray(m, "L").save(os.path.join(seg, cls, name + ".png"))
            recs.append({dec["image_col"]: f"{cls}/{name}.jpg",
                         dec["y_col"]: int(rng.integers(0, 2)),
                         dec["attr_col"]: int(rng.integers(0, 2)),
                         dec["split_col"]: ci})
    pd.DataFrame(recs).to_csv(os.path.join(base, dec["metadata"]), index=False)
    return seg


def _self_test() -> int:
    import tempfile
    from PIL import Image

    print("cub_masks self-test")
    ok = True

    with tempfile.TemporaryDirectory() as td:
        seg = os.path.join(td, "segmentations", "001.Test")
        img = os.path.join(td, "images", "001.Test")
        os.makedirs(seg)
        os.makedirs(img)

        # A 40x50 image whose mask marks exactly 400 of 2000 pixels -> 0.20.
        arr = np.zeros((50, 40), dtype=np.uint8)
        arr[:10, :40] = 255
        Image.fromarray(arr, mode="L").save(os.path.join(seg, "a.png"))
        Image.fromarray(np.zeros((50, 40, 3), dtype=np.uint8), "RGB").save(
            os.path.join(img, "a.jpg"))

        # A mask whose dimensions DISAGREE with its composite: must be caught.
        Image.fromarray(np.full((25, 20), 255, dtype=np.uint8), "L").save(
            os.path.join(seg, "b.png"))
        Image.fromarray(np.zeros((50, 40, 3), dtype=np.uint8), "RGB").save(
            os.path.join(img, "b.jpg"))

        # A composite with no mask at all: must be reported, not crash.
        Image.fromarray(np.zeros((50, 40, 3), dtype=np.uint8), "RGB").save(
            os.path.join(img, "c.jpg"))

        rels = ["001.Test/a.jpg", "001.Test/b.jpg", "001.Test/c.jpg"]
        paths = [os.path.join(td, "images", r) for r in rels]
        rows = measure(paths, rels, os.path.join(td, "segmentations"),
                       DEFAULT_THRESHOLD)

        if abs(rows[0]["bird_frac"] - 0.20) > 1e-9:
            print(f"  FAIL bird fraction: {rows[0]['bird_frac']} != 0.20"); ok = False
        else:
            print("  ok   bird fraction exact on a known mask (0.20)")

        if not rows[0]["size_match"]:
            print("  FAIL matching dimensions reported as a mismatch"); ok = False
        else:
            print("  ok   matching dimensions accepted")

        if rows[1]["size_match"]:
            print("  FAIL dimension mismatch NOT caught"); ok = False
        else:
            print("  ok   dimension mismatch caught")

        if rows[2]["mask_found"]:
            print("  FAIL missing mask reported as found"); ok = False
        else:
            print("  ok   missing mask reported, no crash")

        rep = report(rows, y=[1, 0, 1], place=[1, 0, 0], g=[0, 0, 1],
                     split_code=[0, 0, 0], threshold=DEFAULT_THRESHOLD)
        if rep["alignment_ok"]:
            print("  FAIL alignment declared OK with a mismatch present"); ok = False
        else:
            print("  ok   alignment refused when a mismatch is present")

    # AUC: perfectly separating, perfectly anti-separating, and uninformative.
    f = np.array([0.1, 0.2, 0.3, 0.4])
    if abs(_leakage(f, np.array([0, 0, 1, 1]))["auc"] - 1.0) > 1e-12:
        print("  FAIL AUC on a perfect separation"); ok = False
    elif abs(_leakage(f, np.array([1, 1, 0, 0]))["auc"] - 0.0) > 1e-12:
        print("  FAIL AUC on a perfect anti-separation"); ok = False
    elif abs(_leakage(np.ones(4), np.array([0, 1, 0, 1]))["auc"] - 0.5) > 1e-12:
        print("  FAIL AUC on all-ties"); ok = False
    else:
        print("  ok   AUC = 1.0 / 0.0 / 0.5 on separated, anti-separated, tied")

    # End to end, through the SAME function main() calls: metadata read, join,
    # measurement, report, files written. This is what catches a bad import or a
    # renamed column, which the helper checks above cannot see.
    with tempfile.TemporaryDirectory() as td:
        try:
            seg = _mini_dataset(td)
            if find_segmentations(td) is None:
                print("  FAIL find_segmentations did not locate the mini tree"); ok = False
            else:
                print("  ok   find_segmentations locates a real-shaped tree")
            out = os.path.join(td, "out")
            rep, rows, csv_path = run_on(td, seg, DEFAULT_THRESHOLD, out, "mini")
            if not rep["alignment_ok"]:
                print(f"  FAIL end-to-end alignment not OK: {rep}"); ok = False
            elif rep["n_usable"] != len(rows) or rep["n"] != 12:
                print(f"  FAIL end-to-end row counts: {rep['n']}/{rep['n_usable']}")
                ok = False
            elif not all(os.path.exists(os.path.join(out, f"mini{e}"))
                         for e in (".csv", ".json", ".md")):
                print("  FAIL end-to-end did not write all three outputs"); ok = False
            else:
                import pandas as pd
                d = pd.read_csv(csv_path)
                needed = {"img_filename", "bird_frac", "split", "y", "place", "g_orig"}
                if not needed.issubset(d.columns):
                    print(f"  FAIL csv is missing {needed - set(d.columns)}"); ok = False
                elif abs(float(d["bird_frac"].iloc[0]) - mini_rows_on(0, 6, 50) / 50) > 1e-9:
                    print(f"  FAIL end-to-end fraction {d['bird_frac'].iloc[0]}"); ok = False
                else:
                    print("  ok   end-to-end run over a mini dataset, all columns present")
        except Exception as exc:                        # noqa: BLE001
            print(f"  FAIL end-to-end raised {type(exc).__name__}: {exc}")
            ok = False

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="where the datasets live (default: datasets.DATA_ROOT)")
    ap.add_argument("--seg-root", default=None,
                    help="the segmentations directory, if it is somewhere unusual")
    ap.add_argument("--no-download", action="store_true",
                    help="fail instead of fetching the archive")
    ap.add_argument("--mask-threshold", type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="v5_bird_fraction")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    from datasets import DATA_ROOT
    root = args.data_root or DATA_ROOT

    seg_root = args.seg_root or find_segmentations(root)
    if seg_root is None:
        if args.no_download:
            raise SystemExit(
                f"no segmentations directory under {root} and --no-download was given.\n"
                f"Expected one of: {', '.join(SEG_CANDIDATES)}")
        seg_root = download_segmentations(root)
    print(f"segmentations: {seg_root}")

    rep, rows, out_csv = run_on(root, seg_root, args.mask_threshold,
                                args.out_dir, args.tag)

    print(to_markdown(rep, out_csv))
    print(f"wrote {out_csv}")

    if not rep["alignment_ok"]:
        print("\nALIGNMENT FAILED -- regroup.py will refuse to run on this. "
              "Nothing downstream is valid.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
