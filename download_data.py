"""Fetch Waterbirds and CelebA into the sibling data directory.

    python download_data.py                # both
    python download_data.py --only waterbirds
    python download_data.py --check        # what is on disk, and is it complete
    python download_data.py --force        # re-fetch even if present

STRAIGHT ANSWER ON WHAT IS AUTOMATED
====================================

  WATERBIRDS   fully automated, reliable. One HTTPS GET from Stanford, one tar
               extract. It has no quota and no login. It just works.

  CELEBA       automated with a real chance of failing, through no fault of this
               script. The official distribution is hosted on Google Drive, which
               rate-limits by IP and serves a virus-scan interstitial instead of
               the file once a quota is hit. "Quota exceeded" on a cloud box is
               common, not exceptional.

               Three routes are tried in order, and the first that works wins:
                 1. torchvision's own CelebA downloader -- it carries the
                    maintained Drive file IDs and handles the confirm token.
                 2. gdown, using the IDs read OUT OF torchvision rather than
                    hardcoded here, so they cannot go stale independently.
                 3. Nothing. It prints exactly which files to fetch by hand,
                    where to put them, and what they should look like.

               If you land on route 3 that is a browser download and a file move,
               not a debugging session. The instructions are exact.

Everything goes into a SIBLING of the repo, never inside it -- Waterbirds is
~1.2 GB and derived from CUB-200-2011 + Places, CelebA is ~1.4 GB, and neither is
ours to redistribute:

    <paperspace root>/
    |-- NeurIPS-rebuttals-3/     <- run commands from here
    `-- data/                    <- SPURIOUS_DATA_ROOT, override to move it
        |-- waterbird_complete95_forest2water2/
        `-- celeba/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

from datasets import DATA_ROOT
from progress import pbar

WATERBIRDS_URL = ("https://nlp.stanford.edu/data/dro/"
                  "waterbird_complete95_forest2water2.tar.gz")

# What each dataset must contain to be considered present. Checked by --check
# and after every fetch, so a partial download fails HERE, loudly, rather than
# three steps later as a confusing column error.
REQUIRED = {
    "waterbirds": {
        "dir": "waterbird_complete95_forest2water2",
        "files": ["metadata.csv"],
        "image_probe": "001.Black_footed_Albatross",
        "min_images": 11_000,
    },
    "celeba": {
        "dir": "celeba",
        # Either extension is fine; _find() in datasets.py sniffs it.
        "files": [("list_attr_celeba.txt", "list_attr_celeba.csv"),
                  ("list_eval_partition.txt", "list_eval_partition.csv")],
        "image_probe": "img_align_celeba",
        "min_images": 200_000,
    },
}


_DL = {"bar": None, "total": None}


def _hook(blocks, bs, total):
    """urlretrieve reporthook -> one progress bar per download.

    urlretrieve gives no start/finish callback, so the bar is created lazily on
    the first block and retired when `total` changes or the transfer completes.
    Progress goes to stderr (see progress.py): this module also prints manual
    instructions to stdout that a user may pipe to a file.
    """
    if total <= 0:
        return
    if _DL["bar"] is None or _DL["total"] != total:
        if _DL["bar"] is not None:
            _DL["bar"].close()
        _DL["bar"] = pbar(total=total, unit="B", desc="    download")
        _DL["total"] = total
    bar = _DL["bar"]
    bar.update(max(0, min(blocks * bs, total) - bar.n))
    if blocks * bs >= total:
        bar.close()
        _DL["bar"] = None
        _DL["total"] = None


def _count_images(root: str) -> int:
    n = 0
    for _, _, files in pbar(os.walk(root), unit="dir", desc="    counting", leave=False):
        n += sum(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)
        if n > 250_000:
            break
    return n


def check(dataset: str, root: str, verbose: bool = True) -> bool:
    """Is `dataset` present and complete? Prints what is missing."""
    spec = REQUIRED[dataset]
    base = os.path.join(root, spec["dir"])
    problems = []

    if not os.path.isdir(base):
        problems.append(f"directory missing: {base}")
    else:
        for f in spec["files"]:
            names = (f,) if isinstance(f, str) else f
            if not any(os.path.exists(os.path.join(base, n)) for n in names):
                problems.append("missing " + " or ".join(names))
        probe = os.path.join(base, spec["image_probe"])
        if not os.path.exists(probe):
            problems.append(f"missing images: {spec['image_probe']}")
        else:
            n = _count_images(base)
            if n < spec["min_images"]:
                problems.append(
                    f"only {n} images, expected >= {spec['min_images']} "
                    f"(a partial or interrupted extract)")

    if verbose:
        if problems:
            print(f"  [{dataset}] INCOMPLETE")
            for p in problems:
                print(f"      - {p}")
        else:
            print(f"  [{dataset}] ok  ({base})")
    return not problems


# --------------------------------------------------------------------------
# Waterbirds
# --------------------------------------------------------------------------


def fetch_waterbirds(root: str, force: bool = False) -> bool:
    if check("waterbirds", root, verbose=False) and not force:
        print("  [waterbirds] already present, skipping (--force to re-fetch)")
        return True

    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, "waterbird_complete95_forest2water2.tar.gz")
    print(f"  [waterbirds] downloading {WATERBIRDS_URL}")
    try:
        urllib.request.urlretrieve(WATERBIRDS_URL, tgz, reporthook=_hook)
        print()
    except Exception as e:
        print(f"\n  [waterbirds] DOWNLOAD FAILED: {type(e).__name__}: {e}")
        print(f"      Fetch it manually and place the .tar.gz in {root}, then "
              f"re-run. URL:\n      {WATERBIRDS_URL}")
        return False

    print("  [waterbirds] extracting ...")
    with tarfile.open(tgz, "r:gz") as tf:
        # filter="data" refuses absolute paths and .. traversal. Not paranoia
        # about Stanford; it is the default in 3.14 and silences the warning.
        try:
            tf.extractall(root, filter="data")
        except TypeError:
            tf.extractall(root)
    os.remove(tgz)
    return check("waterbirds", root)


# --------------------------------------------------------------------------
# CelebA
# --------------------------------------------------------------------------


MANUAL_CELEBA = """
  ------------------------------------------------------------------------
  CELEBA MUST BE FETCHED BY HAND. This is a browser download and a file
  move; nothing here is broken and there is nothing to debug.

  1. Open the official page and follow its Google Drive link:

         Google Drive folder: "CelebA/Img" and "CelebA/Eval" and "CelebA/Anno"
         (the CelebA project page links to it; search "CelebA dataset MMLAB")

  2. Download exactly three files:

         Img/img_align_celeba.zip          (~1.4 GB)
         Anno/list_attr_celeba.txt         (~26 MB)
         Eval/list_eval_partition.txt      (~2.8 MB)

  3. Put them here, and unzip the first one IN PLACE:

         {base}/img_align_celeba.zip   ->  unzip  ->  {base}/img_align_celeba/
         {base}/list_attr_celeba.txt
         {base}/list_eval_partition.txt

     After unzipping you should have {base}/img_align_celeba/000001.jpg
     and 202,598 more.

  4. Re-run:  python download_data.py --check

  The Kaggle mirror (jessicali9530/celeba-dataset) ships the same content as
  .csv instead of .txt and needs a kaggle.json credential. datasets.py reads
  either layout, so use whichever you can actually get.
  ------------------------------------------------------------------------
"""


def _celeba_via_torchvision(root: str) -> bool:
    """Route 1. torchvision carries the maintained Drive IDs and confirm-token logic."""
    try:
        from torchvision.datasets import CelebA
    except Exception as e:
        print(f"    route 1 (torchvision) unavailable: {type(e).__name__}: {e}")
        return False
    try:
        print("    route 1: torchvision CelebA(download=True) ...")
        CelebA(root=root, split="all", download=True)
        return True
    except Exception as e:
        print(f"    route 1 failed: {type(e).__name__}: {e}")
        return False


def _celeba_via_gdown(base: str) -> bool:
    """Route 2. gdown, with IDs read out of torchvision so they cannot go stale here."""
    try:
        import gdown
        from torchvision.datasets import CelebA
    except Exception as e:
        print(f"    route 2 (gdown) unavailable: {type(e).__name__}: {e}")
        return False

    wanted = {"img_align_celeba.zip", "list_attr_celeba.txt",
              "list_eval_partition.txt"}
    try:
        table = {fname: fid for (fid, _md5, fname) in CelebA.file_list}
    except Exception as e:
        print(f"    route 2: could not read torchvision's file table: {e}")
        return False

    os.makedirs(base, exist_ok=True)
    ok = True
    for fname in wanted:
        if fname not in table:
            print(f"    route 2: {fname} not in torchvision's table")
            ok = False
            continue
        dest = os.path.join(base, fname)
        if os.path.exists(dest):
            continue
        print(f"    route 2: gdown {fname}")
        try:
            gdown.download(id=table[fname], output=dest, quiet=False)
        except Exception as e:
            print(f"    route 2 failed on {fname}: {type(e).__name__}: {e}")
            ok = False
    return ok


def fetch_celeba(root: str, force: bool = False) -> bool:
    base = os.path.join(root, "celeba")
    if check("celeba", root, verbose=False) and not force:
        print("  [celeba] already present, skipping (--force to re-fetch)")
        return True

    print("  [celeba] Google Drive hosted; trying the automated routes.")
    got = _celeba_via_torchvision(root) or _celeba_via_gdown(base)

    # torchvision extracts into <root>/celeba/ already; gdown leaves a zip.
    zp = os.path.join(base, "img_align_celeba.zip")
    if os.path.exists(zp) and not os.path.isdir(os.path.join(base, "img_align_celeba")):
        print("  [celeba] unzipping img_align_celeba.zip ...")
        with zipfile.ZipFile(zp) as z:
            z.extractall(base)

    if check("celeba", root, verbose=True):
        return True

    print(MANUAL_CELEBA.format(base=base))
    return False


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["waterbirds", "celeba"], default=None)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--check", action="store_true",
                    help="report what is on disk; download nothing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    targets = [args.only] if args.only else ["waterbirds", "celeba"]
    print(f"data root: {root}")
    free = shutil.disk_usage(os.path.dirname(root) or ".").free / 1e9
    print(f"free space: {free:.1f} GB   (need ~6 GB for both, unpacked)")

    if args.check:
        ok = all(check(t, root) for t in targets)
        print("\nALL PRESENT" if ok else "\nSOMETHING IS MISSING (see above)")
        raise SystemExit(0 if ok else 1)

    results = {}
    for t in targets:
        print(f"\n=== {t} " + "=" * 50)
        results[t] = (fetch_waterbirds if t == "waterbirds"
                      else fetch_celeba)(root, force=args.force)

    print("\n" + "=" * 60)
    for t, ok in results.items():
        print(f"  {t:<12} {'OK' if ok else 'NOT COMPLETE -- see instructions above'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
