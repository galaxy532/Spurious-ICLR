"""Per-image CUB-200-2011 metadata, joined to Waterbirds, for the session-6 screen.

WHY THIS EXISTS
===============
Session 5 established what the per-group margin ratio actually measures when only
the group variable changes: the separator `w_hat` is fitted on `(phi, y)` alone,
so it does not move, and `gamma_min / gamma_maj > 1` holds if and only if one
group contains NONE of the points that attain the global margin (421 of 4,795 on
the dinov2 train bundle). Bird size failed that test by 184 points.

Session 6 asks the only remaining question on these features: does ANY
pre-existing, interpretable property of the photographs pick out a set of images
that avoids the margin points more than chance allows? The candidates come from
CUB-200-2011's own annotations, which Waterbirds inherits image for image:

  * 312 binary attributes per image ("has_wing_color::blue", ...), crowd-sourced,
    each with a certainty level (1 not visible, 2 guessing, 3 probably,
    4 definitely);
  * 15 part locations per image, each flagged visible or not;
  * one bounding box per image.

This module only READS and JOINS them. The candidate family and the test live in
`sv_screen.py`, so the family is defined in exactly one place.

WHERE THE FILES COME FROM
=========================
CUB-200-2011's main archive on CaltechDATA (1.2 GB; only the few text files below
are extracted, never the images):

    https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1

The md5 in `CUB_MD5` is the one CaltechDATA publishes on the record page
(checked 23 Sept 2026; the same value is used by independent CUB loaders written
against the old vision.caltech.edu URL). A mismatch therefore means a truncated
or corrupt download: the archive is renamed to `.corrupt` and the run stops, so
rerunning fetches it again. On top of that there is a structural gate: every
Waterbirds image must be found in `images.txt`, and every (image, attribute) and
(image, part) pair must appear exactly once.

Waterbirds keeps CUB's `<class folder>/<image name>` paths, so the join is by
filename, exact -- the same join `cub_masks.py` used for the masks.

Image dimensions and the bird-pixel fraction come from session 5's
`results/v5_bird_fraction.csv` (which `cub_masks.py` writes), so this module never
opens an image. Note that `.gitignore`'s `!results/*.csv` line carries a trailing
comment, which git does not strip, so that negation matches nothing and the csv
is NOT pulled back by git. It exists on the machine that ran session 5.

Usage
-----
    python cub_meta.py --self-test          # no data, no download, seconds
    python cub_meta.py                      # fetch if needed, parse, join
    python cub_meta.py --no-download        # fail instead of fetching
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

CUB_URL = ("https://data.caltech.edu/records/65de6-vp158/files/"
           "CUB_200_2011.tgz?download=1")
CUB_MD5 = "97eceeb196236b17998738112f37df78"   # as published by CaltechDATA
CUB_TGZ = "CUB_200_2011.tgz"

# Relative to the CUB directory. These four are REQUIRED.
REQUIRED = {
    "images": "images.txt",
    "labels": os.path.join("attributes", "image_attribute_labels.txt"),
    "parts": os.path.join("parts", "part_locs.txt"),
    "bbox": "bounding_boxes.txt",
}
# Attribute names live at the archive root in the release, next to CUB_200_2011/;
# other copies put them inside. Optional: ids are used as names if absent.
ATTR_NAME_CANDIDATES = (
    os.path.join("..", "attributes.txt"),
    "attributes.txt",
    os.path.join("attributes", "attributes.txt"),
)
# Members extracted from the archive: matched on the path's tail.
EXTRACT_TAILS = ("/images.txt", "/image_attribute_labels.txt", "/part_locs.txt",
                 "/bounding_boxes.txt", "attributes.txt", "/certainties.txt",
                 "/parts.txt")
CUB_DIR_CANDIDATES = ("CUB_200_2011", "cub_200_2011",
                      os.path.join("CUB_200_2011", "CUB_200_2011"))

# A bounding box within this many pixels of an image edge counts as touching it.
BORDER_PX = 2


# ----------------------------------------------------------------------------
# Locating / fetching
# ----------------------------------------------------------------------------
def find_cub(root: str) -> str | None:
    """Return the CUB_200_2011 directory under `root` holding all REQUIRED files."""
    for cand in CUB_DIR_CANDIDATES:
        d = os.path.join(root, cand)
        if all(os.path.isfile(os.path.join(d, rel)) for rel in REQUIRED.values()):
            return d
    return None


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh, pbar(total=os.path.getsize(path), desc="md5",
                                      unit="B") as bar:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            bar.update(len(b))
    return h.hexdigest()


def extract_text_members(tgz: str, root: str) -> list[str]:
    """Extract only the metadata text files from the archive. Never the images."""
    got = []
    with tarfile.open(tgz, "r:gz") as tf, pbar(desc="scan archive", unit="member") as bar:
        for m in tf:
            bar.update(1)
            if not m.isfile() or not m.name.endswith(EXTRACT_TAILS):
                continue
            if m.name.startswith("/") or ".." in m.name.split("/"):
                continue                               # refuse path traversal
            try:
                tf.extract(m, root, filter="data")
            except TypeError:                          # python < 3.12
                tf.extract(m, root)
            got.append(m.name)
    return got


def fetch_cub(root: str) -> dict:
    """Download (if absent), md5-check (warn only), extract the text members."""
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, CUB_TGZ)
    info = {"tgz": tgz}
    if not os.path.exists(tgz):
        print(f"downloading CUB-200-2011 (1.2 GB; only text files will be extracted) "
              f"into {root} ...")
        part = tgz + ".part"
        with pbar(total=None, desc="download", unit="B") as bar:
            def hook(count, block, total):
                if total and total > 0:
                    bar.total = total
                bar.update(block)
            try:
                urllib.request.urlretrieve(CUB_URL, part, reporthook=hook)
            except Exception as exc:                   # noqa: BLE001
                if os.path.exists(part):
                    os.remove(part)
                raise RuntimeError(
                    f"could not download CUB-200-2011: {exc}\n"
                    f"Fetch it by hand -- one public file:\n"
                    f"    cd {root}\n"
                    f'    curl -L -o {CUB_TGZ} "{CUB_URL}"\n'
                    f"    tar xzf {CUB_TGZ} CUB_200_2011/images.txt "
                    f"CUB_200_2011/bounding_boxes.txt CUB_200_2011/attributes "
                    f"CUB_200_2011/parts attributes.txt\n") from exc
        os.replace(part, tgz)
    got = _md5(tgz)
    info["md5"] = got
    info["md5_expected"] = CUB_MD5
    info["md5_ok"] = bool(got == CUB_MD5)
    if not info["md5_ok"]:
        bad = tgz + ".corrupt"
        os.replace(tgz, bad)
        raise RuntimeError(
            f"md5 mismatch on {CUB_TGZ}\n  expected {CUB_MD5} (published by CaltechDATA)\n"
            f"  got      {got}\nA truncated download is the usual cause. The file was "
            f"moved to {bad}; rerun and it will be fetched again.")
    info["extracted"] = extract_text_members(tgz, root)
    return info


# ----------------------------------------------------------------------------
# Parsing. Deliberately line by line: the attribute file is known to be
# irregular in places, and a strict CSV reader would refuse the whole file.
# ----------------------------------------------------------------------------
def _count_lines(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def parse_images(path: str) -> dict[str, int]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.split()
            if len(t) >= 2:
                out[t[1]] = int(t[0])
    return out


def parse_attribute_labels(path: str, n_img: int):
    """(present, certainty, n_attr, stats). Arrays are (n_img, n_attr) int8.

    Each line is `image_id attribute_id is_present certainty_id time`. Only the
    first four fields are used. A line with a field count other than five is
    counted as IRREGULAR and still used if its first four fields parse; the
    coverage check afterwards is what decides whether the file is usable.
    """
    rows = []
    irregular = unparsable = 0
    total = _count_lines(path)
    with open(path, encoding="utf-8") as fh:
        for line in pbar(fh, total=total, desc="attribute labels", unit="line"):
            t = line.split()
            if not t:
                continue
            if len(t) != 5:
                irregular += 1
            try:
                rows.append((int(t[0]), int(t[1]), int(t[2]), int(t[3])))
            except (ValueError, IndexError):
                unparsable += 1
    a = np.asarray(rows, dtype=np.int64)
    n_attr = int(a[:, 1].max())
    present = np.full((n_img, n_attr), -1, dtype=np.int8)
    cert = np.zeros((n_img, n_attr), dtype=np.int8)
    count = np.zeros((n_img, n_attr), dtype=np.int32)
    ii, jj = a[:, 0] - 1, a[:, 1] - 1
    ok = (ii >= 0) & (ii < n_img) & (jj >= 0)
    np.add.at(count, (ii[ok], jj[ok]), 1)
    present[ii[ok], jj[ok]] = a[ok, 2]
    cert[ii[ok], jj[ok]] = a[ok, 3]
    stats = {"lines": int(total), "irregular_lines": int(irregular),
             "unparsable_lines": int(unparsable), "n_attr": n_attr,
             "pairs_missing": int((count == 0).sum()),
             "pairs_duplicated": int((count > 1).sum())}
    return present, cert, n_attr, stats


def parse_parts(path: str, n_img: int):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.split()
            if len(t) >= 5:
                rows.append((int(t[0]), int(t[1]), int(float(t[4]))))
    a = np.asarray(rows, dtype=np.int64)
    n_parts = int(a[:, 1].max())
    vis = np.zeros((n_img, n_parts), dtype=np.int8)
    count = np.zeros((n_img, n_parts), dtype=np.int32)
    np.add.at(count, (a[:, 0] - 1, a[:, 1] - 1), 1)
    vis[a[:, 0] - 1, a[:, 1] - 1] = a[:, 2]
    return vis, {"n_parts": n_parts, "pairs_missing": int((count == 0).sum()),
                 "pairs_duplicated": int((count > 1).sum())}


def parse_bbox(path: str, n_img: int):
    box = np.full((n_img, 4), np.nan)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.split()
            if len(t) >= 5:
                box[int(t[0]) - 1] = [float(v) for v in t[1:5]]
    return box, {"missing": int(np.isnan(box).any(axis=1).sum())}


def parse_attr_names(cub_dir: str, n_attr: int) -> tuple[list[str], str | None]:
    for rel in ATTR_NAME_CANDIDATES:
        p = os.path.normpath(os.path.join(cub_dir, rel))
        if os.path.isfile(p):
            names = [f"attr_{j + 1}" for j in range(n_attr)]
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    t = line.split(maxsplit=1)
                    if len(t) == 2 and t[0].isdigit() and 1 <= int(t[0]) <= n_attr:
                        names[int(t[0]) - 1] = t[1].strip()
            return names, p
    return [f"attr_{j + 1}" for j in range(n_attr)], None


# ----------------------------------------------------------------------------
# Joining to Waterbirds
# ----------------------------------------------------------------------------
def build(root: str, cub_dir: str, fractions_csv: str) -> tuple[dict, dict]:
    """Every array aligned to Waterbirds `metadata.csv` row order (all splits)."""
    import pandas as pd
    from cub_masks import read_waterbirds_metadata

    rels, _, y, place, g, split = read_waterbirds_metadata(root)
    n = len(rels)
    rep = {"n_waterbirds": n, "cub_dir": cub_dir}

    name2id = parse_images(os.path.join(cub_dir, REQUIRED["images"]))
    n_img = max(name2id.values())
    rep["n_cub_images"] = len(name2id)
    ids = np.array([name2id.get(r, -1) for r in rels], dtype=np.int64)
    rep["n_joined"] = int((ids > 0).sum())

    present, cert, n_attr, st_a = parse_attribute_labels(
        os.path.join(cub_dir, REQUIRED["labels"]), n_img)
    vis, st_p = parse_parts(os.path.join(cub_dir, REQUIRED["parts"]), n_img)
    box, st_b = parse_bbox(os.path.join(cub_dir, REQUIRED["bbox"]), n_img)
    names, names_path = parse_attr_names(cub_dir, n_attr)
    rep.update(attributes=st_a, parts=st_p, bbox=st_b, attr_names_file=names_path)

    fr = pd.read_csv(fractions_csv).set_index("img_filename")
    missing_fr = [r for r in rels if r not in fr.index]
    rep["n_missing_fraction_rows"] = len(missing_fr)

    ok_join = rep["n_joined"] == n and not missing_fr
    sel = np.where(ids > 0, ids - 1, 0)
    # Coverage restricted to the images Waterbirds actually uses.
    cov_attr = bool((present[sel] >= 0).all()) if ok_join else False
    rep["attr_coverage_ok"] = cov_attr
    rep["parts_coverage_ok"] = bool(st_p["pairs_missing"] == 0 and st_p["pairs_duplicated"] == 0)
    rep["bbox_ok"] = bool(st_b["missing"] == 0)
    rep["attr_duplicates_ok"] = bool(st_a["pairs_duplicated"] == 0)

    out = {}
    if ok_join:
        f = fr.loc[rels]
        W = f["img_w"].to_numpy(dtype=float)
        H = f["img_h"].to_numpy(dtype=float)
        bx = box[sel]
        x, yy, bw, bh = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
        touches = ((x <= BORDER_PX) | (yy <= BORDER_PX) |
                   (x + bw >= W - BORDER_PX) | (yy + bh >= H - BORDER_PX))
        out = dict(
            img_filename=np.asarray(rels, dtype=object),
            split=np.asarray(split, int), y=np.asarray(y, int),
            place=np.asarray(place, int), g_orig=np.asarray(g, int),
            attr_present=present[sel].astype(np.int8),
            attr_certainty=cert[sel].astype(np.int8),
            attr_names=np.asarray(names, dtype=object),
            part_visible=vis[sel].astype(np.int8),
            bbox=bx,
            img_w=W, img_h=H,
            bird_frac=f["bird_frac"].to_numpy(dtype=float),
            size_match=f["size_match"].to_numpy(dtype=bool),
            # The four continuous difficulty proxies and one binary one. Their
            # definitions are here; which tails are tested is in sv_screen.py.
            bbox_area_frac=(bw * bh) / (W * H),
            n_visible_parts=vis[sel].sum(axis=1).astype(float),
            frac_attr_not_visible=(cert[sel] == 1).mean(axis=1),
            bird_in_frame=(~touches).astype(int),
        )
        rep["bird_in_frame_rate"] = float(out["bird_in_frame"].mean())
        rep["size_match_all"] = bool(out["size_match"].all())
        for k in ("bird_frac", "bbox_area_frac", "n_visible_parts",
                  "frac_attr_not_visible"):
            v = out[k]
            rep[f"dist_{k}"] = {q: float(np.quantile(v, p)) for q, p in
                                (("min", 0), ("q05", .05), ("q25", .25), ("median", .5),
                                 ("q75", .75), ("q95", .95), ("max", 1))}
    rep["ok"] = bool(ok_join and cov_attr and rep["parts_coverage_ok"] and
                     rep["bbox_ok"] and rep["attr_duplicates_ok"] and
                     rep.get("size_match_all", False))
    return out, rep


def to_markdown(rep: dict, fetch: dict | None) -> str:
    L = ["# CUB-200-2011 metadata joined to Waterbirds (session 6)", "",
         "Read-only join by filename. Nothing is modified; no image is opened.", "",
         "## Verdict", "",
         f"**{'OK -- every check passed' if rep['ok'] else 'FAILED -- sv_screen.py will refuse this file'}**",
         "", "| check | value |", "|---|---|",
         f"| Waterbirds images | {rep['n_waterbirds']} |",
         f"| found in CUB images.txt | {rep['n_joined']} |",
         f"| missing from v5_bird_fraction.csv | {rep['n_missing_fraction_rows']} |",
         f"| attributes per image | {rep['attributes']['n_attr']} |",
         f"| attribute lines / irregular / unparsable | {rep['attributes']['lines']} / "
         f"{rep['attributes']['irregular_lines']} / {rep['attributes']['unparsable_lines']} |",
         f"| every (image, attribute) pair present | {rep['attr_coverage_ok']} |",
         f"| no duplicated (image, attribute) pair | {rep['attr_duplicates_ok']} |",
         f"| every (image, part) pair present exactly once | {rep['parts_coverage_ok']} |",
         f"| every image has a bounding box | {rep['bbox_ok']} |",
         f"| mask/composite dimensions all match (session 5) | {rep.get('size_match_all')} |",
         f"| attribute names file | `{rep.get('attr_names_file')}` |", ""]
    if fetch:
        L += [f"- archive md5: `{fetch.get('md5')}` -- matches the value CaltechDATA "
              "publishes", ""]
    if rep.get("ok"):
        L += ["## The difficulty proxies (all splits)", "",
              "| proxy | min | q05 | q25 | median | q75 | q95 | max |",
              "|---|---|---|---|---|---|---|---|"]
        for k in ("bird_frac", "bbox_area_frac", "n_visible_parts",
                  "frac_attr_not_visible"):
            d = rep[f"dist_{k}"]
            L.append(f"| {k} | " + " | ".join(f"{d[q]:.4g}" for q in
                     ("min", "q05", "q25", "median", "q75", "q95", "max")) + " |")
        L += ["", f"- `bird_in_frame` (bounding box at least {BORDER_PX} px from every "
              f"edge): {100 * rep['bird_in_frame_rate']:.1f}% of images", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# Self-test: a miniature CUB tree, including an irregular line and an archive
# ----------------------------------------------------------------------------
def write_mini_cub(root: str, rels: list[str], n_attr: int, present: np.ndarray,
                   cert: np.ndarray, vis: np.ndarray, box: np.ndarray,
                   irregular_line: bool = True, extra_images: int = 3) -> str:
    """Write a CUB-shaped tree under `root`. Also used by validate_session6.py."""
    d = os.path.join(root, "CUB_200_2011")
    os.makedirs(os.path.join(d, "attributes"), exist_ok=True)
    os.makedirs(os.path.join(d, "parts"), exist_ok=True)
    names = [f"extra.Class/extra_{k}.jpg" for k in range(extra_images)] + list(rels)
    with open(os.path.join(d, "images.txt"), "w") as fh:
        for i, r in enumerate(names):
            fh.write(f"{i + 1} {r}\n")
    n_img = len(names)
    off = extra_images
    with open(os.path.join(d, "attributes", "image_attribute_labels.txt"), "w") as fh:
        for i in range(n_img):
            for j in range(n_attr):
                p = int(present[i - off, j]) if i >= off else 0
                c = int(cert[i - off, j]) if i >= off else 1
                if irregular_line and i == off and j == 0:
                    fh.write(f"{i + 1} {j + 1} {p} {c} 0 17.5\n")   # six fields
                else:
                    fh.write(f"{i + 1} {j + 1} {p} {c} 12.3\n")
    with open(os.path.join(d, "parts", "part_locs.txt"), "w") as fh:
        for i in range(n_img):
            for k in range(vis.shape[1]):
                v = int(vis[i - off, k]) if i >= off else 0
                fh.write(f"{i + 1} {k + 1} {10.0 * v} {5.0 * v} {v}\n")
    with open(os.path.join(d, "bounding_boxes.txt"), "w") as fh:
        for i in range(n_img):
            b = box[i - off] if i >= off else (1.0, 1.0, 5.0, 5.0)
            fh.write(f"{i + 1} {b[0]} {b[1]} {b[2]} {b[3]}\n")
    with open(os.path.join(root, "attributes.txt"), "w") as fh:
        for j in range(n_attr):
            fh.write(f"{j + 1} has_thing::value_{j + 1}\n")
    return d


def _self_test() -> int:
    import tempfile
    print("cub_meta self-test")
    ok = True
    rng = np.random.default_rng(0)
    rels = [f"001.A/A_{k:04d}.jpg" for k in range(6)]
    n_attr, n_parts = 4, 3
    present = rng.integers(0, 2, (6, n_attr))
    cert = rng.integers(1, 5, (6, n_attr))
    vis = rng.integers(0, 2, (6, n_parts))
    box = np.array([[3.0, 3.0, 10.0, 10.0]] * 6)

    with tempfile.TemporaryDirectory() as td:
        d = write_mini_cub(td, rels, n_attr, present, cert, vis, box)
        if find_cub(td) != d:
            print("  FAIL find_cub did not locate the tree"); ok = False
        else:
            print("  ok   find_cub locates a real-shaped tree")

        n2id = parse_images(os.path.join(d, "images.txt"))
        pr, ce, na, st = parse_attribute_labels(
            os.path.join(d, REQUIRED["labels"]), max(n2id.values()))
        sel = np.array([n2id[r] - 1 for r in rels])
        if not (np.array_equal(pr[sel], present) and np.array_equal(ce[sel], cert)):
            print("  FAIL attribute values not recovered"); ok = False
        elif st["irregular_lines"] != 1 or st["pairs_missing"] or st["pairs_duplicated"]:
            print(f"  FAIL irregular-line accounting: {st}"); ok = False
        else:
            print("  ok   attributes recovered exactly; the six-field line is counted, "
                  "not dropped")

        vv, stp = parse_parts(os.path.join(d, REQUIRED["parts"]), max(n2id.values()))
        if not np.array_equal(vv[sel], vis) or stp["pairs_missing"]:
            print("  FAIL part visibility not recovered"); ok = False
        else:
            print("  ok   part visibility recovered exactly")

        names, p = parse_attr_names(d, n_attr)
        if names[2] != "has_thing::value_3" or p is None:
            print(f"  FAIL attribute names from the archive root: {names} {p}"); ok = False
        else:
            print("  ok   attribute names read from the archive root")

        # Missing pair must be caught.
        lab = os.path.join(d, REQUIRED["labels"])
        lines = open(lab).read().splitlines()
        open(lab, "w").write("\n".join(lines[:-1]) + "\n")
        _, _, _, st2 = parse_attribute_labels(lab, max(n2id.values()))
        if st2["pairs_missing"] != 1:
            print(f"  FAIL a missing (image, attribute) pair went unnoticed: {st2}")
            ok = False
        else:
            print("  ok   a missing (image, attribute) pair is caught")

    # Archive: only text members come out, images never do.
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        write_mini_cub(src, rels, n_attr, present, cert, vis, box)
        os.makedirs(os.path.join(src, "CUB_200_2011", "images", "001.A"))
        open(os.path.join(src, "CUB_200_2011", "images", "001.A", "x.jpg"), "wb").write(b"j")
        tgz = os.path.join(td, CUB_TGZ)
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(os.path.join(src, "CUB_200_2011"), arcname="CUB_200_2011")
            tf.add(os.path.join(src, "attributes.txt"), arcname="attributes.txt")
        dst = os.path.join(td, "dst")
        os.makedirs(dst)
        got = extract_text_members(tgz, dst)
        has_img = any(g.endswith(".jpg") for g in got)
        if has_img or find_cub(dst) is None or not os.path.isfile(
                os.path.join(dst, "attributes.txt")):
            print(f"  FAIL selective extraction: {got}"); ok = False
        else:
            print(f"  ok   archive extraction takes the {len(got)} text files and no image")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="where the datasets live (default: datasets.DATA_ROOT)")
    ap.add_argument("--cub-dir", default=None, help="the CUB_200_2011 directory")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--fractions", default="results/v5_bird_fraction.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="v6_cub_meta")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()

    from datasets import DATA_ROOT
    root = args.data_root or DATA_ROOT
    if not os.path.isfile(args.fractions):
        raise SystemExit(f"{args.fractions} not found. It is written by "
                         "`python cub_masks.py --tag v5_bird_fraction` (run_session6.sh "
                         "does this automatically when it is missing).")

    fetch = None
    cub_dir = args.cub_dir or find_cub(root)
    if cub_dir is None:
        if args.no_download:
            raise SystemExit(f"no CUB_200_2011 metadata under {root} and --no-download.")
        fetch = fetch_cub(root)
        cub_dir = find_cub(root)
        if cub_dir is None:
            raise SystemExit(f"extracted {CUB_TGZ} but found no complete CUB_200_2011 "
                             f"directory under {root}. Extracted: {fetch['extracted']}")
    print(f"CUB directory: {cub_dir}")

    out, rep = build(root, cub_dir, args.fractions)
    if fetch:
        rep["fetch"] = {k: v for k, v in fetch.items() if k != "extracted"}
    os.makedirs(args.out_dir, exist_ok=True)
    md = to_markdown(rep, fetch)
    with open(os.path.join(args.out_dir, f"{args.tag}.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(md)
    if not rep["ok"]:
        print("CUB METADATA CHECK FAILED -- nothing downstream is valid.", file=sys.stderr)
        return 1
    npz = os.path.join(args.out_dir, f"{args.tag}.npz")
    np.savez_compressed(npz, **out)
    print(f"wrote {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
