"""Dataset declarations: what is y, what is the group, and what eps that gives.

Definition 3.3 of the manuscript defines EXACTLY TWO groups, G1 and G2, by which
operator carries r to s, with a single eps = P(G2). Every group variable here is
therefore binary. A k-way partition is a different setting and `alpha` is not
defined for it -- Rebuttals_2 got this wrong once and its loader now refuses to
construct with anything but two groups. Same rule holds here.

    python datasets.py            # print both declarations and their cell sizes


WATERBIRDS
==========

    y      = 1[waterbird]                       (from metadata.csv `y`)
    place  = 1[water background]                (from metadata.csv `place`)
    g      = 1[place != y]                      (the standard group definition)
    eps    = P(g = 1) ~ 0.05 on train

Unchanged from `Rebuttals/`. Split cell sizes, (y, place):

    train   3498 / 184 / 56 / 1057      <- the 56 cell is why train is unusable
                                           for anything that conditions on cells
    test    2255 / 2255 / 642 / 642     <- built roughly balanced


CELEBA -- AND WHY THIS REVERSES AN EARLIER DECISION
===================================================

`Rebuttals/README.md` recorded CelebA as unusable, on the grounds that grouping
by whether `Male` agrees with the majority blond-female pairing gives eps ~ 0.45
-- "no imbalance, because CelebA's imbalance lives inside the blond class."

That objection was correct FOR THAT USE and does not apply here. It was about
reading the NATURAL eps off the dataset and comparing it to theory. The
epsilon-sweep does the opposite: it MANUFACTURES eps by subsampling. For that
purpose a naturally balanced group variable is not a defect, it is the ideal
substrate, because it is what gives the sweep room to move. A dataset that
starts at eps = 0.05 can only be swept downward; one that starts at eps = 0.42
can be swept across an order of magnitude with thousands of samples at every
point.

So CelebA is declared here as:

    y      = 1[Blond_Hair]
    g      = 1[Male]                     <- the group IS the spurious attribute
    eps    = P(Male) ~ 0.42 on train

with the four (y, Male) cells on train:

    non-blond female  71629      non-blond male  66874
    blond     female  22880      blond     male   1387

Three consequences, all of which have to be stated rather than discovered later:

1. THE BASE RATES DIFFER ENORMOUSLY ACROSS GROUPS. P(blond | female) ~ 0.24
   against P(blond | male) ~ 0.02, a factor of twelve. Any identification rule
   that compares P(y | Phi_k, g) between groups WITHOUT removing a constant
   log-odds offset will reject every coordinate on this dataset, including
   perfectly causal ones. This is exactly the confound `invariance.py` is built
   to survive and that `validate_invariance.py` check C4 verifies. It is also
   why CelebA, not Waterbirds, is the dataset that would have caught a naive
   implementation.

2. THE GROUP AND THE SPURIOUS CONCEPT COINCIDE. On Waterbirds, g = 1[place != y]
   is DERIVED from the spurious attribute; here g = Male IS it. So the
   two-concept rule's 2x2 factorial over (y, attribute) is a factorial over
   (y, g), i.e. the same cells the coupling test conditions on. The rule still
   runs, but it is no longer using information independent of the group index,
   so the "two rules using different information agree" argument is weaker on
   CelebA than on Waterbirds. Report the three-rule agreement for both datasets
   and do not claim more for CelebA than the construction supports.

3. n IS WHY CELEBA IS HERE. The power curve in `validate_invariance.py` (check
   C2b) puts the invariance rule's F1 on a magnitude-only difference at ~0.37 at
   n = 6,000 and 1.00 at n = 160,000. Waterbirds test is n = 5,794. CelebA train
   is n = 162,770. Waterbirds is UNDERPOWERED for this rule and CelebA is
   comfortable; a null result on Waterbirds alone would therefore be
   uninterpretable. Pooling all three Waterbirds splits reaches n = 11,788,
   which the curve puts at F1 ~ 0.66 -- better, still marginal. See
   `--pool-splits`.
"""

from __future__ import annotations

import os

import numpy as np

DATA_ROOT = os.environ.get(
    "SPURIOUS_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)


DECLARATIONS = {
    "waterbirds": {
        "dir": "waterbird_complete95_forest2water2",
        "metadata": "metadata.csv",
        "y_col": "y",
        "y_desc": "1 = waterbird",
        "attr_col": "place",
        "attr_desc": "1 = water background",
        "group_rule": "place != y",
        "eps_train": 0.05,
        "image_col": "img_filename",
        "split_col": "split",
        "splits": {"train": 0, "val": 1, "test": 2},
    },
    "celeba": {
        "dir": "celeba",
        "metadata": "list_attr_celeba.csv",
        "y_col": "Blond_Hair",
        "y_desc": "1 = blond",
        "attr_col": "Male",
        "attr_desc": "1 = male",
        "group_rule": "Male == 1",
        "eps_train": 0.42,
        "image_col": "image_id",
        "split_col": "partition",       # from list_eval_partition.csv
        "splits": {"train": 0, "val": 1, "test": 2},
    },
}


def _find(base: str, stem: str) -> str:
    """Locate `stem.txt` or `stem.csv`, whichever the mirror shipped.

    CelebA comes in two incompatible layouts and which one you get depends
    entirely on where you downloaded it:

      OFFICIAL (Google Drive)  `list_attr_celeba.txt` -- whitespace-delimited,
          with a leading line giving the row count (202599) and a second line of
          40 attribute names. Rows are `000001.jpg -1  1  1 ...`, so there are
          41 fields against 40 names and the filename lands in the index.
      KAGGLE (jessicali9530)   `list_attr_celeba.csv` -- ordinary CSV with an
          `image_id` column and a proper header.

    Both are supported by sniffing the extension, so a run does not fail three
    steps later with a confusing column error because a mirror was swapped.
    """
    for ext in (".txt", ".csv"):
        p = os.path.join(base, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"neither {stem}.txt nor {stem}.csv found in {base}. "
        f"Expected the CelebA annotation files alongside img_align_celeba/."
    )


def _read_celeba_attr(base: str):
    """Attribute table -> DataFrame with an `image_id` column and {-1,+1} attrs."""
    import pandas as pd

    p = _find(base, "list_attr_celeba")
    if p.endswith(".txt"):
        # skiprows=1 drops the count line; the names line then becomes the
        # header and the filename column becomes the index, because there is one
        # more field per row than there are names.
        df = pd.read_csv(p, sep=r"\s+", skiprows=1)
        df.index.name = "image_id"
        df = df.reset_index()
    else:
        df = pd.read_csv(p)
    if "image_id" not in df.columns:
        raise ValueError(f"{p}: no image_id column after parsing; got "
                         f"{list(df.columns)[:5]}")
    return df


def _read_celeba_partition(base: str):
    """Split table -> DataFrame with `image_id` and `partition` (0/1/2)."""
    import pandas as pd

    p = _find(base, "list_eval_partition")
    if p.endswith(".txt"):
        # No header at all in the official file.
        return pd.read_csv(p, sep=r"\s+", header=None,
                           names=["image_id", "partition"])
    df = pd.read_csv(p)
    return df.rename(columns={df.columns[1]: "partition"})


def load_metadata(dataset: str, split: str | None = "train",
                  root: str | None = None, pool_splits: bool = False):
    """Return (paths, y, g, attr) with y, g, attr as int arrays.

    y and attr are returned in {0, 1}; `g` is in {0, 1} with **g = 1 meaning the
    MINORITY group**, matching `common.FeatureBundle`'s convention and the
    manuscript's G_min. Callers converting y to the +/-1 convention should do it
    explicitly rather than relying on this function.

    `pool_splits=True` concatenates train/val/test. Legitimate for questions
    about the frozen representation Phi -- it is the same function whichever
    images pass through it, the argument already used in `Rebuttals/README.md` to
    justify measuring coupling on the test split. NOT legitimate for anything
    about training dynamics under imbalance, which needs the split where the
    imbalance actually lives. `eps_backbone_sweep.py` therefore never pools.
    """
    import pandas as pd

    root = root or DATA_ROOT
    dec = DECLARATIONS[dataset]
    base = os.path.join(root, dec["dir"])

    if dataset == "waterbirds":
        md = pd.read_csv(os.path.join(base, dec["metadata"]))
        y = md[dec["y_col"]].to_numpy().astype(int)
        attr = md[dec["attr_col"]].to_numpy().astype(int)
        g = (attr != y).astype(int)
        paths = [os.path.join(base, p) for p in md[dec["image_col"]]]
        sp = md[dec["split_col"]].to_numpy().astype(int)

    elif dataset == "celeba":
        md = _read_celeba_attr(base)
        part = _read_celeba_partition(base)
        md = md.merge(part, on="image_id")
        # CelebA attribute files encode absent as -1; recode to {0, 1}.
        y = (md[dec["y_col"]].to_numpy().astype(int) > 0).astype(int)
        attr = (md[dec["attr_col"]].to_numpy().astype(int) > 0).astype(int)
        g = attr.copy()                       # g = 1[Male]; see module docstring
        paths = [os.path.join(base, "img_align_celeba", p)
                 for p in md[dec["image_col"]]]
        sp = md["partition"].to_numpy().astype(int)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    paths = np.asarray(paths)
    if pool_splits or split is None:
        return paths, y, g, attr
    m = sp == dec["splits"][split]
    return paths[m], y[m], g[m], attr[m]


def cell_report(y, g, attr, name="") -> str:
    """The 2x2 cell sizes, printed before anything else runs.

    Every failure in the earlier repos that took real time to diagnose was
    visible in this table: the 56-sample Waterbirds train cell, the 51-way
    partition, the eps ~ 0.45 that made CelebA look useless. Print it, read it.
    """
    L = [f"\n{name}  n = {y.size}",
         f"  eps = P(g=1) = {np.mean(g == 1):.4f}",
         f"  P(y=1 | g=0) = {np.mean(y[g == 0]):.4f}   "
         f"P(y=1 | g=1) = {np.mean(y[g == 1]):.4f}",
         "  cells (y, attr):"]
    for yy in (0, 1):
        row = [int(np.sum((y == yy) & (attr == aa))) for aa in (0, 1)]
        L.append(f"    y={yy}:  attr=0 {row[0]:>7}   attr=1 {row[1]:>7}")
    smallest = min(int(np.sum((y == yy) & (attr == aa)))
                   for yy in (0, 1) for aa in (0, 1))
    L.append(f"  smallest cell = {smallest}"
             + ("   <-- BELOW 200; cell-conditional estimates will be refused"
                if smallest < 200 else ""))
    return "\n".join(L)


def main() -> None:
    print(__doc__.split("\n")[0])
    for ds in DECLARATIONS:
        print(f"\n=== {ds} " + "=" * 50)
        for k, v in DECLARATIONS[ds].items():
            print(f"  {k}: {v}")
        try:
            for split in ("train", "test"):
                _, y, g, attr = load_metadata(ds, split)
                print(cell_report(y, g, attr, f"{ds}/{split}"))
        except Exception as e:
            print(f"  [data not present: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
