"""Produce one frozen FeatureBundle per (dataset, backbone). GPU step.

    python extract_features.py --dataset waterbirds --backbone erm_rn50
    python extract_features.py --dataset waterbirds --backbone clip --splits train,test
    python extract_features.py --dataset celeba     --backbone dinov2
    python extract_features.py --dataset waterbirds --backbone gdro_rn50 --splits train,val,test

Writes `features_<dataset>_<backbone>_<split>.npz`, loadable with
`common.FeatureBundle.load`. Those files are git-ignored (hundreds of MB each,
and there is one per pair); everything downstream reads them and writes small
json/md into `results/`, which IS tracked.

FOUR THINGS THIS FILE IS RESPONSIBLE FOR
========================================

1. STANDARDISATION, and recording that it happened. Every downstream comparison
   at "matched z_t" is meaningless across backbones on different scales, because
   rescaling Phi is exactly rescaling the effective step size. Features are
   standardised here and `meta["standardized"] = True` is written into the
   bundle so downstream scripts can warn if it is ever missing. Do not add a
   flag to turn this off.

   CHANGED 17 Sept 2026: every split is standardised with the TRAIN split's mean
   and standard deviation (`meta["standardized_with"] = "train statistics"`).
   Before, each split was standardised with its OWN statistics, which applies a
   different affine map to train and test -- so a last layer fitted on train was
   evaluated on differently-transformed test features, and with no intercept the
   re-centring alone moves every margin. That did not affect anything computed on
   the train split alone (the sweep, long_horizon, invariance, dfr_gain), and the
   TRAIN bundle is numerically unchanged by the fix. It does matter for any
   train-to-test evaluation (dfr_bridge.py refuses bundles without the new flag).
   The train mean and std are stored in every bundle as `std_mu` / `std_sd`.

2. THE SAME Phi FOR EVERY SPLIT. Trained backbones fit on the TRAIN split only,
   then extract all requested splits with the model frozen. The train split is
   always embedded (its statistics are needed), even if not requested.

3. idx_r / idx_s ARE LEFT EMPTY. `FeatureBundle` carries them, but identifying
   them is the job of `invariance.py` and `identify_rs.py`, which run later and
   disagree with each other on purpose. Writing a split in here would silently
   pick a winner.

4. HOW A TRAINED BACKBONE IS TRAINED (`backbones.REGISTRY[key].train_mode`).
   All three modes share architecture, optimiser, learning rate, weight decay,
   batch size and epochs; only the loss differs:
     erm   mean cross-entropy.
     rwg   cross-entropy with per-sample weight N / (4 * n_cell), where n_cell is
           the size of the sample's (y, g) cell on the train split; batch loss =
           sum(w * ce) / sum(w).
     gdro  online group DRO (Sagawa et al., ICLR 2020, their Algorithm 1): keep a
           distribution q over the four (y, g) cells; for each batch compute the
           mean loss of each cell present, update q_c <- q_c * exp(eta * loss_c),
           renormalise, and minimise sum_c q_c * loss_c. eta = --dro-eta.
           NOT included from that paper: their strong l2 penalty and group
           adjustments, so that only the loss differs from erm_rn50.
   Groups for rwg/gdro are the four (y, g) cells, as in the group-robustness
   literature, not the theorem's two groups g. A trained model is saved to
   models/<dataset>_<backbone>.pt so that re-extracting other splits never
   requires retraining (a retrained model is a different Phi).

The one-epoch `under_rn50` backbone shares the erm_rn50 code path with a
different epoch count; that is the point of it (see `backbones.py`).

Smoke test before a real run (a few batches, no saving of the model):
    python extract_features.py --dataset waterbirds --backbone gdro_rn50 \\
        --epochs 1 --max-batches 5 --splits train --out-prefix smoke --no-save-model
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from backbones import REGISTRY, get_backbone
from progress import pbar
from common import FeatureBundle
from datasets import DATA_ROOT, cell_report, load_metadata


class _ImageList:
    """Minimal path -> tensor dataset. Avoids depending on any dataset library."""

    def __init__(self, paths, transform):
        self.paths, self.tf = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


def _loader(paths, tf, bs, workers, shuffle=False):
    import torch
    return torch.utils.data.DataLoader(
        _ImageList(paths, tf), batch_size=bs, shuffle=shuffle,
        num_workers=workers, pin_memory=True)


def cell_ids(y, g):
    """(y, g) cell index in {0, 1, 2, 3} for y, g in {0, 1}."""
    return (2 * np.asarray(y, int) + np.asarray(g, int)).astype(int)


def train_backbone(model, dim, paths, y, g, tf, epochs, bs, lr, wd, workers, device,
                   mode="erm", dro_eta=0.01, max_batches=None, shuffle=None):
    """Fine-tune backbone + a linear head. Returns the backbone (head discarded).

    `mode` selects the loss; everything else is identical across modes.
    """
    import torch
    import torch.nn as nn

    head = nn.Linear(dim, 2).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss(reduction="none")
    yt = torch.as_tensor(y, dtype=torch.long)
    cid = cell_ids(y, g)
    ct = torch.as_tensor(cid, dtype=torch.long)
    n_cell = np.bincount(cid, minlength=4).astype(float)
    # rwg: N / (4 n_cell); empty cells get weight 0 (never sampled anyway)
    w_cell = np.where(n_cell > 0, len(cid) / (4.0 * np.maximum(n_cell, 1)), 0.0)
    w_cell_t = torch.as_tensor(w_cell, dtype=torch.float32, device=device)
    q = torch.full((4,), 0.25, dtype=torch.float32, device=device)   # gdro
    print(f"  train mode {mode}: cell sizes {n_cell.astype(int).tolist()}"
          + (f", weights {np.round(w_cell, 3).tolist()}" if mode == "rwg" else "")
          + (f", dro eta {dro_eta}" if mode == "gdro" else ""))

    # The original ERM loader did NOT shuffle (batches follow the metadata file's
    # order). erm keeps that by default so erm_rn50 / under_rn50 are reproduced as
    # they were built; rwg and gdro must shuffle, since per-batch cell losses are
    # meaningless on sorted batches. --shuffle yes|no overrides for any mode.
    if shuffle is None:
        shuffle = mode != "erm"
    print(f"  shuffle = {shuffle}")
    dl = _loader(paths, tf, bs, workers, shuffle=shuffle)

    model.train()
    for ep in range(epochs):
        tot = cor = 0
        run = 0.0
        nb = len(dl) if max_batches is None else min(len(dl), max_batches)
        bar = pbar(total=nb, unit="batch", desc=f"  train ep {ep + 1}/{epochs}",
                   leave=False)
        for b, (xb, idx) in enumerate(dl):
            if max_batches is not None and b >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yt[idx].to(device)
            cb = ct[idx].to(device)
            opt.zero_grad(set_to_none=True)
            out = head(model(xb))
            per = ce(out, yb)
            if mode == "erm":
                loss = per.mean()
            elif mode == "rwg":
                wb = w_cell_t[cb]
                loss = (wb * per).sum() / wb.sum().clamp_min(1e-12)
            elif mode == "gdro":
                counts = torch.bincount(cb, minlength=4).float()
                sums = torch.zeros(4, device=device).index_add_(0, cb, per)
                present = counts > 0
                gl = torch.where(present, sums / counts.clamp_min(1.0),
                                 torch.zeros_like(sums))
                with torch.no_grad():
                    q = q * torch.exp(dro_eta * gl.detach() * present.float())
                    q = q / q.sum()
                loss = (q * gl).sum()
            else:
                raise ValueError(f"unknown train mode {mode!r}")
            loss.backward()
            opt.step()
            run += float(per.mean()) * yb.size(0)
            cor += int((out.argmax(1) == yb).sum())
            tot += yb.size(0)
            post = f"ce {run / tot:.4f} acc {cor / tot:.4f}"
            if mode == "gdro":
                post += " q " + ",".join(f"{v:.2f}" for v in q.tolist())
            bar.set_postfix_str(post)
            bar.update(1)
        bar.close()
        print(f"    epoch {ep + 1}/{epochs}  ce {run / tot:.4f}  acc {cor / tot:.4f}"
              + (f"  q {np.round(q.cpu().numpy(), 3).tolist()}" if mode == "gdro" else ""))
    model.eval()
    return model


def embed(model, paths, tf, bs, workers, device) -> np.ndarray:
    """Forward pass only. Returns (N, d) float64."""
    import torch

    dl = _loader(paths, tf, bs, workers)
    out = []
    with torch.no_grad():
        for xb, _ in pbar(dl, unit="batch", desc="  embedding"):
            out.append(model(xb.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def train_stats(phi: np.ndarray):
    """Same arithmetic as common.standardize, but returning the statistics."""
    mu = phi.mean(axis=0, keepdims=True)
    sd = phi.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=["waterbirds", "celeba"])
    ap.add_argument("--backbone", required=True, choices=list(REGISTRY))
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the backbone's declared epoch count")
    ap.add_argument("--dro-eta", type=float, default=0.01,
                    help="group DRO step size for q (Sagawa et al. default 0.01)")
    ap.add_argument("--shuffle", default="auto", choices=["auto", "yes", "no"],
                    help="auto: no for erm (original behaviour), yes for rwg/gdro")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="smoke test only: stop each epoch after this many batches")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--no-save-model", action="store_true")
    ap.add_argument("--load-model", action="store_true",
                    help="load models/<dataset>_<backbone>.pt instead of training")
    ap.add_argument("--out-prefix", default="features")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA. This will take hours.")

    spec = REGISTRY[args.backbone]
    epochs = spec.train_epochs if args.epochs is None else args.epochs
    model, tf, dim = get_backbone(args.backbone, device=device)
    print(f"backbone {args.backbone}: d = {dim}, task epochs = {epochs}, "
          f"train mode = {spec.train_mode}")

    mpath = os.path.join(args.model_dir, f"{args.dataset}_{args.backbone}.pt")
    if epochs > 0:
        if args.load_model:
            model.load_state_dict(torch.load(mpath, map_location=device))
            model.eval()
            print(f"  loaded trained backbone from {mpath}")
        else:
            p_tr, y_tr, g_tr, a_tr = load_metadata(args.dataset, "train", args.root)
            print(cell_report(y_tr, g_tr, a_tr, f"{args.dataset}/train"))
            print(f"  training ({spec.train_mode}) ...")
            model = train_backbone(model, dim, p_tr, y_tr, g_tr, tf, epochs,
                                   args.bs, args.lr, args.wd, args.workers, device,
                                   mode=spec.train_mode, dro_eta=args.dro_eta,
                                   max_batches=args.max_batches,
                                   shuffle={"auto": None, "yes": True, "no": False}[args.shuffle])
            if not args.no_save_model:
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save(model.state_dict(), mpath)
                print(f"  saved trained backbone to {mpath}")

    splits = args.splits.split(",")
    order = ["train"] + [s for s in splits if s != "train"]
    mu = sd = None
    for split in order:
        paths, y01, g, attr = load_metadata(args.dataset, split, args.root)
        print(cell_report(y01, g, attr, f"{args.dataset}/{split}"))
        print(f"  embedding {len(paths)} images ...")
        phi = embed(model, paths, tf, args.bs, args.workers, device)
        if split == "train":
            mu, sd = train_stats(phi)
        # Standardise with TRAIN statistics -- see point 1 in the module docstring.
        phi = (phi - mu) / sd
        if split not in splits:
            continue                      # train embedded only for its statistics

        fb = FeatureBundle(
            phi=phi,
            y=np.where(y01 > 0, 1, -1).astype(int),   # +/-1, the manuscript's convention
            g=g.astype(int),
            idx_r=np.array([], dtype=int),             # deliberately empty
            idx_s=np.array([], dtype=int),
            place=attr.astype(int),
            meta={
                "dataset": args.dataset, "split": split,
                "backbone": args.backbone, "backbone_kind": spec.kind,
                "model_id": spec.model_id, "task_epochs": epochs,
                "train_mode": spec.train_mode,
                "d": int(phi.shape[1]), "n": int(phi.shape[0]),
                "standardized": True,
                "standardized_with": "train statistics",
                "eps": float(np.mean(g == 1)),
            },
        )
        out = f"{args.out_prefix}_{args.dataset}_{args.backbone}_{split}.npz"
        fb.save(out)
        with np.load(out, allow_pickle=True) as z:
            arrays = dict(z)
        np.savez_compressed(out, **arrays, std_mu=mu.ravel(), std_sd=sd.ravel())
        print(f"  wrote {out}  phi {phi.shape}  eps = {fb.meta['eps']:.4f}")


if __name__ == "__main__":
    main()
