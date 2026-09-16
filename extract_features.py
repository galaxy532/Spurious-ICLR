"""Produce one frozen FeatureBundle per (dataset, backbone). GPU step.

    python extract_features.py --dataset waterbirds --backbone erm_rn50
    python extract_features.py --dataset waterbirds --backbone clip --splits train,test
    python extract_features.py --dataset celeba     --backbone dinov2

Writes `features_<dataset>_<backbone>_<split>.npz`, loadable with
`common.FeatureBundle.load`. Those files are git-ignored (hundreds of MB each,
and there is one per pair); everything downstream reads them and writes small
json/md into `results/`, which IS tracked.

THREE THINGS THIS FILE IS RESPONSIBLE FOR
=========================================

1. STANDARDISATION, and recording that it happened. Every downstream comparison
   at "matched z_t" is meaningless across backbones on different scales, because
   rescaling Phi is exactly rescaling the effective step size. Features are
   standardised here and `meta["standardized"] = True` is written into the
   bundle so `eps_backbone_sweep.py` can warn if it is ever missing. Do not add
   a flag to turn this off.

2. THE SAME Phi FOR EVERY SPLIT. Both trained backbones fit on the TRAIN split
   only, then extract all requested splits in one pass with the model frozen. A
   backbone refit per split is a different Phi per split and makes the
   cross-split comparisons meaningless.

3. idx_r / idx_s ARE LEFT EMPTY. `FeatureBundle` carries them, but identifying
   them is the job of `invariance.py` and `identify_rs.py`, which run later and
   disagree with each other on purpose. Writing a split in here would silently
   pick a winner.

The one-epoch `under_rn50` backbone shares the erm_rn50 code path with a
different epoch count; that is the point of it (see `backbones.py`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from backbones import REGISTRY, get_backbone
from progress import pbar
from common import FeatureBundle, standardize
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


def _loader(paths, tf, bs, workers):
    import torch
    return torch.utils.data.DataLoader(
        _ImageList(paths, tf), batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)


def train_erm(model, dim, paths, y, tf, epochs, bs, lr, wd, workers, device):
    """Fine-tune backbone + a linear head with plain ERM. Returns the backbone."""
    import torch
    import torch.nn as nn

    head = nn.Linear(dim, 2).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    yt = torch.as_tensor(y, dtype=torch.long)
    dl = _loader(paths, tf, bs, workers)

    model.train()
    for ep in range(epochs):
        tot = cor = 0
        run = 0.0
        bar = pbar(total=len(dl), unit="batch", desc=f"  train ep {ep + 1}/{epochs}",
                   leave=False)
        for xb, idx in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            out = head(model(xb))
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            run += float(loss) * yb.size(0)
            cor += int((out.argmax(1) == yb).sum())
            tot += yb.size(0)
            bar.set_postfix_str(f"loss {run / tot:.4f} acc {cor / tot:.4f}")
            bar.update(1)
        bar.close()
        print(f"    epoch {ep + 1}/{epochs}  loss {run / tot:.4f}  acc {cor / tot:.4f}")
    model.eval()
    # The head is discarded on purpose: Phi is the backbone output, and every
    # downstream script fits its own last layer.
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
    ap.add_argument("--out-prefix", default="features")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA. This will take hours.")

    spec = REGISTRY[args.backbone]
    epochs = spec.train_epochs if args.epochs is None else args.epochs
    model, tf, dim = get_backbone(args.backbone, device=device)
    print(f"backbone {args.backbone}: d = {dim}, task epochs = {epochs}")

    # Train ONCE on the train split, then freeze. Refitting per split would make
    # every cross-split statement compare two different functions.
    if epochs > 0:
        p_tr, y_tr, g_tr, a_tr = load_metadata(args.dataset, "train", args.root)
        print(cell_report(y_tr, g_tr, a_tr, f"{args.dataset}/train"))
        print("  training ERM ...")
        model = train_erm(model, dim, p_tr, y_tr, tf, epochs,
                          args.bs, args.lr, args.wd, args.workers, device)

    for split in args.splits.split(","):
        paths, y01, g, attr = load_metadata(args.dataset, split, args.root)
        print(cell_report(y01, g, attr, f"{args.dataset}/{split}"))
        print(f"  embedding {len(paths)} images ...")
        phi = embed(model, paths, tf, args.bs, args.workers, device)

        # Standardise, and RECORD that it happened -- see point 1 in the module
        # docstring. Downstream warns if this flag is missing.
        phi = standardize(phi)

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
                "d": int(phi.shape[1]), "n": int(phi.shape[0]),
                "standardized": True,
                "eps": float(np.mean(g == 1)),
            },
        )
        out = f"{args.out_prefix}_{args.dataset}_{args.backbone}_{split}.npz"
        fb.save(out)
        print(f"  wrote {out}  phi {phi.shape}  eps = {fb.meta['eps']:.4f}")


if __name__ == "__main__":
    main()
