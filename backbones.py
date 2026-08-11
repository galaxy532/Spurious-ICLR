"""The four frozen representations the alpha-sweep varies over.

WHY THIS FILE EXISTS
====================

The manuscript's `alpha` is a property of the REPRESENTATION, not of the
dataset. It is built from the group r-margins and the eigenvalue structure of
the coupling operators, all of which are properties of Phi. The dataset is
fixed; Phi is a choice.

That is the whole design of the epsilon-sweep arm. Holding a dataset fixed and
varying Phi should move `alpha`, and therefore should move the regime -- and
therefore should change WHETHER GROUP BALANCING HELPS. That converts the
theory from a description ("this system is epsilon-sensitive") into a
prediction ("balancing will help for this backbone and not that one"), which is
the difference between a section reviewers skim and a result they cite.

It also speaks to something the literature reports without explaining: the
benefit of last-layer retraining (DFR, Kirichenko et al. 2023) varies a lot with
the backbone, and nobody has a theory of why. If `alpha` predicts that variation,
the theory has earned its place.

THE FOUR POINTS
===============

Chosen to span the space of inductive biases as widely as possible for the
compute, not to be a fair architecture comparison.

| key            | Phi                          | trained on the task? | d    |
|----------------|------------------------------|----------------------|------|
| `erm_rn50`     | ResNet-50, ERM to convergence| yes, ~10 epochs      | 2048 |
| `under_rn50`   | ResNet-50, ERM stopped early | yes, 1 epoch         | 2048 |
| `clip`         | CLIP ViT-B/32 image encoder  | no, frozen           |  512 |
| `dinov2`       | DINOv2 ViT-B/14              | no, frozen           |  768 |

`under_rn50` is the cheap way to get a deliberately BAD Phi: same architecture,
same data, same optimiser, one epoch. If `alpha` is a property of the
representation rather than of the architecture, the two ResNet-50 points should
differ, and that is a much sharper test than comparing a ResNet to a ViT (where
architecture, pretraining corpus and objective all change at once).

The two frozen backbones cost a forward pass and no training, which is why four
points fit in the budget where four trained models would not.

A NOTE ON SCALE
===============

Different backbones produce features on wildly different scales -- CLIP
embeddings are near-unit-norm by construction, ResNet penultimate activations
are not. Every downstream quantity that involves a training time (`z_t`, the
decay exponents, anything compared "at matched z_t") is scale-dependent, so
features are ALWAYS standardised before they are used. `common.standardize`
does it, `extract_features.py` calls it, and the bundle records that it was
called. Comparing an un-standardised CLIP run against an un-standardised ResNet
run at "the same z_t" compares nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackboneSpec:
    key: str
    kind: str            # "torchvision" | "openclip" | "timm"
    model_id: str
    dim: int
    train_epochs: int    # 0 => frozen pretrained, no task training
    image_size: int
    note: str


REGISTRY: dict[str, BackboneSpec] = {
    "erm_rn50": BackboneSpec(
        key="erm_rn50", kind="torchvision", model_id="resnet50",
        dim=2048, train_epochs=10, image_size=224,
        note="ImageNet-init ResNet-50 fine-tuned with ERM. The baseline, and "
             "the same Phi the earlier repos used.",
    ),
    "under_rn50": BackboneSpec(
        key="under_rn50", kind="torchvision", model_id="resnet50",
        dim=2048, train_epochs=1, image_size=224,
        note="Identical to erm_rn50 but stopped after one epoch. Isolates the "
             "effect of representation QUALITY with architecture, data and "
             "optimiser held fixed.",
    ),
    "clip": BackboneSpec(
        key="clip", kind="openclip", model_id="ViT-B-32/openai",
        dim=512, train_epochs=0, image_size=224,
        note="Frozen CLIP image encoder. Trained on web image-text pairs, so "
             "its features encode background and context heavily -- a priori "
             "the most spuriously entangled of the four.",
    ),
    "dinov2": BackboneSpec(
        key="dinov2", kind="timm", model_id="vit_base_patch14_dinov2.lvd142m",
        dim=768, train_epochs=0, image_size=518,
        note="Frozen DINOv2. Self-supervised, object-centric, and the natural "
             "contrast to CLIP: same 'no task training' status, very different "
             "pretraining objective.",
    ),
}


def get_backbone(key: str, device: str = "cuda"):
    """Return (nn.Module producing Phi, torchvision transform, dim).

    Imports live inside the function so that `python backbones.py` and any
    analysis script that only wants the registry metadata work on a CPU box with
    no torch installed. Only `extract_features.py` needs the heavy dependencies.
    """
    import torch
    import torchvision.transforms as T

    spec = REGISTRY[key]
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    if spec.kind == "torchvision":
        import torchvision.models as M
        model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()          # penultimate layer IS Phi
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])

    elif spec.kind == "openclip":
        import open_clip
        name, pretrained = spec.model_id.split("/")
        model_full, _, tf = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)

        class _Img(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m.encode_image(x)

        model = _Img(model_full)

    elif spec.kind == "timm":
        import timm
        model = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_model_data_config(model)
        tf = timm.data.create_transform(**cfg, is_training=False)

    else:
        raise ValueError(f"unknown backbone kind {spec.kind!r}")

    return model.to(device).eval(), tf, spec.dim


def main() -> None:
    print(f"{'key':<12} {'kind':<12} {'dim':>5} {'epochs':>7}  note")
    for k, s in REGISTRY.items():
        print(f"{k:<12} {s.kind:<12} {s.dim:>5} {s.train_epochs:>7}  "
              f"{s.note.splitlines()[0][:60]}")


if __name__ == "__main__":
    main()
