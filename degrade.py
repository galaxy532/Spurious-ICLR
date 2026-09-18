"""Group-conditional degradation of the CORE feature, applied to raw images.

WHY THIS EXISTS
===============
Sessions 1-3 found no case of the `alpha > 1` branch of Theorem 5.3 in seven
runs across two datasets and five representations. The branch condition is

    alpha > 1   <=>   gamma~_min / gamma~_maj  >  gamma_crit = (1 + mu_A)/(1 + mu)

i.e. the two groups must have sufficiently DIFFERENT r-margins -- one group
genuinely easier to classify from the core feature than the other -- with the
gap exceeding the coupling ratio. Nothing in that condition is about group SIZE:
the labels "min" and "maj" are assigned by margin, and eps = P(G_min) is free in
[0, 1].

Waterbirds, CelebA and colored MNIST are all built with a group variable
(background, sex, colour) that is a NUISANCE, deliberately uncorrelated with how
hard the core task is. So gamma~_min/gamma~_maj ~ 1 before any training happens,
and the setting sits ON the phase transition. That is a property of the DATASET,
not of the backbone, which is why no choice of backbone moved it.

This module supplies the missing knob: degrade the core feature in ONE GROUP
ONLY, at the image level, leaving the backbone and the spurious structure of
Waterbirds untouched. Sweeping the degradation moves gamma~_min/gamma~_maj by a
controllable amount, and Theorem 5.3 then predicts when beta_min lifts off 1.

WHAT THIS IS NOT
================
This is NOT a surgical operation on `r` alone. Reducing resolution destroys fine
detail (bird shape and texture -- the core feature) much faster than it destroys
coarse colour and layout statistics (land vs water -- the spurious feature), but
it does touch both. The experiment does not need the operation to be pure,
because nothing downstream assumes it is: we MEASURE the resulting per-group
hard margins with `group_margins.py`, and the manuscript's alignment identity

    gamma_min = max(1, alpha_xi(1 + delta_maj) - delta_min) ~= max(1, alpha)

turns that measurement into a parameter-free prediction for beta_min, which
`long_horizon.py` measures independently. The knob only has to MOVE the margin
ratio; it does not have to move it for a reason we can name.

This is also NOT the semi-synthetic `Phi` with a tunable alpha that was proposed
and rejected in August 2026. That one manufactured the exponent inside the
representation. This manufactures a property of the IMAGES, and then lets a real
frozen backbone and a real max-margin problem decide what the exponent is.

THE TWO OPERATORS
=================
`resolution` (default) -- downsample the short side to `level` times its length,
then resample back to the original size, both bicubic. `level = 1.0` is the
identity; `level = 0.25` keeps a quarter of the linear resolution. Scale-free, so
it means the same thing for a 224 px and a 518 px backbone.

`blur` -- Gaussian blur with radius `level * short_side`. `level = 0.0` is the
identity. Offered as a second operator with a different frequency profile, so a
result that depends on which one was used is visible as such.

Both are deterministic, size-preserving, and applied to the PIL image BEFORE the
backbone's own transform.

Self-test (no data, no GPU, a few seconds):

    python degrade.py
"""

from __future__ import annotations

KINDS = ("resolution", "blur")


def is_identity(kind: str, level: float) -> bool:
    """True when this (kind, level) leaves every image untouched."""
    if kind == "resolution":
        return level >= 1.0
    if kind == "blur":
        return level <= 0.0
    raise ValueError(f"unknown degradation kind {kind!r}; expected one of {KINDS}")


def tag(kind: str, level: float) -> str:
    """Short filename-safe tag, e.g. 'res050' or 'blur008'. '' when identity."""
    if is_identity(kind, level):
        return ""
    stem = {"resolution": "res", "blur": "blur"}[kind]
    return f"{stem}{int(round(level * 1000)):03d}"


def make_degrader(kind: str = "resolution", level: float = 1.0):
    """Return a deterministic PIL.Image -> PIL.Image callable, or None.

    None means "identity" and lets callers skip the work entirely rather than
    paying for a no-op decode/encode on every image of a 100-epoch run.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown degradation kind {kind!r}; expected one of {KINDS}")
    if kind == "resolution" and not (0.0 < level <= 1.0):
        raise ValueError("resolution level must be in (0, 1]; "
                         f"got {level}. 1.0 is the identity.")
    if kind == "blur" and level < 0.0:
        raise ValueError(f"blur level must be >= 0; got {level}. 0.0 is the identity.")
    if is_identity(kind, level):
        return None

    from PIL import Image, ImageFilter

    if kind == "resolution":
        def _degrade(im):
            w, h = im.size
            short = min(w, h)
            # At least 8 px on the short side: below that the resampler itself,
            # not the level, decides what survives.
            keep = max(8, int(round(level * short)))
            scale = keep / float(short)
            small = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            return im.resize(small, Image.BICUBIC).resize((w, h), Image.BICUBIC)
    else:
        def _degrade(im):
            radius = level * min(im.size)
            return im.filter(ImageFilter.GaussianBlur(radius=radius))

    return _degrade


# --------------------------------------------------------------------------
# Self-test. Checks the four things that would silently invalidate a sweep:
# identity really is identity, the output size never changes, the operator is
# deterministic, and high-frequency energy falls MONOTONICALLY with the level
# (a knob that is not monotone cannot be swept).
# --------------------------------------------------------------------------
def _self_test() -> int:
    import numpy as np
    from PIL import Image
    from progress import pbar

    rng = np.random.default_rng(0)
    # A texture with energy at every spatial frequency, so the high-frequency
    # measurement below is not dominated by one scale.
    base = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    im = Image.fromarray(base, mode="RGB")

    def hf_energy(pil) -> float:
        """Mean squared Laplacian -- a simple high-spatial-frequency proxy."""
        a = np.asarray(pil.convert("L"), dtype=np.float64)
        lap = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1]
               - a[1:-1, :-2] - a[1:-1, 2:])
        return float(np.mean(lap ** 2))

    fails = []

    if make_degrader("resolution", 1.0) is not None:
        fails.append("resolution level=1.0 should be the identity (None)")
    if make_degrader("blur", 0.0) is not None:
        fails.append("blur level=0.0 should be the identity (None)")
    if tag("resolution", 1.0) != "" or tag("resolution", 0.5) != "res500":
        fails.append(f"tag() wrong: {tag('resolution', 0.5)!r}")

    levels = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.08]
    energies = []
    for lv in pbar(levels, desc="  resolution ladder", unit="level"):
        f = make_degrader("resolution", lv)
        out = im if f is None else f(im)
        if out.size != im.size:
            fails.append(f"level {lv}: size changed {im.size} -> {out.size}")
        if f is not None and np.any(np.asarray(f(im)) != np.asarray(out)):
            fails.append(f"level {lv}: not deterministic")
        energies.append(hf_energy(out))

    for a, b, la, lb in zip(energies, energies[1:], levels, levels[1:]):
        if not b < a:
            fails.append(f"high-frequency energy not decreasing: "
                         f"level {la} -> {lb} gave {a:.1f} -> {b:.1f}")

    blur_e = []
    for lv in pbar([0.0, 0.01, 0.03, 0.08], desc="  blur ladder", unit="level"):
        f = make_degrader("blur", lv)
        out = im if f is None else f(im)
        if out.size != im.size:
            fails.append(f"blur {lv}: size changed")
        blur_e.append(hf_energy(out))
    for a, b in zip(blur_e, blur_e[1:]):
        if not b < a:
            fails.append(f"blur energy not decreasing: {a:.1f} -> {b:.1f}")

    for kind, lv in [("resolution", 0.0), ("resolution", 1.5), ("blur", -1.0),
                     ("nonsense", 0.5)]:
        try:
            make_degrader(kind, lv)
        except ValueError:
            pass
        else:
            fails.append(f"make_degrader({kind!r}, {lv}) should have raised")

    print("\nresolution ladder (high-frequency energy, must decrease):")
    for lv, e in zip(levels, energies):
        print(f"  level {lv:<5} {e:12.1f}")
    print("blur ladder:")
    for lv, e in zip([0.0, 0.01, 0.03, 0.08], blur_e):
        print(f"  level {lv:<5} {e:12.1f}")

    if fails:
        print("\nSELF-TEST FAILED")
        for f in fails:
            print("  -", f)
        return 1
    print("\nSELF-TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
