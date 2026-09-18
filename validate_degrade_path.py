"""Integration check of the group-conditional degradation path.

Unit-testing `degrade.py` (done in that file) does not prove the operator reaches
the right IMAGES. This does: it builds a fake split with a known group vector, runs
it through `extract_features._ImageList`, and checks that exactly the masked indices
changed and the others are byte-identical. It also pins the filename tags, because a
tag collision silently overwrites one degradation level with another.

Needs no dataset, no GPU and no torch. Run:

    python validate_degrade_path.py
"""

import os
import tempfile

import numpy as np
from PIL import Image

from degrade import make_degrader, tag
from extract_features import _ImageList

tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
paths = []
for i in range(6):
    p = os.path.join(tmp, f"{i}.png")
    Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(p)
    paths.append(p)

g = np.array([0, 0, 1, 1, 0, 1])
mask = (g == 0)
deg = make_degrader("resolution", 0.2)
ident = lambda im: np.asarray(im, dtype=np.float64)     # "transform" = raw pixels

plain = _ImageList(paths, ident)
degd = _ImageList(paths, ident, pre=deg, pre_mask=mask)

fails = []
for i in range(6):
    a, _ = plain[i]
    b, _ = degd[i]
    same = np.array_equal(a, b)
    if mask[i] and same:
        fails.append(f"index {i} (g=0) should have been degraded but is unchanged")
    if not mask[i] and not same:
        fails.append(f"index {i} (g=1) must NOT be touched but changed")
    if a.shape != b.shape:
        fails.append(f"index {i}: shape changed {a.shape} -> {b.shape}")

if not np.array_equal(degd[0][0], degd[0][0]):
    fails.append("not deterministic across reads")

try:
    _ImageList(paths, ident, pre=deg)
except ValueError:
    pass
else:
    fails.append("a degrader without a mask should raise")
try:
    _ImageList(paths, ident, pre=deg, pre_mask=mask[:3])
except ValueError:
    pass
else:
    fails.append("a mask of the wrong length should raise")

cases = {("resolution", 1.0): "", ("resolution", 0.2): "res200",
         ("resolution", 0.35): "res350", ("blur", 0.0): "", ("blur", 0.03): "blur030"}
for (k, l), want in cases.items():
    got = tag(k, l)
    if got != want:
        fails.append(f"tag({k!r},{l}) = {got!r}, expected {want!r}")

print(f"degraded {int(mask.sum())} of {len(paths)}; untouched {int((~mask).sum())}")
print("tags:", {f"{k[0]}@{k[1]}": tag(*k) for k in cases})
if fails:
    print("FAILED")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("INTEGRATION OK")
