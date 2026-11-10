"""Occlusal views of the labelled inhouse scans: real colour | human label | PurePTv3 (no colour) | segmentation3d x8.
Writes /s/render/<name>.png and prints per-tooth position agreement between the two models."""
import sys
from pathlib import Path

sys.path[:0] = ["/src", "/pureptv3", "/inference", "/seg3d"]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import json  # noqa: E402
from toothseg import Scan, ToothSeg17  # noqa: E402
from pipeline.config import Placement, SegmentationConfig  # noqa: E402
from pipeline.stages import LoadStage  # noqa: E402
from tools.semantic_view import SemanticModel  # noqa: E402
from tools.preprocess_view import fdi as seg3d_fdi  # noqa: E402

DEV = torch.device("cuda")
PKG = Path("/models/toothseg-pureptv3/1")
AD = ToothSeg17(PKG, json.loads((PKG / "model.json").read_text()), DEV)
S3D = SemanticModel(Path("/seg3d/runs/semantic/best.pt"), 8, DEV)
cfg = SegmentationConfig()
LOAD = LoadStage(cfg, Placement.resolve(cfg))
OUT = Path("/s/render")
OUT.mkdir(exist_ok=True)
PAL = plt.get_cmap("tab10")


def frame(xyz, nrm, tooth):
    """z: mean crown normal (out of the page), x/y: PCA of the teeth in that plane."""
    z = nrm[tooth].mean(0)
    z /= np.linalg.norm(z)
    c = xyz[tooth] - xyz[tooth].mean(0)
    c = c - np.outer(c @ z, z)
    x = np.linalg.svd(c, full_matrices=False)[2][0]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def draw(ax, uv, depth, colours, labels, title):
    order = np.argsort(depth)
    ax.scatter(uv[order, 0], uv[order, 1], c=colours[order], s=0.6, linewidths=0)
    if labels is not None:
        for t in np.unique(labels[labels > 0]):
            m = labels == t
            if m.sum() > 200:
                cx, cy = np.median(uv[m], 0)
                ax.text(cx, cy, str(int(t)), ha="center", va="center", fontsize=11, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def fdi_colours(f):
    out = np.full((len(f), 3), 0.82)
    t = f > 0
    out[t] = np.array([PAL((v % 10) - 1)[:3] for v in f[t]])
    out[t & (f // 10 % 2 == 0)] *= 0.7          # quadrants 2 and 4 darker
    return out


for name in sys.argv[1:]:
    d = np.load(f"/s/inhouse_gt/{name}.npz")
    xyz, nrm, g, jaw = d["xyz"].astype(np.float64), d["normals"].astype(np.float64), d["fdi"].astype(int), str(d["jaw"])
    p = AD.predict(Scan(d["xyz"], d["normals"], None, jaw)).fdi.astype(int)
    scan = LOAD.run(Path(f"/data/ios/inhouse/raw/{name}.ply"))
    cls = S3D.predict(scan)[0]
    lut = np.array([0] + [int(seg3d_fdi(c, jaw == "upper")) for c in range(1, 17)])
    s = lut[cls]
    R = frame(xyz, nrm, g > 0)
    q = (xyz - xyz.mean(0)) @ R.T
    idx = np.random.default_rng(0).choice(len(q), min(len(q), 80000), replace=False)
    uv, depth = q[idx, :2], q[idx, 2]
    rgb = d["rgb"][idx] / 255.0 if "rgb" in d.files else np.full((len(idx), 3), 0.8)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
    draw(axes[0], uv, depth, rgb, None, f"{name}  real colour (occlusal)")
    draw(axes[1], uv, depth, fdi_colours(g[idx]), g[idx], "human label (DDH)")
    draw(axes[2], uv, depth, fdi_colours(p[idx]), p[idx], "PurePTv3 auto-label (no colour)")
    draw(axes[3], uv, depth, fdi_colours(s[idx]), s[idx], "segmentation3d semantic TTA x8")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=80)
    plt.close(fig)
    both = (p > 0) & (s > 0)
    print(f"{name}: PurePTv3 vs segmentation3d same position {np.mean(p[both] % 10 == s[both] % 10):.3f}, "
          f"same side {np.mean(p[both] // 10 == s[both] // 10):.3f}", flush=True)
