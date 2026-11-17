"""Picture of one scan: colour GT next to a prediction, tooth area the prediction called gingiva in red, gingiva it
called tooth in blue, area given to another tooth in orange.  Numpy point splatting, no GPU.

usage: python render.py <probe npz dir> <stages npz dir> <asset> <out.png> [--tooth GT_ID] [variant ...]
With --tooth the second row is a close-up of that colour tooth (and 4 mm around it), seen obliquely from its outer side.
The output contains a scan, so keep it outside the repo.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = 700


def normals(xyz, faces):
    c = np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]])
    n = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(n, faces[:, k], c)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def splat(xyz, nrm, rgb, view):
    """Orthographic, looking along -view[2]; each vertex a square about one mesh edge (0.25 mm) wide,
    painted far to near so the nearest wins."""
    p = xyz @ view.T
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    s = (RES - 3) / (hi - lo).max()
    ij = ((p[:, :2] - lo) * s).astype(int)
    img = np.ones((RES, RES, 3))
    depth = np.full((RES, RES), -np.inf)
    order = np.argsort(p[:, 2])
    shade = np.clip(0.35 + 0.65 * (nrm @ view[2]), 0.15, 1)[:, None]
    size = int(np.clip(np.ceil(0.25 * s), 2, 12))
    for di in range(size):
        for dj in range(size):
            i = np.clip(RES - 1 - (ij[order, 1] + di), 0, RES - 1)
            j = np.clip(ij[order, 0] + dj, 0, RES - 1)
            img[i, j] = (rgb[order] * shade[order])
            depth[i, j] = p[order, 2]
    return img


def main():
    probe, stages, aid, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    args = sys.argv[5:]
    tooth = None
    if "--tooth" in args:
        k = args.index("--tooth")
        tooth = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    variants = args or ["v0_deployed"]
    d, s = np.load(probe / f"{aid}.npz"), np.load(stages / f"{aid}.npz")
    xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
    nrm = normals(xyz, faces)
    gt = s["gt"].astype(int)
    c = xyz - xyz.mean(0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    z = axes[2] * (1 if (nrm[gt > 0] @ axes[2]).mean() > 0 else -1)   # occlusal: teeth face the camera
    x = axes[0]
    occlusal = np.stack([x, np.cross(z, x), z])
    front = np.stack([x, z, -np.cross(z, x)])                          # from the side
    near = np.ones(len(c), bool)
    if tooth is not None:
        m = gt == tooth
        centre = c[m].mean(0)
        out_dir = centre - c[gt > 0].mean(0)                            # away from the arch centre
        out_dir -= (out_dir @ z) * z
        out_dir = out_dir / np.linalg.norm(out_dir)
        v = 0.6 * out_dir + 0.8 * z                                     # oblique: outer side and top
        v /= np.linalg.norm(v)
        u = np.cross(z, v)
        u /= np.linalg.norm(u)
        front = np.stack([u, np.cross(v, u), v])
        near = np.linalg.norm(c - centre, axis=1) < np.linalg.norm(c[m] - centre, axis=1).max() + 4
    def labels(v):
        """A stages.py variant, or "saved": the label the user completed (ios_labels), one id per FDI."""
        if v == "saved":
            return np.unique(d["saved_fdi"].astype(int) * (d["saved_label"] > 0), return_inverse=True)[1].ravel()
        return s[v].astype(int)

    rng = np.random.RandomState(3)
    pal = rng.uniform(0.45, 0.95, (int(max(gt.max(), max(labels(v).max() for v in variants))) + 1, 3))
    gum = np.array([0.93, 0.80, 0.80])

    def colour_gt():
        col = pal[gt].copy()
        col[gt == 0] = gum
        return col

    def colour_pred(pred):
        col = pal[pred % len(pal)].copy()
        col[pred == 0] = gum
        # each pred instance -> its majority GT tooth, to tell "other tooth" from "own tooth"
        owner = np.zeros(pred.max() + 1, int)
        for j in range(1, pred.max() + 1):
            m = pred == j
            if m.any():
                owner[j] = np.bincount(gt[m]).argmax()
        col[(gt > 0) & (pred == 0)] = [0.85, 0.05, 0.05]                  # tooth called gingiva
        col[(gt == 0) & (pred > 0)] = [0.10, 0.30, 0.95]                  # gingiva called tooth
        col[(gt > 0) & (pred > 0) & (owner[pred] != gt)] = [1.0, 0.55, 0.0]  # another tooth's instance
        return col

    panels = [("colour GT", colour_gt())] + [(v, colour_pred(labels(v))) for v in variants]
    fig, ax = plt.subplots(2, len(panels), figsize=(5 * len(panels), 10))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for k, (name, col) in enumerate(panels):
        for r, view in enumerate((occlusal, front)):
            sel = near if r == 1 else np.ones(len(c), bool)
            ax[r, k].imshow(splat(c[sel], nrm[sel], col[sel], view))
            ax[r, k].set_axis_off()
        ax[0, k].set_title(name)
    fig.suptitle(f"{aid}  red: tooth called gingiva  blue: gingiva called tooth  orange: other tooth")
    fig.tight_layout()
    fig.savefig(out, dpi=70)


if __name__ == "__main__":
    main()
