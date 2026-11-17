"""Close-up of one region without overlays (added 2026-11-17, E8): plain shaded geometry, the scan's painted colour,
the live prediction (one colour per predicted tooth, gingiva pink), from three directions around the chosen colour
tooth.  Keep the output outside the repo (it shows a scan).

usage: python closeup.py <probe npz dir> <stages npz dir> <asset> <colour GT id> <out.png> [radius mm] [label key]
label key: an npz array in the probe file (default pred_grey_label / pred_grey_fdi) or a stages variant name.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import render  # noqa: E402

render.RES = 900


def main():
    probe, stages, aid, tooth, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    radius = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
    key = sys.argv[7] if len(sys.argv) > 7 else None
    d, s = np.load(probe / f"{aid}.npz"), np.load(stages / f"{aid}.npz")
    xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
    nrm = render.normals(xyz, faces)
    gt = s["gt"].astype(int)
    if key is None:
        pred = np.where(d["pred_grey_label"] > 0, d["pred_grey_fdi"].astype(int), 0)
    else:
        pred = s[key].astype(int)
    c = xyz - xyz.mean(0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    z = axes[2] * (1 if (nrm[gt > 0] @ axes[2]).mean() > 0 else -1)
    m = gt == tooth
    centre = c[m].mean(0)
    outward = centre - c[gt > 0].mean(0)
    outward -= (outward @ z) * z
    outward /= np.linalg.norm(outward)
    near = np.linalg.norm(c - centre, axis=1) < np.linalg.norm(c[m] - centre, axis=1).max() + radius
    views = []
    for name, v in (("occlusal", z), ("outer side", 0.8 * outward + 0.6 * z), ("inner side", -0.8 * outward + 0.6 * z)):
        v = v / np.linalg.norm(v)
        u = np.cross(z if abs(v @ z) < 0.99 else outward, v)
        u /= np.linalg.norm(u)
        views.append((name, np.stack([u, np.cross(v, u), v])))
    ids = np.unique(pred)
    fixed = [(0.90, 0.15, 0.15), (0.15, 0.35, 0.95), (0.10, 0.75, 0.25), (0.95, 0.80, 0.10), (0.60, 0.20, 0.80),
             (0.10, 0.80, 0.85), (0.95, 0.50, 0.10), (0.45, 0.45, 0.45)]
    order = sorted((int(p) for p in ids if p), key=lambda p: -(pred[near] == p).sum())
    pal = {p: np.array(fixed[k % len(fixed)]) for k, p in enumerate(order)}
    pal[0] = np.array([0.93, 0.80, 0.80])
    plain = np.full((len(c), 3), 0.85)
    painted = d["rgb"].astype(float) / 255 if "rgb" in d.files else plain
    predc = np.array([pal[int(p)] for p in pred])
    panels = [("geometry", plain), ("painted colour", painted), ("live prediction", predc)]
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for r, (vname, view) in enumerate(views):
        for k, (pname, col) in enumerate(panels):
            ax[r, k].imshow(render.splat(c[near], nrm[near], col[near], view))
            ax[r, k].set_axis_off()
            ax[r, k].set_title(f"{pname} ({vname})")
    inside = {int(p): round(float((pred[m] == p).mean()), 2) for p in np.unique(pred[m])}
    names = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "grey"]
    legend = ", ".join(f"{p}={names[k]}" for k, p in enumerate(order[:8]))
    fig.suptitle(f"{aid} colour tooth {tooth}: share per predicted label {inside}\n{legend}")
    fig.tight_layout()
    fig.savefig(out, dpi=60)


if __name__ == "__main__":
    main()
