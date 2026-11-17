"""Are the vertex colours of the unlabelled Teeth3DS scans painted tooth labels?  And how does auto-label score
against them, with and without the colour?

Reads probe.py's /out/<asset>.npz.  For each scan:
  - the distinct colours, their area share and how many mesh-connected pieces each colour has;
  - "colour GT": the most common colour is gingiva, every other colour is one tooth;
  - tooth instance F1 at IoU >= 0.75 (area weighted, 1:1 greedy on IoU) of the colour GT against
    the prediction with colour, the prediction without colour, and the label the user saved (if any).
    Predicted and saved instances follow DDH: same FDI = one tooth.

usage: python colour_regions.py <out dir> <probe jsonl> [--palette]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.stdout.reconfigure(encoding="utf-8")


def vertex_area(xyz, faces):
    t = xyz[faces].astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    w = np.zeros(len(xyz))
    for k in range(3):
        np.add.at(w, faces[:, k], a / 3)
    return w


def pieces(mask, faces, n):
    """Number of mesh-connected pieces of the vertices in mask (pieces under 1% of the region ignored)."""
    f = faces[mask[faces].all(1)]
    if not len(f):
        return 0
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab[mask])
    sizes = sizes[sizes > 0]
    return int((sizes >= 0.01 * mask.sum()).sum())


def f1_75(gt, pred, w):
    """gt, pred: instance ids per vertex (0 = background).  Area-weighted IoU, greedy 1:1 matching."""
    g_ids = [i for i in np.unique(gt) if i]
    p_ids = [i for i in np.unique(pred) if i]
    if not g_ids and not p_ids:
        return 1.0, [], 0, 0
    pairs = []
    wg = {i: w[gt == i].sum() for i in g_ids}
    wp = {j: w[pred == j].sum() for j in p_ids}
    for i in g_ids:
        m = gt == i
        for j in np.unique(pred[m]):
            if not j:
                continue
            inter = w[m & (pred == j)].sum()
            pairs.append((inter / (wg[i] + wp[j] - inter), i, j))
    pairs.sort(reverse=True)
    used_g, used_p, ious = set(), set(), {}
    for iou, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i), used_p.add(j)
        ious[i] = iou
    tp = sum(1 for v in ious.values() if v >= 0.75)
    f1 = 2 * tp / (len(g_ids) + len(p_ids))
    return round(f1, 3), sorted(round(float(v), 2) for v in ious.values()), len(g_ids), len(p_ids)


def main():
    out, probe = Path(sys.argv[1]), sys.argv[2]
    palette = "--palette" in sys.argv
    rows = [json.loads(line) for line in open(probe, encoding="utf-8")]
    pal = {}
    for r in rows:
        d = np.load(out / f"{r['asset_id']}.npz")
        if "rgb" not in d:
            continue
        xyz, faces, rgb = d["xyz"], d["faces"], d["rgb"]
        n = len(xyz)
        w = vertex_area(xyz, faces)
        cols, inv = np.unique(rgb, axis=0, return_inverse=True)
        inv = inv.ravel()
        share = np.bincount(inv, weights=w) / w.sum()
        ging = int(np.argmax(share))
        gt = np.where(inv == ging, 0, inv + 1)
        desc = []
        for c in np.argsort(-share):
            desc.append(f"{tuple(int(x) for x in cols[c])}:{share[c]:.3f}/{pieces(inv == c, faces, n)}")
        rep = {"asset_id": r["asset_id"], "jaw": r["jaw"], "n_colours": len(cols), "gingiva": cols[ging].tolist()}
        for name, key in (("colour", "pred_colour_fdi"), ("grey", "pred_grey_fdi"), ("saved", "saved_fdi")):
            if key in d:
                f1, ious, ng, npred = f1_75(gt, d[key].astype(np.int64), w)
                rep[name] = {"F1@75": f1, "n_gt": ng, "n_pred": npred, "low_iou": [v for v in ious if v < 0.75]}
        # tooth/gingiva agreement of the colour GT with each prediction (area share)
        for name, key in (("colour", "pred_colour_fdi"), ("grey", "pred_grey_fdi")):
            rep[name]["tooth_agree"] = round(float(w[(gt > 0) == (d[key] > 0)].sum() / w.sum()), 3)
        print(json.dumps(rep))
        print("   colours (rgb:area share/pieces):", " ".join(desc))
        if palette:  # colour -> FDI the grey prediction gives it (majority by area)
            p = d["pred_grey_fdi"].astype(np.int64)
            for c in range(len(cols)):
                if c == ging:
                    continue
                m = inv == c
                f = np.bincount(p[m], weights=w[m])
                pal.setdefault((r["jaw"], tuple(int(x) for x in cols[c])), []).append(int(np.argmax(f)))
    if palette:
        print("palette: (jaw, rgb) -> FDI the no-colour prediction gives, per scan")
        for k in sorted(pal):
            print("  ", k, pal[k])


if __name__ == "__main__":
    main()
