"""Experiment: move the predicted tooth/gingiva boundary to the bottom of the concave groove (marker watershed).

Offline, numpy only.  Markers come from a prediction (eval_fix.py output "new"):
  tooth k   = vertices of predicted tooth k farther than R_IN mm (along the mesh) from its boundary
  gingiva   = predicted gingiva farther than R_OUT mm from every predicted tooth
  undecided = the band in between
The band is flooded from the markers in order of elevation = -curvature (smoothed), so the regions grow over
convex and flat surface first and meet where the surface is most concave: the gingival margin and the
interproximal grooves.  Scores the prediction with and without the refinement against the GT in the eval npz.

usage: python refine_watershed.py <probe npz dir> <eval npz dir> <eval jsonl> [R_IN R_OUT]
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).parent))
from boundary_geometry import curvature, mesh_edges, normals  # noqa: E402
from eval_fix_score import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def smooth(v, edges, n, times=2):
    deg = np.bincount(edges.ravel(), minlength=n) + 1
    for _ in range(times):
        v = (v + np.bincount(edges[:, 0], v[edges[:, 1]], n) + np.bincount(edges[:, 1], v[edges[:, 0]], n)) / deg
    return v


def distance_from(mask, xyz, edges, limit):
    n = len(xyz)
    length = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    src = np.nonzero(mask)[0]
    g = coo_matrix((np.r_[length, length, np.full(len(src), 1e-9)],
                    (np.r_[edges[:, 0], edges[:, 1], np.full(len(src), n)], np.r_[edges[:, 1], edges[:, 0], src])),
                   shape=(n + 1, n + 1)).tocsr()
    return dijkstra(g, indices=n, limit=limit)[:n]


def refine(xyz, faces, pred, r_in, r_out):
    n = len(xyz)
    edges = mesh_edges(faces)
    elev = -smooth(curvature(xyz, normals(xyz, faces), edges), edges, n)
    a, b = pred[edges[:, 0]], pred[edges[:, 1]]
    border = np.zeros(n, bool)
    diff = a != b
    border[edges[diff, 0]] = border[edges[diff, 1]] = True
    near_border = distance_from(border, xyz, edges, r_in) < r_in
    near_tooth = distance_from(pred > 0, xyz, edges, r_out) < r_out
    lab = np.full(n, -1, np.int64)
    lab[(pred > 0) & ~near_border] = pred[(pred > 0) & ~near_border]
    lab[(pred == 0) & ~near_tooth] = 0
    if (lab < 0).sum() == 0:
        return pred
    # adjacency lists
    order = np.argsort(np.r_[edges[:, 0], edges[:, 1]], kind="stable")
    nbr = np.r_[edges[:, 1], edges[:, 0]][order]
    start = np.searchsorted(np.r_[edges[:, 0], edges[:, 1]][order], np.arange(n + 1))
    heap = []
    queued = lab >= 0
    for v in np.nonzero(lab >= 0)[0]:
        for u in nbr[start[v]:start[v + 1]]:
            if not queued[u]:
                queued[u] = True
                heapq.heappush(heap, (elev[u], int(u), int(lab[v])))
    while heap:
        _, v, l = heapq.heappop(heap)
        if lab[v] >= 0:
            continue
        lab[v] = l
        for u in nbr[start[v]:start[v + 1]]:
            if lab[u] < 0:
                heapq.heappush(heap, (max(elev[u], elev[v]), int(u), l))
    lab[lab < 0] = pred[lab < 0]           # band pieces no marker reaches keep the prediction
    return lab


def main():
    probe, ev = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in open(sys.argv[3], encoding="utf-8")]
    r_in = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    r_out = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    keys = ["f1_75", "iou_mean", "iou_min", "teeth_below_085", "tooth_to_gum_mm2", "gum_to_tooth_mm2", "merges", "splits"]
    tot = {"new": {k: [] for k in keys}, "refined": {k: [] for k in keys}}
    for r in rows:
        d, e = np.load(probe / f"{r['asset_id']}.npz"), np.load(ev / f"{r['asset_id']}.npz")
        if "faces" not in d.files:
            continue
        xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
        gt, new, area = e["gt"].astype(int), e["new"].astype(int), e["area"].astype(float)
        ref = refine(xyz, faces, new, r_in, r_out)
        for name, lab in (("new", new), ("refined", ref)):
            s = score(gt, lab, area)
            for k in keys:
                tot[name][k].append(s[k])
    print(f"R_IN {r_in} R_OUT {r_out}, {len(tot['new']['f1_75'])} scans")
    for name, t in tot.items():
        print(f"  {name:8s} " + "  ".join(f"{k} {np.mean([x for x in t[k] if x is not None]):.4f}" for k in keys))


if __name__ == "__main__":
    main()
