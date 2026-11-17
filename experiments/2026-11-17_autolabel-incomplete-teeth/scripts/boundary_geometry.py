"""Is the true tooth/gingiva boundary a concave groove, and does the prediction stop above it?

For the painted Teeth3DS scans (colour GT) and the deployed prediction (stages.py v0_deployed):
  - per-vertex signed curvature from the mesh: mean over edges of (n_u - n_v).(x_u - x_v) / |x_u - x_v|^2
    (> 0 convex, < 0 concave; 1/mm);
  - the curvature on the GT boundary (tooth vertices with a gingiva neighbour), on the predicted boundary, and on
    the tooth area the prediction called gingiva ("missed band");
  - the geodesic-ish distance (graph hops x edge length) from each missed vertex to the predicted tooth.

usage: python boundary_geometry.py <probe npz dir> <stages npz dir> <stages jsonl>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.stdout.reconfigure(encoding="utf-8")


def normals(xyz, faces):
    c = np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]])
    n = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(n, faces[:, k], c)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def mesh_edges(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def curvature(xyz, nrm, edges):
    d = xyz[edges[:, 0]] - xyz[edges[:, 1]]
    k = ((nrm[edges[:, 0]] - nrm[edges[:, 1]]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12)
    s = np.bincount(edges[:, 0], k, len(xyz)) + np.bincount(edges[:, 1], k, len(xyz))
    c = np.bincount(edges.ravel(), minlength=len(xyz))
    return s / np.maximum(c, 1)


def boundary(lab, edges):
    """Tooth vertices (lab > 0) with a gingiva neighbour."""
    a, b = lab[edges[:, 0]], lab[edges[:, 1]]
    m = np.zeros(len(lab), bool)
    x = (a > 0) & (b == 0)
    m[edges[x, 0]] = True
    x = (b > 0) & (a == 0)
    m[edges[x, 1]] = True
    return m


def main():
    probe, stages = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in open(sys.argv[3], encoding="utf-8")]
    pct = [10, 25, 50, 75, 90]
    acc = {"gt boundary": [], "pred boundary": [], "missed band": [], "tooth inside": [], "gingiva": []}
    dist_all, w_all = [], []
    for r in rows:
        d, s = np.load(probe / f"{r['asset_id']}.npz"), np.load(stages / f"{r['asset_id']}.npz")
        xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
        edges = mesh_edges(faces)
        nrm = normals(xyz, faces)
        k = curvature(xyz, nrm, edges)
        gt, v0, area = s["gt"].astype(int), s["v0_deployed"].astype(int), s["area"].astype(float)
        gb, pb = boundary(gt, edges), boundary(v0, edges)
        miss = (gt > 0) & (v0 == 0)
        acc["gt boundary"].append(k[gb])
        acc["pred boundary"].append(k[pb])
        acc["missed band"].append(k[miss])
        acc["tooth inside"].append(k[(gt > 0) & ~gb & (v0 > 0)])
        acc["gingiva"].append(k[(gt == 0) & (v0 == 0)])
        if miss.any() and (v0 > 0).any():
            # distance from each missed vertex to the nearest predicted tooth vertex, along the mesh
            n = len(xyz)
            length = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
            g = coo_matrix((np.r_[length, length, np.full((v0 > 0).sum(), 1e-9)],
                            (np.r_[edges[:, 0], edges[:, 1], np.full((v0 > 0).sum(), n)],
                             np.r_[edges[:, 1], edges[:, 0], np.nonzero(v0 > 0)[0]])), shape=(n + 1, n + 1)).tocsr()
            dist = dijkstra(g, indices=n, limit=10.0)[:n]
            dist_all.append(dist[miss])
            w_all.append(area[miss])
    print("signed curvature 1/mm (> 0 convex, < 0 concave), percentiles", pct)
    for name, v in acc.items():
        v = np.concatenate(v)
        print(f"  {name:14s} n={len(v):8d}  " + "  ".join(f"{x:6.2f}" for x in np.percentile(v, pct))
              + f"   share concave(< -0.1) {np.mean(v < -0.1):.2f}")
    dist, w = np.concatenate(dist_all), np.concatenate(w_all)
    dist[~np.isfinite(dist)] = 10.0
    print("\nmissed tooth area by distance from the predicted tooth (mm):")
    for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99)]:
        m = (dist >= lo) & (dist < hi)
        print(f"  {lo:4.1f} - {hi:4.1f}: {w[m].sum():8.1f} mm2  ({w[m].sum() / w.sum():.2f})")


if __name__ == "__main__":
    main()
