"""Prototype (not in inference-server): keep the mesh pieces of one network instance together when they touch in
space.  Added 2026-11-17, E8.

components() splits an instance into its mesh-connected pieces.  On scans with overlapping, unwelded scan sheets
(Case 322: two sheets 0.26 mm apart over a molar, open borders) one tooth lies on two sheets, so it becomes two
teeth and numbering gives them two numbers.  Here pieces of the same instance whose closest points are under
GAP mm apart are joined again; pieces farther apart stay separate.

usage: python merge_close.py <npz dir with xyz, faces> <labels npz dir> <key> <gap mm,...> < asset ids
  labels npz must hold `gt`, and `key` the raw instances (stages.py v0_noclean).
stdout: per scan the score with plain components and with each gap; a summary at the end.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "inference-server"))
sys.path.insert(0, str(Path(__file__).parent))
from toothseg import MIN_FRAGMENT_MM2, absorb_fragments, components, vertex_graph  # noqa: E402
from eval_fix_score import score  # noqa: E402


def join_close(lab, comp, xyz, gap):
    """comp (components of lab) with the pieces of one label that come within `gap` mm merged."""
    parent = np.arange(comp.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for inst in np.unique(lab[lab > 0]):
        pieces = np.unique(comp[lab == inst])
        if len(pieces) < 2:
            continue
        trees = {int(c): cKDTree(xyz[comp == c]) for c in pieces}
        for i, a in enumerate(pieces):
            for b in pieces[i + 1:]:
                if find(a) == find(b):
                    continue
                dist, _ = trees[int(b)].query(xyz[comp == a], distance_upper_bound=gap)
                if np.isfinite(dist).any():
                    parent[find(a)] = find(b)
    roots = np.array([find(c) for c in range(len(parent))])
    out = roots[comp]
    out[lab == 0] = 0
    return np.unique(out, return_inverse=True)[1].ravel()


def main():
    src, lab_dir, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    gaps = [float(g) for g in sys.argv[4].split(",")]
    totals = {"components": []} | {f"gap {g}": [] for g in gaps}
    for line in sys.stdin:
        aid = line.strip()
        if not aid or not (lab_dir / f"{aid}.npz").exists():
            continue
        d, s = np.load(src / f"{aid}.npz"), np.load(lab_dir / f"{aid}.npz")
        xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
        edges, area = vertex_graph(xyz, faces)
        gt = s["gt"].astype(np.int64)
        lab = absorb_fragments(s[key].astype(np.int64), edges, area, MIN_FRAGMENT_MM2)
        comp = components(lab, edges)
        res = {"components": score(gt, comp, area)}
        for g in gaps:
            res[f"gap {g}"] = score(gt, join_close(lab, comp, xyz, g), area)
        for k, v in res.items():
            totals[k].append(v)
        changed = [k for k in res if res[k] != res["components"]]
        if changed:
            print(aid, {k: (v["f1_75"], v["n_pred"], v["iou_min"], v["splits"], v["merges"]) for k, v in res.items()},
                  flush=True)
    print(f"\n{len(totals['components'])} scans: F1@75 mean / scans F1=1 / IoU mean / teeth < 0.85 / n_pred total / "
          "splits / merges")
    for k, rows in totals.items():
        print(f"  {k:10s} {np.mean([r['f1_75'] for r in rows]):.4f}  {sum(r['f1_75'] == 1 for r in rows):3d}  "
              f"{np.mean([r['iou_mean'] for r in rows]):.4f}  {sum(r['teeth_below_085'] for r in rows):3d}  "
              f"{sum(r['n_pred'] for r in rows)}  {sum(r['splits'] for r in rows)}  {sum(r['merges'] for r in rows)}")


if __name__ == "__main__":
    main()
