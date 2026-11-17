"""When does the clean-up split one network instance into several teeth?  (added 2026-11-17, E8)

After absorb_fragments, components() gives every mesh-connected piece of an instance its own tooth.  For every
instance that ends up in two or more pieces of at least 5 mm2 this lists the pieces: area, the closest distance
to the instance's other pieces, the share of the piece lying within 0.5 mm of them (overlapping scan sheets give a
large share), open mesh borders inside the piece, and the GT tooth each piece mostly covers, so the split can be
judged: same GT tooth = a wrong split (Case 322), different GT teeth = a useful split.

usage: python split_pieces.py <npz dir with xyz, faces> <labels npz dir> <key> <gt: colour|src|saved|stages> < asset ids
  key: the raw-instance array in the labels npz (stages.py: v0_noclean).
Needs inference-server/ on the path (toothseg); run from the repo root or set PYTHONPATH.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "inference-server"))
from toothseg import MIN_FRAGMENT_MM2, absorb_fragments, components, vertex_graph  # noqa: E402


def open_border_vertices(faces, n):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    ue, ce = np.unique(e, axis=0, return_counts=True)
    bv = np.zeros(n, bool)
    bv[ue[ce == 1].ravel()] = True
    return bv


def gt_of(d, s, how):
    if how == "stages":
        return s["gt"].astype(int)
    if how == "src":
        lab, fdi = d["src_label"].astype(int), d["src_fdi"].astype(int)
        key = np.where((lab > 0) & (fdi > 0), lab * 100 + fdi, 0)
    else:  # saved
        lab, fdi = d["saved_label"].astype(int), d["saved_fdi"].astype(int)
        key = np.where((lab > 0) & (fdi > 0), lab * 100 + fdi, 0)
    return np.unique(key, return_inverse=True)[1].ravel()


def main():
    src, lab_dir, key, how = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    for line in sys.stdin:
        aid = line.strip()
        if not aid or not (lab_dir / f"{aid}.npz").exists():
            continue
        d, s = np.load(src / f"{aid}.npz"), np.load(lab_dir / f"{aid}.npz")
        xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
        edges, area = vertex_graph(xyz, faces)
        raw = s[key].astype(np.int64)
        gt = gt_of(d, s, how)
        lab = absorb_fragments(raw.copy(), edges, area, MIN_FRAGMENT_MM2)
        comp = components(lab, edges)
        bv = open_border_vertices(faces, len(xyz))
        n_split = 0
        for inst in np.unique(lab[lab > 0]):
            pieces = [c for c in np.unique(comp[lab == inst]) if area[comp == c].sum() >= MIN_FRAGMENT_MM2]
            if len(pieces) < 2:
                continue
            n_split += 1
            for c in pieces:
                p = comp == c
                rest = (lab == inst) & ~p
                dist, _ = cKDTree(xyz[rest]).query(xyz[p], distance_upper_bound=5.0)
                g = gt[p]
                gid = int(np.bincount(g, weights=area[p]).argmax())
                print(f"{aid} inst {inst} piece {area[p].sum():6.1f} mm2  min gap {dist.min():.3f} mm  "
                      f"near(<0.5mm) {(dist < 0.5).mean():.2f}  open-border v {int(bv[p].sum())}  "
                      f"GT {gid} ({np.bincount(g, weights=area[p])[gid] / area[p].sum():.2f})", flush=True)
        print(f"{aid} split instances {n_split}", flush=True)


if __name__ == "__main__":
    main()
