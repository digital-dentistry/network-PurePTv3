"""Islands in a label map on the mesh (added 2026-11-17, E7).  For each label value, its mesh-connected pieces:
a tooth label in more than one piece (one tooth shown as several, or a tooth number on a speck) and small gingiva
pieces enclosed by teeth (holes) are listed with their area and the labels around them.

usage: python islands.py <probe npz dir> grey|colour|src <asset> [...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
from score_src import vertex_area  # noqa: E402


def main():
    src, key = Path(sys.argv[1]), sys.argv[2]
    for aid in sys.argv[3:]:
        d = np.load(src / f"{aid}.npz")
        f = d["faces"].astype(np.int64)
        w = vertex_area(d["xyz"].astype(np.float64), f)
        if key == "src":
            lab = np.where((d["src_label"] > 0) & (d["src_fdi"] > 0), d["src_fdi"].astype(int), 0)
        else:
            lab = np.where(d[f"pred_{key}_label"] > 0, d[f"pred_{key}_fdi"].astype(int), 0)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        same = lab[e[:, 0]] == lab[e[:, 1]]
        n = len(lab)
        _, comp = connected_components(coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n)),
                                       directed=False)
        ca = np.bincount(comp, weights=w)
        cl = np.zeros(comp.max() + 1, int)
        cl[comp] = lab
        cross = e[~same]
        print(f"{aid} [{key}]")
        for v in np.unique(lab):
            cs = np.nonzero(cl == v)[0]
            cs = cs[np.argsort(-ca[cs])]
            extra = [c for c in cs[1:] if ca[c] > 0] if v else [c for c in cs if ca[c] < 30]
            if not extra:
                continue
            out = []
            for c in extra[:6]:
                nb = np.concatenate([lab[cross[comp[cross[:, 0]] == c, 1]], lab[cross[comp[cross[:, 1]] == c, 0]]])
                out.append(f"{ca[c]:.2f} mm2/{(comp == c).sum()} v next to {sorted(set(nb.tolist()))}")
            print(f"  label {v}: {len(cs)} pieces, largest {ca[cs[0]]:.1f} mm2; others: {out}")


if __name__ == "__main__":
    main()
