"""Live auto-label (probe.py PROBE_MODES=grey output) against the scan's own source labels (Teeth3DS label/fdi in
mesh.ply), per tooth.  Added 2026-11-17 for the user's Case 452 / 453 question (E7).

usage: python score_src.py <probe npz dir> <asset> [...]
Per scan the boundary score (eval_fix_score.score) and, per source tooth, IoU and where its area went
("parts": (predicted FDI, share), 0 = gingiva).  Predicted teeth lying mostly on gingiva are listed too.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_fix_score import score  # noqa: E402


def vertex_area(xyz, f):
    a = 0.5 * np.linalg.norm(np.cross(xyz[f[:, 1]] - xyz[f[:, 0]], xyz[f[:, 2]] - xyz[f[:, 0]]), axis=1)
    w = np.zeros(len(xyz))
    for k in range(3):
        np.add.at(w, f[:, k], a / 3)
    return w


def main():
    src = Path(sys.argv[1])
    for aid in sys.argv[2:]:
        d = np.load(src / f"{aid}.npz")
        w = vertex_area(d["xyz"].astype(np.float64), d["faces"])
        lab, fdi = d["src_label"].astype(int), d["src_fdi"].astype(int)
        u, gt = np.unique(np.where((lab > 0) & (fdi > 0), lab * 100 + fdi, 0), return_inverse=True)
        pl, pf = d["pred_grey_label"].astype(int), d["pred_grey_fdi"].astype(int)
        pu, pr = np.unique(np.where((pl > 0) & (pf > 0), pl * 100 + pf, 0), return_inverse=True)
        print(aid, str(d["jaw"]), "score", score(gt, pr, w))
        for i in range(1, len(u)):
            m = gt == i
            tot = w[m].sum()
            sh = np.bincount(pr[m], weights=w[m], minlength=len(pu)) / tot
            best = int(np.argmax(sh[1:])) + 1
            inter = w[m & (pr == best)].sum()
            iou = inter / (tot + w[pr == best].sum() - inter)
            parts = [(int(pu[j] % 100), round(float(sh[j]), 3)) for j in np.argsort(-sh) if sh[j] > 0.01]
            print(f"  src {u[i] % 100}: {tot:6.1f} mm2 IoU {iou:.3f} parts {parts}{' <--' if iou < 0.85 else ''}")
        for j in range(1, len(pu)):
            m = pr == j
            on_gum = w[m & (gt == 0)].sum() / w[m].sum()
            if on_gum > 0.3:
                print(f"  pred {pu[j] % 100}: {w[m].sum():.1f} mm2, {on_gum:.2f} on gingiva")


if __name__ == "__main__":
    main()
