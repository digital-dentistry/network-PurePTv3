"""What did the user change after auto-label?  Saved labels (ios_labels) of the painted Teeth3DS scans against the
no-colour auto-label (probe pred_grey_*) and against the colour map.

For every saved scan: tooth area the user added over auto-label (auto gingiva -> saved tooth), area removed
(auto tooth -> saved gingiva), and per colour tooth the IoU of auto and of the saved label.  The user may have
started from either button, so "added" is an upper bound on hand work against 컬러 미사용.

usage: python saved_vs_auto.py <probe npz dir> <probe jsonl>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def vertex_area(xyz, faces):
    t = xyz[faces].astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    w = np.zeros(len(xyz))
    for k in range(3):
        np.add.at(w, faces[:, k], a / 3)
    return w


def colour_gt(rgb, w):
    _, inv = np.unique(rgb, axis=0, return_inverse=True)
    inv = inv.ravel()
    gum = int(np.argmax(np.bincount(inv, weights=w)))
    gt = inv + 1
    gt[inv == gum] = 0
    return np.unique(gt, return_inverse=True)[1].ravel()


def best_iou(gt, pred, w):
    """Per GT tooth, the IoU of the pred instance overlapping it most (area)."""
    out = []
    for i in range(1, gt.max() + 1):
        m = gt == i
        ids, inv = np.unique(pred[m], return_inverse=True)
        ov = np.bincount(inv.ravel(), weights=w[m])
        ov[ids == 0] = 0
        if ov.max() <= 0:
            out.append(0.0)
            continue
        j = ids[ov.argmax()]
        inter = ov.max()
        out.append(inter / (w[m].sum() + w[pred == j].sum() - inter))
    return np.array(out)


def main():
    src = Path(sys.argv[1])
    rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
    print("asset            jaw   teeth  added mm2  removed mm2  IoU<0.9 auto  IoU<0.9 saved  min IoU auto/saved")
    for r in rows:
        if not r["saved"]:
            continue
        d = np.load(src / f"{r['asset_id']}.npz")
        w = vertex_area(d["xyz"], d["faces"].astype(np.int64))
        gt = colour_gt(d["rgb"], w)
        auto = d["pred_grey_fdi"].astype(np.int64)
        saved = d["saved_fdi"].astype(np.int64) * (d["saved_label"] > 0)
        added = w[(auto == 0) & (saved > 0)].sum()
        removed = w[(auto > 0) & (saved == 0)].sum()
        ia, isv = best_iou(gt, auto, w), best_iou(gt, saved, w)
        print(f"{r['asset_id']} {r['jaw']:5s} {gt.max():5d}  {added:9.1f}  {removed:11.1f}  {(ia < 0.9).sum():12d}"
              f"  {(isv < 0.9).sum():13d}  {ia.min():.2f} / {isv.min():.2f}")


if __name__ == "__main__":
    main()
