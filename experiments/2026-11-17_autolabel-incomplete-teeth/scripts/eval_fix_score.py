"""Boundary score shared by eval_fix.py and refine_watershed.py (numpy only).

score(gt, pred, w): gt / pred are object ids per vertex (0 = gingiva), w the vertex area.
  F1@75 (area-weighted IoU, greedy 1:1), mean / min IoU of the GT teeth, teeth below IoU 0.85 and below 0.5
  ("lost"), tooth area called
  gingiva, gingiva called tooth (mm2), merges (one prediction holds >= 30 % of two GT teeth), splits (one GT tooth
  gives >= 30 % to two predictions).
"""
import numpy as np


def score(gt, pred, w):
    n_g, n_p = int(gt.max()) + 1, int(pred.max()) + 1
    inter = np.zeros((n_g, n_p))
    np.add.at(inter, (gt, pred), w)
    wg, wp = inter.sum(1), inter.sum(0)
    iou = inter / np.maximum(wg[:, None] + wp[None] - inter, 1e-12)
    pairs = sorted(((iou[i, j], i, j) for i in range(1, n_g) for j in range(1, n_p) if inter[i, j] > 0), reverse=True)
    used_g, used_p, best = set(), set(), {}
    for v, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i), used_p.add(j)
            best[i] = v
    g_ious = np.array([best.get(i, 0.0) for i in range(1, n_g)])
    tp = int((g_ious >= 0.75).sum())
    share_g = inter[1:, 1:] / np.maximum(wg[1:, None], 1e-12)      # GT tooth -> pred
    share_p = inter[1:, 1:] / np.maximum(wp[None, 1:], 1e-12)      # pred <- GT tooth
    merges = int(((share_p >= 0.3).sum(0) >= 2).sum())
    splits = int(((share_g >= 0.3).sum(1) >= 2).sum())
    return {"f1_75": round(2 * tp / max((n_g - 1) + (n_p - 1), 1), 4), "n_gt": n_g - 1, "n_pred": n_p - 1,
            "iou_mean": round(float(g_ious.mean()), 4) if len(g_ious) else None,
            "iou_min": round(float(g_ious.min()), 3) if len(g_ious) else None,
            "teeth_below_085": int((g_ious < 0.85).sum()), "teeth_lost": int((g_ious < 0.5).sum()),
            "tooth_to_gum_mm2": round(float(inter[1:, 0].sum()), 1), "gum_to_tooth_mm2": round(float(inter[0, 1:].sum()), 1),
            "merges": merges, "splits": splits}
