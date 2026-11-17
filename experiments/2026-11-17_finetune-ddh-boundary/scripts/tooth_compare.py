"""Per-tooth IoU, base model against a candidate, with the deployed clean-up (absorb + join_close_pieces 0.5 mm)
applied to each run's raw instances (eval_fix.py output dirs).  GT teeth under 5 mm2 are dropped.

usage: python tooth_compare.py <probe npz dir> <base eval dir> <cand eval dir> [--palette] < asset ids
  --palette: the probe npz has the painted rgb; FDI is guessed from the palette (summarise.fdi) to report third molars.
stdout: totals (teeth, IoU mean, below 0.85, lost), third molars if --palette, and the teeth that moved most.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[5] / "inference-server"))
sys.path.insert(0, str(HERE.parents[2] / "2026-11-17_autolabel-incomplete-teeth" / "scripts"))
from toothseg import MIN_FRAGMENT_MM2, absorb_fragments, join_close_pieces, vertex_graph  # noqa: E402

MIN_GT_MM2 = 5.0


def tooth_ious(gt, pred, w):
    """Greedy 1:1 by IoU, as eval_fix_score.score; IoU per GT tooth id 1..G (0 if unmatched)."""
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
    return {i: best.get(i, 0.0) for i in range(1, n_g)}, wg


def deployed(raw, edges, area, xyz):
    return join_close_pieces(absorb_fragments(raw.astype(np.int64), edges, area, MIN_FRAGMENT_MM2), edges, xyz)


def main():
    src, base, cand = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    palette = "--palette" in sys.argv
    if palette:
        from summarise import fdi as palette_fdi
    rows = []
    for line in sys.stdin:
        aid = line.split()[0] if line.strip() else ""
        if not aid or not (base / f"{aid}.npz").exists() or not (cand / f"{aid}.npz").exists():
            continue
        d, b, c = np.load(src / f"{aid}.npz"), np.load(base / f"{aid}.npz"), np.load(cand / f"{aid}.npz")
        xyz = d["xyz"].astype(np.float64)
        faces = d["faces"].astype(np.int64) if "faces" in d.files else None
        edges, _ = vertex_graph(xyz, faces)
        area = b["area"].astype(np.float64)
        gt = b["gt"].astype(np.int64)
        small = np.bincount(gt, weights=area) < MIN_GT_MM2
        small[0] = False
        gt = np.unique(np.where(small[gt], 0, gt), return_inverse=True)[1].ravel()
        ib, wg = tooth_ious(gt, deployed(b["raw_consensus"], edges, area, xyz), area)
        ic, _ = tooth_ious(gt, deployed(c["raw_consensus"], edges, area, xyz), area)
        for t in ib:
            f = 0
            if palette:
                m = gt == t
                cols, cnt = np.unique(d["rgb"][m], axis=0, return_counts=True)
                f = palette_fdi(str(d["jaw"]), cols[cnt.argmax()].tolist())
            rows.append((aid, t, f, wg[t], ib[t], ic[t]))
    ious_b = np.array([r[4] for r in rows])
    ious_c = np.array([r[5] for r in rows])
    print(f"{len({r[0] for r in rows})} scans, {len(rows)} GT teeth")
    print(f"  base: IoU mean {ious_b.mean():.4f}  below 0.85 {(ious_b < 0.85).sum()}  lost {(ious_b < 0.5).sum()}")
    print(f"  cand: IoU mean {ious_c.mean():.4f}  below 0.85 {(ious_c < 0.85).sum()}  lost {(ious_c < 0.5).sum()}")
    print(f"  teeth better by > 0.02: {((ious_c - ious_b) > 0.02).sum()}, worse by > 0.02: {((ious_b - ious_c) > 0.02).sum()}")
    if palette:
        for pos in (7, 8):
            sel = np.array([r[2] % 10 == pos for r in rows])
            if sel.any():
                print(f"  position {pos}: {sel.sum()} teeth  base IoU {ious_b[sel].mean():.4f} (<0.85: "
                      f"{(ious_b[sel] < 0.85).sum()})  cand {ious_c[sel].mean():.4f} (<0.85: {(ious_c[sel] < 0.85).sum()})")
    order = np.argsort(ious_c - ious_b)
    print("\nlargest drops / gains (asset, GT id, FDI guess, area mm2, IoU base -> cand):")
    for k in list(order[:8]) + list(order[-12:]):
        a, t, f, ar, x, y = rows[k]
        print(f"  {a} {t:2d} {f:3d} {ar:6.1f}  {x:.3f} -> {y:.3f}")


if __name__ == "__main__":
    main()
