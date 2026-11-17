"""Rescore eval_fix.py outputs (E8, 2026-11-17) with GT specks removed, and add the "join close pieces" clean-up.

The inhouse saved labels were completed on top of the old auto-label, so some keep its 0-1 mm2 specks as "teeth"
with their own number (c673 GT tooth 8: 14 vertices, 0.0 mm2).  The old code's specks match them and the new
speck rule looks like a lost tooth.  Here GT teeth under MIN_GT_MM2 count as gingiva.

Sides: old, new, nosplit (as saved by eval_fix.py) and gap<g> = new absorb, components, then the pieces of one
network instance that come within g mm joined again (merge_close.join_close).

usage: python rescore.py <probe npz dir> <eval_fix out dir> <gaps mm,...> [<asset dataset file>] < asset ids
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "inference-server"))
sys.path.insert(0, str(Path(__file__).parent))
from toothseg import MIN_FRAGMENT_MM2, absorb_fragments, components, vertex_graph  # noqa: E402
from eval_fix_score import score  # noqa: E402
from merge_close import join_close  # noqa: E402

MIN_GT_MM2 = 5.0
KEYS = ["f1_75", "iou_mean", "teeth_below_085", "teeth_lost", "tooth_to_gum_mm2", "merges", "splits", "n_pred"]
SUM = {"teeth_below_085", "teeth_lost", "merges", "splits", "n_pred"}


def main():
    src, run = Path(sys.argv[1]), Path(sys.argv[2])
    gaps = [float(g) for g in sys.argv[3].split(",")]
    group = {}
    if len(sys.argv) > 4:
        group = dict(line.split()[:2] for line in open(sys.argv[4], encoding="utf-8") if line.strip())
    res = defaultdict(lambda: defaultdict(list))
    changed = []
    for line in sys.stdin:
        aid = line.split()[0] if line.strip() else ""
        if not aid or not (run / f"{aid}.npz").exists():
            continue
        d, e = np.load(src / f"{aid}.npz"), np.load(run / f"{aid}.npz")
        xyz = d["xyz"].astype(np.float64)
        faces = d["faces"].astype(np.int64) if "faces" in d.files else None
        edges, _ = vertex_graph(xyz, faces)
        area = e["area"].astype(np.float64)
        gt = e["gt"].astype(np.int64)
        small = np.bincount(gt, weights=area) < MIN_GT_MM2
        small[0] = False
        gt = np.where(small[gt], 0, gt)
        gt = np.unique(gt, return_inverse=True)[1].ravel()
        labs = {k: e[k].astype(np.int64) for k in ("old", "new", "nosplit")}
        absorbed = absorb_fragments(e["raw_consensus"].astype(np.int64), edges, area, MIN_FRAGMENT_MM2)  # as eval_fix
        comp = components(absorbed, edges)
        for g in gaps:
            labs[f"gap{g}"] = join_close(absorbed, comp, xyz, g)
        sc = {k: score(gt, v, area) for k, v in labs.items()}
        for k, v in sc.items():
            res[group.get(aid, "all")][k].append(v)
        if any(sc[k]["f1_75"] != sc["new"]["f1_75"] or sc[k]["teeth_below_085"] != sc["new"]["teeth_below_085"]
               for k in sc if k.startswith("gap")) or sc["old"]["teeth_lost"] != sc["new"]["teeth_lost"]:
            changed.append((aid, {k: (v["f1_75"], v["teeth_below_085"], v["teeth_lost"], v["n_pred"])
                                  for k, v in sc.items()}))
    for g, sides in sorted(res.items()):
        n = len(sides["old"])
        print(f"{g} ({n} scans, GT teeth under {MIN_GT_MM2} mm2 dropped)")
        print("  side      " + " ".join(f"{k[:11]:>11s}" for k in KEYS))
        for side, rows in sides.items():
            vals = [np.sum([r[k] for r in rows]) if k in SUM else np.mean([r[k] for r in rows]) for k in KEYS]
            print(f"  {side:9s} " + " ".join(f"{v:11.4f}" for v in vals))
    print("\nscans where a gap side differs from new, or old and new lose different numbers of teeth "
          "(F1, below 0.85, lost, n_pred):")
    for aid, v in changed:
        print(f"  {aid} {group.get(aid, '')}: {v}")


if __name__ == "__main__":
    main()
