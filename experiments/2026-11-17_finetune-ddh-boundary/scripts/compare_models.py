"""Base model against a fine-tuned one on the same test scans: eval_fix.py output of each, one post-processing side
(default "nosplit" = the deployed pipeline after 2026-11-17: fuse_consensus + absorb_fragments + keep_instances).

usage: python compare_models.py <base jsonl> <candidate jsonl> [side]
Repeat 0 only.  Prints the means / sums over the scans both files have, then the scans that moved most.
"""
import json
import sys

import numpy as np

KEYS = ["f1_75", "iou_mean", "iou_min", "teeth_below_085", "teeth_lost", "tooth_to_gum_mm2", "gum_to_tooth_mm2",
        "merges", "splits"]
SUM = {"teeth_below_085", "teeth_lost", "merges", "splits"}


def load(path, side):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("repeat", 0) == 0:
            out[r["asset_id"]] = r[side]
    return out


def main():
    side = sys.argv[3] if len(sys.argv) > 3 else "nosplit"
    base, cand = load(sys.argv[1], side), load(sys.argv[2], side)
    ids = sorted(set(base) & set(cand))
    print(f"{len(ids)} scans, side {side}")
    print("        " + " ".join(f"{k[:12]:>12s}" for k in KEYS))
    for name, rows in (("base", base), ("cand", cand)):
        vals = []
        for k in KEYS:
            v = [rows[a][k] for a in ids if rows[a][k] is not None]
            vals.append(np.sum(v) if k in SUM else np.mean(v))
        print(f"{name:7s} " + " ".join(f"{v:12.4f}" for v in vals))
    moved = sorted(ids, key=lambda a: cand[a]["iou_mean"] - base[a]["iou_mean"])
    print("\nlargest IoU-mean drops / gains (scan: base -> cand  F1, IoU mean, IoU min, tooth->gum)")
    for a in moved[:5] + moved[-5:]:
        b, c = base[a], cand[a]
        print(f"  {a}  F1 {b['f1_75']:.3f}->{c['f1_75']:.3f}  IoU {b['iou_mean']:.4f}->{c['iou_mean']:.4f}  "
              f"min {b['iou_min']}->{c['iou_min']}  t->g {b['tooth_to_gum_mm2']}->{c['tooth_to_gum_mm2']}")


if __name__ == "__main__":
    main()
