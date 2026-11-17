"""Summary of eval_fix.py output: old (deployed) / fusion_only / speck_only / new, per dataset, and the scans that
changed.  Repeat 0 (input as is) for the table; all repeats for "lost teeth" stability.

usage: python compare.py <eval jsonl> [<file with "asset dataset" per line>]
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SIDES = ["old", "fusion_only", "speck_only", "new"]
KEYS = ["f1_75", "iou_mean", "iou_min", "teeth_below_085", "teeth_lost", "tooth_to_gum_mm2", "gum_to_tooth_mm2",
        "merges", "splits", "n_pred"]
SUM = {"teeth_below_085", "teeth_lost", "merges", "splits"}


def main():
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    group = {}
    if len(sys.argv) > 2:
        group = dict(line.split()[:2] for line in open(sys.argv[2], encoding="utf-8") if line.strip())
    first = [r for r in rows if r.get("repeat", 0) == 0]
    sides = SIDES + [k for k in ("nosplit",) if k in rows[0]]
    by = defaultdict(list)
    for r in first:
        by[group.get(r["asset_id"], "all")].append(r)
    print("dataset       n  side         " + " ".join(f"{k[:12]:>12s}" for k in KEYS))
    for g, rs in sorted(by.items()):
        for side in sides:
            vals = []
            for k in KEYS:
                v = [r[side][k] for r in rs if r[side][k] is not None]
                vals.append(np.sum(v) if k in SUM else np.mean(v))
            print(f"{g[:12]:12s} {len(rs):3d}  {side:11s} " + " ".join(f"{v:12.4f}" for v in vals))
        print(f"{'':12s}      teeth in GT: {sum(r['old']['n_gt'] for r in rs)}")

    reps = max(r.get("repeat", 0) for r in rows) + 1
    if reps > 1:
        print(f"\nlost teeth (IoU < 0.5) over {reps} runs per scan (runs 2.. with normal noise sd 0.002):")
        for side in sides:
            per = defaultdict(list)
            for r in rows:
                per[r["asset_id"]].append(r[side]["teeth_lost"])
            flaky = {a: v for a, v in per.items() if len(set(v)) > 1}
            print(f"  {side:11s} total {sum(sum(v) for v in per.values()):4d}   scans where it varies between runs: "
                  f"{len(flaky)} {dict(list(flaky.items())[:6])}")

    for a_side, b_side in [("old", "new")] + [("new", "nosplit")] * ("nosplit" in sides):
        diff(first, group, a_side, b_side)


def diff(first, group, a_side, b_side):
    print(f"\nscans where {a_side} and {b_side} differ (repeat 0): F1, IoU min, teeth below 0.85, n pred")
    for r in first:
        b, n = r[a_side], r[b_side]
        if (b["f1_75"] != n["f1_75"] or b["teeth_below_085"] != n["teeth_below_085"]
                or abs((b["iou_min"] or 0) - (n["iou_min"] or 0)) > 0.02):
            print(f"  {r['asset_id']} {group.get(r['asset_id'], ''):12s} F1 {b['f1_75']:.3f} -> {n['f1_75']:.3f}"
                  f"  IoU min {b['iou_min']} -> {n['iou_min']}  below.85 {b['teeth_below_085']} -> {n['teeth_below_085']}"
                  f"  n {b['n_pred']} -> {n['n_pred']} (gt {b['n_gt']})  changed {r['changed_share']:.4f}")


if __name__ == "__main__":
    main()
