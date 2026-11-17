"""Summary of stages.py output: variant scores, the teeth auto-label covers only in part, and why.

usage: python summarise.py <stages jsonl> [--teeth] [--variant NAME]
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RAMP = [0, 56, 85, 113, 141, 170, 198, 226]            # red (255,k,k) / blue (k,k,255): positions 1..8
GREEN = [255, 198, 170, 141, 113, 85, 56, 28]           # lower (0,k,0): 31..38
PURPLE_G = [226, 198, 170, 141, 113, 85, 56, 0]         # lower purple (r,g,255): 41..48 by g
INCOMPLETE_IOU = 0.85                                   # a tooth below this IoU is visibly incomplete or spilled


def fdi(jaw: str, rgb) -> int:
    r, g, b = rgb
    upper = jaw == "upper"
    if r == 255 and g == b and g in RAMP:
        return (10 if upper else 30) + RAMP.index(g) + 1
    if b == 255 and r == g and r in RAMP:
        return (20 if upper else 40) + RAMP.index(r) + 1
    if r == 0 and b == 0 and g in GREEN:
        return 30 + GREEN.index(g) + 1
    if b == 255 and g in PURPLE_G:
        return 40 + PURPLE_G.index(g) + 1
    return 0


def main():
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    show_teeth = "--teeth" in sys.argv
    variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else None
    print(f"{len(rows)} scans, {sum(r['n_gt'] for r in rows)} colour teeth")
    agree = [r["server"]["same_partition"] for r in rows if r.get("server")]
    if agree:
        print(f"v0 vs running server, same partition: min {min(agree):.4f}, mean {np.mean(agree):.4f}")

    names = list(rows[0]["scores"])
    print("\nvariant              F1@75  F1=1  IoU mean  IoU min(mean)  recall  missed->gum mm2/scan  extra gum")
    for n in names:
        s = [r["scores"][n] for r in rows]
        print(f"{n:20s} {np.mean([x['f1_75'] for x in s]):.4f} {sum(x['f1_75'] == 1 for x in s):5d}"
              f"  {np.mean([x['iou_mean'] for x in s]):.4f}    {np.mean([x['iou_min'] for x in s]):.3f}"
              f"        {np.mean([x['recall_area'] for x in s]):.4f}  {np.mean([x['missed_to_gum_mm2'] for x in s]):8.1f}"
              f"             {np.mean([x['extra_from_gum_share'] for x in s]):.4f}")

    teeth = []
    for r in rows:
        for t in r["teeth"]:
            t = dict(t, asset=r["asset_id"], jaw=r["jaw"], fdi=fdi(r["jaw"], t["rgb"]))
            teeth.append(t)
    bad = [t for t in teeth if t["iou"] < INCOMPLETE_IOU]
    print(f"\nv0 teeth below IoU {INCOMPLETE_IOU}: {len(bad)} of {len(teeth)} in "
          f"{len({t['asset'] for t in bad})} scans")
    kind = defaultdict(list)
    for t in bad:
        if t["pred"] == 0 or t["recall"] < 0.3:
            k = "missed (no or tiny match)"
        elif t["missed_to_other"] > 0.15:
            k = "part goes to another tooth (merge / split / boundary)"
        elif t["missed_to_gum"] > t["extra_from_gum"]:
            k = "part called gingiva (incomplete)"
        else:
            k = "spills into gingiva"
        kind[k].append(t)
    for k, ts in kind.items():
        print(f"  {k}: {len(ts)}")

    # causes, area weighted over the tooth area called gingiva (v0), teeth that lost >= 2 mm2
    keys = ["l0_above_full_below", "mask_above_conf_below", "some_pose_above", "half_poses_above", "no_pose_above"]
    acc, tot = defaultdict(float), 0.0
    for t in teeth:
        c = t.get("cause")
        if not c or t["missed_mm2"] < 2:
            continue
        tot += t["missed_mm2"]
        for k in keys:
            acc[k] += c[k] * t["missed_mm2"]
    print(f"\ncauses of tooth area called gingiva (teeth losing >= 2 mm2, {tot:.0f} mm2 in all):")
    for k in keys:
        print(f"  {k:24s} {acc[k] / max(tot, 1e-9):.3f}")

    by_pos = defaultdict(list)
    for t in teeth:
        by_pos[t["fdi"] % 10 if t["fdi"] else 0].append(t)
    print("\nposition  n   IoU mean  below %.2f  recall mean  missed->gum mean" % INCOMPLETE_IOU)
    for p in sorted(by_pos):
        ts = by_pos[p]
        print(f"   {p}     {len(ts):3d}  {np.mean([t['iou'] for t in ts]):.3f}     {sum(t['iou'] < INCOMPLETE_IOU for t in ts):3d}"
              f"        {np.mean([t['recall'] for t in ts]):.3f}       {np.mean([t['missed_to_gum'] for t in ts]):.3f}")

    if show_teeth:
        print("\nasset            jaw   fdi  area  IoU   recall ->gum ->other +gum  missed mm2  causes (l0>full, mask>conf, pose>=1, pose>=4, none) poses_found")
        for t in sorted(bad, key=lambda t: t["iou"]):
            c = t.get("cause") or {}
            cs = " ".join(f"{c.get(k, 0):.2f}" for k in keys)
            print(f"{t['asset']} {t['gt']:2d} {t['jaw']:5s} {t['fdi']:3d} {t['area']:6.1f} {t['iou']:.2f}  {t['recall']:.2f}  {t['missed_to_gum']:.2f}"
                  f"  {t['missed_to_other']:.2f}  {t['extra_from_gum']:.2f}   {t['missed_mm2']:6.1f}     {cs}  {c.get('poses_found', '')}")
    if variant:
        print(f"\nper scan: v0_deployed vs {variant} (F1, IoU min), scans that differ")
        for r in rows:
            a, b = r["scores"]["v0_deployed"], r["scores"][variant]
            if abs(a["f1_75"] - b["f1_75"]) > 1e-9 or abs(a["iou_min"] - b["iou_min"]) > 0.02:
                print(f"  {r['asset_id']} {r['jaw']:5s} F1 {a['f1_75']:.3f} -> {b['f1_75']:.3f}  "
                      f"IoU min {a['iou_min']:.2f} -> {b['iou_min']:.2f}  n {a['n_pred']} -> {b['n_pred']} (gt {r['n_gt']})")


if __name__ == "__main__":
    main()
