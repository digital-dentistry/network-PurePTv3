"""Which painted colours hurt?  Per palette colour: the mean best IoU of that tooth with a predicted tooth (same
FDI = one tooth), with the colour sent and without it, and the share of the tooth's area predicted as gingiva.

usage: python per_colour.py <out dir> <probe jsonl>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colour_regions import vertex_area  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    out, probe = Path(sys.argv[1]), sys.argv[2]
    acc = defaultdict(lambda: defaultdict(list))
    for line in open(probe, encoding="utf-8"):
        r = json.loads(line)
        d = np.load(out / f"{r['asset_id']}.npz")
        w = vertex_area(d["xyz"], d["faces"])
        cols, inv = np.unique(d["rgb"], axis=0, return_inverse=True)
        inv = inv.ravel()
        ging = int(np.argmax(np.bincount(inv, weights=w)))
        for c in range(len(cols)):
            if c == ging:
                continue
            m = inv == c
            key = (r["jaw"], tuple(int(x) for x in cols[c]))
            for mode in ("colour", "grey"):
                p = d[f"pred_{mode}_fdi"].astype(np.int64)
                best = 0.0
                for j in np.unique(p[m]):
                    if j:
                        q = p == j
                        best = max(best, w[m & q].sum() / w[m | q].sum())
                acc[key][f"{mode}_iou"].append(best)
                acc[key][f"{mode}_ging"].append(w[m & (p == 0)].sum() / w[m].sum())
    print("jaw rgb n | IoU colour / grey | share called gingiva colour / grey")
    for key in sorted(acc, key=lambda k: np.mean(acc[k]["colour_iou"])):
        a = acc[key]
        print(key[0], key[1], len(a["colour_iou"]), "|", f"{np.mean(a['colour_iou']):.2f} / {np.mean(a['grey_iou']):.2f}",
              "|", f"{np.mean(a['colour_ging']):.2f} / {np.mean(a['grey_ging']):.2f}")


if __name__ == "__main__":
    main()
