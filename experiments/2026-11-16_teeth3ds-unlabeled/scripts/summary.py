"""Print one row per probed asset from probe.py's JSON lines.  usage: python summary.py a.jsonl [b.jsonl ...]"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
for fn in sys.argv[1:]:
    print(f"== {fn}")
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        c, g = r.get("colour", {}), r.get("grey", {})
        extra = [p for p in r["props"] if p not in ("x", "y", "z", "red", "green", "blue")]
        print(r["asset_id"], r["jaw"][0], r["n_vertices"], "ext", r["extent"], "ctr", r["centre"],
              "edge", r.get("edge_mm_median"), "rgb", r.get("rgb_mean"), r.get("rgb_std"),
              "grey", r.get("grey_share"), "ncol", r.get("n_colours"), "SAVED" if r["saved"] else "",
              "| C", c.get("instances"), c.get("teeth"), "| G", g.get("instances"), g.get("teeth"), extra)
