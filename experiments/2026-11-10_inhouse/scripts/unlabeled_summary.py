"""E11 요약: results/unlabeled_eval.csv를 부분/전악(bbox 긴 변 55 mm 기준)과 색 유무로 나눈다."""
import csv
import statistics as st

rows = list(csv.DictReader(open("/exp/results/unlabeled_eval.csv")))
for r in rows:
    for k in ("extent_mm", "agree_F50", "agree_mIoU", "stab_F50", "conf_teeth", "n_ptv3", "n_s3d"):
        r[k] = float(r[k])
    r["kind"] = "full" if r["extent_mm"] >= 55 else "partial"
print(f"{'group':22s} {'n':>4s} {'agreeF50':>9s} {'<0.7':>5s} {'stabF50':>8s} {'<0.7':>5s} {'teeth P-S':>9s} {'P>S':>4s} {'P<S':>4s} {'conf':>5s}")
for kind in ("partial", "full"):
    for col in ("True", "False", None):
        g = [r for r in rows if r["kind"] == kind and (col is None or r["colour"] == col)]
        if not g:
            continue
        diff = [r["n_ptv3"] - r["n_s3d"] for r in g]
        label = f"{kind} colour={col if col else 'all'}"
        print(f"{label:22s} {len(g):4d} {st.median(r['agree_F50'] for r in g):9.3f} {sum(r['agree_F50'] < .7 for r in g):5d} "
              f"{st.median(r['stab_F50'] for r in g):8.3f} {sum(r['stab_F50'] < .7 for r in g):5d} {st.mean(diff):+9.2f} "
              f"{sum(d > 0 for d in diff):4d} {sum(d < 0 for d in diff):4d} {st.median(r['conf_teeth'] for r in g):5.3f}")
print("\nlowest agreement:")
for r in sorted(rows, key=lambda r: r["agree_F50"])[:12]:
    print(f"  {r['name']} {r['kind']:7s} colour={r['colour']:5s} PurePTv3 {int(r['n_ptv3']):2d} seg3d {int(r['n_s3d']):2d} "
          f"agree {r['agree_F50']:.2f} stab {r['stab_F50']:.2f} conf {r['conf_teeth']:.2f}")
