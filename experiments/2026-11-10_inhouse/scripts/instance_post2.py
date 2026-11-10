"""E10c 재학습 없이 PurePTv3 instance의 병합/분할을 줄일 수 있는가.

  H            E10의 최선 (TTA8 + kNN 업샘플 + 위치합 + 정중선 규칙 + 조각 정리)
  H+colour     같은데 실제 색을 넣음 (색 있는 스캔만 차이)
  H16          TTA 16자세
  H+knnclean   조각 정리를 mesh 면 대신 정점 kNN 그래프(k=8)로 — 추론 서버는 면을 받지 않는다
  H+erode.2c   치아를 잇몸 쪽에서 0.2 mm 깎은 뒤 조각 정리
  H+sharemerge 맞닿은 두 instance가 작은 쪽 둘레의 T 넘게 경계를 공유하면 합친다 (한 치아가 쪼개진 것).
               T는 정답의 실제 이웃 치아 쌍에서 잰 공유 비율 분포로 정한다.
"""
import sys
from pathlib import Path

_src = Path("/exp/scripts/instance_eval.py").read_text(encoding="utf-8")
exec(_src.split("# ------------------------------------------------------------------ run")[0])

ROTS16 = tta_rotations(16).double()
pred = np.load("/exp/results/instance_pred.npz")


def knn_edges(xyz, k=8):
    _, i = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    e = np.stack([np.repeat(np.arange(len(xyz)), k), i[:, 1:].ravel()], 1)
    return np.unique(np.sort(e, 1), axis=0)


def shared_fractions(inst, edges):
    """맞닿은 instance 쌍마다 (a, b, 공유 경계 / 작은 쪽 둘레).  둘레 = 다른 라벨(잇몸 포함)과 닿은 변 수."""
    a, b = inst[edges[:, 0]], inst[edges[:, 1]]
    cross = a != b
    per = np.bincount(np.concatenate([a[cross], b[cross]]), minlength=inst.max() + 1)
    tt = cross & (a > 0) & (b > 0)
    lo, hi = np.minimum(a[tt], b[tt]), np.maximum(a[tt], b[tt])
    pairs, cnt = np.unique(lo.astype(np.int64) * 100000 + hi, return_counts=True)
    out = []
    for p, c in zip(pairs, cnt):
        x, y = divmod(int(p), 100000)
        out.append((x, y, c / max(min(per[x], per[y]), 1)))
    return out


def share_merge(inst, edges, t):
    inst = inst.copy()
    while True:
        best = max(((x, y, f) for x, y, f in shared_fractions(inst, edges) if f > t), key=lambda z: z[2], default=None)
        if best is None:
            return components(inst, edges)
        inst[inst == best[1]] = best[0]


# 1) 정답 이웃 치아 쌍의 공유 비율 (기준 T를 정하려고)
gt_fracs, data = [], []
for f in sorted(GT.glob("INHOUSE*.npz")):
    d = np.load(f)
    scan = LOAD.run(RAW / f"{f.stem}.ply")
    faces = scan.faces.cpu().numpy().astype(np.int64)
    edges = edges_of(faces)
    gt = pred[f"{f.stem}__GT"].astype(np.int64)
    gt_fracs += [fr for _, _, fr in shared_fractions(gt, edges)]
    data.append((f.stem, d, scan, faces, edges))
gt_fracs = np.array(gt_fracs)
print(f"GT neighbour pairs: {len(gt_fracs)}, shared fraction median {np.median(gt_fracs):.3f}, "
      f"p90 {np.percentile(gt_fracs, 90):.3f}, max {gt_fracs.max():.3f}", flush=True)
T = float(max(0.35, gt_fracs.max() * 1.25))
print(f"share-merge threshold T = {T:.3f}", flush=True)

rows = []
for name, d, scan, faces, edges in data:
    jaw = str(d["jaw"])
    xyz, nrm = d["xyz"], d["normals"]
    x64 = xyz.astype(np.float64)
    rgb = d["rgb"] if "rgb" in d.files else None
    area = vertex_area(x64, faces)
    gt = pred[f"{name}__GT"].astype(np.int64)
    H = pred[f"{name}__H_tta8+knn_pos9+rule+clean"].astype(np.int64)
    v = {"H": H}
    if rgb is not None:
        p = vertex_probs(xyz, nrm, rgb, jaw, tta=True, up="knn")
        v["H+colour"] = components(cleanup(with_rule(pos9(p).argmax(1), x64, nrm), edges, area), edges)
    else:
        v["H+colour"] = H
    acc = 0
    for r in ROTS16:
        rep, pr, inv = rep_probs(xyz, nrm, None, jaw, r)
        acc = acc + upsample(xyz, rep, pr, inv, "knn")
    p16 = acc / len(ROTS16)
    lab16 = with_rule(pos9(p16).argmax(1), x64, nrm)
    v["H16"] = components(cleanup(lab16, edges, area), edges)
    p8 = vertex_probs(xyz, nrm, None, jaw, tta=True, up="knn")
    lab8 = with_rule(pos9(p8).argmax(1), x64, nrm)
    ke = knn_edges(x64)
    v["H+knnclean"] = components(cleanup(lab8, ke, area), edges)
    tooth = H > 0
    dist = cKDTree(x64[~tooth]).query(x64[tooth], workers=-1)[0]
    er = H.copy()
    er[np.flatnonzero(tooth)[dist < 0.2]] = 0
    v["H+erode.2c"] = components(cleanup(er, edges, area), edges)
    v["H+sharemerge"] = share_merge(H, edges, T)
    res = {k: score(gt, x, area, x64, edges) for k, x in v.items()}
    rows.append(res)
    print(name, " | ".join(f"{k} F50 {r['F50']:.2f} F75 {r['F75']:.2f} mIoU {r['mIoU']:.3f} m{r['merge']}s{r['split']}"
                           for k, r in res.items()), flush=True)

print(f"\n{'variant':14s}  F50   F75  mIoU  merge split  margin GT>P P>GT  inter GT>P P>GT  miss  extra(mm2)")
for k in rows[0]:
    vals = [r[k] for r in rows]
    mean = lambda m: np.nanmean([x[m] for x in vals])  # noqa: E731
    tot = lambda m: int(sum(x[m] for x in vals))  # noqa: E731
    print(f"{k:14s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
          f"   {mean('margin_gt2p'):.2f} {mean('margin_p2gt'):.2f}      {mean('inter_gt2p'):.2f} {mean('inter_p2gt'):.2f}"
          f"  {mean('miss_mm2'):5.0f} {mean('extra_mm2'):5.0f}")
