"""E10b instance_eval.py가 저장한 예측(results/instance_pred.npz)에 후처리만 바꿔 다시 잰다 + 병합/분할 그림.

  A+clean         대시보드 FDI 영역에 조각 정리만 (화면에 바로 넣을 수 있는 최소 변경)
  S1+clean        segmentation3d instance_ios에도 같은 조각 정리 (공정 비교)
  H+erode d       PurePTv3 최선(H)의 치아 영역을 잇몸 쪽 경계에서 d mm 깎기 (치아가 잇몸으로 넘치는지)
  치아 면적 대비 miss / extra (%)
그림: results/render_instance/<scan>.png (정답 | PurePTv3 H | segmentation3d S1+clean), 병합/분할이 있는 스캔만.
"""
import sys
from pathlib import Path

_src = Path("/exp/scripts/instance_eval.py").read_text(encoding="utf-8")
exec(_src.split("# ------------------------------------------------------------------ run")[0])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pred = np.load("/exp/results/instance_pred.npz")
OUT_R = Path("/exp/results/render_instance")
OUT_R.mkdir(parents=True, exist_ok=True)
PAL = plt.get_cmap("tab20")


def relabel_clean(inst, edges, area):
    """instance 번호를 라벨로 보고 조각 정리 -> 다시 connected component."""
    return components(cleanup(inst.astype(np.int64), edges, area), edges)


def erode(inst, xyz, d):
    tooth = inst > 0
    if d <= 0 or not tooth.any() or tooth.all():
        return inst
    dist = cKDTree(xyz[~tooth]).query(xyz[tooth], workers=-1)[0]
    out = inst.copy()
    out[np.flatnonzero(tooth)[dist < d]] = 0
    return out


def render(name, xyz, nrm, insts, titles):
    t = insts[0] > 0
    z = nrm[t].mean(0)
    z /= np.linalg.norm(z)
    c = xyz - xyz.mean(0)
    x = np.linalg.svd(c - np.outer(c @ z, z), full_matrices=False)[2][0]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    q = c @ np.stack([x, np.cross(z, x), z]).T
    idx = np.random.default_rng(0).choice(len(q), min(len(q), 90000), replace=False)
    o = idx[np.argsort(q[idx, 2])]
    fig, axes = plt.subplots(1, len(insts), figsize=(6.5 * len(insts), 5.5))
    for ax, inst, title in zip(axes, insts, titles):
        col = np.full((len(o), 3), 0.85)
        m = inst[o] > 0
        col[m] = np.array([PAL((i * 7) % 20)[:3] for i in inst[o][m]])
        ax.scatter(q[o, 0], q[o, 1], c=col, s=0.6, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_R / f"{name}.png", dpi=70)
    plt.close(fig)


variants = ["A+clean", "H", "S1", "S1+clean"] + [f"H+erode{d}" for d in (0.1, 0.2, 0.3)]
rows = []
for f in sorted(GT.glob("INHOUSE*.npz")):
    d = np.load(f)
    name = f.stem
    xyz = d["xyz"].astype(np.float64)
    scan = LOAD.run(RAW / f"{name}.ply")
    faces = scan.faces.cpu().numpy().astype(np.int64)
    edges = edges_of(faces)
    area = vertex_area(xyz, faces)
    gt = pred[f"{name}__GT"].astype(np.int64)
    A = pred[f"{name}__A_dashboard_FDI_regions"].astype(np.int64)
    H = pred[f"{name}__H_tta8+knn_pos9+rule+clean"].astype(np.int64)
    S1 = pred[f"{name}__S1_seg3d_instance_ios_tta8"].astype(np.int64)
    v = {"A+clean": relabel_clean(A, edges, area), "H": H, "S1": S1, "S1+clean": relabel_clean(S1, edges, area)}
    for dd in (0.1, 0.2, 0.3):
        v[f"H+erode{dd}"] = components(erode(H, xyz, dd), edges)
    res = {k: score(gt, x, area, xyz, edges) for k, x in v.items()}
    tooth_area = float(area[gt > 0].sum())
    rows.append({"name": name, "full": "0001" in name or "0002" in name, "res": res, "tooth_area": tooth_area})
    print(name, f"tooth area {tooth_area:.0f} mm2 | " + " | ".join(
        f"{k} F50 {r['F50']:.2f} mIoU {r['mIoU']:.3f} m{r['merge']}s{r['split']} miss {100 * r['miss_mm2'] / tooth_area:.1f}% "
        f"extra {100 * r['extra_mm2'] / tooth_area:.1f}%" for k, r in res.items() if k in ("H", "S1+clean", "A+clean")),
        flush=True)
    if res["H"]["merge"] or res["H"]["split"] or res["S1+clean"]["merge"] or res["S1+clean"]["split"]:
        render(name, xyz, d["normals"].astype(np.float64), [gt, H, v["S1+clean"]],
               [f"{name} human label ({gt.max()} teeth)",
                f"PurePTv3 H ({H.max()}; merge {res['H']['merge']} split {res['H']['split']})",
                f"segmentation3d S1+clean ({v['S1+clean'].max()}; merge {res['S1+clean']['merge']} "
                f"split {res['S1+clean']['split']})"])

print(f"\n{'variant':14s}  F50   F75  mIoU  merge split  margin GT>P P>GT  inter GT>P P>GT  miss%  extra%")
for k in variants:
    vals = [r["res"][k] for r in rows]
    mean = lambda m: np.nanmean([x[m] for x in vals])  # noqa: E731
    tot = lambda m: int(sum(x[m] for x in vals))  # noqa: E731
    miss = 100 * np.mean([r["res"][k]["miss_mm2"] / r["tooth_area"] for r in rows])
    extra = 100 * np.mean([r["res"][k]["extra_mm2"] / r["tooth_area"] for r in rows])
    print(f"{k:14s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
          f"   {mean('margin_gt2p'):.2f} {mean('margin_p2gt'):.2f}      {mean('inter_gt2p'):.2f} {mean('inter_p2gt'):.2f}"
          f"  {miss:5.1f}  {extra:5.1f}")
