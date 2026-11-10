"""E11 라벨 없는 inhouse 스캔 전체: PurePTv3 instance가 어디서 무너지는지 정답 없이 찾는다.

스캔마다
  n_ptv3, n_s3d       찾은 치아 수 (PurePTv3: TTA8+kNN+위치+정중선 규칙+조각 정리 = E10의 H, segmentation3d instance_ios TTA x8 + 조각 정리)
  agree_F50, agree_mIoU  두 모델 instance끼리 (segmentation3d를 기준으로 본) F1@50, mIoU.  둘 다 맞으면 높다
  stab_F50            PurePTv3를 다른 두 자세(TTA pose 0, 3)로 따로 돌린 결과끼리 F1@50 — 자기 일관성
  conf_teeth          PurePTv3 치아 정점의 평균 최대 확률
  colour, vertices, extent_mm (bbox 가장 긴 변), teeth_share
결과: results/unlabeled_eval.csv, 일치도가 낮은 스캔은 results/render_unlabeled/에 그림.

    python /exp/scripts/unlabeled_eval.py            (/data/ios/inhouse/raw 전체, 라벨 있는 12개 포함)
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ["/exp/scripts"]
# instance_eval.py의 함수들을 그대로 쓴다 (모듈 맨 아래의 실행부는 건너뛴다)
_src = Path("/exp/scripts/instance_eval.py").read_text(encoding="utf-8")
exec(_src.split("# ------------------------------------------------------------------ run")[0])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_R = Path("/exp/results/render_unlabeled")
OUT_R.mkdir(parents=True, exist_ok=True)
paths = sorted(p for p in RAW.glob("INHOUSE*.ply") if not p.name.endswith(".labeled.ply"))
if len(sys.argv) > 1:
    paths = paths[:int(sys.argv[1])]


def ptv3_instances(xyz, nrm, rgb, jaw, edges, area, poses):
    acc, rep_conf = 0, None
    for r in poses:
        rep, p, inv = rep_probs(xyz, nrm, rgb, jaw, r)
        acc = acc + upsample(xyz, rep, p, inv, "knn")
    prob = acc / len(poses)
    lab = cleanup(with_rule(pos9(prob).argmax(1), xyz.astype(np.float64), nrm), edges, area)
    return components(lab, edges), prob


def pair_score(ref, pr, area):
    G, P = ref.max(), pr.max()
    if G == 0 or P == 0:
        return 0.0, 0.0
    both = (ref > 0) & (pr > 0)
    inter = coo_matrix((area[both], (ref[both] - 1, pr[both] - 1)), shape=(G, P)).toarray()
    ga = np.bincount(ref, weights=area, minlength=G + 1)[1:]
    pa = np.bincount(pr, weights=area, minlength=P + 1)[1:]
    iou = inter / np.maximum(ga[:, None] + pa[None, :] - inter, 1e-9)
    r, c = linear_sum_assignment(-iou)
    return 2 * int((iou[r, c] >= 0.5).sum()) / (G + P), float(iou.max(1).mean())


PAL = plt.get_cmap("tab20")


def render(name, xyz, nrm, insts, titles):
    t = insts[0] > 0
    z = nrm[t].mean(0) if t.any() else np.array([0, 0, 1.0])
    z /= np.linalg.norm(z)
    c = xyz - xyz.mean(0)
    x = np.linalg.svd(c - np.outer(c @ z, z), full_matrices=False)[2][0]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    q = c @ np.stack([x, np.cross(z, x), z]).T
    idx = np.random.default_rng(0).choice(len(q), min(len(q), 70000), replace=False)
    o = idx[np.argsort(q[idx, 2])]
    fig, axes = plt.subplots(1, len(insts), figsize=(6 * len(insts), 5))
    for ax, inst, title in zip(axes, insts, titles):
        col = np.where((inst[o] > 0)[:, None], np.array([PAL(i % 20)[:3] for i in inst[o]]), 0.85)
        ax.scatter(q[o, 0], q[o, 1], c=col, s=0.5, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_R / f"{name}.png", dpi=70)
    plt.close(fig)


rows = []
for k, path in enumerate(paths, 1):
    t0 = time.time()
    try:
        scan = LOAD.run(path)
        xyz = scan.positions.cpu().numpy().astype(np.float32)
        nrm = scan.normals.cpu().numpy().astype(np.float32)
        faces = scan.faces.cpu().numpy().astype(np.int64)
        rgb = None if scan.colours is None else (scan.colours.cpu().numpy() * 255).round().astype(np.uint8)
        edges = edges_of(faces)
        area = vertex_area(xyz.astype(np.float64), faces)
        inst, prob = ptv3_instances(xyz, nrm, None, scan.jaw, edges, area, list(ROTS))
        a, _ = ptv3_instances(xyz, nrm, None, scan.jaw, edges, area, [ROTS[0]])
        b, _ = ptv3_instances(xyz, nrm, None, scan.jaw, edges, area, [ROTS[3]])
        s3d = components(cleanup(S3D_INST.segment(scan).instances.astype(np.int64), edges, area), edges)
    except Exception as e:  # noqa: BLE001
        print(f"[{k}/{len(paths)}] {path.stem}: {type(e).__name__}: {e}", flush=True)
        continue
    f50, miou = pair_score(s3d, inst, area)
    stab, _ = pair_score(a, b, area)
    tooth = inst > 0
    row = {"name": path.stem, "jaw": scan.jaw, "vertices": len(xyz), "colour": rgb is not None,
           "extent_mm": round(float(np.ptp(xyz, 0).max()), 1), "n_ptv3": int(inst.max()), "n_s3d": int(s3d.max()),
           "agree_F50": round(f50, 3), "agree_mIoU": round(miou, 3), "stab_F50": round(stab, 3),
           "conf_teeth": round(float(prob.max(1)[tooth].mean()), 3) if tooth.any() else 0.0,
           "teeth_share": round(float(area[tooth].sum() / area.sum()), 3)}
    rows.append(row)
    print(f"[{k}/{len(paths)}] " + json.dumps(row) + f" {time.time() - t0:.0f}s", flush=True)
    if f50 < 0.7:
        render(path.stem, xyz.astype(np.float64), nrm.astype(np.float64), [s3d, inst, a],
               [f"{path.stem} segmentation3d instance_ios ({s3d.max()})", f"PurePTv3 TTA8 ({inst.max()})",
                f"PurePTv3 one pose ({a.max()})"])

with open("/exp/results/unlabeled_eval.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
arr = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
print(f"\n{len(rows)} scans | agree F50 median {np.median(arr('agree_F50')):.3f}, <0.7: {(arr('agree_F50') < 0.7).sum()} "
      f"| stab F50 median {np.median(arr('stab_F50')):.3f}, <0.7: {(arr('stab_F50') < 0.7).sum()}")
print(f"teeth PurePTv3 - segmentation3d: mean {np.mean(arr('n_ptv3') - arr('n_s3d')):+.2f}, "
      f"fewer {(arr('n_ptv3') < arr('n_s3d')).sum()}, more {(arr('n_ptv3') > arr('n_s3d')).sum()}")
for col in (True, False):
    sel = [r for r in rows if r["colour"] == col]
    if sel:
        print(f"colour={col}: {len(sel)} scans, agree F50 {np.median([r['agree_F50'] for r in sel]):.3f}, "
              f"stab {np.median([r['stab_F50'] for r in sel]):.3f}")
