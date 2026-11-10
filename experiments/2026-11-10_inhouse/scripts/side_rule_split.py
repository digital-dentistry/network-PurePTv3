"""E13 정중선 규칙이 치아를 쪼개는가: 같은 확률(TTA8 + kNN)에서 instance를 세 가지로 뽑아 비교.

  H   위치합 + 정중선 규칙 (늘) + 조각 정리           — E10/E12의 H
  Hn  위치합 + 조각 정리 (좌우 없음)                 — 전악에서는 11/21처럼 같은 위치 둘이 맞닿아 합쳐질 수 있다
  J   위치합 + 정중선 규칙은 경고 없는 "midline"일 때만 (양쪽에 위치가 4개 이상) + 조각 정리
데이터: 인하우스 라벨 12개 (mesh 면, 면적 가중) + 공개 검증 40개 전악 / quadrant / half (kNN, 정점 수).
"""
import sys
from dataclasses import replace
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
sys.argv = sys.argv[:1]
_src = Path("/exp/scripts/instance_eval.py").read_text(encoding="utf-8")
exec(_src.split("# ------------------------------------------------------------------ run")[0])

from dental_ai.dataset.algorithm.augmentation.augmentation import (AugmentationInput, AugmentationOutput,  # noqa: E402
                                                                   AugmentationParams)
from dental_ai.dataset.algorithm.augmentation.segment_crop_cpu import segment_crop_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import ToothSegDataset, _name_generator, patient_split  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402


def knn_edges(xyz, k=8):
    _, i = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    e = np.stack([np.repeat(np.arange(len(xyz)), k), i[:, 1:].ravel()], 1)
    return np.unique(np.sort(e, 1), axis=0)


def side(pos_lab, xyz, nrm, only_confident):
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)), VOX), ds)
    rep = ds.representative.numpy()
    coord = xyz[rep].astype(np.float64)
    n = nrm[rep].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    second_rep, method, warnings = decide_quadrant(coord - coord.mean(0), n, pos_lab[rep])
    if only_confident and (method != "midline" or warnings):
        return np.where(pos_lab > 0, pos_lab, 0), method
    second = second_rep[cKDTree(coord).query(xyz, workers=-1)[1]]
    return np.where(pos_lab > 0, pos_lab + 8 * second, 0), method


def derive(prob, xyz, nrm, edges, area, min_area):
    p9 = pos9(prob).argmax(1)
    x64 = xyz.astype(np.float64)
    H, m = side(p9, x64, nrm, False)
    J, _ = side(p9, x64, nrm, True)
    return {"H": components(cleanup(H, edges, area, min_area), edges),
            "Hn": components(cleanup(p9, edges, area, min_area), edges),
            "J": components(cleanup(J, edges, area, min_area), edges)}, m


rows = {}
# 인하우스
for f in sorted(GT.glob("INHOUSE*.npz")):
    d = np.load(f)
    scan = LOAD.run(RAW / f"{f.stem}.ply")
    faces = scan.faces.cpu().numpy().astype(np.int64)
    edges = edges_of(faces)
    xyz, nrm = d["xyz"], d["normals"]
    area = vertex_area(xyz.astype(np.float64), faces)
    fdi = d["fdi"].astype(np.int64)
    gt = np.where(fdi > 0, np.unique(fdi, return_inverse=True)[1], 0)
    v, m = derive(vertex_probs(xyz, nrm, None, str(d["jaw"]), tta=True, up="knn"), xyz, nrm, edges, area, MIN_AREA)
    res = {k: score(gt, x, area, xyz.astype(np.float64), edges) for k, x in v.items()}
    rows.setdefault("inhouse", []).append(res)
    print(f.stem, m, " | ".join(f"{k} F75 {r['F75']:.2f} mIoU {r['mIoU']:.3f} m{r['merge']}s{r['split']}"
                                 for k, r in res.items()), flush=True)

# 공개 검증
caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))
_, val = patient_split(names, 0.1, seed=0)
val = [n for n in val if not (Path("/data/ios/3d_iosseg/cache") / f"{n}.npz").exists()][::2][:N]
ds = ToothSegDataset(caches, val)
for i, name in enumerate(val):
    s, jaw_upper = ds.load(name)
    for mode in ("full", "quadrant", "half"):
        smp = s
        if mode != "full":
            out = AugmentationOutput()
            if not segment_crop_cpu(AugmentationInput(s, replace(AugmentationParams(), crop_mode=mode),
                                                      _name_generator(name)), out):
                continue
            smp = out.sample
        xyz, nrm = smp.positions.numpy().astype(np.float32), smp.normals.numpy().astype(np.float32)
        ins = smp.instances.numpy()
        gt = np.unique(ins, return_inverse=True)[1].astype(np.int64) + (0 if (ins == 0).any() else 1)
        edges = knn_edges(xyz.astype(np.float64))
        area = np.ones(len(xyz))
        v, m = derive(vertex_probs(xyz, nrm, None, "upper" if jaw_upper else "lower", tta=True, up="knn"),
                      xyz, nrm, edges, area, 60)
        res = {k: score(gt, x, area, xyz.astype(np.float64), edges) for k, x in v.items()}
        rows.setdefault(f"public {mode}", []).append(res)
        print(f"[{i + 1}/{len(val)}] {name} {mode} {m}: " + " | ".join(
            f"{k} F75 {r['F75']:.2f} m{r['merge']}s{r['split']}" for k, r in res.items()), flush=True)

for group, rs in rows.items():
    print(f"\n== {group} ({len(rs)} scans, {sum(r['H']['n_gt'] for r in rs)} teeth)")
    print(f"   {'method':4s}  F50   F75  mIoU  merge split  inter GT>P P>GT")
    for k in rs[0]:
        vals = [r[k] for r in rs]
        mean = lambda m: np.nanmean([x[m] for x in vals])  # noqa: E731
        tot = lambda m: int(sum(x[m] for x in vals))  # noqa: E731
        print(f"   {k:4s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
              f"   {mean('inter_gt2p'):.2f} {mean('inter_p2gt'):.2f}")
