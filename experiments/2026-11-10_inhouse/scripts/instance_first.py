"""E14 instance 먼저, 좌우는 치아 단위로 (K).  정중선 규칙이 점마다 자르면서 생기는 분할(E13의 H)을 없애고,
좌우를 무시해서 생기는 정중선 병합(E13의 Hn, 11+21)만 따로 푼다.

  K   위치합 + 조각 정리 -> instance.  정중선 규칙이 확신할 때만 (경고 없는 "midline": 양쪽에 위치 4개 이상)
      정중선 양쪽에 각각 30% 넘게 걸친 instance를 정중선에서 둘로 나눈다.  좌우 번호는 이후 instance마다 다수결로
      정하면 된다 (여기서는 instance만 잰다).
H, Hn은 E13와 같다.  데이터도 E13와 같다 (인하우스 12, 공개 검증 40 x 전악/quadrant/half).
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


def rule(pos_lab, xyz, nrm):
    """정점마다 둘째 사분면인지, 그리고 규칙이 확신하는지 (경고 없는 midline)."""
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)), VOX), ds)
    rep = ds.representative.numpy()
    coord = xyz[rep].astype(np.float64)
    n = nrm[rep].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    second_rep, method, warnings = decide_quadrant(coord - coord.mean(0), n, pos_lab[rep])
    return second_rep[cKDTree(coord).query(xyz, workers=-1)[1]], method == "midline" and not warnings


def derive(prob, xyz, nrm, edges, area, min_area):
    x64 = xyz.astype(np.float64)
    p9 = pos9(prob).argmax(1)
    second, confident = rule(p9, x64, nrm)
    H = components(cleanup(np.where(p9 > 0, p9 + 8 * second, 0), edges, area, min_area), edges)
    Hn = components(cleanup(p9, edges, area, min_area), edges)
    K = Hn.copy()
    if confident:
        nxt = K.max() + 1
        for k in range(1, Hn.max() + 1):
            m = Hn == k
            share = float(np.average(second[m], weights=area[m]))
            if 0.3 < share < 0.7:
                K[m & second] = nxt
                nxt += 1
        K = components(K, edges)
    return {"H": H, "Hn": Hn, "K": K}


rows = {}
for f in sorted(GT.glob("INHOUSE*.npz")):
    d = np.load(f)
    scan = LOAD.run(RAW / f"{f.stem}.ply")
    faces = scan.faces.cpu().numpy().astype(np.int64)
    edges = edges_of(faces)
    xyz, nrm = d["xyz"], d["normals"]
    area = vertex_area(xyz.astype(np.float64), faces)
    fdi = d["fdi"].astype(np.int64)
    gt = np.where(fdi > 0, np.unique(fdi, return_inverse=True)[1], 0)
    v = derive(vertex_probs(xyz, nrm, None, str(d["jaw"]), tta=True, up="knn"), xyz, nrm, edges, area, MIN_AREA)
    res = {k: score(gt, x, area, xyz.astype(np.float64), edges) for k, x in v.items()}
    rows.setdefault("inhouse", []).append(res)
    print(f.stem, " | ".join(f"{k} F75 {r['F75']:.2f} m{r['merge']}s{r['split']}" for k, r in res.items()), flush=True)

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
        v = derive(vertex_probs(xyz, nrm, None, "upper" if jaw_upper else "lower", tta=True, up="knn"),
                   xyz, nrm, edges, area, 60)
        res = {k: score(gt, x, area, xyz.astype(np.float64), edges) for k, x in v.items()}
        rows.setdefault(f"public {mode}", []).append(res)
        print(f"[{i + 1}/{len(val)}] {name} {mode}: " + " | ".join(
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
