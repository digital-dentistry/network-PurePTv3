"""E15 정중선에서 나누는 치아를 중절치로 한정 (K1).  E14의 K는 공개 quadrant 크롭에서도 "확신하는" 정중선을 잡아
대구치를 나눴다 (분할 32).  정중선을 넘어 합쳐질 수 있는 것은 두 중절치(11+21, 31+41)뿐이므로, 나누는 대상을
위치 1인 instance로 한정한다.

  H, Hn, K   E14와 같다
  K1         Hn에서, 정중선 규칙이 "midline"이면 (경고가 있어도) 위치 1 instance 중 정중선 양쪽에 30~70% 걸친 것만 나눈다
  K1c        K1과 같지만 규칙이 확신할 때만 (경고 없는 midline, E14의 K와 같은 조건)
데이터와 지표는 E14와 같다 (인하우스 12, 공개 검증 40 x 전악/quadrant/half).
"""
import sys
import time
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
    """정점마다 둘째 사분면인지, 판정 방법, 경고."""
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)), VOX), ds)
    rep = ds.representative.numpy()
    coord = xyz[rep].astype(np.float64)
    n = nrm[rep].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    second_rep, method, warnings = decide_quadrant(coord - coord.mean(0), n, pos_lab[rep])
    return second_rep[cKDTree(coord).query(xyz, workers=-1)[1]], method, warnings


def split_at_midline(inst, pos, second, area, edges, only_incisor):
    out = inst.copy()
    nxt = out.max() + 1
    for k in range(1, inst.max() + 1):
        m = inst == k
        if only_incisor and np.bincount(pos[m], weights=area[m]).argmax() != 1:
            continue
        share = float(np.average(second[m], weights=area[m]))
        if 0.3 < share < 0.7:
            out[m & second] = nxt
            nxt += 1
    return components(out, edges)


def derive(prob, xyz, nrm, edges, area, min_area):
    x64 = xyz.astype(np.float64)
    p9 = pos9(prob).argmax(1)
    second, method, warnings = rule(p9, x64, nrm)
    confident = method == "midline" and not warnings
    H = components(cleanup(np.where(p9 > 0, p9 + 8 * second, 0), edges, area, min_area), edges)
    pos = cleanup(p9, edges, area, min_area)
    Hn = components(pos, edges)
    out = {"H": H, "Hn": Hn}
    out["K"] = split_at_midline(Hn, pos, second, area, edges, False) if confident else Hn
    out["K1"] = split_at_midline(Hn, pos, second, area, edges, True) if method == "midline" else Hn
    out["K1c"] = split_at_midline(Hn, pos, second, area, edges, True) if confident else Hn
    return out, method, confident


rows = {}
calls = {}
started = time.time()
for f in sorted(GT.glob("INHOUSE*.npz")):
    d = np.load(f)
    scan = LOAD.run(RAW / f"{f.stem}.ply")
    faces = scan.faces.cpu().numpy().astype(np.int64)
    edges = edges_of(faces)
    xyz, nrm = d["xyz"], d["normals"]
    area = vertex_area(xyz.astype(np.float64), faces)
    fdi = d["fdi"].astype(np.int64)
    gt = np.where(fdi > 0, np.unique(fdi, return_inverse=True)[1], 0)
    v, method, conf = derive(vertex_probs(xyz, nrm, None, str(d["jaw"]), tta=True, up="knn"), xyz, nrm, edges, area,
                             MIN_AREA)
    res = {k: score(gt, x, area, xyz.astype(np.float64), edges) for k, x in v.items()}
    rows.setdefault("inhouse", []).append(res)
    calls.setdefault("inhouse", []).append((method, conf))
    print(f.stem, method, "conf" if conf else "-", " | ".join(
        f"{k} F75 {r['F75']:.2f} m{r['merge']}s{r['split']}" for k, r in res.items()), flush=True)

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
        v, method, conf = derive(vertex_probs(xyz, nrm, None, "upper" if jaw_upper else "lower", tta=True, up="knn"),
                                 xyz, nrm, edges, area, 60)
        res = {k: score(gt, x, area, xyz.astype(np.float64), edges) for k, x in v.items()}
        rows.setdefault(f"public {mode}", []).append(res)
        calls.setdefault(f"public {mode}", []).append((method, conf))
        print(f"[{i + 1}/{len(val)}] {name} {mode}: {method} {'conf' if conf else '-'} " + " | ".join(
            f"{k} F75 {r['F75']:.2f} m{r['merge']}s{r['split']}" for k, r in res.items()), flush=True)

for group, rs in rows.items():
    c = calls[group]
    print(f"\n== {group} ({len(rs)} scans, {sum(r['H']['n_gt'] for r in rs)} teeth) "
          f"rule: midline {sum(m == 'midline' for m, _ in c)}, confident {sum(x for _, x in c)}")
    print(f"   {'method':4s}  F50   F75  mIoU  merge split  inter GT>P P>GT")
    for k in rs[0]:
        vals = [r[k] for r in rs]
        mean = lambda m: np.nanmean([x[m] for x in vals])  # noqa: E731
        tot = lambda m: int(sum(x[m] for x in vals))  # noqa: E731
        print(f"   {k:4s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
              f"   {mean('inter_gt2p'):.2f} {mean('inter_p2gt'):.2f}")
print(f"\n{time.time() - started:.0f}s")
