"""E15b 정중선에서 나눌 중절치 고르기: 폭으로 합쳐진 두 중절치(11+21)와 중절치 하나를 가를 수 있는가.

E15의 K1은 공개 quadrant 크롭에서 분할이 27건 남았다.  중절치가 하나뿐인 스캔에서 정중선(중절치 중앙값)이
그 중절치 한가운데를 지나기 때문으로 보인다.  두 중절치가 합쳐진 instance는 악궁 방향(view frame x) 폭이 중절치
하나의 두 배쯤 된다 (상악 중절치 약 8.5 mm, 하악 약 5.5 mm).

K1의 나누기 후보 (정중선 규칙 "midline", 위치 1, 정중선 양쪽 30~70%)마다 다음을 적는다.
  truth   정답 치아 중 이 instance가 30% 넘게 덮은 수 (2 이상이면 나누는 게 맞다)
  width   x 폭 (면적 가중 5~95 백분위, mm)
  ratio   instance 면적 / 같은 스캔의 위치 2, 3 instance 중 가장 큰 면적
그리고 폭 기준별로 K1w(폭이 기준보다 넓을 때만 나눔)의 F75를 잰다.
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
from toothseg import view_frame  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402

WIDTHS = {"upper": [0, 10, 11, 12, 13, 14], "lower": [0, 7, 7.5, 8, 9, 10]}   # 0 = K1 (폭 무시)


def knn_edges(xyz, k=8):
    _, i = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    e = np.stack([np.repeat(np.arange(len(xyz)), k), i[:, 1:].ravel()], 1)
    return np.unique(np.sort(e, 1), axis=0)


def rule(pos_lab, xyz, nrm):
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)), VOX), ds)
    rep = ds.representative.numpy()
    coord = xyz[rep].astype(np.float64)
    n = nrm[rep].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    centre = coord.mean(0)
    second_rep, method, warnings = decide_quadrant(coord - centre, n, pos_lab[rep])
    xaxis = view_frame(coord - centre, n, pos_lab[rep])[0]
    return second_rep[cKDTree(coord).query(xyz, workers=-1)[1]], method, (xyz - centre) @ xaxis


def run(prob, xyz, nrm, edges, area, min_area, gt, jaw):
    x64 = xyz.astype(np.float64)
    p9 = pos9(prob).argmax(1)
    second, method, x = rule(p9, x64, nrm)
    pos = cleanup(p9, edges, area, min_area)
    Hn = components(pos, edges)
    ga = np.bincount(gt, weights=area, minlength=gt.max() + 1)
    big23 = max([area[Hn == k].sum() for k in range(1, Hn.max() + 1) if pos[Hn == k][0] in (2, 3)], default=np.nan)
    cands = []
    if method == "midline":
        for k in range(1, Hn.max() + 1):
            m = Hn == k
            if pos[m][0] != 1:
                continue
            share = float(np.average(second[m], weights=area[m]))
            if not 0.3 < share < 0.7:
                continue
            cover = np.bincount(gt[m], weights=area[m], minlength=len(ga))[1:] / np.maximum(ga[1:], 1e-9)
            lo, hi = np.percentile(x[m], [5, 95])          # 정점 수 가중 (면적 차이는 작다)
            cands.append({"k": k, "truth": int((cover > 0.3).sum()), "width": float(hi - lo),
                          "ratio": float(area[m].sum() / big23)})
    out = {}
    for w in WIDTHS[jaw]:
        K = Hn.copy()
        nxt = K.max() + 1
        for c in cands:
            if c["width"] > w:
                K[(Hn == c["k"]) & second] = nxt
                nxt += 1
        out[f"w{w}"] = components(K, edges)
    return out, cands


rows, cand_rows = {}, []
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
    jaw = str(d["jaw"])
    v, cands = run(vertex_probs(xyz, nrm, None, jaw, tta=True, up="knn"), xyz, nrm, edges, area, MIN_AREA, gt, jaw)
    rows.setdefault(("inhouse", jaw), []).append({k: score(gt, x, area, xyz.astype(np.float64), edges)
                                                  for k, x in v.items()})
    cand_rows += [dict(c, group="inhouse", jaw=jaw, scan=f.stem) for c in cands]
    print(f.stem, cands, flush=True)

caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))
_, val = patient_split(names, 0.1, seed=0)
val = [n for n in val if not (Path("/data/ios/3d_iosseg/cache") / f"{n}.npz").exists()][::2][:N]
ds = ToothSegDataset(caches, val)
for i, name in enumerate(val):
    s, jaw_upper = ds.load(name)
    jaw = "upper" if jaw_upper else "lower"
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
        v, cands = run(vertex_probs(xyz, nrm, None, jaw, tta=True, up="knn"), xyz, nrm, edges, area, 60, gt, jaw)
        rows.setdefault((f"public {mode}", jaw), []).append({k: score(gt, x, area, xyz.astype(np.float64), edges)
                                                             for k, x in v.items()})
        cand_rows += [dict(c, group=f"public {mode}", jaw=jaw, scan=name) for c in cands]
        print(f"[{i + 1}/{len(val)}] {name} {mode}:", [(c["truth"], round(c["width"], 1)) for c in cands], flush=True)

print("\n== 나누기 후보 (truth 2 이상 = 나누는 게 맞음)")
for jaw in ("upper", "lower"):
    c = [r for r in cand_rows if r["jaw"] == jaw]
    for t, lab in ((True, "합쳐진 둘"), (False, "하나")):
        w = sorted(round(r["width"], 1) for r in c if (r["truth"] >= 2) == t)
        q = sorted(round(r["ratio"], 2) for r in c if (r["truth"] >= 2) == t)
        print(f"  {jaw} {lab} ({len(w)}): width {w}\n      ratio {q}")
print("\n== F75 (병합/분할), 폭 기준별 (w0 = E15의 K1)")
for (group, jaw), rs in sorted(rows.items()):
    line = " | ".join(f"{k} {np.mean([r[k]['F75'] for r in rs]):.3f} "
                      f"({sum(r[k]['merge'] for r in rs)}/{sum(r[k]['split'] for r in rs)})" for k in rs[0])
    print(f"  {group:16s} {jaw:5s} n={len(rs):2d}  {line}")
print(f"\n{time.time() - started:.0f}s")
