"""E16 바꾼 추론 서버 어댑터 (inference-server/toothseg.py, 2026-11-10)를 그대로 돌려 E15b와 같은 데이터로 잰다.

어댑터가 실험의 방법(TTA8 + kNN, 조각 정리, instance 먼저, 넓은 중절치만 정중선에서 나눔)을 그대로 재현하는지,
그리고 DDH가 저장하는 방식으로 봐도 좋은지 본다.
  DDH   instance = 같은 FDI (DDH의 assign_instances와 같다.  번호가 겹치면 한 치아로 합쳐진다)
  CC    instance = 같은 FDI끼리 이어진 영역 (E10 A와 같은 추출)
인하우스는 면을 넘기고 (대시보드와 같다), 색은 넘기지 않은 것과 넘긴 것 둘 다.  공개 데이터는 면이 없어 어댑터가
가까운 점 그래프로 정리한다 (실험 E12~E15는 정점 수 60 기준, 어댑터는 추정 면적 5 mm² 기준).
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

assert AD.tta == 8, AD.tta


def knn_edges(xyz, k=8):
    _, i = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    e = np.stack([np.repeat(np.arange(len(xyz)), k), i[:, 1:].ravel()], 1)
    return np.unique(np.sort(e, 1), axis=0)


def as_instances(fdi, edges):
    ddh = np.where(fdi > 0, np.unique(fdi, return_inverse=True)[1], 0)
    return {"DDH": ddh, "CC": components(fdi.astype(np.int64), edges)}


rows, times, warn = {}, {}, {}
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
    res = {}
    variants = [("", None)] + ([("+colour ", d["rgb"])] if "rgb" in d.files else [])
    for tag, rgb in variants:
        torch.cuda.synchronize()
        t = time.time()
        pr = AD.predict(Scan(xyz, nrm, rgb, jaw, faces))
        torch.cuda.synchronize()
        times.setdefault("inhouse" + (" colour" if tag else ""), []).append(time.time() - t)
        for k, inst in as_instances(pr.fdi.astype(np.int64), edges).items():
            res[tag + k] = score(gt, inst, area, xyz.astype(np.float64), edges)
        if not tag:
            warn[f.stem] = (pr.info["quadrant_method"], pr.info["teeth"], pr.warnings)
    rows.setdefault("inhouse", []).append(res)
    print(f.stem, len(xyz), f"{times['inhouse'][-1]:.1f}s", pr.info["quadrant_method"], pr.info["teeth"],
          " | ".join(f"{k} F75 {r['F75']:.2f} m{r['merge']}s{r['split']}" for k, r in res.items()), flush=True)

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
        t = time.time()
        pr = AD.predict(Scan(xyz, nrm, None, "upper" if jaw_upper else "lower", None))
        times.setdefault(f"public {mode}", []).append(time.time() - t)
        res = {k: score(gt, inst, area, xyz.astype(np.float64), edges)
               for k, inst in as_instances(pr.fdi.astype(np.int64), edges).items()}
        rows.setdefault(f"public {mode}", []).append(res)
        print(f"[{i + 1}/{len(val)}] {name} {mode}: " + " | ".join(
            f"{k} F75 {r['F75']:.2f} m{r['merge']}s{r['split']}" for k, r in res.items()), flush=True)

print("\n== 인하우스 정중선 판정과 경고")
for k, (m, teeth, w) in warn.items():
    print(f"  {k} {m} {teeth} {w}")
for group, rs in rows.items():
    print(f"\n== {group} ({len(rs)} scans)")
    print(f"   {'method':14s}  F50   F75  mIoU  merge split")
    for k in rs[0]:
        vals = [r[k] for r in rs if k in r]
        mean = lambda m: np.nanmean([x[m] for x in vals])  # noqa: E731
        tot = lambda m: int(sum(x[m] for x in vals))  # noqa: E731
        print(f"   {k:14s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
              f"   (n={len(vals)})")
print("\n== 어댑터 시간 (s, 평균 / 최대)")
for k, v in times.items():
    print(f"   {k:16s} {np.mean(v):.2f} / {np.max(v):.2f}")
