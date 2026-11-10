"""E12 공개 검증 스캔에서 instance 품질: PurePTv3 (E10의 H) vs segmentation3d instance_ios, 전악과 인공 크롭.

라벨이 믿을 만한 데이터에서도 PurePTv3의 병합/분할이 나오는지 (인하우스 라벨 문제가 아닌지) 본다.
학습 캐시에는 면이 없어서 인접은 정점 kNN 그래프 (k=8), 면적 가중 없이 정점 수로 잰다 — 두 모델에 같은 기준.
검증 스캔은 PurePTv3 학습의 patient_split 검증셋 (teeth3ds / poseidon3d / cbct_ios_fusion; 3d_iosseg는 성겨서 뺌).
segmentation3d instance_ios도 같은 4개 데이터셋, 같은 split으로 학습했다 (train.log).

    python /exp/scripts/public_instance_eval.py 40
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
from pipeline.types import Scan as S3DScan  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402


def knn_edges(xyz, k=8):
    _, i = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    e = np.stack([np.repeat(np.arange(len(xyz)), k), i[:, 1:].ravel()], 1)
    return np.unique(np.sort(e, 1), axis=0)


caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))
_, val = patient_split(names, 0.1, seed=0)
val = [n for n in val if not (Path("/data/ios/3d_iosseg/cache") / f"{n}.npz").exists()][::2][:N]
ds = ToothSegDataset(caches, val)

rows = {"full": [], "quadrant": [], "half": []}
for i, name in enumerate(val):
    s, jaw_upper = ds.load(name)
    jaw = "upper" if jaw_upper else "lower"
    for mode in rows:
        if mode == "full":
            smp = s
        else:
            out = AugmentationOutput()
            if not segment_crop_cpu(AugmentationInput(s, replace(AugmentationParams(), crop_mode=mode),
                                                      _name_generator(name)), out):
                continue
            smp = out.sample
        xyz = smp.positions.numpy().astype(np.float32)
        nrm = smp.normals.numpy().astype(np.float32)
        x64 = xyz.astype(np.float64)
        gt = np.unique(smp.instances.numpy(), return_inverse=True)[1].astype(np.int64)
        if (smp.instances.numpy() == 0).sum() == 0:
            gt = gt + 1
        edges = knn_edges(x64)
        area = np.ones(len(xyz))
        p = vertex_probs(xyz, nrm, None, jaw, tta=True, up="knn")
        H = components(cleanup(with_rule(pos9(p).argmax(1), x64, nrm), edges, area, min_area=60), edges)
        scan = S3DScan(path=Path(name), name=name, jaw=jaw, positions=smp.positions.float(),
                       normals=unit_normals(smp.normals.float()), faces=None, colours=None)
        S = components(cleanup(S3D_INST.segment(scan).instances.astype(np.int64), edges, area, min_area=60), edges)
        res = {"PurePTv3 H": score(gt, H, area, x64, edges), "seg3d S1+clean": score(gt, S, area, x64, edges)}
        rows[mode].append(res)
        print(f"[{i + 1}/{len(val)}] {name} {mode}: " + " | ".join(
            f"{k} F75 {r['F75']:.2f} mIoU {r['mIoU']:.3f} m{r['merge']}s{r['split']} n {r['n_pr']}/{r['n_gt']}"
            for k, r in res.items()), flush=True)

for mode, rs in rows.items():
    if not rs:
        continue
    print(f"\n== {mode} ({len(rs)} scans, {sum(r['PurePTv3 H']['n_gt'] for r in rs)} teeth)")
    print(f"   {'method':16s}  F50   F75  mIoU  merge split  margin GT>P P>GT  inter GT>P P>GT")
    for k in rs[0]:
        vals = [r[k] for r in rs]
        mean = lambda m: np.nanmean([x[m] for x in vals])  # noqa: E731
        tot = lambda m: int(sum(x[m] for x in vals))  # noqa: E731
        print(f"   {k:16s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
              f"   {mean('margin_gt2p'):.2f} {mean('margin_p2gt'):.2f}      {mean('inter_gt2p'):.2f} {mean('inter_p2gt'):.2f}")
