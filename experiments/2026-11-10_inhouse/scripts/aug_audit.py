"""E17 학습 augmentation 점검 (GPU 없음).  배포 모델(toothseg-pureptv3/1)이 본 학습 입력이 어땠는지 다시 만든다.

1. 데이터셋마다 원래 좌표계의 교합면 방향: 치아 정점 법선 평균이 어느 축에 가까운가.
   wedge_crop_cpu는 회전 전에 xy 평면의 각도로 자르므로, 교합면이 xy가 아닌 데이터셋에서는 엉뚱하게 자른다.
2. 배포 모델 설정(crop_p 0.35, 나머지 AugmentationParams 기본)과 지금 기본(crop_p 0.5)으로 train_recipe를
   돌려, 샘플마다 어떤 크롭이 실제로 적용됐는지(남은 치아 비율로 판정), 크롭 뒤 중심, 회전을 센다.
   판정: 치아가 원래의 85% 이상 남으면 "전악", 그 아래는 남은 치아 수로 1~5개 bridge급 / 그 이상 부분 악궁.

    docker run --rm --entrypoint sh (마운트는 README와 같고 --gpus 없음) hdl-gpu:latest \
        -c "python /exp/scripts/aug_audit.py 40"
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path[:0] = ["/src", "/pureptv3"]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dental_ai.dataset.algorithm.augmentation.augmentation import (AugmentationInput, AugmentationOutput,  # noqa: E402
                                                                   AugmentationParams)
from dental_ai.dataset.algorithm.augmentation.train_recipe_cpu import train_recipe_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import ToothSegDataset  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402

PER = int(sys.argv[1]) if len(sys.argv) > 1 else 40
caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))
ds = ToothSegDataset(caches, names)
by_set = {}
for n in names:
    key = next(c.parent.name for c in caches if (c / f"{n}.npz").exists())
    by_set.setdefault(key, []).append(n)
print("데이터셋별 스캔 수:", {k: len(v) for k, v in by_set.items()}, "합", len(names))

print("\n== 1. 원래 좌표계의 교합면 축 (치아 법선 평균, 스캔마다 단위벡터 -> 가장 큰 성분의 축)")
rng = np.random.default_rng(0)
picks = {k: list(rng.choice(v, min(PER, len(v)), replace=False)) for k, v in by_set.items()}
for key, ns in picks.items():
    axes = []
    for n in ns:
        s, _ = ds.load(n)
        t = s.instances > 0
        m = s.normals[t].double().mean(0)
        m = m / m.norm()
        axes.append(m.numpy())
    a = np.array(axes)
    dom = np.abs(a).argmax(1)
    print(f"  {key:26s} n={len(a):3d}  주축 x/y/z = {np.bincount(dom, minlength=3).tolist()}  "
          f"|z| 평균 {np.abs(a[:, 2]).mean():.2f}  (1이면 교합면이 xy)")

print("\n== 2. train_recipe가 실제로 만든 입력")
for label, params in (("배포 모델 (crop_p 0.35)", AugmentationParams()),
                      ("지금 기본 (crop_p 0.5)", replace(AugmentationParams(), crop_p=0.5))):
    print(f"\n  -- {label}: mirror {params.mirror_p}, crop {params.crop_p} (그중 wedge {params.wedge_p}, "
          f"segment mix bridge/quadrant/half {params.crop_mix}), trim {params.trim_p}, rotation {params.rotation}")
    total = {}
    for key, ns in picks.items():
        kinds, centre, zs = [], [], []
        for i, n in enumerate(ns):
            s, _ = ds.load(n)
            n_teeth = len(torch.unique(s.instances[s.instances > 0]))
            gen = torch.Generator().manual_seed(1000 + i)
            out = AugmentationOutput()
            train_recipe_cpu(AugmentationInput(s, params, gen), out)
            a = out.sample
            kept = len(torch.unique(a.instances[a.instances > 0]))
            frac = kept / max(n_teeth, 1)
            kinds.append("full" if frac >= 0.85 else "bridge(<=5)" if kept <= 5 else "partial(>5)")
            t = a.instances > 0
            m = a.normals[t].double().mean(0)
            zs.append(float(m[2] / m.norm()))
            centre.append(float((a.positions.double() - a.positions.double().mean(0)).mean(0).norm()))
        c = {k: kinds.count(k) for k in ("full", "partial(>5)", "bridge(<=5)")}
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
        print(f"  {key:26s} " + "  ".join(f"{k} {v / len(ns):.0%}" for k, v in c.items())
              + f"  | 회전 뒤 교합면 z성분 평균 {np.mean(zs):+.2f}, |z| 평균 {np.mean(np.abs(zs)):.2f} (균일이면 0, 0.5)")
    n_all = sum(total.values())
    print("  " + "합계".ljust(26) + "  ".join(f"{k} {v / n_all:.0%}" for k, v in total.items()))
