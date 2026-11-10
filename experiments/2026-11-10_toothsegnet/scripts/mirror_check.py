"""E5 dental_ai 학습 augmentation이 좌우 라벨을 지키는지: 좌우를 아는 모델(S0)로 증강한 샘플을 잰다.

라벨이 맞으면 S0의 side-only wrong이 원본과 같이 낮다.  증강이 좌우 라벨을 깨뜨리면(예: 메시는 뒤집고
class는 그대로) 그 샘플에서만 높아진다.  데이터셋마다 검증 전악 --n개, 증강은 seed --seeds개.

    mirror      mirror_cpu만
    recipe      train_recipe_cpu (semantic_ios와 같은 crop 0.5 / trim 0.5, colour 없음, mirror_p 기본값)

    python mirror_check.py --package /models/toothsegnet-ios/1 --n 8 --out /exp/results/mirror_check.json
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["/app", "/pureptv3", "/exp/scripts"]
import registry  # noqa: E402
from eval_models import fdi_to_cls, score  # noqa: E402
from toothseg import Scan  # noqa: E402
from dental_ai.dataset.algorithm.augmentation.augmentation import (AugmentationInput, AugmentationOutput,  # noqa: E402
                                                                   AugmentationParams)
from dental_ai.dataset.algorithm.augmentation.mirror_cpu import mirror_cpu  # noqa: E402
from dental_ai.dataset.algorithm.augmentation.train_recipe_cpu import train_recipe_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import ToothSegDataset, patient_split  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pkg = next(x for x in registry.discover(a.package.parent.parent) if x.path == a.package)
    ad = registry.load(pkg, torch.device("cuda"))
    recipe = replace(AugmentationParams(), crop_p=0.5, trim_p=0.5, colour_p=0.0)
    print("recipe params", recipe, flush=True)

    caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))
    _, val = patient_split(names, 0.1, seed=0)
    by_set = {}
    for name in val:
        ds = next(c.parent.name for c in map(Path, caches) if (c / f"{name}.npz").exists())
        by_set.setdefault(ds, []).append(name)

    rows = []
    for ds_name, picked in sorted(by_set.items()):
        picked = picked[:: max(1, len(picked) // a.n)][: a.n]
        data = ToothSegDataset(caches, picked)
        for name in picked:
            s, jaw_upper = data.load(name)
            jaw = "upper" if jaw_upper else "lower"
            variants = [("original", s)]
            out = AugmentationOutput()
            assert mirror_cpu(AugmentationInput(s, AugmentationParams()), out)
            variants.append(("mirror", out.sample))
            for seed in range(a.seeds):
                out = AugmentationOutput()
                gen = torch.Generator().manual_seed(seed)
                assert train_recipe_cpu(AugmentationInput(s, recipe, gen), out)
                variants.append(("recipe", out.sample))
            for kind, smp in variants:
                pred = ad.predict(Scan(smp.positions.numpy(), smp.normals.numpy(), None, jaw))
                pr = fdi_to_cls(pred.fdi.astype(np.int64), jaw)
                pr[(pred.label > 0) & (pr == 0)] = 17
                rows.append({"dataset": ds_name, "name": name, "kind": kind, **score(smp.classes.numpy(),
                                                                                    smp.instances.numpy(), pr)})
        print(ds_name, len(picked), flush=True)

    summary = {}
    for ds_name in sorted(by_set):
        for kind in ("original", "mirror", "recipe"):
            r = [x for x in rows if x["dataset"] == ds_name and x["kind"] == kind]
            summary.setdefault(ds_name, {})[kind] = {
                "samples": len(r), "tooth_fdi": round(float(np.mean([x["tooth_fdi"] for x in r])), 3),
                "side_wrong": round(float(np.mean([x["side_wrong"] for x in r])), 3)}
            s = summary[ds_name][kind]
            print(f"  {ds_name:24s} {kind:8s} samples {s['samples']:3d} | tooth FDI {s['tooth_fdi']:.3f} | "
                  f"side-only wrong {s['side_wrong']:.3f}")
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
