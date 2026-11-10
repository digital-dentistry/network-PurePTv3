"""E4 데이터셋마다 좌우(사분면)가 맞는지: 좌우를 배운 모델로 학습 데이터 4개를 따로 잰다.

가설: dental_ai 로더로 4개 데이터셋을 함께 학습한 17-class 모델(PurePTv3, semantic_ios)은 좌우를 끝내 배우지 못했다.
Teeth3DS만으로 학습한 segmentation3d semantic(runs/semantic)은 배웠다.
어떤 데이터셋의 메시가 다른 데이터셋의 거울상이면(축을 바꿔 변환하면 det = -1), SO(3) 회전만으로는 좌우 라벨이
서로 모순된다.  그러면 좌우를 배운 모델이 그 데이터셋에서만 사분면을 거의 다 반대로 낸다.

전악만 쓴다 (정중선이 있어 좌우가 분명하다).  데이터셋마다 patient_split(seed 0)의 검증 스캔 --n개.

    python side_by_dataset.py --packages /models/toothsegnet-ios/1 /pkgs/s1/1 --n 20 --out /exp/results/side_by_dataset.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["/app", "/pureptv3", "/exp/scripts"]
import registry  # noqa: E402
from eval_models import fdi_to_cls, score  # noqa: E402
from toothseg import Scan  # noqa: E402
from dental_ai.dataset.ToothSegLoader import ToothSegDataset, patient_split  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    adapters = {}
    for p in map(Path, a.packages):
        pkg = next(x for x in registry.discover(p.parent.parent) if x.path == p)
        adapters[pkg.ref] = registry.load(pkg, dev)

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
            scan = Scan(s.positions.numpy(), s.normals.numpy(), None, jaw)
            cls, ins = s.classes.numpy(), s.instances.numpy()
            for ref, ad in adapters.items():
                pred = ad.predict(scan)
                pr = fdi_to_cls(pred.fdi.astype(np.int64), jaw)
                pr[(pred.label > 0) & (pr == 0)] = 17
                rows.append({"dataset": ds_name, "name": name, "model": ref, **score(cls, ins, pr)})
        print(ds_name, len(picked), flush=True)

    summary = {}
    for ref in adapters:
        print(ref)
        for ds_name in sorted(by_set):
            r = [x for x in rows if x["model"] == ref and x["dataset"] == ds_name]
            side = np.array([x["side_wrong"] for x in r])
            summary.setdefault(ref, {})[ds_name] = {
                "scans": len(r), "tooth_fdi": round(float(np.mean([x["tooth_fdi"] for x in r])), 3),
                "side_wrong": round(float(side.mean()), 3),
                "scans_mostly_mirrored": int((side > 0.5).sum())}
            s = summary[ref][ds_name]
            print(f"  {ds_name:24s} scans {s['scans']:3d} | tooth FDI {s['tooth_fdi']:.3f} | side-only wrong "
                  f"{s['side_wrong']:.3f} | scans with >50% teeth on the wrong side {s['scans_mostly_mirrored']}")
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
