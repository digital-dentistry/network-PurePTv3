"""자동 라벨 모델 패키지 비교: 추론 서버 어댑터를 그대로 불러 번호까지 맞는지 잰다.

공개 검증 스캔 (학습 4개 데이터셋의 patient_split 검증셋, PurePTv3 / segmentation3d instance_ios와 같은 split)을
전악, half, quadrant, bridge로 잘라 넣고, 인하우스 라벨 12개(있으면)도 넣는다.
학습 캐시에는 면이 없어 어댑터는 정점 kNN 그래프로 조각을 정리한다 (대시보드는 mesh 면).

점수 (정점은 수, 치아는 정답 instance 기준)
    tooth_f1     치아/치은 binary F1 (정점)
    vertex_fdi   정답이 치아(위치 아는)인 정점 중 번호까지 맞은 비율
    tooth_fdi    정답 치아마다 예측 번호 다수결이 정답 번호인 비율
    tooth_ok75   정답 치아와 같은 번호로 예측된 정점 집합의 IoU >= 0.75인 비율 (영역과 번호가 둘 다 맞음)
    side_wrong   정답 치아 중 위치는 맞고 사분면만 틀린 비율

    python eval_models.py --packages /models/toothseg-pureptv3/1 /tmp/pkgs/s0/1 --n 60 --out /exp/results/x.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["/app", "/pureptv3"]
import registry  # noqa: E402
from toothseg import Scan  # noqa: E402
from dental_ai.dataset.algorithm.augmentation.augmentation import (AugmentationInput, AugmentationOutput,  # noqa: E402
                                                                   AugmentationParams)
from dental_ai.dataset.algorithm.augmentation.segment_crop_cpu import segment_crop_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import ToothSegDataset, _name_generator, patient_split  # noqa: E402
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402

MODES = ("full", "half", "quadrant", "bridge")


def fdi_to_cls(fdi: np.ndarray, jaw: str) -> np.ndarray:
    first, second = (1, 2) if jaw == "upper" else (3, 4)
    q, t = fdi // 10, fdi % 10
    out = np.zeros(len(fdi), np.int64)
    ok = (t >= 1) & (t <= 8)
    out[ok & (q == first)] = t[ok & (q == first)]
    out[ok & (q == second)] = t[ok & (q == second)] + 8
    return out


def score(gt_cls: np.ndarray, gt_ins: np.ndarray, pr_cls: np.ndarray) -> dict:
    gt_t, pr_t = gt_cls > 0, pr_cls > 0
    tp = (gt_t & pr_t).sum()
    known = (gt_cls > 0) & (gt_cls < 17)
    teeth = [k for k in np.unique(gt_ins[known]) if k > 0]
    fdi_ok = ok75 = side = 0
    for k in teeth:
        m = (gt_ins == k) & known
        c = np.bincount(gt_cls[m]).argmax()
        vote = np.bincount(pr_cls[m], minlength=17)
        vote[0] = 0                                          # 치아로 본 정점만 투표
        p = vote.argmax() if vote.sum() else 0
        fdi_ok += p == c
        side += p > 0 and p != c and (p - 1) % 8 == (c - 1) % 8
        region = pr_cls == c
        ok75 += (m & region).sum() / max((m | region).sum(), 1) >= 0.75
    n = max(len(teeth), 1)
    return {"tooth_f1": float(2 * tp / max(gt_t.sum() + pr_t.sum(), 1)),
            "vertex_fdi": float((pr_cls[known] == gt_cls[known]).mean()) if known.any() else float("nan"),
            "tooth_fdi": fdi_ok / n, "tooth_ok75": ok75 / n, "side_wrong": side / n, "n_teeth": len(teeth)}


def public_scans(n: int):
    caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))
    _, val = patient_split(names, 0.1, seed=0)
    val = [v for v in val if not (Path("/data/ios/3d_iosseg/cache") / f"{v}.npz").exists()][::2][:n]
    ds = ToothSegDataset(caches, val)
    for name in val:
        s, jaw_upper = ds.load(name)
        jaw = "upper" if jaw_upper else "lower"
        for mode in MODES:
            smp = s
            if mode != "full":
                out = AugmentationOutput()
                if not segment_crop_cpu(AugmentationInput(s, replace(AugmentationParams(), crop_mode=mode),
                                                          _name_generator(name)), out):
                    continue
                smp = out.sample
            yield mode, name, Scan(smp.positions.numpy(), smp.normals.numpy(), None, jaw), \
                smp.classes.numpy(), smp.instances.numpy()


def inhouse_scans(root: Path):
    for f in sorted(root.glob("*.npz")):
        d = np.load(f)
        jaw = str(d["jaw"])
        fdi = d["fdi"].astype(np.int64)
        cls = np.where(d["label"] == 1, fdi_to_cls(fdi, jaw), 0)
        cls[(d["label"] == 1) & (cls == 0)] = 17                # 치아인데 번호가 이 턱이 아님 -> 위치 모름
        ins = np.where(d["label"] == 1, fdi, 0)                 # DDH는 번호마다 치아 하나
        rgb = d["rgb"] if "rgb" in d.files else None
        yield "inhouse", f.stem, Scan(d["xyz"], d["normals"], rgb, jaw), cls, ins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", nargs="+", required=True, help="패키지 폴더 (.../name/version)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--inhouse", type=Path, default=Path("/gt"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    adapters = {}
    for p in map(Path, a.packages):
        pkg = next(x for x in registry.discover(p.parent.parent) if x.path == p)
        adapters[pkg.ref] = registry.load(pkg, dev)
    rows = []
    scans = list(public_scans(a.n)) + (list(inhouse_scans(a.inhouse)) if a.inhouse.is_dir() else [])
    print(f"{len(scans)} inputs", flush=True)
    for i, (mode, name, scan, cls, ins) in enumerate(scans):
        for ref, ad in adapters.items():
            t0 = time.time()
            pred = ad.predict(scan)
            dt = time.time() - t0
            pr = fdi_to_cls(pred.fdi.astype(np.int64), scan.jaw)
            pr[(pred.label > 0) & (pr == 0)] = 17
            rows.append({"model": ref, "mode": mode, "name": name, "s": dt, **score(cls, ins, pr)})
        if i % 20 == 0:
            print(f"  {i}/{len(scans)}", flush=True)
    summary = {}
    for ref in adapters:
        for mode in MODES + ("inhouse",):
            r = [x for x in rows if x["model"] == ref and x["mode"] == mode]
            if not r:
                continue
            teeth = sum(x["n_teeth"] for x in r)
            summary.setdefault(ref, {})[mode] = {
                "scans": len(r), "teeth": teeth, "s_per_scan": round(float(np.mean([x["s"] for x in r])), 2),
                "tooth_f1": round(float(np.mean([x["tooth_f1"] for x in r])), 4),
                "vertex_fdi": round(float(np.nanmean([x["vertex_fdi"] for x in r])), 4),
                **{k: round(sum(x[k] * x["n_teeth"] for x in r) / max(teeth, 1), 4)
                   for k in ("tooth_fdi", "tooth_ok75", "side_wrong")}}
    for ref, by in summary.items():
        print(ref)
        for mode, s in by.items():
            print(f"  {mode:9s} scans {s['scans']:3d} teeth {s['teeth']:4d} | tooth F1 {s['tooth_f1']:.3f} "
                  f"| vertex FDI {s['vertex_fdi']:.3f} | tooth FDI {s['tooth_fdi']:.3f} ok75 {s['tooth_ok75']:.3f} "
                  f"side-only wrong {s['side_wrong']:.3f} | {s['s_per_scan']:.2f}s")
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
