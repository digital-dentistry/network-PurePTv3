"""E1: 학습이 끝난 체크포인트에서 평가 직렬화 순서만 바꾸면 점수가 달라지는가.

학습은 stage마다 순서를 섞었고(shuffle_orders), 평가는 order 튜플 그대로라서 블록이 2개인 stage는
앞의 두 순서만 쓴다.  같은 가중치로 순서 배치만 바꿔 검증 158 스캔을 다시 잰다.
stage별 점 수와 패치 수도 기록한다 (순서가 영향을 주는 stage가 어디인지).

    python eval_order.py --ckpt /runs/pureptv3_semantic/best.pt --out /exp/results/eval_order.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/pureptv3")
import train_toothseg as T  # noqa: E402
from dental_ai.dataset.algorithm.augmentation.augmentation import AugmentationParams  # noqa: E402
from pureptv3 import PurePTv3Segmentor  # noqa: E402

VARIANTS = {
    "z_first": dict(order=("z", "z-trans", "hilbert", "hilbert-trans")),          # 학습 설정 그대로
    "hilbert_first": dict(order=("hilbert", "hilbert-trans", "z", "z-trans")),
    "shuffle_s0": dict(order=("z", "z-trans", "hilbert", "hilbert-trans"), seed=0),  # Pointcept 평가 방식
    "shuffle_s1": dict(order=("z", "z-trans", "hilbert", "hilbert-trans"), seed=1),
    "shuffle_s2": dict(order=("z", "z-trans", "hilbert", "hilbert-trans"), seed=2),
}
POSITION = np.array([0] + list(range(1, 9)) * 2)


def fold9(cm: np.ndarray) -> np.ndarray:
    out = np.zeros((9, 9), np.int64)
    np.add.at(out, (POSITION[:, None].repeat(17, 1), POSITION[None, :].repeat(17, 0)), cm)
    return out


def miou(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(1) + cm.sum(0) - tp, 1)
    return float(iou[cm.sum(1) > 0].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/app/data/ios"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta, args = ck["meta"], ck["args"]

    caches, names = T.ios_names(a.data_root, args["ios"])
    _, val_names = T.patient_split(names, args["val_frac"], seed=0)
    base = dict(voxel_mm=meta["voxel_mm"], features=tuple(meta["features"]), max_points=meta["max_points"])
    # 첫 학습의 검증과 같은 색: legacy synth colour (AugmentationParams 기본값)
    vals = {"grey": T.ToothSegDataset(caches, val_names, T.ToothSegConfig(**base, train=False)),
            "colour": T.ToothSegDataset(caches, val_names, T.ToothSegConfig(**base, train=False, colour_force=True,
                                                                            augmentation=AugmentationParams()))}
    loaders = {k: T.tooth_seg_loader(v, 4, workers=3) for k, v in vals.items()}

    result = {"ckpt": str(a.ckpt), "epoch": ck["epoch"], "trained_order": meta["backbone"]["order"],
              "n_val": len(val_names), "variants": {}, "stages": None}
    stage_n = []
    for vname in a.variants:
        v = VARIANTS[vname]
        bb = dict(meta["backbone"], order=v["order"])
        model = PurePTv3Segmentor(num_classes=meta["num_classes"], backbone_out_channels=64, backbone=bb)
        model.load_state_dict(ck["state_dict"])
        model = model.to(dev).eval()
        gen = None
        if "seed" in v:
            gen = torch.Generator().manual_seed(v["seed"])
        scored, t0 = {}, time.time()
        for tag, loader in loaders.items():
            cm = np.zeros((17, 17), np.int64)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(dev, non_blocking=True)
                    inputs = model.prepare(batch.grid, batch=batch.batch_index,
                                           shuffle_orders=gen is not None, generator=gen)
                    if vname == a.variants[0] and tag == "grey":
                        for b in range(len(batch.counts)):
                            stage_n.append([int((inputs[f"s{s}_unpool"] if s else batch.batch_index).shape[0])
                                            for s in range(1)])
                        stage_n[-len(batch.counts):] = [[None]] * len(batch.counts)
                        sizes = [int(inputs[f"s{s}_inverse0"].shape[0]) for s in range(5)]
                        patches = [int(inputs[f"s{s}_attn_bias"].shape[0]) for s in range(5)]
                        stage_n.append({"batch_scans": len(batch.counts), "points": sizes, "patches": patches})
                    with torch.autocast("cuda", torch.bfloat16):
                        pred = model(batch.features, inputs).float().argmax(1)
                    cm += T.confusion(pred, batch.classes).numpy()
            s = T.summarise(cm)
            scored[tag] = {"miou17": s["miou"], "acc17": s["acc"], "tsa": s["tsa"], "miou9": miou(fold9(cm))}
        result["variants"][vname] = scored
        print(vname, " | ".join(f"{t} mIoU17 {x['miou17'] * 100:.2f} mIoU9 {x['miou9'] * 100:.2f} "
                                f"TSA {x['tsa'] * 100:.2f}" for t, x in scored.items()),
              f"| {time.time() - t0:.0f}s", flush=True)
    rows = [r for r in stage_n if isinstance(r, dict)]
    pts = np.array([np.array(r["points"]) / r["batch_scans"] for r in rows])
    pat = np.array([np.array(r["patches"]) / r["batch_scans"] for r in rows])
    result["stages"] = {"points_per_scan_mean": pts.mean(0).round(0).tolist(),
                        "patches_per_scan_mean": pat.mean(0).round(2).tolist(),
                        "patch_size": meta["backbone"]["enc_patch_size"][0]}
    print("points per scan by stage", result["stages"]["points_per_scan_mean"],
          "| patches per scan", result["stages"]["patches_per_scan_mean"])
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
