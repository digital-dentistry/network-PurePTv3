"""PurePTv3 (Pointcept PT-v3m1 base) 치아 semantic segmentation 학습.

데이터는 dental_ai.dataset.ToothSegLoader가 만든다 (augmentation, 0.25 mm voxel downsample,
feature, target).  PurePTv3의 인덱스 전처리(직렬화, 이웃, 패치, 풀링)는 batch마다 GPU에서
model.prepare로 만든다.

target은 segmentation3d semantic 단계와 같은 17-class (0 gingiva, 1..16 FDI 위치)이고,
위치를 모르는 치아(class 17)는 loss와 점수에서 뺀다.  loss도 같다: 가중 cross-entropy
(빈도의 역제곱근, 평균 1, [0.2, 6]) + class soft dice.  optimizer는 Pointcept PTv3 semseg
레시피 (AdamW wd 0.05, OneCycle, "block" 파라미터는 lr x 0.1).

    python train_toothseg.py --out /app/runs/pureptv3_semantic
    python train_toothseg.py --out /app/runs/pureptv3_semantic --resume      # last.pt에서 이어서
    python train_toothseg.py --out /tmp/smoke --limit 16 --epochs 1          # 스모크 테스트
    python train_toothseg.py --out /app/runs/pureptv3_semantic_scannet         --pretrained /app/checkpoints/pointcept/scannet-semseg-pt-v3m1-0-base/model_best.pth   # 사전 가중치에서 시작

체크포인트의 "state_dict"는 Pointcept DefaultSegmentorV2와 키가 같아서
pureptv3.load_pointcept_checkpoint와 Pointcept 양쪽에서 읽힌다.
"""
from __future__ import annotations

import os

# batch마다 점 수가 달라 고정 블록 할당은 조각난다 (segmentation3d train.py와 같은 이유).  torch보다 먼저
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dental_ai.dataset.algorithm.augmentation.augmentation import AugmentationParams  # noqa: E402
from dental_ai.dataset.ToothSegLoader import (FEATURE_CHANNELS, ToothSegConfig, ToothSegDataset,  # noqa: E402
                                              patient_split, read_split, tooth_seg_loader)

from pureptv3 import PurePTv3Segmentor, init_from_pretrained  # noqa: E402

NUM_SEM = 17                     # 0 gingiva + 16 위치.  ToothSegLoader class 17(위치 모름)은 ignore
# --synth-colour clinical: 라벨이 있는 in-house 컬러 스캔 10개의 실제 색에 맞춘 값 (2026-11-10).
# 색만으로 치아/치은을 가르는 AUC (경계 0.5 mm 안 / 1.5 mm 밖)와 전이 중간점:
#   실제 0.62 / 0.82 / +0.79 mm,  이 값 0.55 / 0.84 / +0.85 mm,  legacy 0.98 / 0.98 / 0 mm
CLINICAL_COLOUR = dict(colour_edge_mm=(0.4, 1.2), colour_edge_width_mm=(1.2, 2.2), colour_drift_scale=(1.5, 3.5))
# --ios 데이터셋마다 학습에 쓰는 공식 split (segmentation3d/training/dental_ai_data.py와 같다).
# test split은 쓰지 않는다.  None = split이 없어 전부
IOS_TRAIN_SPLITS: dict[str, list[str] | None] = {
    "teeth3ds": ["splits/Teeth3DS_train_test_split/training_upper.txt",
                 "splits/Teeth3DS_train_test_split/training_lower.txt"],
    "3d_iosseg": ["splits/train.txt"],
    "poseidon3d": ["splits/train.txt", "splits/validation.txt"],
    "cbct_ios_fusion_liu2023": None,
}


# 직렬화 순서.  블록 i는 i % 4번째를 쓴다.  학습 때는 stage마다 순서를 섞지만(shuffle_orders) 평가·추론은
# 이 순서 그대로라서, 블록이 2개인 stage(인코더 0·1·2·4, 디코더 전부)는 앞의 두 개만 쓴다.
# 첫 학습(pureptv3_semantic)은 z 먼저였다: 추론에서 hilbert는 블록 6개인 인코더 stage 3에서만 쓰였다
ORDER_HILBERT_FIRST = ("hilbert", "hilbert-trans", "z", "z-trans")
ORDER_Z_FIRST = ("z", "z-trans", "hilbert", "hilbert-trans")      # Pointcept 설정의 순서


def backbone_config(in_ch: int, patch: int, drop_path: float, order=ORDER_HILBERT_FIRST) -> dict:
    """Pointcept semseg-pt-v3m1-0-base의 backbone (in_channels, patch, 순서만 바꾼다)."""
    return dict(
        in_channels=in_ch,
        order=tuple(order),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(patch,) * 5,
        dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16), dec_patch_size=(patch,) * 4,
        mlp_ratio=4, qkv_bias=True, drop_path=drop_path, shuffle_orders=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/app/data/ios"))
    ap.add_argument("--ios", nargs="+", default=list(IOS_TRAIN_SPLITS), choices=list(IOS_TRAIN_SPLITS))
    ap.add_argument("--features", nargs="+", default=["xyz", "normal", "jaw", "colour"])
    ap.add_argument("--voxel-mm", type=float, default=0.25)
    ap.add_argument("--max-points", type=int, default=70000)
    # 부분 스캔(bridge, quadrant, half) 대비.  샘플을 읽을 때마다 이 확률로 crop한다 (고정된 일부 스캔이 아니다).
    # 나머지 augmentation은 ToothSegLoader 기본값: SO(3) 회전은 항상, crop 뒤 원점 중심 정렬도 항상
    ap.add_argument("--crop-p", type=float, default=0.5, help="crop 확률. ToothSegLoader 기본값과 첫 학습(pureptv3_semantic)은 0.35")
    # 색을 쓸 확률.  쓸 때 실제 색이 없는 스캔(지금 학습 캐시 전부)은 in-house 실측 컬러맵으로 칠하고(synth_colour_cpu),
    # 안 쓰면 실제 색이 있어도 0으로 넣는다 — 색 없는 STL 스캔도 그만큼 배운다
    ap.add_argument("--colour-p", type=float, default=0.5, help="색 사용 확률. ToothSegLoader 기본값과 첫 학습은 0.6")
    ap.add_argument("--synth-colour", choices=["clinical", "legacy"], default="clinical",
                    help="칠한 색. clinical: 실제 스캔처럼 경계에서 서서히 바뀌고 덜 깨끗하다 (CLINICAL_COLOUR). "
                         "legacy: 라벨 경계에서 바로 바뀐다 (첫 학습)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="처음 N개 스캔만 (스모크 테스트)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3, help="Pointcept 0.006 @ batch 12를 batch 4로 줄인 값")
    ap.add_argument("--block-lr-scale", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--w-dice", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--order", nargs=4, default=list(ORDER_HILBERT_FIRST),
                    choices=["z", "z-trans", "hilbert", "hilbert-trans"],
                    help="직렬화 순서 (평가·추론에서 블록 i가 i %% 4번째를 쓴다). 첫 학습은 z z-trans hilbert hilbert-trans")
    # 사전 가중치: Pointcept PT-v3m1 체크포인트 (DefaultSegmentorV2).  backbone은 전부, seg_head는 class 수가 달라
    # 새로 시작한다.  stem conv만 입력 채널이 달라서 같은 이름의 feature 채널을 옮기고 나머지는 0에서 시작한다
    ap.add_argument("--pretrained", type=Path, default=None,
                    help="Pointcept PT-v3m1 체크포인트 (예: scannet-semseg-pt-v3m1-0-base/model/model_best.pth)")
    ap.add_argument("--pretrained-features", nargs="+", default=["colour", "normal"],
                    help="체크포인트의 입력 feature 순서. Pointcept ScanNet은 color(0..1) normal")
    ap.add_argument("--resume", action="store_true", help="out/last.pt에서 이어서")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ------------------------------------------------------------------ data
def ios_names(root: Path, datasets: list[str]) -> tuple[list[Path], list[str]]:
    caches, names = [], []
    for d in datasets:
        cache = root / d / "cache"
        caches.append(cache)
        files = IOS_TRAIN_SPLITS[d]
        if files is None:
            names += sorted(p.stem for p in cache.glob("*.npz"))
        else:
            for f in files:
                names += read_split(root / d / f)
    return caches, names


def pretrained_channel_map(features: tuple[str, ...], source: list[str], cfg: ToothSegConfig) -> list[int | None]:
    """모델 입력 채널마다 가중치를 가져올 체크포인트 입력 채널.  체크포인트에 없던 feature는 None (0에서 시작)."""
    def width(f: str) -> int:
        return FEATURE_CHANNELS[f] * (len(cfg.curvature_k) if f == "curvature" else 1)

    source = ["colour" if f == "color" else f for f in source]
    start, off = {}, 0
    for f in source:
        start[f], off = off, off + width(f)
    return [start[f] + c if f in start else None for f in features for c in range(width(f))]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def class_weights(out: Path, dataset: ToothSegDataset) -> torch.Tensor:
    """학습 스캔 (최대 300개)의 class 빈도 -> 역제곱근, 평균 1, [0.2, 6].  run 폴더에 캐시한다."""
    path = out / "class_weights.json"
    if path.exists():
        freq = torch.tensor(json.loads(path.read_text()), dtype=torch.float64)
    else:
        cnt = torch.zeros(NUM_SEM, dtype=torch.float64)
        for n in dataset.names[:300]:
            c = dataset.load(n)[0].classes
            cnt += torch.bincount(c[c < NUM_SEM], minlength=NUM_SEM).double()
        freq = cnt / cnt.sum()
        path.write_text(json.dumps(freq.tolist()))
    w = 1.0 / freq.clamp_min(1e-6).sqrt()
    return (w / w.mean()).clamp(0.2, 6.0).float()


def one_cycle(total: int, pct_start: float = 0.05, div: float = 10.0, final_div: float = 1000.0):
    """OneCycleLR (cos)와 같은 모양의 배율: max/div -> max (pct_start) -> max/(div*final_div).
    Pointcept PTv3 semseg 설정값.  LambdaLR용이라 step 수가 아주 적어도 0으로 나누지 않는다."""
    warm = max(1, int(total * pct_start))

    def factor(step: int) -> float:
        if step < warm:
            lo, t = 1.0 / div, step / warm
            return lo + (1.0 - lo) * (1.0 - math.cos(math.pi * t)) / 2
        lo, t = 1.0 / (div * final_div), min(1.0, (step - warm) / max(1, total - warm))
        return lo + (1.0 - lo) * (1.0 + math.cos(math.pi * t)) / 2

    return factor


# ------------------------------------------------------------------ loss
def class_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """target에 있는 class들의 soft dice 평균 (segmentation3d class_dice_loss)."""
    p = logits.softmax(1)
    t = F.one_hot(target, logits.shape[1]).to(p.dtype)
    d = 1 - (2 * (p * t).sum(0) + 1) / (p.sum(0) + t.sum(0) + 1)
    present = t.sum(0) > 0
    return d[present].mean()


def criterion(logits: torch.Tensor, classes: torch.Tensor, cw: torch.Tensor, w_dice: float):
    known = classes < NUM_SEM
    logits, classes = logits[known].float(), classes[known]
    ce = F.cross_entropy(logits, classes, weight=cw)
    dice = class_dice(logits, classes)
    return ce + w_dice * dice, {"ce": ce.detach(), "dice": dice.detach()}


# ------------------------------------------------------------- metrics
def confusion(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    known = gt < NUM_SEM
    k = gt[known] * NUM_SEM + pred[known]
    return torch.bincount(k, minlength=NUM_SEM * NUM_SEM).view(NUM_SEM, NUM_SEM).cpu()


def summarise(cm: np.ndarray) -> dict:
    """mIoU (본 class만), acc, TSA (치아/치은 binary F1) — segmentation3d ConfusionMeter와 같다."""
    tp = np.diag(cm).astype(np.float64)
    iou = tp / np.maximum(cm.sum(1) + cm.sum(0) - tp, 1)
    seen = cm.sum(1) > 0
    gt_t, pr_t, tp_t = cm[1:, :].sum(), cm[:, 1:].sum(), cm[1:, 1:].sum()
    return {"miou": float(iou[seen].mean()) if seen.any() else 0.0,
            "acc": float(tp.sum() / max(cm.sum(), 1)),
            "tsa": float(2 * tp_t / max(gt_t + pr_t, 1)), "iou": iou.tolist()}


@torch.no_grad()
def validate(model: nn.Module, loaders: dict, source: dict[str, str], dev: torch.device) -> dict:
    model.eval()
    scored = {}
    for tag, loader in loaders.items():
        cms: dict[str, np.ndarray] = {}
        for batch in loader:
            batch = batch.to(dev, non_blocking=True)
            inputs = model.prepare(batch.grid, batch=batch.batch_index)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(batch.features, inputs)
            pred = logits.float().argmax(1)
            for name, p, c in zip(batch.names, torch.split(pred, batch.counts),
                                  torch.split(batch.classes, batch.counts)):
                cm = confusion(p, c).numpy()
                for key in ("all", source[name]):
                    cms[key] = cms.get(key, 0) + cm
        scored[tag] = {k: summarise(v) for k, v in cms.items()}
    model.train()
    return scored


def describe(scored: dict) -> str:
    parts = []
    for tag, by in scored.items():
        a = by["all"]
        per = " ".join(f"{k} {v['miou'] * 100:.1f}" for k, v in by.items() if k != "all")
        parts.append(f"{tag.upper()} mIoU {a['miou'] * 100:.2f} acc {a['acc'] * 100:.2f} "
                     f"TSA {a['tsa'] * 100:.2f} ({per})")
    return "  |  ".join(parts)


# ------------------------------------------------------------------ main
def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log = open(a.out / "train.log", "a")

    def say(s: str) -> None:
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    # ---- data
    caches, names = ios_names(a.data_root, a.ios)
    train_names, val_names = patient_split(names, a.val_frac, seed=0)
    if a.limit:
        train_names, val_names = train_names[:a.limit], val_names[:max(4, a.limit // 4)]
    base = dict(voxel_mm=a.voxel_mm, features=tuple(a.features), max_points=a.max_points)
    aug = AugmentationParams(crop_p=a.crop_p, colour_p=a.colour_p,
                             **(CLINICAL_COLOUR if a.synth_colour == "clinical" else {}))
    train_cfg = ToothSegConfig(**base, augmentation=aug)
    ds_train = ToothSegDataset(caches, train_names, train_cfg)
    vals = {"grey": ToothSegDataset(caches, val_names, ToothSegConfig(**base, train=False))}
    if "colour" in a.features:
        # 검증의 칠한 색도 학습과 같은 방식으로 (train=False라 crop·회전은 돌지 않는다)
        vals["colour"] = ToothSegDataset(caches, val_names,
                                         ToothSegConfig(**base, train=False, colour_force=True, augmentation=aug))
    missing = len(ds_train.missing) + len(vals["grey"].missing)
    dl_train = tooth_seg_loader(ds_train, a.batch, workers=a.workers)
    dl_vals = {k: tooth_seg_loader(v, a.batch, workers=max(1, a.workers // 4)) for k, v in vals.items()}
    source = {n: next(c.parent.name for c in caches if (c / f"{n}.npz").exists()) for n in vals["grey"].names}
    if len(dl_train) == 0:
        raise SystemExit("no training batches: check --data-root and the caches")

    # ---- model
    bb = backbone_config(train_cfg.in_ch, a.patch, a.drop_path, a.order)
    model = PurePTv3Segmentor(num_classes=NUM_SEM, backbone_out_channels=64, backbone=bb)
    pretrained = None
    if a.pretrained is not None and not (a.resume and (a.out / "last.pt").exists()):
        cmap = pretrained_channel_map(tuple(a.features), a.pretrained_features, train_cfg)
        rep = init_from_pretrained(model, str(a.pretrained), in_channel_map=cmap)
        pretrained = dict(path=str(a.pretrained), sha256=sha256(a.pretrained), features=a.pretrained_features,
                          in_channel_map=cmap, copied=len(rep["copied"]), adapted=rep["adapted"], fresh=rep["fresh"])
    model = model.to(dev)
    cw = class_weights(a.out, ds_train).to(dev)

    block = [p for n, p in model.named_parameters() if "block" in n]
    other = [p for n, p in model.named_parameters() if "block" not in n]
    opt = torch.optim.AdamW([{"params": other, "lr": a.lr}, {"params": block, "lr": a.lr * a.block_lr_scale}],
                            lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(1, len(dl_train) // a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, one_cycle(a.epochs * steps_per_epoch))

    start_ep, best = 0, -1.0
    last = a.out / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        say(f"resumed from {last} @ epoch {start_ep}")

    say(f"=== PurePTv3 semantic -> {a.out} | train {len(ds_train)} / val {len(vals['grey'])} scans"
        f"{f' ({missing} names without a cache)' if missing else ''} | {a.voxel_mm * 1000:.0f} um "
        f"| max {a.max_points} pts | features {' '.join(a.features)} (in_ch {train_cfg.in_ch}) "
        f"| batch {a.batch}x{a.accum} | {a.epochs} epochs | lr {a.lr} (block x{a.block_lr_scale}) "
        f"| patch {a.patch} ===")
    say(f"    caches {' '.join(str(c) for c in caches)}")
    say(f"    augmentation rotation {aug.rotation} (every sample) | crop p {aug.crop_p} (bridge/quadrant/half "
        f"{'/'.join(str(x) for x in aug.crop_mix)}, wedge {aug.wedge_p}) | mirror {aug.mirror_p} | trim {aug.trim_p} "
        f"| colour {aug.colour_p} (synth {a.synth_colour}) | centred after crop")
    say(f"    params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | order {' '.join(a.order)} "
        f"| class weights {[round(float(x), 2) for x in cw]}")
    if pretrained:
        say(f"    pretrained {pretrained['path']} (sha256 {pretrained['sha256'][:16]}) | {pretrained['copied']} tensors "
            f"copied | stem input channels from checkpoint {pretrained['in_channel_map']} (None = 0) "
            f"| fresh {' '.join(pretrained['fresh'])}")
    elif a.pretrained is not None:
        say("    pretrained: skipped (resuming from last.pt)")

    meta = dict(num_classes=NUM_SEM, backbone_out_channels=64, backbone=bb, features=list(a.features),
                voxel_mm=a.voxel_mm, max_points=a.max_points)
    if a.resume and last.exists():
        meta["pretrained"] = ck["meta"].get("pretrained")
    elif pretrained:
        meta["pretrained"] = pretrained
    args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}

    model.train()
    for ep in range(start_ep, a.epochs):
        t0, t_data, sums, nb, npts = time.time(), 0.0, {}, 0, 0
        torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        t_it = time.time()
        for it, batch in enumerate(dl_train):
            t_data += time.time() - t_it
            batch = batch.to(dev, non_blocking=True)
            inputs = model.prepare(batch.grid, batch=batch.batch_index)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(batch.features, inputs)
            loss, terms = criterion(logits, batch.classes, cw, a.w_dice)
            (loss / a.accum).backward()
            if (it + 1) % a.accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                if torch.isfinite(gnorm):
                    opt.step()
                else:
                    say(f"    ep {ep} it {it}: non-finite grad norm, step skipped")
                opt.zero_grad(set_to_none=True)
                sched.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            nb += 1
            npts += batch.features.shape[0]
            if a.log_every and (it + 1) % a.log_every == 0:
                say(f"    ep {ep} it {it + 1}/{len(dl_train)} | " + " ".join(f"{k} {v / nb:.4f}" for k, v in sums.items())
                    + f" | {(time.time() - t0) / nb:.2f}s/it (data wait {t_data / nb:.2f}) "
                    f"| peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            t_it = time.time()
        terms = " ".join(f"{k} {v / max(nb, 1):.4f}" for k, v in sums.items())
        say(f"ep {ep:3d} | {terms} | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s "
            f"(data wait {t_data:.0f}s) | {npts / max(nb, 1) / a.batch:.0f} pts/scan "
            f"| peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

        if ep % a.val_every == a.val_every - 1 or ep == a.epochs - 1:
            t1 = time.time()
            scored = validate(model, dl_vals, source, dev)
            score = float(np.mean([by["all"]["miou"] for by in scored.values()]))
            say(f"      {describe(scored)} | {time.time() - t1:.0f}s")
            ck = dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                      epoch=ep, best=max(best, score), val=scored, meta=meta, args=args)
            torch.save(ck, a.out / "last.pt.tmp")
            os.replace(a.out / "last.pt.tmp", last)
            if score > best:
                best = score
                torch.save(ck, a.out / "best.pt")
                say(f"      new best mIoU {best * 100:.2f}")
    say("done")


if __name__ == "__main__":
    main()
