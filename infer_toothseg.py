"""train_toothseg.py 체크포인트로 검증 스캔을 추론하고 결과를 그린다.

학습과 같은 검증 split (체크포인트 meta/args로 다시 만든다), 같은 전처리 (0.25 mm voxel, 최대 70k 점)로
grey(색 없이) / colour 두 조건을 추론해서 다음을 남긴다.

    out/summary.json            조건 x 데이터셋별 점수
    out/per_scan.csv            스캔마다 점수
    out/confusion_<tag>.png     17x17 confusion (행 정규화)
    out/overview_<tag>.png      데이터셋마다 중앙값 스캔 + 가장 나쁜 스캔: 정답 / 예측 / 오류
    out/renders/<tag>/*.png     모든 검증 스캔의 정답 / 예측 / 오류

점수는 17-class (학습 target)와, 좌우를 합친 9-class 악궁 위치 (0 gingiva, 1 중절치 .. 8 제3대구치)
두 가지다.  "L/R swap"은 치아 점 중 위치는 맞고 사분면만 반대로 예측한 비율이다.

    python infer_toothseg.py --ckpt /app/runs/pureptv3_semantic/best.pt --out /app/runs/pureptv3_semantic/infer
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_toothseg as T  # PYTORCH_CUDA_ALLOC_CONF, sys.path, 학습과 같은 split / metric
from dental_ai.dataset import ToothSegLoader as L
from pureptv3 import PurePTv3Segmentor

NUM_SEM = T.NUM_SEM
ARCH = np.array([0] + list(range(1, 9)) * 2)                 # 0..16 -> 악궁 위치 0..8
ARCH18 = np.r_[ARCH, -1]                                       # + 17 (위치 모름) -> -1
MIRROR = np.array([0] + list(range(9, 17)) + list(range(1, 9)))
CLASS_NAMES = ["G"] + [f"1-{i}" for i in range(1, 9)] + [f"2-{i}" for i in range(1, 9)]

BG, INK = np.array(L.DEBUG_BACKGROUND), L.DEBUG_TEXT
MUTED = (0.765, 0.761, 0.718)                                 # #c3c2b7
# 오류 종류 (dark surface의 categorical 1..3번 slot)
ERR = {"tooth <-> gingiva": (0.224, 0.529, 0.898),            # #3987e5
       "L/R swap (same position)": (0.851, 0.349, 0.149),    # #d95926
       "wrong position": (0.098, 0.620, 0.439)}              # #199e70
CORRECT_TOOTH, CORRECT_GUM = (0.55, 0.55, 0.53), (0.30, 0.30, 0.29)
CLASS_RGB = np.array([L._class_colour(c) for c in range(NUM_SEM)])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/app/data/ios"))
    ap.add_argument("--tags", nargs="+", default=["grey", "colour"], choices=["grey", "colour"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="처음 N개 검증 스캔만")
    ap.add_argument("--no-renders", action="store_true", help="스캔마다 그림은 건너뛴다 (overview는 그린다)")
    ap.add_argument("--crop", choices=L.CROP_MODES, default=None,
                    help="부분 스캔 흉내: loader의 crop_force (bridge 2~5 치아, quadrant, half).  스캔마다 고정")
    return ap.parse_args()


# ------------------------------------------------------------------ metrics
def fold9(cm: np.ndarray) -> np.ndarray:
    out = np.zeros((9, 9), cm.dtype)
    np.add.at(out, (ARCH[:, None].repeat(NUM_SEM, 1), ARCH[None, :].repeat(NUM_SEM, 0)), cm)
    return out


def miou(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    iou = tp / np.maximum(cm.sum(1) + cm.sum(0) - tp, 1)
    seen = cm.sum(1) > 0
    return float(iou[seen].mean()) if seen.any() else 0.0


def scores(cm: np.ndarray) -> dict:
    """17-class 점수 + 9-class (좌우 합침) 점수 + 치아 점이 어디로 갔는지."""
    s = T.summarise(cm)
    teeth = cm[1:].sum()
    swap = sum(cm[c, MIRROR[c]] for c in range(1, NUM_SEM))
    to_gum = cm[1:, 0].sum()
    right = np.trace(cm[1:, 1:])
    c9 = fold9(cm)
    return {"miou17": s["miou"], "acc17": s["acc"], "tsa": s["tsa"],
            "miou9": miou(c9), "acc9": float(np.trace(c9) / max(c9.sum(), 1)),
            "tooth_correct": float(right / max(teeth, 1)), "tooth_lr_swap": float(swap / max(teeth, 1)),
            "tooth_wrong_pos": float((teeth - right - swap - to_gum) / max(teeth, 1)),
            "tooth_to_gingiva": float(to_gum / max(teeth, 1)), "points": int(cm.sum())}


def side_by_geometry(coord: np.ndarray, normals: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """예측의 위치(1..8)만 쓰고 사분면은 기하로 정한다 (정답은 쓰지 않는다).

    loader의 교합면 frame (z = 치아 normal 평균, y = 위치 번호가 작아지는 쪽)을 예측 위치로 만들면
    위·아래턱 모두 1-x 사분면이 화면 왼쪽(x < 정중선)에 온다.  정중선 = 예측 중절치 점들의 x 중앙값.
    """
    pos = ARCH[pred]
    t = pos > 0
    if t.sum() < 50:
        return pred
    x = (coord @ L._view_frame(coord, normals, pos, pos, "occlusal").T)[:, 0]
    inc = pos == 1
    mid = np.median(x[inc]) if inc.sum() >= 20 else np.median(x[t])
    return np.where(t, np.where(x < mid, pos, pos + 8), 0)


def confusion_np(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    k = gt < NUM_SEM
    return np.bincount(gt[k] * NUM_SEM + pred[k], minlength=NUM_SEM ** 2).reshape(NUM_SEM, NUM_SEM)


# ------------------------------------------------------------------ rendering
def project(coord: np.ndarray, normals: np.ndarray, gt: np.ndarray, size: int):
    """교합면 view (loader의 render_debug와 같다): 정답 위치 번호로 전치 쪽이 위, 1-x 사분면이 왼쪽."""
    frame = L._view_frame(coord, normals, gt, ARCH18[gt], "occlusal")
    q = coord @ frame.T
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    scale = 0.92 * size / max(float((hi - lo).max()), 1e-6)
    centre = (lo + hi) / 2
    u = np.round((q[:, 0] - centre[0]) * scale + size / 2).astype(np.int64)
    v = np.round(size / 2 - (q[:, 1] - centre[1]) * scale).astype(np.int64)
    spacing = np.sqrt(float(np.prod(hi - lo)) * 0.4 / len(q)) * scale
    radius = int(np.clip(round(0.6 * spacing), 1, 4))
    shade = (0.45 + 0.55 * np.abs((normals @ frame.T)[:, 2]))[:, None]
    return u, v, q[:, 2], radius, shade


def error_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.where((gt > 0)[:, None], CORRECT_TOOTH, CORRECT_GUM).astype(np.float64)
    wrong = pred != gt
    tg = wrong & ((gt == 0) | (pred == 0))
    swap = wrong & ~tg & (MIRROR[gt] == pred)
    rgb[tg] = ERR["tooth <-> gingiva"]
    rgb[swap] = ERR["L/R swap (same position)"]
    rgb[wrong & ~tg & ~swap] = ERR["wrong position"]
    return rgb


def panels(scan: dict, size: int) -> tuple[list[np.ndarray], list[list[tuple[int, int, str]]]]:
    """[정답, 예측, 오류] 그림과 그림마다 (u, v, 글자) 목록 (class 번호를 그 class 점들의 중앙에)."""
    coord, normals, gt, pred = scan["coord"], scan["normals"], scan["gt"], scan["pred"]
    u, v, depth, r, shade = project(coord, normals, gt, size)
    known = gt < NUM_SEM
    gt_c = np.where(known, gt, 0)
    imgs = [L._splat(u, v, depth, CLASS_RGB[gt_c] * shade * np.where(known, 1, 0.3)[:, None], size, r),
            L._splat(u, v, depth, CLASS_RGB[pred] * shade, size, r),
            L._splat(u, v, depth, np.where(known[:, None], error_rgb(gt_c, pred), 0.15) * shade, size, r)]
    labels = []
    for lab in (gt_c, pred):
        marks = []
        for c in range(1, NUM_SEM):
            m = lab == c
            if m.sum() >= 30:
                marks.append((int(np.median(u[m])), int(np.median(v[m])), str(c)))
        labels.append(marks)
    return imgs, labels + [[]]


def draw_row(axes, scan: dict, size: int, show_titles: bool) -> None:
    imgs, labels = panels(scan, size)
    s = scan["scores"]
    titles = ["ground truth", "prediction", "errors"]
    notes = [f"{scan['name']}  ({scan['source']}, {'upper' if scan['upper'] else 'lower'}, {len(scan['gt'])} pts)",
             f"mIoU17 {s['miou17'] * 100:.1f}   mIoU9 {s['miou9'] * 100:.1f}   TSA {s['tsa'] * 100:.1f}",
             f"teeth: correct {s['tooth_correct'] * 100:.0f}%  L/R swap {s['tooth_lr_swap'] * 100:.0f}%  "
             f"wrong pos {s['tooth_wrong_pos'] * 100:.0f}%  ->gum {s['tooth_to_gingiva'] * 100:.0f}%"]
    for ax, img, marks, title, note in zip(axes, imgs, labels, titles, notes):
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
        for x, y, t in marks:
            ax.text(x, y, t, color=INK, fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=(0.1, 0.1, 0.1, 0.75), ec="none"))
        ax.text(6, 14, (title + "\n" if show_titles else "") + note, color=INK if show_titles else MUTED,
                fontsize=8, va="top", linespacing=1.5)
        ax.set_axis_off()


def legend_axes(ax) -> None:
    import matplotlib.patches as mp
    ax.set_axis_off()
    # legend는 열 우선으로 채운다: 열마다 [1-x, 2-x]
    hs = [mp.Patch(color=L.DEBUG_GINGIVA, label="0 gingiva"), mp.Patch(alpha=0, label="")]
    for p in range(1, 9):
        hs += [mp.Patch(color=CLASS_RGB[p], label=f"{p} = 1-{p}"), mp.Patch(color=CLASS_RGB[p + 8], label=f"{p + 8} = 2-{p}")]
    leg1 = ax.legend(handles=hs, loc="upper left", ncol=9, frameon=False, labelcolor=INK, fontsize=8,
                     title="classes: hue = position (1 central incisor .. 8 third molar), light = quadrant 1-x, "
                           "dark = quadrant 2-x", title_fontsize=8)
    leg1.get_title().set_color(MUTED)
    ax.add_artist(leg1)
    hs = [mp.Patch(color=CORRECT_TOOTH, label="correct tooth"), mp.Patch(color=CORRECT_GUM, label="correct gingiva")]
    hs += [mp.Patch(color=c, label=k) for k, c in ERR.items()]
    leg2 = ax.legend(handles=hs, loc="lower left", ncol=5, frameon=False, labelcolor=INK, fontsize=8,
                     title="errors panel", title_fontsize=8)
    leg2.get_title().set_color(MUTED)


def render_scan(path: Path, scan: dict, size: int = 460) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(3 * size / 100, size / 100), dpi=100, facecolor=BG)
    fig.subplots_adjust(0.003, 0.003, 0.997, 0.997, 0.01, 0.01)
    draw_row(axes, scan, size, show_titles=True)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def render_overview(path: Path, rows: list[dict], tag: str, headline: str, size: int = 400) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3 * size / 100, (len(rows) * size + 280) / 100), dpi=100, facecolor=BG)
    top = 1 - 90 / (len(rows) * size + 280)
    bottom = 210 / (len(rows) * size + 280)
    gs = fig.add_gridspec(len(rows), 3, left=0.003, right=0.997, top=top, bottom=bottom, wspace=0.01, hspace=0.01)
    for i, scan in enumerate(rows):
        draw_row([fig.add_subplot(gs[i, j]) for j in range(3)], scan, size, show_titles=(i == 0))
    fig.text(0.006, 1 - 12 / (len(rows) * size + 280), headline, color=INK, fontsize=10, va="top", linespacing=1.6)
    legend_axes(fig.add_axes([0.006, 0.004, 0.99, bottom - 0.01]))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def render_confusion(path: Path, cm: np.ndarray, tag: str) -> None:
    """행(정답)마다 정규화한 %.  sequential blue (dataviz 기본 ramp), 칸 사이 2px 간격."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ramp = ["#f7f6f3", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue", ramp)
    rows = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(8.6, 7.6), dpi=110, facecolor="white")
    im = ax.imshow(rows, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(NUM_SEM), CLASS_NAMES, fontsize=8, color="#3d3d3a")
    ax.set_yticks(range(NUM_SEM), CLASS_NAMES, fontsize=8, color="#3d3d3a")
    ax.set_xticks(np.arange(-0.5, NUM_SEM), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_SEM), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(NUM_SEM):
        for j in range(NUM_SEM):
            if rows[i, j] >= 5:
                ax.text(j, i, f"{rows[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if rows[i, j] >= 45 else "#3d3d3a")
    ax.set_xlabel("predicted class", fontsize=9, color="#3d3d3a")
    ax.set_ylabel("ground-truth class", fontsize=9, color="#3d3d3a")
    ax.set_title(f"PurePTv3 val confusion ({tag}), % of each ground-truth row\n"
                 "1-x / 2-x = quadrant, x = position (1 central incisor .. 8 third molar)",
                 fontsize=9.5, color="#141413", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors="#3d3d3a", length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
@torch.no_grad()
def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta, targs = ck["meta"], ck["args"]
    model = PurePTv3Segmentor(meta["num_classes"], meta["backbone_out_channels"], backbone=meta["backbone"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    print(f"checkpoint {a.ckpt} | epoch {ck['epoch']} | val at save: "
          + " ".join(f"{t} {v['all']['miou'] * 100:.2f}" for t, v in ck["val"].items()), flush=True)

    caches, names = T.ios_names(a.data_root, targs["ios"])
    _, val_names = L.patient_split(names, targs["val_frac"], seed=0)
    if a.limit:
        val_names = val_names[:a.limit]
    base = dict(voxel_mm=meta["voxel_mm"], features=tuple(meta["features"]), max_points=meta["max_points"],
                crop_force=a.crop)
    configs = {"grey": L.ToothSegConfig(**base, train=False),
               "colour": L.ToothSegConfig(**base, train=False, colour_force=True)}
    sl = L.feature_slices(configs["grey"])["normal"]

    summary, table = {"checkpoint": str(a.ckpt), "epoch": ck["epoch"]}, []
    for tag in a.tags:
        ds = L.ToothSegDataset(caches, val_names, configs[tag])
        source = {n: next(c.parent.name for c in caches if (c / f"{n}.npz").exists()) for n in ds.names}
        loader = L.tooth_seg_loader(ds, a.batch, workers=a.workers, shuffle=False)
        cms: dict[str, np.ndarray] = {}
        cms_rule: dict[str, np.ndarray] = {}
        scans, t0, t_model = [], time.time(), 0.0
        for batch in loader:
            batch = batch.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.time()
            inputs = model.prepare(batch.grid, batch=batch.batch_index)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(batch.features, inputs)
            pred = logits.float().argmax(1)
            torch.cuda.synchronize()
            t_model += time.time() - t1
            split = lambda x: [t.cpu().numpy() for t in torch.split(x, batch.counts)]  # noqa: E731
            for name, upper, p, c, xyz, f in zip(batch.names, batch.jaw_upper.tolist(), split(pred),
                                                 split(batch.classes), split(batch.coord), split(batch.features)):
                xyz, nrm = xyz.astype(np.float64), f[:, sl].astype(np.float64)
                cm = confusion_np(c, p)
                cm_rule = confusion_np(c, side_by_geometry(xyz, nrm, p))
                for key in ("all", source[name]):
                    cms[key] = cms.get(key, 0) + cm
                    cms_rule[key] = cms_rule.get(key, 0) + cm_rule
                sc = scores(cm)
                sc["miou17_side_rule"] = scores(cm_rule)["miou17"]
                scans.append(dict(name=name, source=source[name], upper=bool(upper), gt=c, pred=p,
                                  coord=xyz, normals=nrm, scores=sc))
        n = len(scans)
        print(f"[{tag}] {n} scans | model {t_model / n * 1000:.0f} ms/scan (prepare + forward, batch {a.batch}) "
              f"| {time.time() - t0:.0f}s total", flush=True)

        order = lambda d: sorted(d.items(), key=lambda kv: kv[0] != "all")  # noqa: E731
        summary[tag] = {k: scores(v) for k, v in order(cms)}
        summary[f"{tag}+side_rule"] = {k: scores(v) for k, v in order(cms_rule)}
        np.save(a.out / f"confusion_{tag}.npy", cms["all"])
        render_confusion(a.out / f"confusion_{tag}.png", cms["all"], tag)
        render_confusion(a.out / f"confusion_{tag}_side_rule.png", cms_rule["all"], f"{tag}, geometric side rule")
        for s in scans:
            table.append({"tag": tag, "name": s["name"], "source": s["source"],
                          "jaw": "upper" if s["upper"] else "lower",
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s["scores"].items()}})

        # overview: 데이터셋마다 mIoU17 중앙값 스캔, 마지막 줄은 전체에서 가장 나쁜 스캔
        rows = []
        for src in dict.fromkeys(s["source"] for s in scans):
            group = sorted((s for s in scans if s["source"] == src), key=lambda s: s["scores"]["miou17"])
            rows.append(group[len(group) // 2])
        rows.append(min(scans, key=lambda s: s["scores"]["miou17"]))
        al = summary[tag]["all"]
        headline = (f"PurePTv3 semantic, {tag} input{f', {a.crop} crop' if a.crop else ''}, "
                    f"epoch {ck['epoch']}, {n} val scans: "
                    f"mIoU17 {al['miou17'] * 100:.1f}  mIoU9 (L/R merged) {al['miou9'] * 100:.1f}  "
                    f"TSA {al['tsa'] * 100:.1f}\n"
                    f"tooth points: correct {al['tooth_correct'] * 100:.1f}%  L/R swap {al['tooth_lr_swap'] * 100:.1f}%  "
                    f"wrong position {al['tooth_wrong_pos'] * 100:.1f}%  to gingiva {al['tooth_to_gingiva'] * 100:.1f}%"
                    f"\nquadrant from geometry instead (predicted position + side of the midline): mIoU17 "
                    f"{summary[f'{tag}+side_rule']['all']['miou17'] * 100:.1f}"
                    f"      rows: median scan of each dataset, last row = worst scan")
        render_overview(a.out / f"overview_{tag}.png", rows, tag, headline)
        if not a.no_renders:
            rd = a.out / "renders" / tag
            rd.mkdir(parents=True, exist_ok=True)
            for s in scans:
                render_scan(rd / f"{s['source']}__{s['name']}.png", s)
        print(f"[{tag}] rendered | " + "  ".join(
            f"{k}: mIoU17 {v['miou17'] * 100:.1f} mIoU9 {v['miou9'] * 100:.1f} swap {v['tooth_lr_swap'] * 100:.1f}%"
            for k, v in summary[tag].items()), flush=True)
        print(f"[{tag}] geometric side rule | " + "  ".join(
            f"{k}: mIoU17 {v['miou17'] * 100:.1f} swap {v['tooth_lr_swap'] * 100:.1f}%"
            for k, v in summary[f"{tag}+side_rule"].items()), flush=True)

    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(a.out / "per_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
