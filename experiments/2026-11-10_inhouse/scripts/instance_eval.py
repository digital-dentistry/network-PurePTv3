"""E10 치아 instance 품질 (번호 무시): 라벨 있는 inhouse 스캔에서 PurePTv3 instance 추출법·후처리별, segmentation3d와 비교.

정답 instance = 사람 라벨의 FDI마다 하나 (번호가 틀려도 경계는 그대로 쓴다).
예측 instance = 라벨 영역의 mesh connected component (면을 공유하는 이웃).
면적 가중 (정점 면적 = 닿은 삼각형 넓이 / 3) — 스캐너가 치아를 더 촘촘히 meshing해도 치우치지 않게.

지표
  P/R/F1@50, @75   IoU로 1:1 매칭 (Hungarian), IoU >= 0.5 / 0.75면 맞음
  mIoU             정답 치아마다 가장 잘 맞는 예측의 IoU 평균
  merge            예측 하나가 정답 치아 둘 이상을 각각 30% 넘게 덮음 (그 예측 수)
  split            정답 치아 하나를 예측 둘 이상이 각각 20% 넘게 덮음 (그 정답 수)
  margin GT->P     정답 치아-잇몸 경계 정점에서 예측 치아-잇몸 경계까지 거리 평균 (mm).  P->GT는 반대 방향
  inter  GT->P     정답 치아-치아 경계에서 예측 치아-치아 경계까지 (놓친 경계 = merge).  P->GT: 헛 경계 = split
  tooth area +/-   치아인데 잇몸으로 / 잇몸인데 치아로 (mm²)

    docker run --gpus all ... hdl-gpu:latest python /exp/scripts/instance_eval.py /gt
    (/gt: export_gt.py가 만든 INHOUSE*.npz, /data: AI/data, /exp: 이 실험 폴더)
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path[:0] = ["/src", "/pureptv3", "/inference", "/seg3d"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.sparse import coo_matrix, diags  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from dental_ai.dataset.algorithm.voxel_downsample.voxel_downsample import (VoxelDownsampleInput,  # noqa: E402
                                                                            VoxelDownsampleOutput)
from dental_ai.dataset.algorithm.voxel_downsample.voxel_grid_cpu import voxel_grid_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import unit_normals  # noqa: E402
from toothseg import NUM_CLASSES, Scan, ToothSeg17, decide_quadrant  # noqa: E402

DEV = torch.device("cuda")
GT = Path(sys.argv[1] if len(sys.argv) > 1 else "/gt")
OUT = Path("/exp/results")
RAW = Path("/data/ios/inhouse/raw")
PKG = Path("/models/toothseg-pureptv3/1")
spec = json.loads((PKG / "model.json").read_text())
AD = ToothSeg17(PKG, spec, DEV)
VOX = float(spec["preprocessing"]["voxel_mm"])
CAP = int(spec["preprocessing"].get("max_points") or 0)
MIN_AREA = 5.0          # mm², 이보다 작은 조각은 정리 대상
PAIR = np.array([0] + list(range(1, 9)) * 2)          # 17-class -> 위치 0..8

from pipeline.config import Placement, SegmentationConfig  # noqa: E402
from pipeline.rotations import tta_rotations  # noqa: E402
from pipeline.segmenter import Segmenter  # noqa: E402
from pipeline.stages import LoadStage  # noqa: E402
from tools.semantic_view import SemanticModel  # noqa: E402

ROTS = tta_rotations(8).double()
_cfg = SegmentationConfig()
LOAD = LoadStage(_cfg, Placement.resolve(_cfg))
S3D_SEM = SemanticModel(Path("/seg3d/runs/semantic/best.pt"), 8, DEV)
S3D_INST = Segmenter(SegmentationConfig(checkpoint=Path("/seg3d/runs/instance_ios/best.pt"), tta=8), write_json=False)


# ------------------------------------------------------------------ mesh
def edges_of(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def vertex_area(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = 0.5 * np.linalg.norm(np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]]), axis=1)
    return sum(np.bincount(faces[:, c], weights=a / 3, minlength=len(xyz)) for c in range(3))


def components(lab: np.ndarray, edges: np.ndarray, include_zero: bool = False) -> np.ndarray:
    """같은 라벨끼리 이어진 조각 번호 (1..K).  include_zero=False면 라벨 0은 0."""
    n = len(lab)
    same = lab[edges[:, 0]] == lab[edges[:, 1]]
    if not include_zero:
        same &= lab[edges[:, 0]] > 0
    e = edges[same]
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    comp = comp + 1
    if not include_zero:
        comp[lab == 0] = 0
    _, comp = np.unique(comp, return_inverse=True)
    return comp if include_zero or (lab == 0).any() else comp + 1


def cleanup(lab: np.ndarray, edges: np.ndarray, area: np.ndarray, min_area: float = MIN_AREA) -> np.ndarray:
    """min_area보다 작은 조각 (치아 조각, 치아 속 잇몸 구멍 모두)을 가장 길게 맞닿은 이웃 라벨로."""
    lab = lab.copy()
    for _ in range(4):
        comp = components(lab, edges, include_zero=True)
        carea = np.bincount(comp, weights=area)
        small = carea < min_area
        if not small[comp].any():
            break
        a, b = comp[edges[:, 0]], comp[edges[:, 1]]
        cross = a != b
        src = np.concatenate([a[cross], b[cross]])
        nbr_lab = np.concatenate([lab[edges[cross, 1]], lab[edges[cross, 0]]])
        keep = small[src]
        src, nbr_lab = src[keep], nbr_lab[keep]
        if not len(src):
            break
        base = int(lab.max()) + 1                     # 라벨 번호가 몇이든 (instance 번호도 된다)
        key = src.astype(np.int64) * base + nbr_lab
        k, cnt = np.unique(key, return_counts=True)
        best = {}
        for kk, c in zip(k, cnt):
            s, l = divmod(int(kk), base)
            if c > best.get(s, (0, -1))[0]:
                best[s] = (c, l)
        new = np.zeros(len(carea), np.int64)          # 이웃이 없는 조각은 잇몸으로
        for s, (_, l) in best.items():
            new[s] = l
        m = small[comp]
        lab[m] = new[comp[m]]
    return lab


def smooth(prob: np.ndarray, edges: np.ndarray, area: np.ndarray, iters: int) -> np.ndarray:
    """mesh 1-ring에서 면적 가중 평균을 iters번."""
    n = len(prob)
    w = np.concatenate([area[edges[:, 1]], area[edges[:, 0]], area])
    r = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    c = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    A = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    A = diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    for _ in range(iters):
        prob = A @ prob
    return prob


# ------------------------------------------------------------- PurePTv3
def rep_probs(xyz, nrm, rgb, jaw, rot=None):
    """voxel 대표점의 확률 (어댑터와 같은 전처리).  -> rep 정점 번호, [m,17], 정점 -> rep."""
    pos = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32))
    n = unit_normals(torch.from_numpy(np.ascontiguousarray(nrm, dtype=np.float32)))
    if rot is not None:
        pos, n = (pos.double() @ rot.T).float(), (n.double() @ rot.T).float()
    colour = None if rgb is None else torch.from_numpy(rgb.astype(np.float32) / 255.0)
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(pos, VOX), ds)
    cell = torch.floor((pos - pos.amin(0)) / VOX).long()
    dims = cell.amax(0) + 1
    inverse = torch.unique((cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2], sorted=True,
                           return_inverse=True)[1]
    rep = ds.representative
    m = rep.numel()
    chunks = max(1, math.ceil(m / (CAP or m)))
    order = torch.randperm(m, generator=torch.Generator().manual_seed(0)) if chunks > 1 else torch.arange(m)
    prob = torch.empty(m, NUM_CLASSES)
    for idx in torch.tensor_split(order, chunks):
        r = rep[idx]
        prob[idx] = AD._logits(pos[r], n[r], None if colour is None else colour[r], ds.cells[idx], jaw == "upper")
    return rep.numpy(), prob.numpy(), inverse.numpy()


def upsample(xyz, rep, prob, inverse, mode):
    if mode == "nearest":
        return prob[inverse]
    d, i = cKDTree(xyz[rep]).query(xyz, k=5, workers=-1)
    w = 1.0 / (d + 0.05)
    return (prob[i] * w[..., None]).sum(1) / w.sum(1, keepdims=True)


def vertex_probs(xyz, nrm, rgb, jaw, tta=False, up="nearest"):
    poses = list(ROTS) if tta else [None]
    acc = 0
    for r in poses:
        rep, p, inv = rep_probs(xyz, nrm, rgb, jaw, r)
        acc = acc + upsample(xyz, rep, p, inv, up)
    return acc / len(poses)


def pos9(prob17):
    p9 = np.zeros((len(prob17), 9), prob17.dtype)
    for c in range(NUM_CLASSES):
        p9[:, PAIR[c]] += prob17[:, c]
    return p9


def with_rule(pos_lab, xyz, nrm):
    """위치 라벨 + 어댑터의 정중선 규칙으로 좌우 -> 17-class 라벨 (정중선의 두 중절치를 가르려고)."""
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)), VOX), ds)
    rep = ds.representative.numpy()
    coord = xyz[rep].astype(np.float64)
    n = nrm[rep].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    second_rep, _, _ = decide_quadrant(coord - coord.mean(0), n, pos_lab[rep])
    # 대표점의 좌우를 가장 가까운 대표점으로 정점에 옮긴다
    second = second_rep[cKDTree(coord).query(xyz, workers=-1)[1]]
    return np.where(pos_lab > 0, pos_lab + 8 * second, 0)


# ------------------------------------------------------------------ metrics
def boundary(inst, edges, kind):
    """kind: margin (치아-잇몸) / inter (치아-치아) 경계 정점."""
    a, b = inst[edges[:, 0]], inst[edges[:, 1]]
    if kind == "margin":
        m = (a > 0) != (b > 0)
    else:
        m = (a > 0) & (b > 0) & (a != b)
    v = np.zeros(len(inst), bool)
    v[edges[m, 0]] = True
    v[edges[m, 1]] = True
    return v


def directed(xyz, src, dst):
    if not src.any():
        return float("nan")
    if not dst.any():
        return float("inf")
    return float(cKDTree(xyz[dst]).query(xyz[src], workers=-1)[0].mean())


def score(gt, pr, area, xyz, edges):
    G, P = gt.max(), pr.max()
    both = (gt > 0) & (pr > 0)
    inter = coo_matrix((area[both], (gt[both] - 1, pr[both] - 1)), shape=(G, P)).toarray()
    ga = np.bincount(gt, weights=area, minlength=G + 1)[1:]
    pa = np.bincount(pr, weights=area, minlength=P + 1)[1:]
    iou = inter / np.maximum(ga[:, None] + pa[None, :] - inter, 1e-9)
    r, c = linear_sum_assignment(-iou)
    m = iou[r, c]
    res = {}
    for t in (0.5, 0.75):
        tp = int((m >= t).sum())
        res[f"P{int(t*100)}"] = tp / max(P, 1)
        res[f"R{int(t*100)}"] = tp / max(G, 1)
        res[f"F{int(t*100)}"] = 2 * tp / max(G + P, 1)
    res["mIoU"] = float(iou.max(1).mean()) if P else 0.0
    res["merge"] = int(((inter / ga[:, None]) > 0.3).sum(0).__ge__(2).sum())
    res["split"] = int(((inter / ga[:, None]) > 0.2).sum(1).__ge__(2).sum())
    res["n_gt"], res["n_pr"] = int(G), int(P)
    gm, pm = boundary(gt, edges, "margin"), boundary(pr, edges, "margin")
    gi, pi = boundary(gt, edges, "inter"), boundary(pr, edges, "inter")
    res["margin_gt2p"], res["margin_p2gt"] = directed(xyz, gm, pm), directed(xyz, pm, gm)
    res["inter_gt2p"], res["inter_p2gt"] = directed(xyz, gi, pi), directed(xyz, pi, gi)
    res["miss_mm2"] = float(area[(gt > 0) & (pr == 0)].sum())
    res["extra_mm2"] = float(area[(gt == 0) & (pr > 0)].sum())
    return res


# ------------------------------------------------------------------ run
rows, keep_pred = [], {}
for f in sorted(GT.glob("INHOUSE*.npz")):
    d = np.load(f)
    name, jaw = f.stem, str(d["jaw"])
    xyz, nrm, fdi = d["xyz"], d["normals"], d["fdi"].astype(np.int64)
    rgb = d["rgb"] if "rgb" in d.files else None
    scan = LOAD.run(RAW / f"{name}.ply")
    faces = scan.faces.cpu().numpy().astype(np.int64)
    assert scan.positions.shape[0] == len(xyz)
    edges = edges_of(faces)
    area = vertex_area(xyz.astype(np.float64), faces)
    x64 = xyz.astype(np.float64)
    gt = np.where(fdi > 0, np.unique(fdi, return_inverse=True)[1], 0)   # FDI 하나 = instance 하나
    full = len({int(t) // 10 for t in np.unique(fdi[fdi > 0])}) == 2
    t0 = time.time()
    preds = {}

    # 대시보드 그대로 (색은 대시보드처럼: 실제 색이 있을 때만)
    dash = AD.predict(Scan(xyz, nrm, rgb, jaw)).fdi.astype(np.int64)
    preds["A dashboard FDI regions"] = components(dash, edges)

    p_grey = vertex_probs(xyz, nrm, None, jaw)
    p_col = vertex_probs(xyz, nrm, rgb, jaw)
    preds["B grey 17-class regions"] = components(p_grey.argmax(1), edges)
    preds["C grey pos9 regions"] = components(pos9(p_grey).argmax(1), edges)
    preds["D grey pos9+rule"] = components(with_rule(pos9(p_grey).argmax(1), x64, nrm), edges)
    preds["E grey pos9+rule+clean"] = components(cleanup(with_rule(pos9(p_grey).argmax(1), x64, nrm), edges, area), edges)
    preds["F colour pos9+rule+clean"] = components(cleanup(with_rule(pos9(p_col).argmax(1), x64, nrm), edges, area), edges)
    p_knn = vertex_probs(xyz, nrm, None, jaw, up="knn")
    preds["G knn pos9+rule+clean"] = components(cleanup(with_rule(pos9(p_knn).argmax(1), x64, nrm), edges, area), edges)
    p_tta = vertex_probs(xyz, nrm, None, jaw, tta=True, up="knn")
    preds["H tta8+knn pos9+rule+clean"] = components(cleanup(with_rule(pos9(p_tta).argmax(1), x64, nrm), edges, area), edges)
    for it in (3, 10):
        ps = smooth(pos9(p_tta), edges, area, it)
        preds[f"I tta8+knn+smooth{it} pos9+rule+clean"] = components(cleanup(with_rule(ps.argmax(1), x64, nrm), edges, area), edges)

    preds["S1 seg3d instance_ios tta8"] = components(S3D_INST.segment(scan).instances.astype(np.int64), edges)
    cls = S3D_SEM.predict(scan)[0]
    preds["S2 seg3d semantic tta8 pos9+rule+clean"] = components(cleanup(with_rule(PAIR[cls], x64, nrm), edges, area), edges)

    res = {k: score(gt, v, area, x64, edges) for k, v in preds.items()}
    rows.append({"name": name, "full": full, "n_teeth": int(gt.max()), "res": res, "s": round(time.time() - t0, 1)})
    keep_pred[name] = {k.replace(" ", "_"): v.astype(np.int16) for k, v in preds.items()} | {"GT": gt.astype(np.int16)}
    print(f"{name} full={full} teeth={gt.max()} ({time.time() - t0:.0f}s)", flush=True)
    for k, v in res.items():
        print(f"   {k:40s} F50 {v['F50']:.2f} F75 {v['F75']:.2f} mIoU {v['mIoU']:.3f} n {v['n_pr']:2d}/{v['n_gt']:2d} "
              f"merge {v['merge']} split {v['split']} | margin {v['margin_gt2p']:.2f}/{v['margin_p2gt']:.2f} "
              f"inter {v['inter_gt2p']:.2f}/{v['inter_p2gt']:.2f} | miss {v['miss_mm2']:.0f} extra {v['extra_mm2']:.0f} mm2",
              flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "instance_eval.json").write_text(json.dumps(rows, indent=1, default=float))
np.savez_compressed(OUT / "instance_pred.npz", **{f"{n}__{k}": v for n, d in keep_pred.items() for k, v in d.items()})


def table(sel, title):
    rs = [r for r in rows if sel(r)]
    print(f"\n== {title} ({len(rs)} scans, {sum(r['n_teeth'] for r in rs)} teeth)")
    print(f"   {'method':40s}  F50   F75  mIoU  merge split  margin GT>P P>GT  inter GT>P P>GT  miss  extra(mm2/scan)")
    for k in rows[0]["res"]:
        v = [r["res"][k] for r in rs]
        mean = lambda m: np.nanmean([x[m] for x in v])  # noqa: E731
        tot = lambda m: int(sum(x[m] for x in v))  # noqa: E731
        print(f"   {k:40s} {mean('F50'):.3f} {mean('F75'):.3f} {mean('mIoU'):.3f} {tot('merge'):5d} {tot('split'):5d}"
              f"   {mean('margin_gt2p'):.2f} {mean('margin_p2gt'):.2f}      {mean('inter_gt2p'):.2f} {mean('inter_p2gt'):.2f}"
              f"  {mean('miss_mm2'):5.0f} {mean('extra_mm2'):5.0f}")


table(lambda r: True, "all")
table(lambda r: not r["full"], "partial (one quadrant)")
table(lambda r: r["full"], "full arch")
