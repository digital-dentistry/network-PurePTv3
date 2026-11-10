"""Inhouse failure analysis: PurePTv3 auto-label (toothseg-pureptv3/1) vs segmentation3d on the 12 labelled
inhouse scans, with one factor changed at a time, plus the same model on synthetic crops of the public val set.

Mounts: /src dental_ai, /pureptv3, /inference (DDH inference-server), /models, /seg3d, /data (AI/data), /s scratch.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path[:0] = ["/src", "/pureptv3", "/inference", "/seg3d"]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dental_ai.dataset.algorithm.voxel_downsample.voxel_downsample import (VoxelDownsampleInput,  # noqa: E402
                                                                            VoxelDownsampleOutput)
from dental_ai.dataset.algorithm.voxel_downsample.voxel_grid_cpu import voxel_grid_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import (ARCH_POSITION, ToothSegConfig, ToothSegDataset,  # noqa: E402
                                              feature_slices, patient_split, unit_normals)
from toothseg import NUM_CLASSES, POSITION, SECOND, Scan, ToothSeg17, decide_quadrant, to_fdi  # noqa: E402

DEV = torch.device("cuda")
GT_DIR = Path("/s/inhouse_gt")
RAW = Path("/data/ios/inhouse/raw")
PKG = Path("/models/toothseg-pureptv3/1")
spec = json.loads((PKG / "model.json").read_text())
AD = ToothSeg17(PKG, spec, DEV)
VOX = float(spec["preprocessing"]["voxel_mm"])
CAP = int(spec["preprocessing"].get("max_points") or 0)

from pipeline.rotations import tta_rotations  # noqa: E402  (segmentation3d's TTA poses, used for both models)
ROTS = tta_rotations(8).double()


# ------------------------------------------------------------------ metrics
def metrics(g: np.ndarray, p: np.ndarray, keep: np.ndarray | None = None) -> dict:
    if keep is not None:
        g, p = g[keep], p[keep]
    gt, pt = g > 0, p > 0
    both = gt & pt
    teeth = [int(t) for t in np.unique(g[gt])]
    ok = []
    for t in teeth:
        v, c = np.unique(p[g == t], return_counts=True)
        ok.append(int(v[c.argmax()]) == t)
    return {"f1": 2 * both.sum() / max(gt.sum() + pt.sum(), 1),
            "pos": float((g[both] % 10 == p[both] % 10).mean()) if both.any() else 0.0,
            "side": float((g[both] // 10 == p[both] // 10).mean()) if both.any() else 0.0,
            "fdi": float((g[gt] == p[gt]).mean()),
            "teeth_ok": float(np.mean(ok)) if ok else 0.0}


# ------------------------------------------------------------- PurePTv3 core
def grid(pos: torch.Tensor):
    ds = VoxelDownsampleOutput()
    if not voxel_grid_cpu(VoxelDownsampleInput(pos, VOX), ds):
        raise ValueError("empty")
    cell = torch.floor((pos - pos.amin(0)) / VOX).long()
    dims = cell.amax(0) + 1
    key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    inverse = torch.unique(key, sorted=True, return_inverse=True)[1]
    return ds, inverse


def vertex_probs(xyz, nrm, rgb, jaw, rot=None, normal_mode="rep") -> np.ndarray:
    """[V,17] like the adapter (one pose).  rot: [3,3] applied to positions and normals first."""
    pos = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32))
    n = unit_normals(torch.from_numpy(np.ascontiguousarray(nrm, dtype=np.float32)))
    if rot is not None:
        pos = (pos.double() @ rot.T).float()
        n = (n.double() @ rot.T).float()
    colour = None if rgb is None else torch.from_numpy(rgb.astype(np.float32) / 255.0)
    ds, inverse = grid(pos)
    rep = ds.representative
    if normal_mode == "mean":
        acc = torch.zeros(rep.numel(), 3).index_add_(0, inverse, n)
        nrep = unit_normals(acc)
    else:
        nrep = n[rep]
    m = rep.numel()
    chunks = max(1, math.ceil(m / (CAP or m)))
    order = torch.randperm(m, generator=torch.Generator().manual_seed(0)) if chunks > 1 else torch.arange(m)
    prob = torch.empty(m, NUM_CLASSES)
    for idx in torch.tensor_split(order, chunks):
        r = rep[idx]
        prob[idx] = AD._logits(pos[r], nrep[idx], None if colour is None else colour[r], ds.cells[idx], jaw == "upper")
    return prob[inverse].numpy()


def finish(prob: np.ndarray, xyz, nrm, jaw, side: str = "rule", gt=None) -> tuple[np.ndarray, str]:
    """vertex probs -> FDI.  side: "rule" (adapter's midline rule on the voxel reps), "raw" (17-class argmax),
    "oracle" (the GT quadrant of each vertex; non-GT-tooth vertices keep the rule)."""
    pred = prob.argmax(1)
    pos = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32))
    ds, inverse = grid(pos)
    rep = ds.representative.numpy()
    coord = xyz[rep].astype(np.float64)
    n = nrm[rep].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    second_rep, method, _ = decide_quadrant(coord - coord.mean(0), n, pred[rep])
    second = second_rep[inverse.numpy()]
    if side == "raw":
        second = SECOND[pred]
    label, fdi = to_fdi(pred, second, jaw)
    if side == "oracle":
        t = (gt > 0) & (fdi > 0)
        fdi = fdi.copy()
        fdi[t] = (gt[t] // 10) * 10 + fdi[t] % 10
    return fdi, method


def near(xyz: np.ndarray, sel: np.ndarray, mm: float) -> np.ndarray:
    """vertices within mm of any selected vertex (chunked GPU cdist on a 0.5 mm subsample of the selection)."""
    pts = np.ascontiguousarray(xyz[sel], dtype=np.float32)
    ds = VoxelDownsampleOutput()
    voxel_grid_cpu(VoxelDownsampleInput(torch.from_numpy(pts), 0.5), ds)
    src = torch.from_numpy(pts[ds.representative.numpy()]).to(DEV)
    q = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)).to(DEV)
    keep = torch.zeros(len(xyz), dtype=torch.bool, device=DEV)
    for i in range(0, len(xyz), 20000):
        keep[i:i + 20000] = torch.cdist(q[i:i + 20000], src).amin(1) < mm
    return keep.cpu().numpy() | sel


# ------------------------------------------------------- segmentation3d
from pipeline.config import Placement, SegmentationConfig  # noqa: E402
from pipeline.stages import LoadStage  # noqa: E402
from tools.semantic_view import SemanticModel  # noqa: E402
from tools.preprocess_view import fdi as seg3d_fdi  # noqa: E402

S3D = {t: SemanticModel(Path("/seg3d/runs/semantic/best.pt"), t, DEV) for t in (8, 1)}
_cfg = SegmentationConfig()
LOAD = LoadStage(_cfg, Placement.resolve(_cfg))


def seg3d(name: str, xyz: np.ndarray, tta: int, keep: np.ndarray | None = None) -> np.ndarray:
    scan = LOAD.run(RAW / f"{name}.ply")
    if scan.positions.shape[0] != len(xyz) or float((scan.positions.cpu().numpy() - xyz).__abs__().max()) > 1e-4:
        raise RuntimeError(f"{name}: raw PLY vertices differ from DDH mesh")
    cls, _, _ = S3D[tta].predict(scan)
    upper = scan.jaw == "upper"
    lut = np.array([0] + [int(seg3d_fdi(c, upper)) for c in range(1, 17)])
    return lut[cls]


# ------------------------------------------------------------------ inhouse
rows = []
for f in sorted(GT_DIR.glob("*.npz")):
    d = np.load(f)
    name, jaw = f.stem, str(d["jaw"])
    xyz, nrm, g = d["xyz"], d["normals"], d["fdi"].astype(np.int64)
    rgb = d["rgb"] if "rgb" in d.files else None
    full = len({int(t) // 10 for t in np.unique(g[g > 0])}) == 2
    ds0, _ = grid(torch.from_numpy(xyz))
    ging_share = float((g[ds0.representative.numpy()] == 0).mean())
    t0 = time.time()
    res = {}

    pr = AD.predict(Scan(xyz, nrm, rgb, jaw))
    res["P0 dashboard (as is)"] = metrics(g, pr.fdi.astype(np.int64))
    method0 = pr.info["quadrant_method"]

    p_rgb = vertex_probs(xyz, nrm, rgb, jaw)
    p_grey = vertex_probs(xyz, nrm, None, jaw)
    res["P1 replica of P0"] = metrics(g, finish(p_rgb, xyz, nrm, jaw)[0])
    res["P2 no colour"] = metrics(g, finish(p_grey, xyz, nrm, jaw)[0])
    res["P3 side = raw model output"] = metrics(g, finish(p_grey, xyz, nrm, jaw, "raw")[0])
    res["P4 side = GT (oracle)"] = metrics(g, finish(p_grey, xyz, nrm, jaw, "oracle", g)[0])
    p_tta = np.mean([vertex_probs(xyz, nrm, None, jaw, r) for r in ROTS], 0)
    res["P5 TTA x8"] = metrics(g, finish(p_tta, xyz, nrm, jaw)[0])
    res["P5o TTA x8 + side GT"] = metrics(g, finish(p_tta, xyz, nrm, jaw, "oracle", g)[0])
    res["P6 voxel-mean normals"] = metrics(g, finish(vertex_probs(xyz, nrm, None, jaw, normal_mode="mean"),
                                                     xyz, nrm, jaw)[0])
    flip = "lower" if jaw == "upper" else "upper"
    pf = finish(vertex_probs(xyz, nrm, None, flip), xyz, nrm, flip)[0].astype(np.int64)
    pf = np.where(pf > 0, (pf // 10 + (2 if jaw == "lower" else -2)) * 10 + pf % 10, 0)   # back to true jaw digits
    res["P7 jaw flag flipped"] = metrics(g, pf)

    # gingiva: GT-based crop (diagnostic) and label-free two-pass crop
    k_gt = near(xyz, g > 0, 3.0)
    pk = vertex_probs(xyz[k_gt], nrm[k_gt], None, jaw)
    fk, _ = finish(pk, xyz[k_gt], nrm[k_gt], jaw, "oracle", g[k_gt])
    res["P8 GT teeth+3mm crop, side GT"] = metrics(g[k_gt], fk)
    first = finish(p_grey, xyz, nrm, jaw)[0] > 0
    k_pr = near(xyz, first, 3.0)
    pk = vertex_probs(xyz[k_pr], nrm[k_pr], None, jaw)
    fk, _ = finish(pk, xyz[k_pr], nrm[k_pr], jaw, "oracle", g[k_pr])
    out = np.zeros_like(g)
    out[k_pr] = fk
    res["P9 2-pass pred teeth+3mm crop, side GT"] = metrics(g, out)
    ging_share_crop = float((g[k_gt] == 0).mean())

    s8 = seg3d(name, xyz, 8)
    res["S8 segmentation3d TTA x8"] = metrics(g, s8)
    res["S1 segmentation3d TTA x1"] = metrics(g, seg3d(name, xyz, 1))
    s8o = np.where((g > 0) & (s8 > 0), (g // 10) * 10 + s8 % 10, s8)
    res["S8o segmentation3d TTA x8, side GT"] = metrics(g, s8o)
    # side agreement between models (independent of the human side label)
    both = (g > 0) & (s8 > 0) & (pr.fdi > 0)
    agree_side = float((s8[both] // 10 == pr.fdi.astype(np.int64)[both] // 10).mean())

    row = {"name": name, "full": full, "method": method0, "gingiva_share_vox": round(ging_share, 3),
           "gingiva_share_after_crop": round(ging_share_crop, 3), "colour": rgb is not None,
           "seg3d_vs_pureptv3_side_agree": round(agree_side, 3), "res": res, "s": round(time.time() - t0, 1)}
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "res"}), flush=True)
    for k, v in res.items():
        print(f"    {k:42s} " + " ".join(f"{m} {x:.3f}" for m, x in v.items()), flush=True)

(Path("/s/inhouse_eval.json")).write_text(json.dumps(rows, indent=1, default=float))


def mean_table(rows, sel, title):
    print(f"\n== {title} ({len([r for r in rows if sel(r)])} scans)")
    keys = list(rows[0]["res"])
    for k in keys:
        vals = [r["res"][k] for r in rows if sel(r)]
        print(f"  {k:42s} " + " ".join(f"{m} {np.mean([v[m] for v in vals]):.3f}" for m in vals[0]))


mean_table(rows, lambda r: True, "all")
mean_table(rows, lambda r: not r["full"], "partial (one quadrant)")
mean_table(rows, lambda r: r["full"], "full arch")

# --------------------------------------------- public val, synthetic crops
from train_toothseg import IOS_TRAIN_SPLITS, ios_names  # noqa: E402

caches, names = ios_names(Path("/data/ios"), list(IOS_TRAIN_SPLITS))     # same split as training
_, val = patient_split(names, 0.1, seed=0)
val = [n for n in val if not (Path("/data/ios/3d_iosseg/cache") / f"{n}.npz").exists()]   # too sparse to crop
val = val[::3][:40]
sl = None
print(f"\n== public val, synthetic crops ({len(val)} scans, grey)")
for mode in (None, "half", "quadrant", "bridge"):
    cfg = ToothSegConfig(voxel_mm=VOX, features=AD.features, max_points=CAP or None, train=False, crop_force=mode)
    sl = feature_slices(cfg)
    ds = ToothSegDataset(caches, val, cfg)
    acc = {"pos": [], "side_rule": [], "side_raw": [], "teeth_ok_oracle_side": [], "gingiva_share": [], "n_teeth": []}
    for i in range(len(ds)):
        it = ds[i]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            logits = AD.model(it.features.to(DEV), AD.model.prepare(it.grid.to(DEV)))
        pred = logits.float().argmax(1).cpu().numpy()
        cls = it.classes.numpy()
        known = cls < 17
        gpos = ARCH_POSITION[torch.from_numpy(cls)].numpy()
        ppos = POSITION[pred]
        both = known & (cls > 0) & (pred > 0)
        coord = it.coord.double().numpy()
        nrm = it.features[:, sl["normal"]].double().numpy()
        second, _, _ = decide_quadrant(coord, nrm, pred)
        gsec = (cls >= 9) & (cls <= 16)
        acc["pos"].append((gpos[both] == ppos[both]).mean())
        acc["side_rule"].append((gsec[both] == second[both]).mean())
        acc["side_raw"].append((gsec[both] == SECOND[pred][both]).mean())
        ok = []
        for t in np.unique(it.instances.numpy()[cls > 0]):
            m = (it.instances.numpy() == t) & known
            if m.any():
                v, c = np.unique(ppos[m], return_counts=True)
                ok.append(v[c.argmax()] == gpos[m][0])
        acc["teeth_ok_oracle_side"].append(np.mean(ok))
        acc["gingiva_share"].append((cls == 0).mean())
        acc["n_teeth"].append(len(np.unique(it.instances.numpy()[cls > 0])))
    print(f"  {str(mode or 'full'):9s} " + " ".join(f"{k} {np.mean(v):.3f}" for k, v in acc.items()), flush=True)
