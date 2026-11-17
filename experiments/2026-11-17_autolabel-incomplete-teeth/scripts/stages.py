"""Where does auto-label lose part of a tooth?  Stage-by-stage diagnosis on the painted Teeth3DS scans.

Runs in a throwaway ddh-inference-server:2026-11-10 container with the GPU (see README for the command). Reads the
probe npz files (../2026-11-16_teeth3ds-unlabeled/scripts/probe.py output: xyz, faces, rgb, pred_grey_*), never
touches the DB, S3 or the running server.

Ground truth = the painted colour map (2026-11-16 finding): the most common colour is gingiva, every other
colour is one tooth.

The instance pipeline of toothsegnet-ios/1, rgb removed (= the dashboard's "컬러 미사용"):
  L0      voxel 0.25 mm cloud, 8 TTA poses, network -> per pose  conf_t = sigmoid(mask) * P(tooth)
  fuse    pose t's queries Hungarian-matched to pose 0, conf = mean_t conf_t            (QueryFusion)
  up      kNN(5) inverse-distance interpolation of conf onto every vertex              (MaskUpsample)
  assign  vertex -> argmax query if conf > 0.5, else gingiva; instances < 0.5 % dropped (InstanceExtraction)
  clean   absorb_fragments(5 mm2) + components                                          (toothsegnet.py)

For every colour tooth it records IoU / recall against the final prediction and, for the tooth area that was
called gingiva, which stage dropped it (see `causes`).  It also scores post-processing variants on the same
network output (`VARIANTS`).

usage: python stages.py <npz dir> <out dir> < asset ids
stdout: one JSON line per scan.  <out dir>/<asset>.npz: GT and the final labels of every variant.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app")
from toothseg import MIN_FRAGMENT_MM2, Scan, absorb_fragments, components, vertex_graph  # noqa: E402
from toothsegnet import ToothSegNet  # noqa: E402
from algorithm.instance_extraction.common.labels import assign, renumber  # noqa: E402
from algorithm.instance_network.instance_network import InstanceNetworkInput, InstanceNetworkOutput  # noqa: E402
from algorithm.mask_upsample.common.interpolation import knn_interpolate  # noqa: E402
from algorithm.query_fusion.common.matching import query_confidence  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

PKG = Path(os.environ.get("PKG", "/models/toothsegnet-ios/1"))
THRESH = 0.5
MIN_FRAC = 0.005
KNN = 5


def vertex_normals(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """ddh.model.ios.read_mesh: area-weighted (unnormalised cross product) vertex normals."""
    p = xyz.astype(np.float64)
    c = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, faces[:, k], c)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def colour_gt(rgb: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[V] 0 = gingiva (largest colour by area), 1..T one per other colour; and [T+1,3] the colour of each id."""
    colours, inv = np.unique(rgb, axis=0, return_inverse=True)
    inv = inv.ravel()
    gum = int(np.argmax(np.bincount(inv, weights=area)))
    gt = inv + 1
    gt[inv == gum] = 0
    ids, gt = np.unique(gt, return_inverse=True)  # 0 stays 0 (it is the smallest), teeth 1..T
    colour_of = np.array([colours[gum]] + [colours[i - 1] for i in ids[1:]])
    return gt.ravel(), colour_of


def align_poses(conf: torch.Tensor) -> list[torch.Tensor]:
    """conf [T,N,K] -> per pose the column order that lines its queries up with pose 0 (QueryFusion.fuse_poses
    matching, kept so every pose stays visible and the bare mask can be aligned the same way)."""
    base = conf[0]
    base_sum = base.sum(0)
    orders = [torch.arange(conf.shape[2], device=conf.device)]
    for t in range(1, conf.shape[0]):
        other = conf[t]
        inter = base.T @ other
        union = base_sum[:, None] + other.sum(0)[None] - inter
        rows, cols = linear_sum_assignment(-(inter / union.clamp_min(1e-6)).cpu().numpy())
        order = torch.empty(conf.shape[2], dtype=torch.int64, device=conf.device)
        order[torch.as_tensor(rows, device=conf.device)] = torch.as_tensor(cols, device=conf.device)
        orders.append(order)
    return orders


def agreement(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Area share on which two instance maps are the same partition (each instance -> its majority partner,
    both directions, the smaller)."""
    def one(x, y):
        ny = int(y.max()) + 1
        inter = np.bincount(x.astype(np.int64) * ny + y, weights=w, minlength=(int(x.max()) + 1) * ny)
        return inter.reshape(-1, ny).max(1).sum() / w.sum()
    return float(min(one(a, b), one(b, a)))


def finish(conf: torch.Tensor, points, query, thresh, edges, area, clean=True, active=None, voxel_mm=0.25):
    """conf [N,K] at L0 -> final instance ids [V] (MaskUpsample KNN + InstanceExtraction + toothsegnet clean)."""
    if active is None:
        active = torch.nonzero(conf.amax(0) > thresh).squeeze(1)
    if active.numel() == 0:
        return np.zeros(query.shape[0], np.int64), None
    full = knn_interpolate(conf[:, active], points, query, KNN, voxel_mm)
    ins = assign(full, active, query.shape[0], thresh)
    ins, _ = renumber(ins, conf.shape[1], MIN_FRAC)
    ins = ins.cpu().numpy().astype(np.int64)
    if clean:
        ins = components(absorb_fragments(ins, edges, area, MIN_FRAGMENT_MM2), edges)
    return ins, full


def match(gt: np.ndarray, pred: np.ndarray, w: np.ndarray):
    """Per GT tooth: best pred instance by IoU (greedy 1:1), IoU, recall, where its missed area went."""
    n_g, n_p = int(gt.max()) + 1, int(pred.max()) + 1
    inter = np.zeros((n_g, n_p))
    np.add.at(inter, (gt, pred), w)
    wg, wp = inter.sum(1), inter.sum(0)
    iou = inter / np.maximum(wg[:, None] + wp[None] - inter, 1e-12)
    pairs = sorted(((iou[i, j], i, j) for i in range(1, n_g) for j in range(1, n_p) if inter[i, j] > 0),
                   reverse=True)
    used_g, used_p, best = set(), set(), {}
    for v, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i), used_p.add(j)
        best[i] = j
    teeth = []
    for i in range(1, n_g):
        j = best.get(i, 0)
        to_gum = inter[i, 0] / wg[i]
        to_other = 1 - (inter[i, j] if j else 0) / wg[i] - to_gum
        teeth.append({"gt": i, "pred": int(j), "iou": float(iou[i, j]) if j else 0.0,
                      "recall": float(inter[i, j] / wg[i]) if j else 0.0,
                      "missed_to_gum": float(to_gum), "missed_to_other": float(max(to_other, 0.0)),
                      "extra_from_gum": float(inter[0, j] / wp[j]) if j else 0.0, "area": float(wg[i])})
    tp = sum(1 for t in teeth if t["iou"] >= 0.75)
    f1 = 2 * tp / max((n_g - 1) + (n_p - 1), 1)
    return teeth, f1, n_p - 1


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    spec = json.loads((PKG / "model.json").read_text())
    tsn = ToothSegNet(PKG, spec, dev)
    seg = tsn.instance
    for line in sys.stdin:
        aid = line.strip()
        if not aid:
            continue
        t0 = time.time()
        d = np.load(src / f"{aid}.npz")
        xyz, faces, jaw = d["xyz"], d["faces"].astype(np.int64), str(d["jaw"])
        scan = Scan(xyz=xyz, normals=vertex_normals(xyz, faces), rgb=None, jaw=jaw, faces=faces)
        edges, area = vertex_graph(xyz.astype(np.float64), faces)
        gt, colour_of = colour_gt(d["rgb"], area)

        with torch.inference_mode():
            s3 = tsn._seg3d_scan(scan)
            prep = seg._prepare.run(s3)
            cls, mask = [], []
            for b in prep.batches:
                o = InstanceNetworkOutput()
                seg._infer._network.execute(InstanceNetworkInput(b.features, b.geometry), o)
                cls.append(o.cls_logits)
                mask.append(o.mask_logits)
            cls, mask = torch.cat(cls).float(), torch.cat(mask).float()
            n = prep.points.shape[0]
            conf_t, prob_t = query_confidence(cls, mask, n)                 # [T,N,K], [T,K]
            sig_t = mask.sigmoid().view(cls.shape[0], n, -1)
            orders = align_poses(conf_t)
            al_c = torch.stack([conf_t[t][:, o] for t, o in enumerate(orders)])   # [T,N,K]
            al_s = torch.stack([sig_t[t][:, o] for t, o in enumerate(orders)])    # bare sigmoid(mask)
            al_p = torch.stack([prob_t[t][o] for t, o in enumerate(orders)])      # [T,K]
            fused = al_c.mean(0)                                            # == QueryFusion output
            points = prep.points
            query = s3.positions.to(dev) - prep.centre.to(dev)
            inverse = prep.inverse.to(dev)

            variants = {}
            # V0: the deployed pipeline (must match the server's pred_grey)
            v0, full = finish(fused, points, query, THRESH, edges, area)
            variants["v0_deployed"] = v0
            variants["v0_noclean"], _ = finish(fused, points, query, THRESH, edges, area, clean=False)
            # V1: decide the mask extent on sigmoid(mask) alone; P(tooth) only picks the queries
            p_mean = al_p.mean(0)
            keep = p_mean > 0.5
            s_mean = (al_s * (al_p[:, None, :] > 0.5)).sum(0) / (al_p > 0.5).sum(0).clamp_min(1)[None]
            v1_conf = torch.where(keep[None], s_mean, torch.zeros_like(s_mean))
            variants["v1_mask_only"], _ = finish(v1_conf, points, query, THRESH, edges, area)
            # V2: average each query only over the poses that found it (P(tooth) > 0.5 in that pose)
            found = (al_p > 0.5).float()                                     # [T,K]
            v2_conf = (al_c * found[:, None, :]).sum(0) / found.sum(0).clamp_min(1)[None]
            variants["v2_found_poses"], _ = finish(v2_conf, points, query, THRESH, edges, area)
            # V3: lower threshold on the deployed confidence
            for th in (0.4, 0.3):
                variants[f"v3_thresh{th}"], _ = finish(fused, points, query, th, edges, area)
            # V4: no TTA (pose 0 only)
            variants["v4_pose0"], _ = finish(conf_t[0], points, query, THRESH, edges, area)

            # ---- causes for tooth area called gingiva by V0
            full_np = full.cpu().numpy() if full is not None else None
            active = torch.nonzero(fused.amax(0) > THRESH).squeeze(1)
            act = active.cpu().numpy()
            l0 = fused[inverse][:, active].cpu().numpy()                     # voxel copy, no kNN
            s_l0 = al_s[:, :, active][:, inverse].cpu().numpy()              # [T,V,A] bare mask per pose
            cpose = al_c[:, :, active][:, inverse].cpu().numpy()             # [T,V,A] conf per pose
            p_act = al_p[:, active].cpu().numpy()                            # [T,A]

        teeth, f1, n_pred = match(gt, v0, area)
        for t in teeth:
            t["rgb"] = colour_of[t["gt"]].tolist()
        server = None
        if "pred_grey_fdi" in d.files:   # the running server, same scan, same settings: should agree
            sf = d["pred_grey_fdi"].astype(np.int64)
            server = {"same_partition": round(agreement(v0, sf, area), 4),
                      "n_server": int(len(np.unique(sf)) - 1), "n_v0": int(v0.max())}
        causes_all = {}
        for t in teeth:
            i = t["gt"]
            miss = (gt == i) & (v0 == 0)
            t["missed_mm2"] = float(area[miss].sum())
            if t["missed_mm2"] < 0.5 or full_np is None or not len(act):
                continue
            col = int(np.argmax(full_np[gt == i].T @ area[gt == i]))     # query that best covers this tooth
            w = area[miss]
            c_full = full_np[miss, col]
            c_l0 = l0[miss, col]
            m_mean = s_l0[:, miss, col].mean(0)
            poses_hi = (cpose[:, miss, col] > THRESH).sum(0)
            pose_found = p_act[:, col] > 0.5
            tot = w.sum()
            c = {
                "query_p_tooth_pose": np.round(p_act[:, col], 2).tolist(),
                "l0_above_full_below": float(w[(c_l0 > THRESH) & (c_full <= THRESH)].sum() / tot),
                "mask_above_conf_below": float(w[(m_mean > THRESH) & (c_l0 <= THRESH)].sum() / tot),
                "some_pose_above": float(w[(poses_hi >= 1) & (c_l0 <= THRESH)].sum() / tot),
                "half_poses_above": float(w[(poses_hi >= 4) & (c_l0 <= THRESH)].sum() / tot),
                "no_pose_above": float(w[poses_hi == 0].sum() / tot),
                "full_conf_hist": np.histogram(c_full, bins=[0, .1, .2, .3, .4, .5, 1.01], weights=w)[0].round(2).tolist(),
                "poses_found": int(pose_found.sum()),
            }
            t["cause"] = c
            causes_all[i] = c
        scores = {}
        for name, lab in variants.items():
            tv, f1v, npv = match(gt, lab, area)
            ious = np.array([x["iou"] for x in tv])
            scores[name] = {"f1_75": round(f1v, 4), "n_pred": npv, "iou_min": round(float(ious.min()), 3),
                            "iou_mean": round(float(ious.mean()), 4),
                            "recall_area": round(float(sum(x["recall"] * x["area"] for x in tv) /
                                                       sum(x["area"] for x in tv)), 4),
                            "missed_to_gum_mm2": round(float(sum(x["missed_to_gum"] * x["area"] for x in tv)), 1),
                            "extra_from_gum_share": round(float(np.mean([x["extra_from_gum"] for x in tv])), 4)}
        np.savez_compressed(dst / f"{aid}.npz", gt=gt.astype(np.int16), area=area.astype(np.float32),
                            **{k: v.astype(np.int16) for k, v in variants.items()})
        print(json.dumps({"asset_id": aid, "jaw": jaw, "n_vertices": len(xyz), "n_l0": int(n),
                          "n_gt": int(gt.max()), "server": server, "scores": scores,
                          "teeth": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}
                                    for t in teeth],
                          "s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
