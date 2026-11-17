"""One colour tooth, pose by pose: which query covers it, with what P(tooth), and where pose-0 alignment sends it.

usage: python pose_debug.py <probe npz dir> <asset> <gt tooth id> [<asset> <gt id> ...]
Runs in ddh-inference-server:2026-11-10 with the GPU, like stages.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/s")
from stages import PKG, align_poses, colour_gt, vertex_normals  # noqa: E402
from toothseg import Scan, vertex_graph  # noqa: E402
from toothsegnet import ToothSegNet  # noqa: E402
from algorithm.instance_network.instance_network import InstanceNetworkInput, InstanceNetworkOutput  # noqa: E402
from algorithm.query_fusion.common.matching import query_confidence  # noqa: E402


def main():
    src = Path(sys.argv[1])
    pairs = list(zip(sys.argv[2::2], map(int, sys.argv[3::2])))
    dev = torch.device("cuda")
    tsn = ToothSegNet(PKG, json.loads((PKG / "model.json").read_text()), dev)
    seg = tsn.instance
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    for aid, tooth in pairs:
        d = np.load(src / f"{aid}.npz")
        xyz, faces = d["xyz"], d["faces"].astype(np.int64)
        scan = Scan(xyz=xyz, normals=vertex_normals(xyz, faces), rgb=None, jaw=str(d["jaw"]), faces=faces)
        _, area = vertex_graph(xyz.astype(np.float64), faces)
        gt, _ = colour_gt(d["rgb"], area)
        with torch.inference_mode():
            prep = seg._prepare.run(tsn._seg3d_scan(scan))
            cls, mask = [], []
            for b in prep.batches:
                o = InstanceNetworkOutput()
                seg._infer._network.execute(InstanceNetworkInput(b.features, b.geometry), o)
                cls.append(o.cls_logits)
                mask.append(o.mask_logits)
            cls, mask = torch.cat(cls).float(), torch.cat(mask).float()
            n = prep.points.shape[0]
            conf, prob = query_confidence(cls, mask, n)
            orders = align_poses(conf)
        inv = prep.inverse.cpu().numpy()
        # GT at L0: a voxel is "the tooth" if most of its vertices are
        m = np.bincount(inv, weights=(gt == tooth) * area, minlength=n) > 0.5 * np.bincount(inv, weights=area, minlength=n)
        others = np.bincount(inv, weights=((gt > 0) & (gt != tooth)) * area, minlength=n) > 0.5 * np.bincount(inv, weights=area, minlength=n)
        c = conf.cpu().numpy()
        p = prob.cpu().numpy()
        print(f"\n{aid} tooth {tooth}: {m.sum()} L0 points")
        base_cover = None
        for t in range(c.shape[0]):
            fg = (c[t] > 0.5)
            cover = fg[m].sum(0) / m.sum()                     # share of the tooth claimed by each query
            spill = fg[others].sum(0) / np.maximum(fg.sum(0), 1)  # share of the query on other teeth
            top = np.argsort(-cover)[:3]
            order = orders[t].cpu().numpy()
            to_base = {int(k): int(np.nonzero(order == k)[0][0]) for k in top}
            desc = ", ".join(f"q{k} cover {cover[k]:.2f} P {p[t, k]:.2f} other-teeth {spill[k]:.2f} -> base q{to_base[k]}"
                             for k in top if cover[k] > 0.02)
            any_fg = (c[t].max(1) > 0.5)[m].mean()
            print(f"  pose {t}: tooth claimed {any_fg:.2f} | {desc}")
        fused = np.stack([c[t][:, orders[t].cpu().numpy()] for t in range(c.shape[0])]).mean(0)
        cover = (fused > 0.5)[m].sum(0) / m.sum()
        k = int(np.argmax((fused[m]).sum(0)))
        print(f"  fused: best base q{k}, mean conf on tooth {fused[m, k].mean():.2f}, claimed (>0.5) {cover.sum():.2f}")
        fg_free = c.max(2).mean(0)
        print(f"  alignment-free foreground mean_t max_k conf > 0.5 on tooth: {(fg_free[m] > 0.5).mean():.2f}")


if __name__ == "__main__":
    main()
