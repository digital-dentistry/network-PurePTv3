"""Old against new post-processing on the same network output (paired), boundaries only.

Runs in a throwaway ddh-inference-server:2026-11-10 container (the image before the 2026-11-17 change) with the
GPU, the changed inference-server folder mounted at
/new (see README).  The image's own /app/toothseg.py is the deployed (old) code.  Per scan the instance network
runs once (8 TTA poses), then four post-processings are applied to that same output:
  old          segmentation3d fuse_poses (every pose matched to pose 0)  + old absorb_fragments   (deployed)
  fusion_only  fuse_consensus (poses matched to the running mean)        + old absorb_fragments
  speck_only   fuse_poses                                                + new absorb_fragments
  new          fuse_consensus                                            + new absorb_fragments  (this change)
Each is upsampled and extracted by segmentation3d's FinishStage, cleaned and split into mesh-connected instances
as toothsegnet.py does.  Numbering is skipped: it gives every instance its own FDI, so it does not move boundaries.

--repeats R > 1 runs the network R times; repeats after the first add tiny noise to the normals (sd 0.002), to
see how often a borderline tooth flips.  The live server is not called: two runs of the same code differ by GPU
non-determinism, which would blur the comparison (the deployed server lost molar 46 of 4e325c26 once and kept it
the next time).

usage: python eval_fix.py <npz dir> <out dir> --gt colour|src|saved [--rgb grey|scan] [--repeats R] < asset ids
stdout: one JSON line per scan (and repeat).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/new")
from toothseg import MIN_FRAGMENT_MM2, Scan, absorb_fragments, components, vertex_graph  # noqa: E402
from toothsegnet import ToothSegNet, fuse_consensus  # noqa: E402
from algorithm.instance_network.instance_network import InstanceNetworkInput, InstanceNetworkOutput  # noqa: E402
from algorithm.query_fusion.common.matching import fuse_poses, query_confidence  # noqa: E402
from pipeline.types import InferredScan  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_fix_score import score  # noqa: E402

PKG = Path(os.environ.get("PKG", "/models/toothsegnet-ios/1"))


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod              # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


absorb_old = _module("old_toothseg", "/app/toothseg.py").absorb_fragments


def vertex_normals(xyz, faces):
    p = xyz.astype(np.float64)
    c = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, faces[:, k], c)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def colour_gt(rgb, area):
    _, inv = np.unique(rgb, axis=0, return_inverse=True)
    inv = inv.ravel()
    gum = int(np.argmax(np.bincount(inv, weights=area)))
    gt = inv + 1
    gt[inv == gum] = 0
    return np.unique(gt, return_inverse=True)[1].ravel()


def renumber(lab):
    """Instance ids as 1..G (0 stays gingiva), without splitting an id into its mesh pieces."""
    _, inv = np.unique(lab, return_inverse=True)
    inv = inv.ravel()
    return inv if (lab == 0).any() else inv + 1


def by_fdi(label, fdi):
    """DDH: one object per (class, FDI); gingiva/unlabelled 0."""
    key = np.where((label > 0) & (fdi > 0), label.astype(np.int64) * 100 + fdi, 0)
    return np.unique(key, return_inverse=True)[1].ravel() if key.any() else np.zeros(len(key), np.int64)


@torch.inference_mode()
def raw_instances(tsn: ToothSegNet, scan: Scan) -> dict[str, np.ndarray]:
    """Network once; [V] instance ids after FinishStage for each fusion."""
    prepared = tsn.prepare.run(tsn._seg3d_scan(scan))
    cls, mask = [], []
    for batch in prepared.batches:
        out = InstanceNetworkOutput()
        tsn.network.execute(InstanceNetworkInput(batch.features, batch.geometry), out)
        cls.append(out.cls_logits)
        mask.append(out.mask_logits)
    conf, prob = query_confidence(torch.cat(cls), torch.cat(mask), prepared.points.shape[0])
    old = tsn.finish.run(InferredScan(prepared=prepared, confidence=fuse_poses(conf), scores=prob[0]))
    fused, scores = fuse_consensus(conf, prob)
    new = tsn.finish.run(InferredScan(prepared=prepared, confidence=fused, scores=scores))
    return {"pose0": old.instances.astype(np.int64), "consensus": new.instances.astype(np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--gt", choices=("colour", "src", "saved"), required=True)
    ap.add_argument("--rgb", choices=("grey", "scan"), default="grey")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    tsn = ToothSegNet(PKG, json.loads((PKG / "model.json").read_text()), torch.device("cuda"))
    rng = np.random.default_rng(0)
    for line in sys.stdin:
        aid = line.split()[0] if line.strip() else ""
        if not aid:
            continue
        d = np.load(a.src / f"{aid}.npz")
        xyz, jaw = d["xyz"].astype(np.float32), str(d["jaw"])
        faces = d["faces"].astype(np.int64) if "faces" in d.files else None
        normals = d["normals"].astype(np.float32) if "normals" in d.files else vertex_normals(xyz, faces)
        rgb = d["rgb"] if (a.rgb == "scan" and "rgb" in d.files) else None
        edges, area = vertex_graph(xyz.astype(np.float64), faces)
        if faces is None:
            area = np.ones(len(xyz))
        if a.gt == "colour":
            gt = colour_gt(d["rgb"], area)
        elif a.gt == "src":
            if "src_fdi" not in d.files:
                continue
            gt = by_fdi(d["src_label"].astype(np.int64), d["src_fdi"].astype(np.int64))
        else:
            if "saved_fdi" not in d.files:
                continue
            gt = by_fdi(d["saved_label"].astype(np.int64), d["saved_fdi"].astype(np.int64))
        for rep in range(a.repeats):
            n = normals
            if rep:
                n = normals + rng.normal(0, 0.002, normals.shape).astype(np.float32)
                n /= np.linalg.norm(n, axis=1, keepdims=True)
            t0 = time.time()
            raw = raw_instances(tsn, Scan(xyz=xyz, normals=n, rgb=rgb, jaw=jaw, faces=faces))
            labs = {
                "old": components(absorb_old(raw["pose0"], edges, area, MIN_FRAGMENT_MM2), edges),
                "fusion_only": components(absorb_old(raw["consensus"], edges, area, MIN_FRAGMENT_MM2), edges),
                "speck_only": components(absorb_fragments(raw["pose0"], edges, area, MIN_FRAGMENT_MM2), edges),
            }
            absorbed = absorb_fragments(raw["consensus"], edges, area, MIN_FRAGMENT_MM2)
            labs["new"] = components(absorbed, edges)
            labs["nosplit"] = renumber(absorbed)
            if rep == 0:
                np.savez_compressed(a.dst / f"{aid}.npz", gt=gt.astype(np.int16), area=area.astype(np.float32),
                                    raw_consensus=raw["consensus"].astype(np.int16),
                                    **{k: v.astype(np.int16) for k, v in labs.items()})
            print(json.dumps({"asset_id": aid, "jaw": jaw, "rgb": rgb is not None, "repeat": rep,
                              "changed_share": round(float(area[(labs["old"] > 0) != (labs["new"] > 0)].sum()
                                                           / area.sum()), 5),
                              **{k: score(gt, v, area) for k, v in labs.items()},
                              "s": round(time.time() - t0, 2)}), flush=True)


if __name__ == "__main__":
    main()
