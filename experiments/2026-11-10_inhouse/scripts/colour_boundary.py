"""Real vs synthetic colour around the tooth/gingiva boundary, on the labelled inhouse colour scans.

Per scan: colour as the network sees it (standardise_colour), projected on the axis from the far-gingiva mean
to the far-tooth mean (0 = gingiva colour, 1 = tooth colour), against the signed distance to the GT boundary
(+ inside a tooth).  The synthetic colour is synth_colour_cpu painted on the same scan with the same labels.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/src")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from dental_ai.dataset.algorithm.augmentation.augmentation import (AugmentationInput, AugmentationOutput,  # noqa: E402
                                                                   AugmentationParams, ScanSample)
from dental_ai.dataset.algorithm.augmentation.synth_colour_cpu import synth_colour_cpu  # noqa: E402
from dental_ai.dataset.ToothSegLoader import standardise_colour  # noqa: E402

BINS = np.arange(-3.0, 3.01, 0.25)
CENTRES = (BINS[:-1] + BINS[1:]) / 2


def profile(xyz, tooth, colour01):
    x = standardise_colour(torch.from_numpy(colour01.astype(np.float32))).numpy().astype(np.float64)
    d_t = cKDTree(xyz[tooth]).query(xyz, workers=-1)[0]
    d_g = cKDTree(xyz[~tooth]).query(xyz, workers=-1)[0]
    sd = np.where(tooth, d_g, -d_t)                     # + inside a tooth
    mt, mg = x[sd > 1.5].mean(0), x[sd < -1.5].mean(0)
    u = mt - mg
    s = (x - mg) @ u / (u @ u)
    curve = np.array([np.median(s[(sd >= a) & (sd < b)]) if ((sd >= a) & (sd < b)).sum() > 50 else np.nan
                      for a, b in zip(BINS[:-1], BINS[1:])])
    # colour-only separability right at the boundary (|sd| < 0.5 mm) and far from it, as AUC of s
    def auc(m):
        pos, neg = s[m & tooth], s[m & ~tooth]
        r = np.argsort(np.argsort(np.concatenate([pos, neg])))
        return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return curve, auc(np.abs(sd) < 0.5), auc(np.abs(sd) > 1.5)


def crossing(curve, level):
    ok = ~np.isnan(curve)
    c, z = CENTRES[ok], np.maximum.accumulate(curve[ok])
    return float(np.interp(level, z, c))


real, synth = [], []
print("scan                 | real: AUC|d|<0.5 AUC|d|>1.5 10%..90% (mm) | synth: AUC|d|<0.5 AUC|d|>1.5 10%..90%")
for f in sorted(Path("/s/inhouse_gt").glob("*.npz")):
    d = np.load(f)
    if "rgb" not in d.files or not standardise_colour(torch.from_numpy(d["rgb"] / 255.0).float()).any():
        continue
    xyz, g = d["xyz"].astype(np.float64), d["fdi"].astype(np.int64)
    tooth = g > 0
    cr, a_near, a_far = profile(xyz, tooth, d["rgb"] / 255.0)
    cls = torch.from_numpy(np.where(tooth, 1, 0))
    ins = torch.from_numpy(g)
    sample = ScanSample(torch.from_numpy(d["xyz"]), torch.from_numpy(d["normals"]), cls, ins)
    out = AugmentationOutput()
    synth_colour_cpu(AugmentationInput(sample, AugmentationParams(), torch.Generator().manual_seed(1)), out)
    cs, s_near, s_far = profile(xyz, tooth, out.sample.colour.numpy())
    real.append(cr), synth.append(cs)
    print(f"{f.stem:20s} | {a_near:.3f} {a_far:.3f} {crossing(cr, .1):+.2f}..{crossing(cr, .9):+.2f} "
          f"| {s_near:.3f} {s_far:.3f} {crossing(cs, .1):+.2f}..{crossing(cs, .9):+.2f}", flush=True)

r, s = np.nanmedian(real, 0), np.nanmedian(synth, 0)
print("\nsigned distance (mm, + = inside tooth) : median position on gingiva(0)->tooth(1) colour axis")
for c, a, b in zip(CENTRES, r, s):
    print(f"  {c:+.3f}  real {a:6.3f}  synth {b:6.3f}")
print(f"real transition 10%..90%: {crossing(r, .1):+.2f} .. {crossing(r, .9):+.2f} mm, 50% at {crossing(r, .5):+.2f} mm")
print(f"synth transition 10%..90%: {crossing(s, .1):+.2f} .. {crossing(s, .9):+.2f} mm, 50% at {crossing(s, .5):+.2f} mm")
