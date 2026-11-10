"""Synthetic colour (edge mode) vs real, on the 10 labelled inhouse colour scans: AUC near/far and transition."""
import sys
sys.argv = [sys.argv[0]]
exec(open("/s/colour_boundary.py").read().split("real, synth = [], []")[0])   # profile(), crossing(), imports
from dataclasses import replace

scans = []
for f in sorted(Path("/s/inhouse_gt").glob("*.npz")):
    d = np.load(f)
    if "rgb" not in d.files or not standardise_colour(torch.from_numpy(d["rgb"] / 255.0).float()).any():
        continue
    scans.append(d)
real = [profile(d["xyz"].astype(np.float64), d["fdi"] > 0, d["rgb"] / 255.0) for d in scans]
print(f"real             AUC near {np.median([r[1] for r in real]):.3f} far {np.median([r[2] for r in real]):.3f} "
      f"50% at {crossing(np.nanmedian([r[0] for r in real], 0), .5):+.2f} mm")
base = AugmentationParams()
for label, params in [("legacy", base)] + [(f"edge drift {s}..3.5", replace(base, colour_edge_mm=(0.4, 1.2),
                                            colour_edge_width_mm=(1.2, 2.2), colour_drift_scale=(s, 3.5)))
                                           for s in (1.5,)]:
    res = []
    for i, d in enumerate(scans):
        tooth = d["fdi"] > 0
        sample = ScanSample(torch.from_numpy(d["xyz"]), torch.from_numpy(d["normals"]),
                            torch.from_numpy(tooth.astype(np.int64)), torch.from_numpy(d["fdi"].astype(np.int64)))
        for seed in range(2):
            out = AugmentationOutput()
            synth_colour_cpu(AugmentationInput(sample, params, torch.Generator().manual_seed(100 * i + seed)), out)
            res.append(profile(d["xyz"].astype(np.float64), tooth, out.sample.colour.numpy()))
    curve = np.nanmedian([r[0] for r in res], 0)
    print(f"{label:16s} AUC near {np.median([r[1] for r in res]):.3f} far {np.median([r[2] for r in res]):.3f} "
          f"50% at {crossing(curve, .5):+.2f} mm, 10..90% {crossing(curve, .1):+.2f}..{crossing(curve, .9):+.2f}", flush=True)
