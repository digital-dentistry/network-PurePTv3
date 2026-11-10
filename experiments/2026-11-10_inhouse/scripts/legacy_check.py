import sys
sys.path.insert(0, "/tmp/src")
import numpy as np, torch
from dataclasses import replace
from pathlib import Path
from dental_ai.dataset.algorithm.augmentation.augmentation import AugmentationInput, AugmentationOutput, AugmentationParams
from dental_ai.dataset.algorithm.augmentation.synth_colour_cpu import synth_colour_cpu
from dental_ai.dataset.algorithm.augmentation.synth_colour_orig import synth_colour_cpu as orig
from dental_ai.dataset.ToothSegLoader import ToothSegDataset
names = sorted(p.stem for p in Path("/data/ios/teeth3ds/cache").glob("*.npz"))[:5]
ds = ToothSegDataset(["/data/ios/teeth3ds/cache"], names)
same = []
for i, n in enumerate(names):
    s, _ = ds.load(n)
    a, b = AugmentationOutput(), AugmentationOutput()
    orig(AugmentationInput(s, AugmentationParams(), torch.Generator().manual_seed(i)), a)
    synth_colour_cpu(AugmentationInput(s, AugmentationParams(), torch.Generator().manual_seed(i)), b)
    same.append(torch.equal(a.sample.colour, b.sample.colour))
print("legacy mode identical to the committed code:", same)
p = replace(AugmentationParams(), colour_edge_mm=(0.4, 1.2), colour_edge_width_mm=(1.2, 2.2), colour_drift_scale=(1.5, 3.5))
s, _ = ds.load(names[0]); o = AugmentationOutput()
import time; t = time.time()
synth_colour_cpu(AugmentationInput(s, p, torch.Generator().manual_seed(0)), o)
c = o.sample.colour
print(f"edge mode: {s.n} vertices, {1000*(time.time()-t):.0f} ms, colour range {float(c.min()):.2f}..{float(c.max()):.2f}, finite {bool(torch.isfinite(c).all())}")
# crops can leave no gingiva or no teeth
keep = s.classes > 0
o2 = AugmentationOutput(); synth_colour_cpu(AugmentationInput(s.select(keep), p, None), o2)
o3 = AugmentationOutput(); synth_colour_cpu(AugmentationInput(s.select(~keep), p, None), o3)
print("teeth only finite:", bool(torch.isfinite(o2.sample.colour).all()), "| gingiva only finite:", bool(torch.isfinite(o3.sample.colour).all()))
