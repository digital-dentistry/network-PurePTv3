"""How often each crop mode actually applies, per training dataset (ToothSegLoader defaults)."""
import random
from dataclasses import replace
from pathlib import Path

import torch

from dental_ai.dataset.algorithm.augmentation.augmentation import AugmentationInput, AugmentationOutput, AugmentationParams
from dental_ai.dataset.algorithm.augmentation.segment_crop_cpu import segment_crop_cpu
from dental_ai.dataset.algorithm.augmentation.wedge_crop_cpu import wedge_crop_cpu
from dental_ai.dataset.ToothSegLoader import ToothSegDataset

ROOT = Path("/data/ios")
N = 30
random.seed(0)
p0 = AugmentationParams()
print("defaults: crop_p", p0.crop_p, "wedge_p", p0.wedge_p, "crop_mix (bridge, quadrant, half)", p0.crop_mix,
      "crop_min_points", p0.crop_min_points)
for ds in ("teeth3ds", "3d_iosseg", "poseidon3d", "cbct_ios_fusion_liu2023"):
    names = sorted(x.stem for x in (ROOT / ds / "cache").glob("*.npz"))
    names = random.sample(names, min(N, len(names)))
    data = ToothSegDataset([ROOT / ds / "cache"], names)
    row = {}
    for mode in ("bridge", "quadrant", "half", "wedge"):
        ok, teeth, share = 0, 0, 0.0
        for i, n in enumerate(names):
            s, _ = data.load(n)
            gen = torch.Generator().manual_seed(i)
            out = AugmentationOutput()
            if mode == "wedge":
                applied = wedge_crop_cpu(AugmentationInput(s, p0, gen), out)
            else:
                applied = segment_crop_cpu(AugmentationInput(s, replace(p0, crop_mode=mode), gen), out)
            if applied:
                ok += 1
                teeth += int(torch.unique(out.sample.instances[out.sample.instances > 0]).numel())
                share += out.sample.n / s.n
        row[mode] = f"{ok}/{len(names)} applied, {teeth / max(ok, 1):.1f} teeth, {100 * share / max(ok, 1):.0f}% pts"
    pts = sum(data.load(n)[0].n for n in names[:10]) / 10
    print(f"{ds:24s} (~{pts:.0f} vertices)  " + " | ".join(f"{k}: {v}" for k, v in row.items()))
