import sys
sys.argv = ["train_toothseg.py", "--out", "/tmp/x"]
sys.path.insert(0, "/pureptv3")
import numpy as np, torch
import train_toothseg as T
from dental_ai.dataset.ToothSegLoader import ToothSegConfig, ToothSegDataset
a = T.parse_args()
aug = T.AugmentationParams(crop_p=a.crop_p)
cfg = ToothSegConfig(voxel_mm=a.voxel_mm, features=tuple(a.features), max_points=a.max_points, augmentation=aug)
names = sorted(p.stem for p in __import__("pathlib").Path("/data/ios/teeth3ds/cache").glob("*.npz"))[:20]
ds = ToothSegDataset(["/data/ios/teeth3ds/cache"], names, cfg)
torch.manual_seed(0)
cropped, total, centre, axes = 0, 0, [], []
for r in range(10):
    for i in range(len(ds)):
        full = int(torch.unique(ds.load(ds.names[i])[0].instances).numel())
        it = ds[i]
        cropped += int(torch.unique(it.instances).numel()) < full
        total += 1
        centre.append(float(it.coord.double().mean(0).abs().max()))
        if i == 0:
            axes.append(np.linalg.svd(it.coord.numpy() - it.coord.numpy().mean(0), full_matrices=False)[2][2])
print("crop_p", a.crop_p, "| rotation", aug.rotation)
print(f"teeth removed (crop applied) in {cropped}/{total} draws = {cropped/total:.2f}")
print(f"max |mean coord| after crop+rotation: {max(centre):.2e} mm")
ax = np.array(axes); print("scan 0 occlusal axis over 10 draws (|cos| to first draw):", np.round(np.abs(ax @ ax[0]), 2).tolist())
