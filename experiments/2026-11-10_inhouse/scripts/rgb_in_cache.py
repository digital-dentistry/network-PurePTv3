import zipfile
from pathlib import Path
for ds in ("teeth3ds", "3d_iosseg", "poseidon3d", "cbct_ios_fusion_liu2023"):
    files = sorted((Path("/data/ios") / ds / "cache").glob("*.npz"))
    n = sum("rgb.npy" in zipfile.ZipFile(f).namelist() for f in files)
    print(f"{ds:24s} {n}/{len(files)} caches have real rgb")
