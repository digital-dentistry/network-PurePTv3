"""segmentation3d semantic prediction (colour-coded _sem.ply on stdin) vs DDH human label for one upper scan.
argv: asset_id upper|lower"""
import base64
import io
import json
import ssl
import sys
import urllib.request

import numpy as np
from plyfile import PlyData

from ddh.model import ios

BASE = "https://localhost:8443"
CTX = ssl._create_unverified_context()
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GINGIVA, UNKNOWN = "#d6d5ce", "#898781"
rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
CLASS_RGB = np.array([rgb(h) for h in (GINGIVA, *SLOTS, *SLOTS[::-1], UNKNOWN)])


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return r.read()


asset, jaw = sys.argv[1], sys.argv[2]
sem = PlyData.read(io.BytesIO(sys.stdin.buffer.read()))["vertex"].data
pred_rgb = np.stack([sem["red"], sem["green"], sem["blue"]], 1).astype(int)
pred_cls = np.argmin(((pred_rgb[:, None, :] - CLASS_RGB[None]) ** 2).sum(2), 1)  # colour -> class (first match)

vertex, _ = ios.read_mesh(get(f"/api/assets/{asset}/file"))
d = np.abs(np.stack([vertex[k] for k in "xyz"], 1) - np.stack([sem[k] for k in "xyz"], 1)).max()
print("xyz max diff vs DDH mesh:", float(d))

fdi = np.frombuffer(base64.b64decode(json.loads(get(f"/api/assets/{asset}/labels"))["fdi"]), dtype="u1").astype(int)
q, p = fdi // 10, fdi % 10
first = np.isin(q, (1, 3))
g = np.where(fdi > 0, np.where(first, p, p + 8), 0)
mirror = np.where(fdi > 0, np.where(first, p + 8, p), 0)
t = fdi > 0
same = (pred_rgb[t] == CLASS_RGB[g[t]]).all(1).mean()
flip = (pred_rgb[t] == CLASS_RGB[mirror[t]]).all(1).mean()
print(f"{asset}: segmentation3d agrees with human label incl. side {same:.3f}, agrees on position but opposite side {flip:.3f}")
