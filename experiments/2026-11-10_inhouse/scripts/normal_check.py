"""Normal orientation per scan: mean dot(normal, vertex - own tooth centroid) over GT tooth vertices.

Positive = normals point out of the crown (what training caches have). Negative = inward (flipped winding).
Also the same statistic for the public training data in DDH (labels from mesh.ply) is not needed here.
"""
import base64
import json
import ssl
import sys
import urllib.request

import numpy as np

from ddh.model import ios

BASE = "https://localhost:8443"
CTX = ssl._create_unverified_context()


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return r.read()


for asset in sys.argv[1:]:
    gt = json.loads(get(f"/api/assets/{asset}/labels"))
    fdi = np.frombuffer(base64.b64decode(gt["fdi"]), dtype="u1")
    vertex, normals = ios.read_mesh(get(f"/api/assets/{asset}/file"))
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    dots = []
    for t in np.unique(fdi[fdi > 0]):
        m = fdi == t
        c = xyz[m].mean(0)
        d = xyz[m] - c
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        dots.append(float((normals[m] * d).sum(1).mean()))
    print(asset, "outward" if np.mean(dots) > 0 else "INWARD", round(float(np.mean(dots)), 3))
