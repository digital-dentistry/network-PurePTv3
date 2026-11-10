"""Full-arch scans: on which side of the midline do the GT first-quadrant teeth lie, in the crown-up frame
built from GT positions (same math as inference-server toothseg.view_frame)?  Convention: first quadrant
(1x upper / 3x lower) at x < midline for both jaws.  'mirrored' means mesh or label is chirality-flipped."""
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


def view_frame(coord, normals, position):
    pca = np.linalg.svd(coord - coord.mean(0), full_matrices=False)[2]
    tooth = position > 0
    z = normals[tooth].mean(0)
    z = z / np.linalg.norm(z)
    p, c = position[tooth], coord[tooth]
    y = -((p - p.mean())[:, None] * (c - c.mean(0))).sum(0)
    y = y - (y @ z) * z
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


for asset in sys.argv[1:]:
    gt = json.loads(get(f"/api/assets/{asset}/labels"))
    fdi = np.frombuffer(base64.b64decode(gt["fdi"]), dtype="u1").astype(int)
    vertex, normals = ios.read_mesh(get(f"/api/assets/{asset}/file"))
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    xyz -= xyz.mean(0)
    pos = np.where(fdi > 0, fdi % 10, 0)
    x = (xyz @ view_frame(xyz, normals, pos).T)[:, 0]
    mid = np.median(x[pos == 1])
    first = np.isin(fdi // 10, (1, 3)) & (fdi > 0)
    second = np.isin(fdi // 10, (2, 4)) & (fdi > 0)
    share = float((x[first] < mid).mean())
    print(asset, "first-quadrant teeth at x<mid:", round(share, 3),
          "consistent" if share > 0.5 else "MIRRORED", "| second at x>=mid:", round(float((x[second] >= mid).mean()), 3))
