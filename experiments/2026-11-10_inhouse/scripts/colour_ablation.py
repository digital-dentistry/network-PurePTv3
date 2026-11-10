"""Same scans, inference server called directly with and without the real colour; scored against completed labels."""
import base64, io, json, ssl, sys, urllib.request
import numpy as np
from ddh.model import ios
CTX = ssl._create_unverified_context()
def get(path):
    req = urllib.request.Request("https://localhost:8443" + path, headers={"Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return r.read()
def segment(arrays, jaw):
    buf = io.BytesIO(); np.savez(buf, **arrays)
    req = urllib.request.Request(f"http://inference-server:8000/v1/ios/segment?jaw={jaw}", data=buf.getvalue(),
                                 method="POST", headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())
    return np.frombuffer(base64.b64decode(res["fdi"]), "u1"), res["info"]["mean_confidence"]
def score(g, p):
    gt, pt = g > 0, p > 0
    both = gt & pt
    return 2 * (gt & pt).sum() / (gt.sum() + pt.sum()), (g[both] % 10 == p[both] % 10).mean()
jaws = dict(a.split(":") for a in sys.argv[1:])
tot = {"rgb": [], "none": []}
for asset, jaw in jaws.items():
    g = np.frombuffer(base64.b64decode(json.loads(get(f"/api/assets/{asset}/labels"))["fdi"]), "u1")
    vertex, normals = ios.read_mesh(get(f"/api/assets/{asset}/file"))
    base = {"xyz": np.stack([vertex[k] for k in "xyz"], 1).astype(np.float32), "normals": normals.astype(np.float32)}
    rgb = np.stack([vertex[k] for k in ("red", "green", "blue")], 1).astype(np.uint8)
    line = [asset]
    for tag, arrays in (("rgb", {**base, "rgb": rgb}), ("none", base)):
        p, conf = segment(arrays, jaw)
        tsa, pos = score(g, p)
        tot[tag].append((tsa, pos, conf))
        line.append(f"{tag}: F1 {tsa:.3f} pos {pos:.3f} conf {conf:.3f}")
    print(" | ".join(line), flush=True)
for tag, v in tot.items():
    v = np.array(v)
    print(f"mean {tag:4s}: tooth/gingiva F1 {v[:, 0].mean():.3f}  position {v[:, 1].mean():.3f}  confidence {v[:, 2].mean():.3f}")
