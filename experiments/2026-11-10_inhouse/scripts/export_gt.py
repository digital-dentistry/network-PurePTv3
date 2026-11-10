import base64, io, json, ssl, sys, urllib.request
import numpy as np
from ddh.model import ios
CTX = ssl._create_unverified_context()
def get(path):
    req = urllib.request.Request("https://localhost:8443" + path, headers={"Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return r.read()
asset, jaw = sys.argv[1], sys.argv[2]
lab = json.loads(get(f"/api/assets/{asset}/labels"))
vertex, normals = ios.read_mesh(get(f"/api/assets/{asset}/file"))
out = {"xyz": np.stack([vertex[k] for k in "xyz"], 1).astype(np.float32), "normals": normals.astype(np.float32),
       "fdi": np.frombuffer(base64.b64decode(lab["fdi"]), "u1"), "label": np.frombuffer(base64.b64decode(lab["label"]), "i1"),
       "jaw": np.array(jaw), "asset": np.array(asset)}
if "red" in vertex.dtype.names:
    out["rgb"] = np.stack([vertex[k] for k in ("red", "green", "blue")], 1).astype(np.uint8)
buf = io.BytesIO(); np.savez(buf, **out); sys.stdout.buffer.write(buf.getvalue())
