"""Compare dashboard auto-label (PurePTv3) with completed human labels for given assets.

Runs inside dashboard-server container; reads only (auto-label does not write to DB).
"""
import base64
import json
import ssl
import sys
import urllib.request

import numpy as np

BASE = "https://localhost:8443"
CTX = ssl._create_unverified_context()


def call(path, method="GET"):
    req = urllib.request.Request(BASE + path, method=method, data=b"" if method == "POST" else None,
                                 headers={"X-DDH-Client": "dashboard", "Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return json.loads(r.read())


def arr(b64, dtype):
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


rows = []
for asset in sys.argv[1:]:
    gt = call(f"/api/assets/{asset}/labels")
    pr = call(f"/api/assets/{asset}/auto-label", "POST")
    gl, gf = arr(gt["label"], "i1"), arr(gt["fdi"], "u1")
    pl, pf = arr(pr["label"], "i1"), arr(pr["fdi"], "u1")
    gtooth, ptooth = gf > 0, pf > 0
    tp = (gtooth & ptooth).sum()
    tsa = 2 * tp / max(gtooth.sum() + ptooth.sum(), 1)
    both = gtooth & ptooth
    pos_ok = (gf[both] % 10 == pf[both] % 10).mean() if both.any() else float("nan")
    quad_ok = (gf[both] // 10 == pf[both] // 10).mean() if both.any() else float("nan")
    fdi_ok = (gf[gtooth] == pf[gtooth]).mean()
    gteeth = sorted(int(t) for t in np.unique(gf) if t)
    pteeth = sorted(int(t) for t in np.unique(pf) if t)
    # position-only accuracy if quadrant were right (mirror quadrant digit)
    rows.append(dict(asset=asset, n=len(gf), gt=gteeth, pred=pteeth, tsa=round(float(tsa), 3),
                     pos=round(float(pos_ok), 3), quad=round(float(quad_ok), 3), fdi=round(float(fdi_ok), 3),
                     method=pr["info"].get("quadrant_method"), colour=pr["info"].get("colour"),
                     conf=pr["info"].get("mean_confidence"), warn=len(pr.get("warnings", []))))
    print(json.dumps(rows[-1]), flush=True)

for k in ("tsa", "pos", "quad", "fdi"):
    v = [r[k] for r in rows]
    print(k, round(float(np.nanmean(v)), 3))
