"""E16b 대시보드 경로 그대로 자동 라벨을 받는다 (dashboard-server 컨테이너 안에서, 읽기만 한다).

    docker exec -i dental-data-hub-dashboard-server-1 python - INHOUSE0001=01f49b17593f8eab ... \
        < dashboard_fetch.py > autolabel.npz

인자는 이름=asset_id.  출력 npz: <이름>__fdi, <이름>__label, <이름>__meta (응답의 warnings, info, elapsed_ms JSON).
"""
import base64
import io
import json
import ssl
import sys
import time
import urllib.request

import numpy as np

CTX = ssl._create_unverified_context()


def call(path, method="GET"):
    req = urllib.request.Request("https://localhost:8443" + path, method=method,
                                 data=b"" if method == "POST" else None,
                                 headers={"X-DDH-Client": "dashboard", "Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return json.loads(r.read())


out = {}
for arg in sys.argv[1:]:
    name, asset = arg.split("=")
    t = time.time()
    r = call(f"/api/assets/{asset}/auto-label", "POST")
    wall = time.time() - t
    out[f"{name}__fdi"] = np.frombuffer(base64.b64decode(r["fdi"]), "u1")
    out[f"{name}__label"] = np.frombuffer(base64.b64decode(r["label"]), "i1")
    out[f"{name}__meta"] = np.array(json.dumps({"warnings": r["warnings"], "info": r["info"],
                                                "elapsed_ms": r["elapsed_ms"], "wall_s": round(wall, 2)},
                                               ensure_ascii=False))
    print(name, r["elapsed_ms"], "ms", r["info"].get("graph"), r["info"].get("teeth"), file=sys.stderr, flush=True)
buf = io.BytesIO()
np.savez(buf, **out)
sys.stdout.buffer.write(buf.getvalue())
