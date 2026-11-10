import base64, json, ssl, sys, urllib.request
import numpy as np
CTX = ssl._create_unverified_context()
def call(path, method="GET"):
    req = urllib.request.Request("https://localhost:8443" + path, method=method, data=b"" if method == "POST" else None,
                                 headers={"X-DDH-Client": "dashboard", "Host": "localhost"})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
        return json.loads(r.read())
for a in sys.argv[1:]:
    g = np.frombuffer(base64.b64decode(call(f"/api/assets/{a}/labels")["fdi"]), "u1")
    p = np.frombuffer(base64.b64decode(call(f"/api/assets/{a}/auto-label", "POST")["fdi"]), "u1")
    out = []
    for t in sorted(set(g[g > 0].tolist())):
        v, c = np.unique(p[g == t], return_counts=True)
        out.append(f"{t}->{int(v[c.argmax()])}")
    print(a, " ".join(out))
