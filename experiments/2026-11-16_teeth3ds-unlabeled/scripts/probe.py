"""Read-only probe: run DDH auto-label on the given IOS assets, with the scan's colour and without it.

Runs in a throwaway ddh-dashboard-server:dev container on dental-data-hub_ddh-net (see README for the command).
Reads mesh.ply from ddh-curated and the saved label (ios_labels) with SELECT only. Sends the same arrays the
dashboard sends (dashboard-server/app.py _scan_arrays), once as is and once with rgb removed, to
POST /v1/ios/segment. Never writes the DB or S3.

stdin: asset ids, one per line.
stdout: one JSON line per asset (numbers only).
/out/<asset_id>.npz: xyz, normals, faces, rgb, source labels from mesh.ply, saved label, both predictions.
"""
import base64
import io
import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np
import psycopg
from plyfile import PlyData, PlyElementParseError

from ddh.model import ios
from ddh.service import s3

URL = os.environ["DDH_INFERENCE_URL"].rstrip("/")
MODES = os.environ.get("PROBE_MODES", "colour,grey").split(",")


def mesh_faces(ply: bytes):
    """Same as dashboard-server/app.py _mesh_faces."""
    end = ply.find(b"end_header")
    element, name = None, None
    for line in ply[:max(end, 0)].decode("ascii", "replace").splitlines():
        words = line.split()
        if words[:1] == ["element"] and len(words) >= 2:
            element = words[1]
        elif element == "face" and words[:2] == ["property", "list"] and len(words) == 5:
            name = words[4]
    if name is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        f.write(ply)
        f.flush()
        try:
            parsed = PlyData.read(f.name, mmap=True, known_list_len={"face": {name: 3}})
        except PlyElementParseError:
            return None
        faces = parsed["face"].data[name]
        faces = np.vstack(faces) if faces.dtype == object else np.array(faces)
        del parsed
    return faces.astype(np.int32) if faces.ndim == 2 and faces.shape[1] == 3 and len(faces) else None


def post(jaw: str, arrays: dict) -> dict:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    req = urllib.request.Request(f"{URL}/v1/ios/segment?jaw={jaw}", data=buf.getvalue(), method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())
    return {"label": np.frombuffer(base64.b64decode(res["label"]), np.int8),
            "fdi": np.frombuffer(base64.b64decode(res["fdi"]), np.uint8),
            "info": res.get("info", {}), "warnings": res.get("warnings", []), "s": round(time.time() - t0, 2)}


def main():
    cli = s3.client()
    con = psycopg.connect()
    for line in sys.stdin:
        aid = line.strip()
        if not aid:
            continue
        row = con.execute("""SELECT a.object_prefix || a.main_file, s.jaw, s.has_color, a.institution
                             FROM assets a JOIN ios_scans s USING (asset_id) WHERE a.asset_id = %s""", [aid]).fetchone()
        key, jaw, has_color, inst = row
        saved = con.execute("SELECT data FROM ios_labels WHERE asset_id = %s", [aid]).fetchone()
        ply = s3.get_bytes(cli, s3.CURATED, key)
        vertex, normals = ios.read_mesh(ply)
        names = list(vertex.dtype.names)
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float32)
        arrays = {"xyz": xyz, "normals": np.asarray(normals, np.float32)}
        rgb = None
        if has_color:
            rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], 1).astype(np.uint8)
            arrays["rgb"] = rgb
        faces = mesh_faces(ply)
        if faces is not None:
            arrays["faces"] = faces
        out = {"xyz": xyz, "jaw": np.array(jaw), "normals": arrays["normals"]}  # normals: added 2026-11-17
        if faces is not None:
            out["faces"] = faces
        if rgb is not None:
            out["rgb"] = rgb
        for f in ("label", "fdi", "instance"):
            if f in names:
                out[f"src_{f}"] = np.asarray(vertex[f])
        if saved is not None:
            lab = ios.unpack_labels(bytes(saved[0]))
            out["saved_label"], out["saved_fdi"] = lab["label"], lab["fdi"]
        rep = {"asset_id": aid, "inst": inst, "jaw": jaw, "n_vertices": len(xyz),
               "n_faces": None if faces is None else len(faces), "props": names, "saved": saved is not None,
               "extent": np.round(np.ptp(xyz, 0), 1).tolist(), "centre": np.round(xyz.mean(0), 1).tolist()}
        if faces is not None:
            e = np.linalg.norm(xyz[faces[:, 0]] - xyz[faces[:, 1]], axis=1)
            rep["edge_mm_median"] = round(float(np.median(e)), 3)
        if rgb is not None:
            c = rgb.astype(np.float32)
            rep["rgb_mean"] = np.round(c.mean(0), 1).tolist()
            rep["rgb_std"] = np.round(c.std(0), 1).tolist()
            rep["grey_share"] = round(float((np.ptp(c, 1) < 8).mean()), 3)
            rep["n_colours"] = int(len(np.unique(rgb, axis=0)))
        for mode in MODES:  # PROBE_MODES=stats: no inference, only the numbers above
            if mode not in ("colour", "grey"):
                continue
            a = dict(arrays)
            if mode == "grey":
                a.pop("rgb", None)
            elif mode == "colour" and "rgb" not in a:
                continue
            r = post(jaw, a)
            out[f"pred_{mode}_label"], out[f"pred_{mode}_fdi"] = r["label"], r["fdi"]
            rep[mode] = {k: r["info"].get(k) for k in ("instances", "raw_instances", "teeth", "instance_scores")}
            rep[mode]["warnings"], rep[mode]["s"] = r["warnings"], r["s"]
        np.savez_compressed(f"/out/{aid}.npz", **out)
        print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
