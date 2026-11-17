"""Read-only export of DDH IOS scans into the training cache format (dental_ai ToothSegLoader npz).

Runs in a throwaway ddh-dashboard-server:dev container on dental-data-hub_ddh-net, like probe.py: reads mesh.ply
from ddh-curated and labels from ios_labels with SELECT only, never writes the DB or S3, and builds no DDH release
(releases cannot be removed).  The output holds patient scans: write it to a Docker volume, not into the repo.

stdin: "<asset_id> <source>" per line.  source:
  saved    the label a person completed (ios_labels)
  src      the label shipped with the public dataset (label/fdi/instance in mesh.ply)
  colour   the painted Teeth3DS colour map: the most common colour (by area) is gingiva, every other colour one
           tooth (2026-11-16_teeth3ds-unlabeled).  The colour is never written to the cache, it is the label.
  auto     saved if there is one, else src if every vertex is labelled, else colour if the scan has colour.
/out/cache/<name>.npz   xyz float32, nrm float16 (area-weighted vertex normals, as DDH and the dashboard),
                        cls uint8 (0 gingiva, 1..16, 17 unknown tooth), ins uint8 (1..K), jaw uint8, rgb uint8 (only
                        real scanner colour; one-colour scans and painted scans get none).
/out/index.jsonl        one line per scan: name, asset, institution, patient, jaw, source, teeth, skipped reason.
name = <patient hex>_<jaw>_<asset_id>; the training split groups by the part before the first "_" (the patient).

Labels: 1 tooth -> tooth.  2 Prep-Tooth -> tooth.  3 Scanbody, 4 Peek -> not tooth (0), so the model learns that
they are not teeth.  A scan with unlabelled vertices (-1) is skipped.  Painted teeth get cls 17 (position unknown):
their FDI comes only from a palette guess, and 17 is ignored by the position loss.
"""
import json
import os
import sys

import numpy as np
import psycopg

from ddh.model import ios
from ddh.service import s3

UNKNOWN_TOOTH = 17
TOOTH_LABELS = (1, 2)


def fdi_to_class(fdi: np.ndarray, jaw_upper: bool) -> np.ndarray:
    """Same as dental_ai ToothSegPreProcessing.fdi_to_class."""
    fdi = fdi.astype(np.int64)
    quadrant, tooth = fdi // 10, fdi % 10
    first, second = (1, 2) if jaw_upper else (3, 4)
    valid = (tooth >= 1) & (tooth <= 8)
    out = np.full_like(fdi, UNKNOWN_TOOTH)
    out[valid & (quadrant == first)] = tooth[valid & (quadrant == first)]
    out[valid & (quadrant == second)] = tooth[valid & (quadrant == second)] + 8
    out[fdi == 0] = 0
    return out


def ordered_instances(instances: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Same as dental_ai ToothSegPreProcessing.ordered_instances: ids 1..K in order of each tooth's class."""
    teeth = [i for i in np.unique(instances) if i != 0]
    teeth.sort(key=lambda i: int(np.bincount(classes[instances == i]).argmax()))
    out = np.zeros(len(instances), np.uint8)
    for k, i in enumerate(teeth, start=1):
        out[instances == i] = k
    return out


def real_colour(rgb: np.ndarray | None) -> np.ndarray | None:
    """Same rule as dental_ai real_colour: no colour or a single colour (Teeth3DS grey) -> None."""
    if rgb is None or float((rgb.astype(np.float32) / 255.0).std(0).mean()) < 0.01:
        return None
    return rgb.astype(np.uint8)


def vertex_area(xyz: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    if faces is None:
        return np.ones(len(xyz))
    p = xyz.astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]]), axis=1)
    return sum(np.bincount(faces[:, c], weights=a / 3, minlength=len(p)) for c in range(3))


def mesh_faces(ply: bytes):
    """The dashboard's _mesh_faces, without plyfile's mmap: only the vertex count and triangles are needed."""
    import tempfile
    from plyfile import PlyData
    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        f.write(ply)
        f.flush()
        try:
            data = PlyData.read(f.name)
        except Exception:  # noqa: BLE001
            return None
        if "face" not in data:
            return None
        face = data["face"].data
        name = face.dtype.names[0]
        faces = face[name]
        faces = np.vstack(faces) if faces.dtype == object else np.array(faces)
    return faces.astype(np.int64) if faces.ndim == 2 and faces.shape[1] == 3 else None


def colour_labels(rgb: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Painted colour map -> instance ids (0 gingiva)."""
    _, inv = np.unique(rgb, axis=0, return_inverse=True)
    inv = inv.ravel()
    gum = int(np.argmax(np.bincount(inv, weights=area)))
    ins = inv + 1
    ins[inv == gum] = 0
    return np.unique(ins, return_inverse=True)[1].ravel()


def main():
    cli = s3.client()
    con = psycopg.connect()
    os.makedirs("/out/cache", exist_ok=True)
    index = open("/out/index.jsonl", "a", encoding="utf-8")
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        aid, want = parts[0], (parts[1] if len(parts) > 1 else "auto")
        row = con.execute("""SELECT a.object_prefix || a.main_file, s.jaw, s.has_color, s.n_labeled, s.n_vertices,
                                    a.institution, u.patient_pid
                             FROM assets a JOIN ios_scans s USING (asset_id) JOIN uploads u ON u.upload_id = a.upload_id
                             WHERE a.asset_id = %s""", [aid]).fetchone()
        key, jaw, has_color, n_labeled, n_vertices, inst, pid = row
        rec = {"asset": aid, "institution": inst, "patient": pid, "jaw": jaw}
        if jaw not in ("upper", "lower"):
            print(json.dumps(rec | {"skipped": "bite scan"}), file=index, flush=True)
            continue
        name = f"{pid.removeprefix('p_')}_{jaw}_{aid}"
        rec["name"] = name
        if os.path.exists(f"/out/cache/{name}.npz"):
            continue
        saved = con.execute("SELECT data FROM ios_labels WHERE asset_id = %s", [aid]).fetchone()
        ply = s3.get_bytes(cli, s3.CURATED, key)
        vertex, normals = ios.read_mesh(ply)
        names = set(vertex.dtype.names)
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float32)
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], 1) if {"red", "green", "blue"} <= names else None
        source = want
        if want == "auto":
            if saved is not None:
                source = "saved"
            elif {"label", "fdi"} <= names and n_labeled == n_vertices:
                source = "src"
            elif has_color:
                source = "colour"
            else:
                source = None
        rec["source"] = source
        if source is None or normals is None:
            print(json.dumps(rec | {"skipped": "no labels" if source is None else "no normals"}), file=index,
                  flush=True)
            continue
        jaw_upper = jaw == "upper"
        if source == "colour":
            faces = mesh_faces(ply)
            ins = colour_labels(rgb, vertex_area(xyz, faces))
            cls = np.where(ins > 0, UNKNOWN_TOOTH, 0)
            ins = ordered_instances(ins, cls)
            out_rgb = None                                   # the colour is the label
        else:
            if source == "saved":
                lab = ios.unpack_labels(bytes(saved[0]))
                label, fdi = lab["label"], lab["fdi"]
            else:
                label, fdi = np.asarray(vertex["label"]), np.asarray(vertex["fdi"])
            if (label < 0).any():
                print(json.dumps(rec | {"skipped": "unlabelled vertices"}), file=index, flush=True)
                continue
            tooth = np.isin(label, TOOTH_LABELS)
            cls = fdi_to_class(np.where(tooth, fdi, 0), jaw_upper)
            cls[tooth & (cls == 0)] = UNKNOWN_TOOTH
            cls[~tooth] = 0
            ins = ordered_instances(np.where(tooth, ios.assign_instances(np.where(tooth, 1, 0), fdi), 0), cls)
            # Teeth3DS has no scanner colour: its scans with colour are the painted ones, and that colour is a label
            out_rgb = real_colour(rgb) if has_color and inst != "teeth3ds" else None
        arrays = dict(xyz=xyz, nrm=np.asarray(normals, np.float16), cls=cls.astype(np.uint8),
                      ins=ins.astype(np.uint8), jaw=np.uint8(1 if jaw_upper else 0))
        if out_rgb is not None:
            arrays["rgb"] = out_rgb
        part = f"/out/cache/{name}.{os.getpid()}.part.npz"
        np.savez(part, **arrays)
        os.replace(part, f"/out/cache/{name}.npz")
        rec |= {"teeth": int(ins.max()), "rgb": out_rgb is not None, "n_vertices": len(xyz)}
        print(json.dumps(rec), file=index, flush=True)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
