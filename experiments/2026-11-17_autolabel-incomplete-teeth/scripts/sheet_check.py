"""Ad hoc checks for E8 (2026-11-17).  sheet_check.py: are the two pieces of one instance overlapping, unwelded
scan sheets (open borders, offset along the normal)?  missed_region.py: is a missed tooth area a mesh defect
(separate piece, open border) or a clean surface?  usage: python <script> <scratch t3probe dir> <asset> <colour GT id>
The scratch dir holds out100/ (probe npz) and stages/ (stages.py npz)."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5] / "inference-server"))
from toothseg import vertex_graph, components
from scipy.spatial import cKDTree
S, aid, tooth = sys.argv[1], sys.argv[2], int(sys.argv[3])
d, s = np.load(f"{S}/out100/{aid}.npz"), np.load(f"{S}/stages/{aid}.npz")
xyz, f = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
edges, area = vertex_graph(xyz, f)
gt = s["gt"].astype(int); raw = s["v0_noclean"].astype(int)
inst = np.bincount(raw[gt == tooth]).argmax(); r = raw == inst
comp = components(raw, edges)
u, c = np.unique(comp[r], return_counts=True); big = [a for a, _ in sorted(zip(u, c), key=lambda x: -x[1])[:2]]
A, B = comp == big[0], comp == big[1]
# boundary edges (used by one face) of the whole mesh
e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
ue, ce = np.unique(e, axis=0, return_counts=True)
bnd = ue[ce == 1]; nonman = ue[ce > 2]
bv = np.zeros(len(xyz), bool); bv[bnd.ravel()] = True
print("mesh boundary edges", len(bnd), "non-manifold edges", len(nonman))
print("boundary vertices in A", int(bv[A].sum()), "in B", int(bv[B].sum()), "of", int(A.sum()), int(B.sum()))
# how far is each boundary vertex of A from B
dA, _ = cKDTree(xyz[B]).query(xyz[A & bv]); dB, _ = cKDTree(xyz[A]).query(xyz[B & bv])
print("A boundary vertices: distance to B median %.3f, <0.3 mm %.2f" % (np.median(dA), (dA < 0.3).mean()) if len(dA) else "A no boundary")
print("B boundary vertices: distance to A median %.3f, <0.3 mm %.2f" % (np.median(dB), (dB < 0.3).mean()) if len(dB) else "B no boundary")
# Is the boundary of A / B (where they don't connect) a crack: the edges between A and B regions?
fn = np.cross(xyz[f[:,1]]-xyz[f[:,0]], xyz[f[:,2]]-xyz[f[:,0]]); vn = np.zeros_like(xyz)
for k in range(3): np.add.at(vn, f[:,k], fn)
vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
dist, idx = cKDTree(xyz[B]).query(xyz[A]); iB = np.nonzero(B)[0][idx]
sel = dist < 0.6
off = ((xyz[A][sel] - xyz[iB[sel]]) * vn[A][sel]).sum(1)
print("A vertices within 0.6 mm of B: %d; offset along normal: median %.3f, |offset|>0.1: %.2f, frac A above B %.2f" % (sel.sum(), np.median(off), (np.abs(off) > 0.1).mean(), (off > 0).mean()))
# whole-mesh: boundary loops count and total length
L = np.linalg.norm(xyz[bnd[:,0]] - xyz[bnd[:,1]], axis=1)
inside_teeth = (raw[bnd[:,0]] > 0) & (raw[bnd[:,1]] > 0)
print("boundary edge length total %.1f mm, inside predicted teeth %.1f mm" % (L.sum(), L[inside_teeth].sum()))
