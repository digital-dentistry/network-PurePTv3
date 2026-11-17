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
gt = s["gt"].astype(int); v0 = s["v0_deployed"].astype(int)
m = gt == tooth; miss = m & (v0 == 0); hit = m & (v0 > 0)
print(aid, tooth, "tooth %.1f mm2, missed %.1f mm2" % (area[m].sum(), area[miss].sum()))
# GT tooth region pieces on the mesh
cg = components(np.where(m, 1, 0), edges)
u, c = np.unique(cg[m], return_counts=True)
print("  GT tooth mesh pieces:", [(int(c_), round(float(area[cg == a].sum()), 1)) for a, c_ in zip(u, c)])
# missed region: connected pieces and their contact with the predicted part
cm = components(np.where(miss, 1, 0), edges)
um = np.unique(cm[miss])
for a in um:
    P = cm == a
    if area[P].sum() < 1: continue
    e = edges[P[edges[:, 0]] ^ P[edges[:, 1]]]
    other = np.where(P[e[:, 0]], e[:, 1], e[:, 0])
    print("  missed piece %.1f mm2: border edges to predicted tooth %d, to GT gum %d, to other" % (area[P].sum(), int((hit[other]).sum()), int((gt[other] == 0).sum())), int(((gt[other] > 0) & ~m[other]).sum()))
    dd, ii = cKDTree(xyz[hit]).query(xyz[P])
    print("    distance to predicted part of the tooth: median %.2f max %.2f" % (np.median(dd), dd.max()))
# boundary edges in/around the tooth
e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
ue, ce = np.unique(e, axis=0, return_counts=True); bnd = ue[ce == 1]
bv = np.zeros(len(xyz), bool); bv[bnd.ravel()] = True
print("  open-boundary vertices: in tooth %d, in missed part %d" % (bv[m].sum(), bv[miss].sum()))
# overlapping sheets: vertices of the tooth with a non-neighbour vertex within 0.4 mm along the normal
fn = np.cross(xyz[f[:,1]]-xyz[f[:,0]], xyz[f[:,2]]-xyz[f[:,0]]); vn = np.zeros_like(xyz)
for k in range(3): np.add.at(vn, f[:,k], fn)
vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
# curvature proxy and edge length in missed vs hit
L = np.linalg.norm(xyz[edges[:,0]] - xyz[edges[:,1]], axis=1)
def med_edge(M):
    sel = M[edges[:,0]] & M[edges[:,1]]
    return np.median(L[sel]) if sel.any() else float('nan')
print("  median edge length: predicted part %.3f, missed part %.3f, gum %.3f" % (med_edge(hit), med_edge(miss), med_edge(gt == 0)))
