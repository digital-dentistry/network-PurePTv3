"""Why does the clean-up step (absorb_fragments + components) lower F1?  Lists the tiny predicted instances and the
mesh shell (connected piece of the mesh) each one sits on.

usage: python specks.py <probe npz dir> <stages npz dir> <asset> [...]
Needs inference-server/ on the path (toothseg.vertex_graph); run from the repo root or set PYTHONPATH.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "inference-server"))
from toothseg import vertex_graph  # noqa: E402


def main():
    probe, stages = Path(sys.argv[1]), Path(sys.argv[2])
    for aid in sys.argv[3:]:
        d, s = np.load(probe / f"{aid}.npz"), np.load(stages / f"{aid}.npz")
        xyz, faces = d["xyz"].astype(np.float64), d["faces"].astype(np.int64)
        edges, area = vertex_graph(xyz, faces)
        n = len(xyz)
        _, shell = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
                                        directed=False)
        shell_area = np.bincount(shell, weights=area)
        v0 = s["v0_deployed"].astype(int)
        sizes = np.bincount(v0, weights=area)
        print(f"{aid}: {shell.max() + 1} mesh shells, {(shell_area < 5).sum()} under 5 mm2")
        for j in np.nonzero((sizes < 1) & (np.arange(len(sizes)) > 0))[0]:
            sh = np.unique(shell[v0 == j])
            print(f"  instance {j}: {(v0 == j).sum()} vertices, {sizes[j]:.3f} mm2, on shell(s) {sh.tolist()} of "
                  f"{shell_area[sh].round(2).tolist()} mm2 holding labels {np.unique(v0[np.isin(shell, sh)]).tolist()}")


if __name__ == "__main__":
    main()
