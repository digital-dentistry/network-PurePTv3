"""Shared helpers for the PurePTv3 tests (no pytest needed)."""

import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_cloud(n: int, seed: int, grid_size: float = 0.02, bumps: int = 6):
    """Surface-like point cloud (bumpy closed shell), grid-sampled to unique voxels.

    Returns ``coord [M,3]``, ``grid_coord [M,3] int32``, ``feat [M,6]``
    (coord + outward normal), one point per voxel like Pointcept GridSample.
    """
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(n, 3, generator=g)
    d = d / d.norm(dim=1, keepdim=True)
    centers = torch.randn(bumps, 3, generator=g)
    centers = centers / centers.norm(dim=1, keepdim=True)
    radius = 1.0 + 0.25 * torch.exp(-((d @ centers.T - 1.0) ** 2) / 0.02).sum(1)
    coord = d * radius[:, None] * torch.tensor([1.0, 0.8, 0.6])
    grid = torch.div(coord - coord.min(0)[0], grid_size, rounding_mode="trunc").int()
    _, first = _unique_first(grid)
    coord, grid, d = coord[first], grid[first], d[first]
    feat = torch.cat([coord, d], dim=1)
    return coord, grid, feat


def _unique_first(grid):
    key = (grid[:, 0].long() << 42) | (grid[:, 1].long() << 21) | grid[:, 2].long()
    uniq, inverse = torch.unique(key, return_inverse=True)
    first = torch.full((len(uniq),), len(key), dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(key)), reduce="amin")
    return uniq, first.sort().values


def randomize_(module: torch.nn.Module, seed: int = 0):
    """Perturb every parameter and BN statistic so parity tests are not trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
        for name, b in module.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(0.1 * torch.randn(b.shape, generator=g))
            elif name.endswith("running_var"):
                b.copy_(0.5 + torch.rand(b.shape, generator=g))


def max_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).abs().max().item()
    scale = b.abs().max().item()
    return diff, diff / max(scale, 1e-12)


def run_all(namespace: dict):
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            msg = fn()
            print(f"PASS {name}" + (f"  {msg}" if msg else ""))
        except SkipTest as e:
            print(f"SKIP {name}  {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed or skipped")
    sys.exit(1 if failed else 0)


class SkipTest(Exception):
    pass
