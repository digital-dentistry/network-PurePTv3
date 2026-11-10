"""Index precomputation for PurePTv3.

Everything whose *shape or value depends on the data* lives here, outside the
model graph: serialization (argsort), neighbour search for the sparse
convolutions, patch padding for attention, and the parent/child grouping for
pooling. The model itself then only needs Gather, MatMul, Softmax and Max.

The algorithms reproduce Pointcept PT-v3m1 step by step:

- ``Point.serialization``                   -> stage-0 codes / orders / inverses
- ``SerializedAttention.get_padding_and_inverse`` -> ``attention_padding``
- ``SerializedPooling.forward`` (index part) -> ``_pool_step``
- spconv ``SubMConv3d`` rulebook            -> ``kernel_map``

Output: a flat ``dict[str, Tensor]`` (see ``build_inputs``) that is passed
unchanged to ``PurePTv3.forward`` and, flattened, to an exported ONNX graph.
"""

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .serialization import encode

__all__ = [
    "build_inputs",
    "kernel_map",
    "attention_padding",
    "grid_coord_from_coord",
    "offset2batch",
    "batch2offset",
]

ATTN_MASK_VALUE = float("-inf")


# --------------------------------------------------------------------------- #
# Small helpers (same semantics as pointcept.models.utils.misc)
# --------------------------------------------------------------------------- #
def offset2bincount(offset: torch.Tensor) -> torch.Tensor:
    return torch.diff(offset, prepend=offset.new_zeros(1))


def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


def batch2offset(batch: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(batch.bincount(), dim=0).long()


@torch.no_grad()
def grid_coord_from_coord(coord: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Pointcept's fallback when ``grid_coord`` is missing (``Point.serialization``)."""
    return torch.div(
        coord - coord.min(0)[0], grid_size, rounding_mode="trunc"
    ).int()


# --------------------------------------------------------------------------- #
# Sparse convolution neighbour map
# --------------------------------------------------------------------------- #
@torch.no_grad()
def kernel_map(
    grid_coord: torch.Tensor, batch: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    """Neighbour table of a submanifold convolution.

    Returns ``kmap`` of shape ``[N, kernel_size**3]``. ``kmap[i, j]`` is the
    index of the point at ``grid_coord[i] + offset[j]`` in the same sample, or
    ``N`` when that voxel is empty. Offsets run over ``(dx, dy, dz)`` in
    ``[-r, r]^3`` with ``dz`` fastest, which is the flattening order of the
    spconv 2.x weight ``[Cout, kx, ky, kz, Cin]`` (verified against spconv
    2.3.8: input voxel = output voxel + (kernel index - centre), i.e. the same
    cross-correlation as ``torch.nn.functional.conv3d``).
    """
    if kernel_size % 2 != 1:
        raise ValueError("submanifold convolution needs an odd kernel size")
    r = kernel_size // 2
    n = grid_coord.shape[0]
    device = grid_coord.device
    g = grid_coord.long() + r  # shift so that every neighbour stays >= 0
    b = batch.long()
    bits = max(int(g.max()) + r + 1, 1).bit_length()
    batch_bits = max(int(b.max()) + 1, 1).bit_length()
    if 3 * bits + batch_bits > 63:
        raise ValueError("grid too large to hash into int64")

    def key(c: torch.Tensor, bb: torch.Tensor) -> torch.Tensor:
        return (
            (bb << (3 * bits))
            | (c[..., 0] << (2 * bits))
            | (c[..., 1] << bits)
            | c[..., 2]
        )

    sorted_key, sort_index = torch.sort(key(g, b))
    if n > 1 and bool((sorted_key[1:] == sorted_key[:-1]).any()):
        raise ValueError(
            "duplicate grid_coord inside one sample; grid-sample the points first "
            "(Pointcept GridSample) so that every voxel holds one point"
        )
    rng = torch.arange(-r, r + 1, device=device)
    offsets = torch.cartesian_prod(rng, rng, rng).reshape(-1, 3)  # dz fastest

    kmap = torch.empty((n, offsets.shape[0]), dtype=torch.long, device=device)
    step = max(1, (1 << 24) // max(n, 1))  # offsets per pass: ~16M queries
    for j0 in range(0, offsets.shape[0], step):
        query = key(g[:, None, :] + offsets[None, j0 : j0 + step], b[:, None])
        pos = torch.searchsorted(sorted_key, query).clamp_(max=n - 1)
        kmap[:, j0 : j0 + step] = torch.where(sorted_key[pos] == query, sort_index[pos], n)
    return kmap


# --------------------------------------------------------------------------- #
# Serialized attention: patch padding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_padding(bincount: torch.Tensor, patch_size: int):
    """Patch layout of serialized attention.

    Mirrors ``SerializedAttention.get_padding_and_inverse`` and the FlashAttention
    varlen call that consumes it:

    - a sample with ``n > K`` points is padded to a multiple of ``K``; the last
      patch is filled up with the points that precede it (Pointcept's
      duplication trick), so it holds the last ``K`` points of the sample;
    - a sample with ``n <= K`` is a single patch of ``n`` points. Flash handles
      that as a shorter sequence; here it is padded to ``K`` and the padding
      keys are masked out, which is the same computation.

    Returns ``pad`` ``[P*K]`` (serialized position for every padded slot),
    ``unpad`` ``[N]`` (padded slot of every serialized position) and
    ``bias`` ``[P, 1, 1, K]`` (additive key mask, 0 or -inf).
    """
    k = int(patch_size)
    device = bincount.device
    bincount = bincount.long()
    bincount_pad = torch.div(bincount + k - 1, k, rounding_mode="trunc") * k
    offset = F.pad(torch.cumsum(bincount, 0), (1, 0))
    offset_pad = F.pad(torch.cumsum(bincount_pad, 0), (1, 0))
    total = int(offset_pad[-1])

    pad = torch.arange(total, device=device)
    unpad = torch.arange(int(offset[-1]), device=device)
    valid = torch.ones(total, dtype=torch.bool, device=device)
    for i in range(len(bincount)):
        n = int(bincount[i])
        s0, s1 = int(offset[i]), int(offset[i + 1])
        p0, p1 = int(offset_pad[i]), int(offset_pad[i + 1])
        unpad[s0:s1] += p0 - s0
        rem = n % k
        if rem != 0:
            if n > k:
                pad[p1 - k + rem : p1] = pad[p1 - 2 * k + rem : p1 - k].clone()
            else:
                pad[p0 + n : p1] = p0
                valid[p0 + n : p1] = False
        pad[p0:p1] -= p0 - s0

    bias = torch.zeros(total, dtype=torch.float32, device=device)
    bias[~valid] = ATTN_MASK_VALUE
    return pad, unpad, bias.view(-1, 1, 1, k)


# --------------------------------------------------------------------------- #
# Serialization and pooling
# --------------------------------------------------------------------------- #
def _order_and_inverse(code: torch.Tensor):
    order = torch.argsort(code, dim=1, stable=True)
    inverse = torch.empty_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(code.shape[1], device=code.device).repeat(code.shape[0], 1),
    )
    return order, inverse


def _maybe_shuffle(code, order, inverse, shuffle, generator):
    if not shuffle:
        return code, order, inverse
    perm = torch.randperm(code.shape[0], generator=generator).to(code.device)
    return code[perm], order[perm], inverse[perm]


@torch.no_grad()
def _pool_step(code, grid_coord, batch, depth, stride):
    """Index part of ``SerializedPooling.forward``."""
    if stride != 2 ** (math.ceil(stride) - 1).bit_length():
        raise ValueError(f"pooling stride must be a power of two, got {stride}")
    pooling_depth = (math.ceil(stride) - 1).bit_length()
    if pooling_depth > depth:
        pooling_depth = 0

    code = code >> pooling_depth * 3
    _, cluster, counts = torch.unique(
        code[0], sorted=True, return_inverse=True, return_counts=True
    )
    _, indices = torch.sort(cluster, stable=True)
    idx_ptr = F.pad(torch.cumsum(counts, 0), (1, 0))
    head = indices[idx_ptr[:-1]]

    # [M, W] children of every parent; empty slots repeat the first child, which
    # leaves a max/min reduction unchanged and needs no sentinel row.
    width = max(8**pooling_depth, int(counts.max()))
    sorted_cluster = cluster[indices]
    rank = torch.arange(indices.numel(), device=code.device) - idx_ptr[sorted_cluster]
    children = head[:, None].repeat(1, width)
    children[sorted_cluster, rank] = indices

    return (
        children,
        cluster,
        code[:, head],
        grid_coord[head] >> pooling_depth,
        batch[head],
        depth - pooling_depth,
    )


@torch.no_grad()
def build_inputs(
    grid_coord: Optional[torch.Tensor] = None,
    *,
    batch: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    coord: Optional[torch.Tensor] = None,
    grid_size: Optional[float] = None,
    order: Sequence[str] = ("z", "z-trans"),
    stride: Sequence[int] = (2, 2, 2, 2),
    patch_size: Sequence[int] = (48, 48, 48, 48, 48),
    stem_kernel_size: int = 5,
    cpe_kernel_size: int = 3,
    depth: Optional[int] = None,
    shuffle_orders: bool = False,
    generator: Optional[torch.Generator] = None,
    index_dtype: torch.dtype = torch.long,
) -> Dict[str, torch.Tensor]:
    """Compute every index tensor PurePTv3 needs for one (batched) point cloud.

    Points are given flat, Pointcept style: ``grid_coord`` ``[N, 3]`` (or
    ``coord`` + ``grid_size``) and ``batch`` ``[N]`` or ``offset`` ``[B]``.
    ``grid_coord`` must be unique within a sample (run GridSample first).

    Keys of the returned dict (``s`` = stage, ``j`` = serialization slot):

    ==================  ===================  ======================================
    key                 shape                meaning
    ==================  ===================  ======================================
    stem_kmap           [N0, 125]            5x5x5 neighbours for the stem conv
    s{s}_kmap           [Ns, 27]             3x3x3 neighbours for the CPE convs
    s{s}_order{j}       [Ps*Ks]              point index per padded patch slot
    s{s}_inverse{j}     [Ns]                 patch slot holding each point
    s{s}_attn_bias      [Ps, 1, 1, Ks]       additive key mask (0 / -inf)
    s{s}_pool  (s>=1)   [Ns, 8]              children (stage s-1 indices)
    s{s}_unpool (s>=1)  [N(s-1)]             parent (stage s index) of each child
    ==================  ===================  ======================================

    ``shuffle_orders`` permutes the serialization slots at every stage like
    Pointcept does in training (and, by default, in Pointcept inference too).
    Keep it ``False`` for deterministic deployment.
    """
    if grid_coord is None:
        if coord is None or grid_size is None:
            raise ValueError("pass grid_coord, or coord together with grid_size")
        grid_coord = grid_coord_from_coord(coord, grid_size)
    if batch is None:
        if offset is None:
            batch = torch.zeros(
                grid_coord.shape[0], dtype=torch.long, device=grid_coord.device
            )
        else:
            batch = offset2batch(offset.to(grid_coord.device))
    batch = batch.long().to(grid_coord.device)

    order = [order] if isinstance(order, str) else list(order)
    num_stages = len(stride) + 1
    if len(patch_size) != num_stages:
        raise ValueError("patch_size needs one entry per stage")

    if depth is None:
        depth = int(grid_coord.max() + 1).bit_length()
    num_samples = int(batch.max()) + 1
    if depth > 16 or depth * 3 + num_samples.bit_length() > 63:
        raise ValueError(f"serialization depth {depth} too large")

    code = torch.stack([encode(grid_coord, batch, depth, order=o) for o in order])
    ser_order, ser_inverse = _order_and_inverse(code)
    code, ser_order, ser_inverse = _maybe_shuffle(
        code, ser_order, ser_inverse, shuffle_orders, generator
    )

    out: Dict[str, torch.Tensor] = {
        "stem_kmap": kernel_map(grid_coord, batch, stem_kernel_size)
    }
    for s in range(num_stages):
        out[f"s{s}_kmap"] = kernel_map(grid_coord, batch, cpe_kernel_size)
        bincount = torch.bincount(batch, minlength=num_samples)
        pad, unpad, bias = attention_padding(bincount, patch_size[s])
        for j in range(len(order)):
            out[f"s{s}_order{j}"] = ser_order[j][pad]
            out[f"s{s}_inverse{j}"] = unpad[ser_inverse[j]]
        out[f"s{s}_attn_bias"] = bias
        if s == num_stages - 1:
            break

        children, cluster, code, grid_coord, batch, depth = _pool_step(
            code, grid_coord, batch, depth, stride[s]
        )
        ser_order, ser_inverse = _order_and_inverse(code)
        code, ser_order, ser_inverse = _maybe_shuffle(
            code, ser_order, ser_inverse, shuffle_orders, generator
        )
        out[f"s{s + 1}_pool"] = children
        out[f"s{s + 1}_unpool"] = cluster

    return {
        k: (v if v.is_floating_point() else v.to(index_dtype)) for k, v in out.items()
    }
