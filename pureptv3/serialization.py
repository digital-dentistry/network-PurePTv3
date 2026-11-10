"""Space-filling-curve serialization (Z-order / Hilbert), pure PyTorch.

Copied from Pointcept ``pointcept/models/utils/serialization`` so that PurePTv3
produces bit-identical codes without importing Pointcept:

- Z-order: from OCNN (MIT, Copyright (c) 2022 Peng-Shuai Wang).
- Hilbert: from numpy-hilbert-curve (MIT), ported to torch by Xiaoyang Wu and Kaixin Xu.

This runs in preprocessing only. Nothing here is part of the exported graph.
"""

from typing import Optional, Union

import torch

__all__ = ["encode", "z_order_encode", "hilbert_encode"]


# --------------------------------------------------------------------------- #
# Z-order (OCNN)
# --------------------------------------------------------------------------- #
class _KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        cpu = torch.device("cpu")
        self._encode = {
            cpu: (
                self._xyz2key(r256, zero, zero, 8),
                self._xyz2key(zero, r256, zero, 8),
                self._xyz2key(zero, zero, r256, 8),
            )
        }

    def encode_lut(self, device):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    @staticmethod
    def _xyz2key(x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key


_key_lut = _KeyLUT()


def _xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    ex, ey, ez = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = ex[x & mask] | ey[y & mask] | ez[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = ex[(x >> 8) & mask] | ey[(y >> 8) & mask] | ez[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        key = b.long() << 48 | key
    return key


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    return _xyz2key(x, y, z, b=None, depth=depth)


# --------------------------------------------------------------------------- #
# Hilbert (numpy-hilbert-curve, torch port from Pointcept)
# --------------------------------------------------------------------------- #
def _right_shift(binary, k=1, axis=-1):
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )


def _gray2binary(gray, axis=-1):
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def _hilbert_encode(locs, num_dims, num_bits):
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            f"last dimension of locs is {orig_shape[-1]}, expected num_dims={num_dims}"
        )
    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims={num_dims} and num_bits={num_bits} do not fit into int64"
        )

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )
    return hh_uint8.view(torch.int64).squeeze()


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    return _hilbert_encode(grid_coord, num_dims=3, num_bits=depth)


# --------------------------------------------------------------------------- #
# Public entry point (same contract as pointcept.models.utils.serialization.encode)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode(
    grid_coord: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    depth: int = 16,
    order: str = "z",
) -> torch.Tensor:
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise ValueError(f"unknown serialization order {order!r}")
    # A single point makes the Hilbert path squeeze down to a 0-d tensor.
    code = code.reshape(-1)
    if batch is not None:
        code = batch.long() << depth * 3 | code
    return code
