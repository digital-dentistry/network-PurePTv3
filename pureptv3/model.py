"""Point Transformer V3 (Pointcept PT-v3m1) in plain PyTorch.

No spconv, torch_scatter, flash_attn, timm or addict. All data-dependent index
work is done beforehand by ``pureptv3.preprocess.build_inputs``; the graph here
uses only Linear / LayerNorm / BatchNorm1d / GELU, Gather (``x[idx]``),
Concat, Reshape, MatMul, Softmax (via ``scaled_dot_product_attention``) and
ReduceMax. Index tensors are inputs, so no tensor shape depends on tensor
values and the model exports to ONNX with dynamic point counts.

Parameter names and shapes match Pointcept exactly, including the spconv 2.x
weight layout ``[Cout, kx, ky, kz, Cin]``, so a Pointcept ``state_dict`` loads
with ``load_state_dict`` as is.

Module-by-module replacement:

=====================================  =====================================
Pointcept                              PurePTv3
=====================================  =====================================
spconv.SubMConv3d (stem k=5, CPE k=3)  gather by ``kmap`` -> one MatMul
flash_attn varlen / masked attention   ``[P, K]`` patches -> SDPA with key mask
torch_scatter.segment_csr(max)         gather ``[M, 8]`` children -> amax
Point.serialization, pooling grouping  preprocessing (outside the graph)
=====================================  =====================================

Behaviour kept on purpose to stay numerically identical to trained weights:

- In the first block of every decoder stage the CPE convolution reads the
  skip branch only (``proj_skip(parent)``), not ``skip + upsampled``. Pointcept
  does this because ``SerializedUnpooling`` updates ``point.feat`` but not
  ``point.sparse_conv_feat``. Set ``dec_cpe_from_skip=False`` to use the
  summed feature instead (only for training from scratch).
- Decoder attention at stage ``s`` reuses the encoder's patch layout at stage
  ``s`` (Pointcept caches it on the shared ``Point``), so ``enc_patch_size[s]``
  is what counts there.
"""

import math
import warnings
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import build_inputs

__all__ = [
    "PurePTv3",
    "PurePTv3Segmentor",
    "SubMConv3d",
    "SerializedAttention",
    "Block",
    "SerializedPooling",
    "SerializedUnpooling",
    "Embedding",
]

Inputs = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------- #
# Basic layers
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    """Stochastic depth, same as ``timm.layers.DropPath`` (drops per point)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


# --------------------------------------------------------------------------- #
# Training-time backward
#
# Autograd's backward of ``x[index]`` is a sort-based ``index_put_`` that runs
# the repeats of one index one after another. A kernel map repeats the
# "empty neighbour" index N millions of times, so that backward took 97 % of a
# training step. The two Functions below replace it; they are used only when a
# gradient is needed, so inference and ONNX export trace the plain ops above.
# Set FAST_BACKWARD = False to fall back to plain autograd (tests compare both).
# --------------------------------------------------------------------------- #
FAST_BACKWARD = True


def _needs_fast_backward(*tensors) -> bool:
    return (
        FAST_BACKWARD
        and torch.is_grad_enabled()
        and any(t is not None and t.requires_grad for t in tensors)
    )


def _autocast_dtype(t: torch.Tensor) -> torch.dtype:
    """The dtype ``F.linear`` would compute in here (autocast aware)."""
    dev = t.device.type
    try:
        if torch.is_autocast_enabled(dev):
            return torch.get_autocast_dtype(dev)
    except TypeError:  # torch < 2.4: no device argument, CUDA only
        if dev == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return t.dtype


class _GatherRows(torch.autograd.Function):
    """``x[index]`` with an atomic ``index_add_`` backward (accumulated in fp32)."""

    @staticmethod
    def forward(ctx, x, index):
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        return x[index]

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        tail = grad.shape[index.dim():]
        out = grad.new_zeros((ctx.rows, *tail), dtype=torch.float32)
        out.index_add_(0, index.reshape(-1), grad.reshape(-1, *tail).float())
        return out.to(grad.dtype), None


def gather_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if _needs_fast_backward(x):
        return _GatherRows.apply(x, index)
    return x[index]


class _SubMConvFunction(torch.autograd.Function):
    """Gather conv that recomputes the gather in backward instead of storing it.

    For a submanifold convolution ``kmap[i, k] == m`` exactly when
    ``kmap[m, K-1-k] == i`` (offset k and its mirror K-1-k cancel), so the
    input gradient is the same gather with the kernel flipped and transposed.
    Backward is then two gathers and two matmuls, with no scatter at all.
    """

    @staticmethod
    def forward(ctx, feat, weight, bias, kmap, dtype):
        cin = feat.shape[1]
        padded = torch.cat([feat.to(dtype), feat.new_zeros((1, cin), dtype=dtype)])
        out = padded[kmap].flatten(1) @ weight.to(dtype).flatten(1).t()
        if bias is not None:
            out = out + bias.to(dtype)
        ctx.save_for_backward(feat, weight, kmap)
        ctx.dtype = dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        feat, weight, kmap = ctx.saved_tensors
        dtype = ctx.dtype
        g = grad.to(dtype)
        cout, cin, k = weight.shape[0], weight.shape[-1], kmap.shape[1]
        grad_feat = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            gp = torch.cat([g, g.new_zeros((1, cout))])
            w_flip = weight.to(dtype).reshape(cout, k, cin).flip(1)
            w_flip = w_flip.permute(1, 0, 2).reshape(k * cout, cin)
            grad_feat = (gp[kmap].flatten(1) @ w_flip).to(feat.dtype)
        if ctx.needs_input_grad[1]:
            padded = torch.cat([feat.to(dtype), feat.new_zeros((1, cin), dtype=dtype)])
            grad_weight = (g.t() @ padded[kmap].flatten(1)).view(weight.shape)
            grad_weight = grad_weight.to(weight.dtype)
        if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
            grad_bias = g.float().sum(0).to(ctx.bias_dtype)
        return grad_feat, grad_weight, grad_bias, None, None


class SubMConv3d(nn.Module):
    """Submanifold sparse 3D convolution as gather + matmul.

    ``weight`` keeps the spconv 2.x layout ``[Cout, k, k, k, Cin]``. ``kmap``
    ``[N, k**3]`` comes from ``preprocess.kernel_map``; the value ``N`` marks an
    empty neighbour and reads a zero row.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        k = int(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.weight = nn.Parameter(torch.empty(out_channels, k, k, k, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size**3
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, feat: torch.Tensor, kmap: torch.Tensor) -> torch.Tensor:
        if _needs_fast_backward(feat, self.weight, self.bias):
            return _SubMConvFunction.apply(
                feat, self.weight, self.bias, kmap, _autocast_dtype(feat)
            )
        padded = torch.cat([feat, feat.new_zeros((1, feat.shape[1]))], dim=0)
        x = padded[kmap].flatten(1)  # [N, K*Cin], Cin fastest
        return F.linear(x, self.weight.flatten(1), self.bias)

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, bias={self.bias is not None}"
        )


# --------------------------------------------------------------------------- #
# PTv3 modules
# --------------------------------------------------------------------------- #
class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = int(patch_size)
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.attn_drop = float(attn_drop)
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, order, inverse, bias):
        h, k, c = self.num_heads, self.patch_size, self.channels
        qkv = gather_rows(self.qkv(feat), order)  # [P*K, 3C], point -> patch slot
        q, kk, v = qkv.reshape(-1, k, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q,
            kk,
            v,
            attn_mask=bias.to(q.dtype),
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale,
        )  # [P, H, K, D]
        out = gather_rows(out.transpose(1, 2).reshape(-1, c), inverse)  # slot -> point
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.order_index = order_index
        self.cpe = nn.ModuleList(
            [
                SubMConv3d(channels, channels, kernel_size=3, bias=True),
                nn.Linear(channels, channels),
                norm_layer(channels),
            ]
        )
        self.norm1 = nn.Sequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
        )
        self.norm2 = nn.Sequential(norm_layer(channels))
        self.mlp = nn.Sequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, feat, kmap, order, inverse, bias, cpe_feat=None):
        x = feat if cpe_feat is None else cpe_feat
        x = self.cpe[2](self.cpe[1](self.cpe[0](x, kmap)))
        feat = feat + x

        shortcut = feat
        if self.pre_norm:
            feat = self.norm1(feat)
        feat = shortcut + self.drop_path(self.attn(feat, order, inverse, bias))
        if not self.pre_norm:
            feat = self.norm1(feat)

        shortcut = feat
        if self.pre_norm:
            feat = self.norm2(feat)
        feat = shortcut + self.drop_path(self.mlp(feat))
        if not self.pre_norm:
            feat = self.norm2(feat)
        return feat


class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
    ):
        super().__init__()
        if reduce not in ("max", "min"):
            raise NotImplementedError(
                f"reduce={reduce!r}: only max/min pooling is implemented"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduce = reduce
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.Sequential(norm_layer(out_channels)) if norm_layer else None
        self.act = nn.Sequential(act_layer()) if act_layer else None

    def forward(self, feat, children):
        x = gather_rows(self.proj(feat), children)  # [M, W, C]
        x = x.amax(dim=1) if self.reduce == "max" else x.amin(dim=1)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SerializedUnpooling(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        proj = [nn.Linear(in_channels, out_channels)]
        proj_skip = [nn.Linear(skip_channels, out_channels)]
        if norm_layer is not None:
            proj.append(norm_layer(out_channels))
            proj_skip.append(norm_layer(out_channels))
        if act_layer is not None:
            proj.append(act_layer())
            proj_skip.append(act_layer())
        self.proj = nn.Sequential(*proj)
        self.proj_skip = nn.Sequential(*proj_skip)

    def forward(self, feat, skip_feat, parent_index):
        """Returns ``(skip + upsampled, skip)``; the second goes to the CPE."""
        skip = self.proj_skip(skip_feat)
        return skip + gather_rows(self.proj(feat), parent_index), skip


class Embedding(nn.Module):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        stem = {"conv": SubMConv3d(in_channels, embed_channels, 5, bias=False)}
        if norm_layer is not None:
            stem["norm"] = norm_layer(embed_channels)
        if act_layer is not None:
            stem["act"] = act_layer()
        self.stem = nn.ModuleDict(stem)

    def forward(self, feat, kmap):
        feat = self.stem["conv"](feat, kmap)
        if "norm" in self.stem:
            feat = self.stem["norm"](feat)
        if "act" in self.stem:
            feat = self.stem["act"](feat)
        return feat


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
class PurePTv3(nn.Module):
    """Drop-in for Pointcept ``PT-v3m1`` (same arguments, same state_dict).

    ``forward(feat, inputs)`` takes the per-point input features ``[N, C_in]``
    and the dict from ``prepare`` / ``preprocess.build_inputs`` and returns the
    per-point output features (decoder) or the coarsest-stage features
    (``enc_mode=True``).

    Accepted but ignored: ``enable_flash``, ``upcast_attention``,
    ``upcast_softmax`` (attention always runs as SDPA in the input dtype).
    Not supported: ``enable_rpe`` and PDNorm (``pdnorm_bn``/``pdnorm_ln``).
    """

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        dec_cpe_from_skip=True,
    ):
        super().__init__()
        if enable_rpe:
            raise NotImplementedError("enable_rpe=True is not supported")
        if pdnorm_bn or pdnorm_ln:
            raise NotImplementedError("PDNorm (pdnorm_bn / pdnorm_ln) is not supported")

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.stride = tuple(stride)
        self.enc_depths = tuple(enc_depths)
        self.dec_depths = tuple(dec_depths)
        self.enc_patch_size = tuple(int(p) for p in enc_patch_size)
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.dec_cpe_from_skip = dec_cpe_from_skip

        s_ = self.num_stages
        assert s_ == len(stride) + 1
        assert s_ == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)
        if not enc_mode:
            assert s_ == len(dec_depths) + 1 == len(dec_channels) + 1
            assert s_ == len(dec_num_head) + 1 == len(dec_patch_size) + 1
            for s in range(s_ - 1):
                if dec_patch_size[s] != enc_patch_size[s]:
                    warnings.warn(
                        f"dec_patch_size[{s}]={dec_patch_size[s]} differs from "
                        f"enc_patch_size[{s}]={enc_patch_size[s]}; Pointcept reuses "
                        "the encoder patch layout in the decoder, and so does PurePTv3"
                    )

        bn_layer = lambda c: nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)  # noqa: E731
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU
        n_orders = len(self.order)

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = nn.ModuleDict()
        for s in range(s_):
            dp = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            stage = nn.ModuleDict()
            if s > 0:
                stage["down"] = SerializedPooling(
                    in_channels=enc_channels[s - 1],
                    out_channels=enc_channels[s],
                    stride=stride[s - 1],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                )
            for i in range(enc_depths[s]):
                stage[f"block{i}"] = Block(
                    channels=enc_channels[s],
                    num_heads=enc_num_head[s],
                    patch_size=enc_patch_size[s],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dp[i],
                    norm_layer=ln_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    order_index=i % n_orders,
                )
            if len(stage) != 0:
                self.enc[f"enc{s}"] = stage

        if not enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = nn.ModuleDict()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(s_ - 1)):
                dp = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
                dp.reverse()
                stage = nn.ModuleDict()
                stage["up"] = SerializedUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                )
                for i in range(dec_depths[s]):
                    stage[f"block{i}"] = Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        # Pointcept reuses the encoder's cached padding here.
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dp[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % n_orders,
                    )
                self.dec[f"dec{s}"] = stage

    # ------------------------------------------------------------------ #
    def prepare(self, grid_coord=None, *, shuffle_orders=None, **kwargs) -> Inputs:
        """``build_inputs`` with this model's order / stride / patch sizes.

        ``shuffle_orders=None`` shuffles only in training mode (and only when
        the model was built with ``shuffle_orders=True``), so evaluation is
        deterministic.
        """
        if shuffle_orders is None:
            shuffle_orders = self.shuffle_orders and self.training
        return build_inputs(
            grid_coord,
            order=self.order,
            stride=self.stride,
            patch_size=self.enc_patch_size,
            shuffle_orders=shuffle_orders,
            **kwargs,
        )

    def input_names(self) -> List[str]:
        """Index keys ``forward`` reads, in a fixed order (for ONNX export)."""
        names = ["stem_kmap"]

        def attn_keys(s, depth):
            keys = [f"s{s}_kmap", f"s{s}_attn_bias"]
            for i in range(depth):
                j = i % len(self.order)
                keys += [f"s{s}_order{j}", f"s{s}_inverse{j}"]
            return keys

        for s in range(self.num_stages):
            if s > 0:
                names.append(f"s{s}_pool")
            names += attn_keys(s, self.enc_depths[s])
        if not self.enc_mode:
            for s in reversed(range(self.num_stages - 1)):
                names.append(f"s{s + 1}_unpool")
                names += attn_keys(s, self.dec_depths[s])
        return list(dict.fromkeys(names))

    def _run_blocks(self, stage, depth, feat, s, inputs: Inputs, first_cpe_feat=None):
        for i in range(depth):
            block = stage[f"block{i}"]
            j = block.order_index
            feat = block(
                feat,
                inputs[f"s{s}_kmap"],
                inputs[f"s{s}_order{j}"],
                inputs[f"s{s}_inverse{j}"],
                inputs[f"s{s}_attn_bias"],
                cpe_feat=first_cpe_feat if i == 0 else None,
            )
        return feat

    def forward_encoder(self, feat: torch.Tensor, inputs: Inputs) -> List[torch.Tensor]:
        """Per-stage encoder outputs, finest first."""
        feat = self.embedding(feat, inputs["stem_kmap"])
        stages = []
        for s in range(self.num_stages):
            if s > 0:
                feat = self.enc[f"enc{s}"]["down"](feat, inputs[f"s{s}_pool"])
            if f"enc{s}" in self.enc:
                feat = self._run_blocks(
                    self.enc[f"enc{s}"], self.enc_depths[s], feat, s, inputs
                )
            stages.append(feat)
        return stages

    def forward(self, feat: torch.Tensor, inputs: Inputs) -> torch.Tensor:
        stages = self.forward_encoder(feat, inputs)
        feat = stages[-1]
        if self.enc_mode:
            return feat
        for s in reversed(range(self.num_stages - 1)):
            stage = self.dec[f"dec{s}"]
            feat, skip = stage["up"](feat, stages[s], inputs[f"s{s + 1}_unpool"])
            feat = self._run_blocks(
                stage,
                self.dec_depths[s],
                feat,
                s,
                inputs,
                first_cpe_feat=skip if self.dec_cpe_from_skip else None,
            )
        return feat


# --------------------------------------------------------------------------- #
# Segmentor (Pointcept DefaultSegmentorV2: backbone + linear head)
# --------------------------------------------------------------------------- #
class PurePTv3Segmentor(nn.Module):
    """Same keys as Pointcept ``DefaultSegmentorV2`` (``backbone.*``, ``seg_head.*``).

    ``backbone`` is either a ``PurePTv3`` or the backbone config dict from a
    Pointcept config (its ``type`` key is ignored).
    """

    def __init__(self, num_classes, backbone_out_channels, backbone=None):
        super().__init__()
        if isinstance(backbone, dict):
            backbone = PurePTv3(**{k: v for k, v in backbone.items() if k != "type"})
        self.backbone = backbone if backbone is not None else PurePTv3()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    @classmethod
    def from_config(cls, model_cfg: dict) -> "PurePTv3Segmentor":
        """Build from a Pointcept ``model = dict(type="DefaultSegmentorV2", ...)``."""
        return cls(
            num_classes=model_cfg["num_classes"],
            backbone_out_channels=model_cfg["backbone_out_channels"],
            backbone=dict(model_cfg["backbone"]),
        )

    def prepare(self, grid_coord=None, **kwargs) -> Inputs:
        return self.backbone.prepare(grid_coord, **kwargs)

    def input_names(self) -> List[str]:
        names = self.backbone.input_names()
        if self.backbone.enc_mode:
            names += [f"s{s}_unpool" for s in range(1, self.backbone.num_stages)]
        return list(dict.fromkeys(names))

    def forward(self, feat: torch.Tensor, inputs: Inputs) -> torch.Tensor:
        if not self.backbone.enc_mode:
            return self.seg_head(self.backbone(feat, inputs))
        # enc_mode: concat every stage, upsampled to the input points (V2 logic)
        stages = self.backbone.forward_encoder(feat, inputs)
        feat = stages[-1]
        for s in reversed(range(self.backbone.num_stages - 1)):
            feat = torch.cat([stages[s], feat[inputs[f"s{s + 1}_unpool"]]], dim=-1)
        return self.seg_head(feat)
