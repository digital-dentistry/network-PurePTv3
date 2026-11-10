"""ONNX export for PurePTv3.

The exported graph takes ``feat`` plus every index tensor from
``build_inputs`` as named inputs (names from ``model.input_names()``), with
the point count of every stage as a dynamic axis. Preprocessing stays outside
the graph and must run in the application before each inference.
"""

import re
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["FlatInputs", "dynamic_axes_for", "export_onnx", "onnx_feed"]


class FlatInputs(nn.Module):
    """Positional-argument wrapper: ``(feat, *indices)`` in ``names`` order."""

    def __init__(self, model: nn.Module, names: Sequence[str]):
        super().__init__()
        self.model = model
        self.names = list(names)

    def forward(self, feat, *indices):
        return self.model(feat, dict(zip(self.names, indices)))


def dynamic_axes_for(names: Sequence[str], output_stage: int = 0) -> Dict[str, Dict[int, str]]:
    """Axis names: ``n{s}`` points of stage s, ``l{s}`` padded patch slots, ``p{s}`` patches."""
    axes = {"feat": {0: "n0"}, "out": {0: f"n{output_stage}"}}
    for name in names:
        if name == "stem_kmap":
            axes[name] = {0: "n0"}
            continue
        m = re.fullmatch(r"s(\d+)_(kmap|order\d+|inverse\d+|attn_bias|pool|unpool)", name)
        if m is None:
            raise ValueError(f"unexpected input name {name!r}")
        s, kind = int(m.group(1)), m.group(2)
        if kind.startswith("order"):
            axes[name] = {0: f"l{s}"}
        elif kind == "attn_bias":
            axes[name] = {0: f"p{s}"}
        elif kind == "unpool":
            axes[name] = {0: f"n{s - 1}"}
        else:  # kmap, inverse, pool
            axes[name] = {0: f"n{s}"}
    return axes


def onnx_feed(feat: torch.Tensor, inputs: Dict[str, torch.Tensor], names: Sequence[str]):
    """numpy feed dict for onnxruntime from ``feat`` and ``build_inputs`` output."""
    feed = {"feat": feat.detach().cpu().numpy()}
    for name in names:
        feed[name] = inputs[name].detach().cpu().numpy()
    return feed


@torch.no_grad()
def export_onnx(
    model: nn.Module,
    feat: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    path: str,
    opset: int = 18,
    dynamic: bool = True,
    dynamo: bool = True,
) -> List[str]:
    """Export ``model`` (PurePTv3 or PurePTv3Segmentor) to ``path``.

    ``feat`` / ``inputs`` are an example batch (``inputs`` from ``prepare``).
    Returns the ordered index input names (after ``feat``).

    ``dynamo=True`` (needs ``onnxscript``) uses ``torch.export`` with every
    first axis set to ``Dim.AUTO``. ``dynamo=False`` uses the older TorchScript
    exporter with named ``dynamic_axes`` (deprecated by PyTorch since 2.9).
    Both were checked against onnxruntime with torch 2.11.
    """
    model = model.eval()
    names = model.input_names()
    wrapper = FlatInputs(model, names).eval()
    args = (feat,) + tuple(inputs[n] for n in names)
    backbone = getattr(model, "backbone", model)
    output_stage = (
        backbone.num_stages - 1
        if backbone.enc_mode and not hasattr(model, "seg_head")
        else 0
    )
    kwargs = dict(
        input_names=["feat"] + names,
        output_names=["out"],
        opset_version=opset,
    )
    if dynamo:
        dims = None
        if dynamic:
            auto = torch.export.Dim.AUTO
            # forward(feat, *indices): the varargs are one tuple for torch.export
            dims = ({0: auto}, tuple({0: auto} for _ in names))
        torch.onnx.export(wrapper, args, path, dynamo=True, dynamic_shapes=dims, **kwargs)
    else:
        torch.onnx.export(
            wrapper,
            args,
            path,
            dynamo=False,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes_for(names, output_stage) if dynamic else None,
            **kwargs,
        )
    return names
