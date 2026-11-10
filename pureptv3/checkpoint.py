"""Load Pointcept PT-v3m1 checkpoints into PurePTv3.

PurePTv3 uses Pointcept's parameter names and the spconv 2.x conv weight
layout, so loading is only key cleanup (``module.`` from DDP, optional
``backbone.`` selection) plus shape checks with readable errors.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = ["read_pointcept_state_dict", "load_pointcept_checkpoint", "init_from_pretrained"]

STEM_KEY = "backbone.embedding.stem.conv.weight"  # [Cout, k, k, k, Cin], the only layer that sees in_channels


def _pointcept_safe_globals() -> list:
    """Non-tensor globals pickled in a Pointcept *training* checkpoint (``model_best.pth``).

    The OneCycleLR state stores its anneal function as ``getattr(OneCycleLR, name)``
    and the best metric is a numpy scalar.  Allowing exactly these keeps
    ``weights_only=True`` (no arbitrary code) for the released Pointcept weights.
    """
    import codecs

    import numpy as np
    from torch.optim.lr_scheduler import OneCycleLR

    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    dtype_classes = [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]  # numpy >= 1.25
    return [getattr, OneCycleLR, codecs.encode, np.dtype, (scalar, "numpy.core.multiarray.scalar"),
            *(c for c in dtype_classes if c is not np.dtype)]


def read_pointcept_state_dict(
    checkpoint: Union[str, dict],
    prefix: Optional[str] = None,
    weights_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Return a clean state_dict from a Pointcept checkpoint file or dict.

    ``prefix`` selects and strips a sub-module, e.g. ``"backbone."`` to load a
    ``DefaultSegmentorV2`` checkpoint into a bare ``PurePTv3``.
    Pointcept training checkpoints load with ``weights_only=True`` (their few
    non-tensor objects are allow-listed). ``weights_only=False`` is only needed
    for other pickles; only use it for files you trust.
    """
    if isinstance(checkpoint, str):
        with torch.serialization.safe_globals(_pointcept_safe_globals()):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=weights_only)
    state = checkpoint.get("state_dict", checkpoint)
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if prefix is not None:
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
        out[key] = value
    if not out:
        raise ValueError(f"no parameters found (prefix={prefix!r})")
    return out


def _check_shapes(model: nn.Module, state: Dict[str, torch.Tensor]):
    own = model.state_dict()
    problems = []
    for key, value in state.items():
        if key not in own:
            continue
        want = tuple(own[key].shape)
        got = tuple(value.shape)
        if want == got:
            continue
        hint = ""
        if len(want) == 5:
            hint = (
                " (sparse conv weight: PurePTv3 expects spconv>=2.2 layout "
                "[Cout, kx, ky, kz, Cin])"
            )
        problems.append(f"{key}: checkpoint {got} vs model {want}{hint}")
    if problems:
        raise ValueError("shape mismatch:\n  " + "\n  ".join(problems))
    unsupported = [k for k in state if ".rpe." in k]
    if unsupported:
        raise ValueError(
            "checkpoint uses RPE (enable_rpe=True), which PurePTv3 does not "
            "implement: " + ", ".join(unsupported[:5])
        )


def load_pointcept_checkpoint(
    model: nn.Module,
    checkpoint: Union[str, dict],
    prefix: Optional[str] = None,
    strict: bool = True,
    weights_only: bool = True,
):
    """Load a Pointcept checkpoint into ``model`` (PurePTv3 or PurePTv3Segmentor).

    With ``prefix=None`` the prefix is guessed: a bare ``PurePTv3`` takes the
    ``backbone.`` part of a segmentor checkpoint.
    """
    state = read_pointcept_state_dict(checkpoint, weights_only=weights_only)
    if prefix is None:
        own = set(model.state_dict())
        if not (own & set(state)) and any(k.startswith("backbone.") for k in state):
            prefix = "backbone."
    if prefix:
        state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    _check_shapes(model, state)
    return model.load_state_dict(state, strict=strict)


@torch.no_grad()
def init_from_pretrained(
    model: nn.Module,
    checkpoint: Union[str, dict],
    in_channel_map: Optional[Sequence[Optional[int]]] = None,
    fresh: Sequence[str] = ("seg_head.",),
    weights_only: bool = True,
) -> Dict[str, List[str]]:
    """Initialise a ``PurePTv3Segmentor`` from a Pointcept checkpoint for fine-tuning.

    For a model whose input features or classes differ from the checkpoint's:

    - every parameter and buffer with a same-shape checkpoint tensor is copied
      (BatchNorm running statistics included);
    - the stem conv (``STEM_KEY``, the only layer that sees the input channels)
      is rebuilt channel by channel when ``in_channels`` differ:
      ``in_channel_map[i]`` is the checkpoint input channel that model input
      channel ``i`` takes its weights from, or ``None`` for zeros (a feature
      the checkpoint never saw starts with no influence and is learned);
    - keys starting with one of ``fresh`` (the classifier by default) keep
      their new initialisation.

    Any other missing or mismatched backbone key raises, so a wrong
    architecture never trains silently from scratch.
    Returns ``{"copied", "adapted", "fresh", "unused"}`` key lists.
    """
    state = read_pointcept_state_dict(checkpoint, weights_only=weights_only)
    if not any(k.startswith("backbone.") for k in state):
        state = {"backbone." + k: v for k, v in state.items()}  # bare backbone checkpoint
    unsupported = [k for k in state if ".rpe." in k]
    if unsupported:
        raise ValueError("checkpoint uses RPE, which PurePTv3 does not implement")
    own = model.state_dict()
    new, report = {}, {"copied": [], "adapted": [], "fresh": [], "unused": []}
    problems = []
    for key, want in own.items():
        if any(key.startswith(p) for p in fresh):
            report["fresh"].append(key)
            continue
        if key not in state:
            problems.append(f"{key}: not in checkpoint")
            continue
        got = state[key]
        if got.shape == want.shape:
            new[key] = got.to(want.dtype)
            report["copied"].append(key)
        elif key == STEM_KEY and got.shape[:-1] == want.shape[:-1]:
            if in_channel_map is None or len(in_channel_map) != want.shape[-1]:
                raise ValueError(
                    f"stem input channels differ (checkpoint {got.shape[-1]}, model {want.shape[-1]}): "
                    f"pass in_channel_map with {want.shape[-1]} entries"
                )
            w = torch.zeros_like(want)
            for i, src in enumerate(in_channel_map):
                if src is not None:
                    if not 0 <= src < got.shape[-1]:
                        raise ValueError(f"in_channel_map[{i}] = {src} is not a checkpoint channel")
                    w[..., i] = got[..., src].to(want.dtype)
            new[key] = w
            report["adapted"].append(key)
        else:
            problems.append(f"{key}: checkpoint {tuple(got.shape)} vs model {tuple(want.shape)}")
    if problems:
        raise ValueError("checkpoint does not fit this architecture:\n  " + "\n  ".join(problems[:20]))
    report["unused"] = [k for k in state if k not in own or any(k.startswith(p) for p in fresh)]
    model.load_state_dict(new, strict=False)
    return report
