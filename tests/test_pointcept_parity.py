"""PurePTv3 vs the original Pointcept PT-v3m1, same weights, same input.

Needs spconv and a Pointcept checkout (``POINTCEPT_ROOT``, default
``../Pointcept`` next to this repo). flash_attn is not needed: the original
runs with ``enable_flash=False``, which for these inputs computes the same
patches as the flash path. torch_scatter, timm, addict, the Pointcept model
registry and PDNorm are stubbed if missing, so no Pointcept install is needed.

spconv's CPU kernels race under multi-threading (results change run to run),
so this test pins torch to one thread.

    python tests/test_pointcept_parity.py
"""

import importlib.util
import os
import sys
import types

import torch

from _common import ROOT, SkipTest, make_cloud, max_diff, randomize_, run_all

from pureptv3 import PurePTv3, PurePTv3Segmentor, load_pointcept_checkpoint

POINTCEPT_ROOT = os.environ.get(
    "POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept")
)
TOL = 1e-4  # relative to the output's max magnitude


# --------------------------------------------------------------------------- #
# Import Pointcept's PT-v3m1 file without importing all of pointcept
# --------------------------------------------------------------------------- #
def _module(name, path=None, **attrs):
    mod = types.ModuleType(name)
    if path is not None:
        mod.__path__ = [path]
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _segment_csr(src, indptr, reduce="sum"):
    counts = indptr[1:] - indptr[:-1]
    index = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
    index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = src.new_zeros((len(counts),) + tuple(src.shape[1:]))
    op = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}[reduce]
    return out.scatter_reduce(0, index, src, reduce=op, include_self=False)


class _DropPath(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()

    def forward(self, x):
        return x


def load_original():
    try:
        import spconv.pytorch  # noqa: F401
    except ImportError as e:
        raise SkipTest(f"spconv not installed ({e})")
    src = os.path.join(
        POINTCEPT_ROOT, "pointcept", "models", "point_transformer_v3",
        "point_transformer_v3m1_base.py",
    )
    if not os.path.exists(src):
        raise SkipTest(f"Pointcept not found at {POINTCEPT_ROOT}")
    if "pointcept_ptv3m1_original" in sys.modules:
        return sys.modules["pointcept_ptv3m1_original"]

    pc = os.path.join(POINTCEPT_ROOT, "pointcept")
    _module("pointcept", pc)
    _module("pointcept.models", os.path.join(pc, "models"))
    _module(
        "pointcept.models.builder",
        MODELS=types.SimpleNamespace(register_module=lambda *a, **k: (lambda c: c)),
    )
    _module("pointcept.models.point_prompt_training", PDNorm=None)
    _module("pointcept.engines", os.path.join(pc, "engines"))
    _module("pointcept.engines.hooks", HookBase=type("HookBase", (), {}))
    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        _module("torch_scatter", segment_csr=_segment_csr)
    try:
        import timm.layers  # noqa: F401
    except ImportError:
        _module("timm", os.devnull)
        _module("timm.layers", DropPath=_DropPath)
    try:
        import addict  # noqa: F401
    except ImportError as e:
        raise SkipTest(f"addict not installed ({e})")

    spec = importlib.util.spec_from_file_location("pointcept_ptv3m1_original", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
def _move_conv_bias_out(model):
    """spconv's CPU kernels refuse a bias; add it after the conv instead (same math)."""
    import spconv.pytorch as spconv

    for m in model.modules():
        if isinstance(m, spconv.SubMConv3d) and m.bias is not None:
            bias = m.bias.detach().clone()
            m.register_parameter("bias", None)
            m.register_forward_hook(
                lambda mod, inp, out, b=bias: out.replace_feature(out.features + b)
            )


def _pair(cfg, seed=0):
    """Original and native model with identical (perturbed) weights.

    Returns ``(original, native, original_state_dict)``.
    """
    orig = load_original().PointTransformerV3(
        **cfg, enable_flash=False, upcast_attention=False, upcast_softmax=False
    )
    randomize_(orig, seed)
    state = {k: v.clone() for k, v in orig.state_dict().items()}
    native = PurePTv3(**cfg)
    native.load_state_dict(state, strict=True)
    if not torch.cuda.is_available():
        _move_conv_bias_out(orig)
    return orig.eval(), native.eval(), state


def _batch(sizes, seed):
    clouds = [make_cloud(n, seed + i) for i, n in enumerate(sizes)]
    coord = torch.cat([c[0] for c in clouds])
    grid = torch.cat([c[1] for c in clouds])
    feat = torch.cat([c[2] for c in clouds])
    offset = torch.cumsum(torch.tensor([len(c[0]) for c in clouds]), 0)
    return coord, grid, feat, offset


def _set_pooling_shuffle(orig, flag):
    for m in orig.modules():
        if type(m).__name__ == "SerializedPooling":
            m.shuffle_orders = flag


def _run_original(orig, coord, grid, feat, offset, seed=None):
    data = dict(coord=coord, grid_coord=grid, feat=feat, offset=offset)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        return orig(data).feat


# --------------------------------------------------------------------------- #
def test_batched_four_orders_with_shuffle():
    """Two samples, 4 serialization orders, Pointcept's random order shuffle."""
    torch.set_num_threads(1)
    coord, grid, feat, offset = _batch([7000, 5000], seed=10)
    cfg = dict(
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        enc_patch_size=(48, 48, 48, 32, 16),
        dec_patch_size=(48, 48, 48, 32),
        shuffle_orders=True,
    )
    orig, native, _ = _pair(cfg, seed=1)
    _set_pooling_shuffle(orig, True)

    inputs = native.prepare(
        grid, offset=offset, shuffle_orders=True,
        generator=torch.Generator().manual_seed(1234),
    )
    # Pointcept's masked (non-flash) path shrinks K to the smallest sample, flash
    # does not; keep every sample >= K so both paths are the same computation.
    for s in range(5):
        assert inputs[f"s{s}_attn_bias"].isfinite().all(), f"stage {s} has a short sample"

    ref = _run_original(orig, coord, grid, feat, offset, seed=1234)
    with torch.no_grad():
        out = native(feat, inputs)
    d, rel = max_diff(out, ref)
    assert rel < TOL, f"max abs diff {d:.3e} (rel {rel:.3e})"
    sizes = [len(inputs[f"s{s}_kmap"]) for s in range(5)]
    return f"N per stage {sizes}, max abs diff {d:.2e}, rel {rel:.2e}"


def test_single_sample_large_patch_masked():
    """ScanNet-style patch 1024: deep stages are shorter than K (masked path)."""
    torch.set_num_threads(1)
    coord, grid, feat, offset = _batch([9000], seed=20)
    cfg = dict(
        in_channels=6,
        enc_patch_size=(1024,) * 5,
        dec_patch_size=(1024,) * 4,
        shuffle_orders=False,
    )
    orig, native, _ = _pair(cfg, seed=2)
    _set_pooling_shuffle(orig, False)
    inputs = native.prepare(grid, offset=offset)
    short = [s for s in range(5) if not inputs[f"s{s}_attn_bias"].isfinite().all()]
    assert short, "expected at least one stage with fewer than K points"

    ref = _run_original(orig, coord, grid, feat, offset)
    with torch.no_grad():
        out = native(feat, inputs)
    d, rel = max_diff(out, ref)
    assert rel < TOL, f"max abs diff {d:.3e} (rel {rel:.3e})"
    return f"masked stages {short}, max abs diff {d:.2e}, rel {rel:.2e}"


def test_decoder_cpe_quirk_is_needed():
    """Using skip+up for the first decoder CPE (the 'obvious' version) breaks parity."""
    torch.set_num_threads(1)
    coord, grid, feat, offset = _batch([6000], seed=30)
    cfg = dict(in_channels=6, enc_patch_size=(32,) * 5, dec_patch_size=(32,) * 4,
               shuffle_orders=False)
    orig, native, _ = _pair(cfg, seed=3)
    _set_pooling_shuffle(orig, False)
    inputs = native.prepare(grid, offset=offset)
    ref = _run_original(orig, coord, grid, feat, offset)
    with torch.no_grad():
        good = native(feat, inputs)
        native.dec_cpe_from_skip = False
        bad = native(feat, inputs)
    d_good, rel_good = max_diff(good, ref)
    d_bad, rel_bad = max_diff(bad, ref)
    assert rel_good < TOL, f"faithful version off by {d_good:.3e}"
    assert rel_bad > 100 * TOL, f"quirk had no effect ({d_bad:.3e})"
    return f"faithful rel {rel_good:.2e}, without quirk rel {rel_bad:.2e}"


def test_segmentor_checkpoint_roundtrip():
    """DDP-style DefaultSegmentorV2 checkpoint -> PurePTv3Segmentor."""
    torch.set_num_threads(1)
    coord, grid, feat, offset = _batch([5000], seed=40)
    backbone_cfg = dict(type="PT-v3m1", in_channels=6, enc_patch_size=(32,) * 5,
                        dec_patch_size=(32,) * 4, shuffle_orders=False)
    cfg = {k: v for k, v in backbone_cfg.items() if k != "type"}
    orig, _, state = _pair(cfg, seed=4)
    _set_pooling_shuffle(orig, False)
    head = torch.nn.Linear(64, 17)
    randomize_(head, 5)
    ckpt = {"epoch": 3, "state_dict": {}}
    for k, v in state.items():
        ckpt["state_dict"]["module.backbone." + k] = v
    for k, v in head.state_dict().items():
        ckpt["state_dict"]["module.seg_head." + k] = v

    seg = PurePTv3Segmentor.from_config(
        dict(type="DefaultSegmentorV2", num_classes=17, backbone_out_channels=64,
             backbone=backbone_cfg)
    ).eval()
    load_pointcept_checkpoint(seg, ckpt)
    with torch.no_grad():
        ref = head(_run_original(orig, coord, grid, feat, offset))
        out = seg(feat, seg.prepare(grid, offset=offset))
    d, rel = max_diff(out, ref)
    assert rel < TOL, f"max abs diff {d:.3e}"
    return f"logits max abs diff {d:.2e}"


if __name__ == "__main__":
    run_all(dict(globals()))
