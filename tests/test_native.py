"""PurePTv3 checks that need only torch (onnx/onnxruntime optional).

    python tests/test_native.py
"""

import os
import tempfile

import torch
import torch.nn.functional as F

from _common import SkipTest, make_cloud, max_diff, randomize_, run_all

from pureptv3 import PurePTv3, PurePTv3Segmentor, build_inputs, init_from_pretrained, kernel_map
from pureptv3.model import SerializedAttention, SerializedPooling, SubMConv3d
from pureptv3.preprocess import attention_padding

SMALL = dict(
    in_channels=6,
    enc_depths=(1, 1, 1, 1, 1),
    enc_channels=(16, 16, 32, 32, 64),
    enc_num_head=(2, 2, 4, 4, 8),
    enc_patch_size=(32, 32, 32, 16, 16),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(16, 16, 32, 32),
    dec_num_head=(2, 2, 4, 4),
    dec_patch_size=(32, 32, 32, 16),
)


def _small_model(**overrides):
    model = PurePTv3(**{**SMALL, **overrides})
    randomize_(model, 0)
    return model.eval()


# --------------------------------------------------------------------------- #
def test_subm_conv_equals_dense_conv3d():
    """Gather conv == torch conv3d on the occupied voxels (k = 3 and 5)."""
    g = torch.Generator().manual_seed(0)
    shape = 14
    grid = torch.unique(torch.randint(0, shape, (600, 3), generator=g), dim=0)
    batch = torch.zeros(len(grid), dtype=torch.long)
    for k in (3, 5):
        conv = SubMConv3d(5, 7, kernel_size=k, bias=True)
        feat = torch.randn(len(grid), 5, generator=g)
        out = conv(feat, kernel_map(grid, batch, k))

        dense = torch.zeros(1, 5, shape, shape, shape)
        dense[0, :, grid[:, 0], grid[:, 1], grid[:, 2]] = feat.T
        w = conv.weight.detach().permute(0, 4, 1, 2, 3)  # [Cout, Cin, kx, ky, kz]
        ref = F.conv3d(dense, w, conv.bias.detach(), padding=k // 2)
        ref = ref[0, :, grid[:, 0], grid[:, 1], grid[:, 2]].T
        d, _ = max_diff(out, ref)
        assert d < 1e-4, f"k={k}: {d}"


def test_kernel_map_keeps_samples_apart():
    grid = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]])
    batch = torch.tensor([0, 0, 1, 1])
    kmap = kernel_map(grid, batch, 3)
    centre = 13
    assert kmap[:, centre].tolist() == [0, 1, 2, 3]
    neighbours = {(i, int(j)) for i in range(4) for j in kmap[i] if j < 4 and j != i}
    assert neighbours == {(0, 1), (1, 0), (2, 3), (3, 2)}, neighbours


def _pointcept_padding(bincount, k):
    """Verbatim index logic of SerializedAttention.get_padding_and_inverse."""
    offset = torch.cumsum(bincount, 0)
    bincount_pad = torch.div(bincount + k - 1, k, rounding_mode="trunc") * k
    mask_pad = bincount > k
    bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
    _offset = F.pad(offset, (1, 0))
    _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
    pad = torch.arange(_offset_pad[-1])
    unpad = torch.arange(_offset[-1])
    for i in range(len(offset)):
        unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
            pad[
                _offset_pad[i + 1] - k + (bincount[i] % k) : _offset_pad[i + 1]
            ] = pad[
                _offset_pad[i + 1] - 2 * k + (bincount[i] % k) : _offset_pad[i + 1] - k
            ]
        pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
    return pad, unpad


def test_attention_padding_matches_pointcept():
    k = 48
    bincount = torch.tensor([100, 48, 97, 1000, 49])  # every sample >= K
    pad, unpad, bias = attention_padding(bincount, k)
    ref_pad, ref_unpad = _pointcept_padding(bincount, k)
    assert torch.equal(pad, ref_pad) and torch.equal(unpad, ref_unpad)
    assert bias.isfinite().all()

    # short samples become one masked patch of K slots
    pad, unpad, bias = attention_padding(torch.tensor([10, 60]), 16)
    assert bias.shape == (1 + 4, 1, 1, 16)
    assert torch.isinf(bias[0, 0, 0, 10:]).all() and (bias[0, 0, 0, :10] == 0).all()
    assert torch.equal(pad[unpad], torch.arange(70))


def _reference_attention(attn, feat, order, bincount):
    """Per-sample loop with the patch rule written out; ``order`` = serialized order."""
    h, k, c = attn.num_heads, attn.patch_size, attn.channels
    qkv = attn.qkv(feat)
    out = torch.empty(feat.shape[0], c)
    start = 0
    for n in bincount.tolist():
        idx = order[start : start + n]
        patches = []
        if n <= k:
            patches.append((idx, idx))
        else:
            for p in range((n + k - 1) // k):
                lo, hi = p * k, min((p + 1) * k, n)
                keys = idx[lo:hi] if hi - lo == k else idx[n - k :]
                patches.append((idx[lo:hi], keys))
        for queries, keys in patches:
            q = qkv[queries, :c].view(-1, h, c // h).transpose(0, 1)
            kk = qkv[keys, c : 2 * c].view(-1, h, c // h).transpose(0, 1)
            v = qkv[keys, 2 * c :].view(-1, h, c // h).transpose(0, 1)
            w = torch.softmax(q @ kk.transpose(1, 2) * attn.scale, -1)
            out[queries] = (w @ v).transpose(0, 1).reshape(-1, c)
        start += n
    return attn.proj(out)


def test_attention_matches_patch_definition():
    g = torch.Generator().manual_seed(1)
    attn = SerializedAttention(32, 4, patch_size=16).eval()
    bincount = torch.tensor([50, 7, 64, 33])
    n = int(bincount.sum())
    feat = torch.randn(n, 32, generator=g)
    serial = torch.cat(
        [torch.randperm(int(b), generator=g) + int(o) for b, o in
         zip(bincount, F.pad(torch.cumsum(bincount, 0), (1, 0))[:-1])]
    )
    inverse_serial = torch.empty_like(serial)
    inverse_serial[serial] = torch.arange(n)
    pad, unpad, bias = attention_padding(bincount, 16)
    with torch.no_grad():
        out = attn(feat, serial[pad], unpad[inverse_serial], bias)
        ref = _reference_attention(attn, feat, serial, bincount)
    d, _ = max_diff(out, ref)
    assert d < 1e-5, d


def test_pooling_equals_scatter_max():
    coord, grid, feat = make_cloud(3000, seed=2)
    inputs = build_inputs(grid, order=("z",), stride=(2,), patch_size=(32, 32))
    children, parent = inputs["s1_pool"], inputs["s1_unpool"]
    pool = SerializedPooling(6, 8, norm_layer=None, act_layer=None)
    with torch.no_grad():
        x = pool.proj(feat)
        ref = torch.full((len(children), 8), float("-inf")).scatter_reduce(
            0, parent[:, None].expand(-1, 8), x, reduce="amax"
        )
        out = pool(feat, children)
    assert torch.equal(out, ref)
    assert children.shape[1] == 8


def test_storage_order_does_not_matter():
    """Shuffling the input points permutes the output the same way."""
    coord, grid, feat = make_cloud(6000, seed=3)
    model = _small_model()
    perm = torch.randperm(len(grid), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(feat, model.prepare(grid))
        out_p = model(feat[perm], model.prepare(grid[perm]))
    d, rel = max_diff(out_p, out[perm])
    assert rel < 1e-5, (d, rel)


def test_batched_equals_separate():
    """Two clouds in one batch give the same result as one at a time."""
    a, b = make_cloud(4000, seed=4), make_cloud(2500, seed=5)
    model = _small_model()
    grid = torch.cat([a[1], b[1]])
    feat = torch.cat([a[2], b[2]])
    offset = torch.tensor([len(a[1]), len(a[1]) + len(b[1])])
    with torch.no_grad():
        both = model(feat, model.prepare(grid, offset=offset))
        sep = torch.cat([model(c[2], model.prepare(c[1])) for c in (a, b)])
    d, rel = max_diff(both, sep)
    assert rel < 1e-5, (d, rel)


def test_int32_indices():
    coord, grid, feat = make_cloud(3000, seed=6)
    model = _small_model()
    with torch.no_grad():
        ref = model(feat, model.prepare(grid))
        out = model(feat, model.prepare(grid, index_dtype=torch.int32))
    assert torch.equal(out, ref)


def test_backward_runs():
    coord, grid, feat = make_cloud(2000, seed=7)
    model = PurePTv3(**SMALL).train()
    inputs = model.prepare(grid)
    out = model(feat, inputs)
    out.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g_ is not None and torch.isfinite(g_).all() for g_ in grads)


def test_fast_backward_gradcheck():
    """float64 gradcheck of the custom conv / gather backward."""
    from pureptv3.model import _GatherRows, _SubMConvFunction

    g = torch.Generator().manual_seed(10)
    grid = torch.unique(torch.randint(0, 6, (60, 3), generator=g), dim=0)
    batch = (torch.arange(len(grid)) % 2).long()  # two samples on one grid
    dt = torch.float64
    for k in (3, 5):
        kmap = kernel_map(grid, batch, k)
        feat = torch.randn(len(grid), 3, dtype=dt, generator=g, requires_grad=True)
        w = torch.randn(4, k, k, k, 3, dtype=dt, generator=g, requires_grad=True)
        b = torch.randn(4, dtype=dt, generator=g, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda f, w_, b_: _SubMConvFunction.apply(f, w_, b_, kmap, dt), (feat, w, b)
        )
    x = torch.randn(9, 2, 3, dtype=dt, generator=g, requires_grad=True)
    idx = torch.randint(0, 9, (5, 4), generator=g)  # repeats on purpose
    assert torch.autograd.gradcheck(lambda x_: _GatherRows.apply(x_, idx), (x,))


def test_fast_backward_matches_autograd():
    """Whole-model gradients with the custom backward == plain autograd."""
    import pureptv3.model as M

    _, grid, feat = make_cloud(3000, seed=11)
    model = _small_model(drop_path=0.0)  # eval: BN running stats, no shuffle
    inputs = model.prepare(grid)
    grads = []
    for fast in (True, False):
        M.FAST_BACKWARD = fast
        try:
            model.zero_grad(set_to_none=True)
            f = feat.clone().requires_grad_(True)
            out = model(f, inputs)
            (out.square().mean() + out[:, 0].sum() * 1e-3).backward()
        finally:
            M.FAST_BACKWARD = True
        grads.append([f.grad] + [p.grad for p in model.parameters()])
    worst = max(max_diff(a, b)[1] for a, b in zip(*grads))
    assert worst < 1e-5, worst
    return f"worst rel {worst:.1e} over {len(grads[0])} tensors"


def test_onnx_roundtrip_dynamic_size():
    try:
        import onnxruntime as ort
    except ImportError:
        raise SkipTest("onnxruntime not installed")
    from pureptv3.export import export_onnx, onnx_feed

    try:
        import onnxscript  # noqa: F401

        exporters = (True, False)
    except ImportError:
        exporters = (False,)

    model = _small_model()
    _, grid, feat = make_cloud(4000, seed=8)
    _, grid2, feat2 = make_cloud(9000, seed=9)  # different N at every stage
    msgs = []
    for dynamo in exporters:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pureptv3.onnx")
            names = export_onnx(model, feat, model.prepare(grid), path, dynamo=dynamo)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            for g_, f_ in ((grid, feat), (grid2, feat2)):
                inputs = model.prepare(g_)
                (onnx_out,) = sess.run(None, onnx_feed(f_, inputs, names))
                with torch.no_grad():
                    ref = model(f_, inputs)
                d, rel = max_diff(torch.from_numpy(onnx_out), ref)
                assert rel < 1e-4, (dynamo, d, rel)
                msgs.append(f"dynamo={dynamo} N={len(g_)} rel {rel:.1e}")
            sess = None
            ops = sorted({n.op_type for n in __import__("onnx").load(path).graph.node})
            msgs.append(f"ops: {' '.join(ops)}")
    return "; ".join(msgs)


def test_init_from_pretrained_adapts_stem_and_head():
    """Fine-tuning init: same-shape tensors copied, stem input channels remapped, head fresh, wrong arch refused."""
    src = PurePTv3Segmentor(num_classes=20, backbone_out_channels=16, backbone=dict(SMALL, in_channels=6))
    randomize_(src, 1)
    ck = {"state_dict": {"module." + k: v for k, v in src.state_dict().items()}}  # DDP keys, like Pointcept
    dst = PurePTv3Segmentor(num_classes=17, backbone_out_channels=16, backbone=dict(SMALL, in_channels=10))
    head = dst.seg_head.weight.detach().clone()
    cmap = [None, None, None, 3, 4, 5, None, 0, 1, 2]  # xyz normal jaw colour <- color normal
    rep = init_from_pretrained(dst, ck, in_channel_map=cmap)
    assert rep["adapted"] == ["backbone.embedding.stem.conv.weight"], rep["adapted"]
    assert rep["fresh"] == ["seg_head.weight", "seg_head.bias"], rep["fresh"]
    s, d = src.state_dict(), dst.state_dict()
    for k in rep["copied"]:
        assert torch.equal(s[k], d[k]), k
    w_src, w_dst = s["backbone.embedding.stem.conv.weight"], d["backbone.embedding.stem.conv.weight"]
    for i, j in enumerate(cmap):
        want = torch.zeros_like(w_dst[..., i]) if j is None else w_src[..., j]
        assert torch.equal(w_dst[..., i], want), i
    assert torch.equal(dst.seg_head.weight, head)
    # a feature the checkpoint never saw starts with no influence: changing it does not change the output
    dst.eval()
    _, grid, _ = make_cloud(800, 0)
    inputs = dst.prepare(grid)
    gen = torch.Generator().manual_seed(0)
    feat = torch.randn(grid.shape[0], 10, generator=gen)
    feat2 = feat.clone()
    feat2[:, [0, 1, 2, 6]] = torch.randn(grid.shape[0], 4, generator=gen)
    with torch.no_grad():
        assert max_diff(dst(feat, inputs), dst(feat2, inputs))[0] == 0
    bad = PurePTv3Segmentor(num_classes=17, backbone_out_channels=16,
                            backbone=dict(SMALL, in_channels=10, enc_channels=(16, 16, 32, 32, 48)))
    try:
        init_from_pretrained(bad, ck, in_channel_map=cmap)
    except ValueError as e:
        assert "does not fit" in str(e)
    else:
        raise AssertionError("a different architecture must be refused")


if __name__ == "__main__":
    run_all(dict(globals()))
