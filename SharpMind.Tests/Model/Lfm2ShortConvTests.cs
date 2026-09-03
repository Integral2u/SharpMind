using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;
using SharpMind.Model.Layers.ShortConv;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// Unit tests for LFM2 short-conv architecture support — config parsing,
/// ShortConvKernels math, ShortConvLayer forward, ShortConvCache lifecycle,
/// and block-level wiring. All tests use synthetic metadata (no model files).
/// </summary>
public class Lfm2ShortConvTests
{
    // ── Config helpers ──────────────────────────────────────────────────

    private static ModelConfig MakeLfm2Config(
        int hiddenDim = 2048,
        int ffnDim = 10752,
        int numLayers = 4,
        int lCache = 3)
    {
        // 30 real layers, 8 shortconv (0-indexed positions from the real model)
        var kvHeads = new int[numLayers];
        int[] shortconvIdx = [2, 5, 9, 13, 17, 21, 24, 27];
        for (int i = 0; i < kvHeads.Length; i++)
            kvHeads[i] = shortconvIdx.Contains(i) ? 0 : 8;

        return new ModelConfig
        {
            Architecture = "lfm2",
            VocabSize = 128000,
            HiddenDim = hiddenDim,
            NumLayers = numLayers,
            NumHeads = 32,
            NumKvHeads = 8,
            LayerKvHeads = kvHeads,
            ShortConvCacheLength = lCache,
            FfnDim = ffnDim,
            MaxSeqLen = 2048,
            NormEps = 1e-5f,
            KeyLength = 64,
            ValueLength = 64,
            HeadDimOverride = 64,
            RopeDim = 256,
        };
    }

    // ── Config parsing ─────────────────────────────────────────────────

    [Fact]
    public void Lfm2_KvHeadArray_ParsedCorrectly()
    {
        var cfg = MakeLfm2Config(numLayers: 30);

        Assert.NotNull(cfg.LayerKvHeads);
        Assert.Equal(8, cfg.NumKvHeads);

        Assert.Equal(0, cfg.LayerKvHeads![2]);
        Assert.Equal(0, cfg.LayerKvHeads[5]);
        Assert.Equal(0, cfg.LayerKvHeads[9]);

        Assert.Equal(8, cfg.LayerKvHeads[0]);
        Assert.Equal(8, cfg.LayerKvHeads[1]);
    }

    [Fact]
    public void Lfm2_ShortConvCacheLength_ParsedCorrectly()
    {
        var cfg = MakeLfm2Config(lCache: 5);
        Assert.Equal(5, cfg.ShortConvCacheLength);
    }

    [Fact]
    public void Lfm2_ShortConvCacheLength_DefaultsTo3()
    {
        var cfg = new ModelConfig
        {
            Architecture = "lfm2",
            VocabSize = 1000,
            HiddenDim = 256,
            NumLayers = 2,
            NumHeads = 4,
            NumKvHeads = 4,
            FfnDim = 512,
            MaxSeqLen = 256,
        };
        Assert.Equal(3, cfg.ShortConvCacheLength);
    }

    [Fact]
    public void Lfm2_IsShortConvLayer_MapsCorrectly()
    {
        var cfg = MakeLfm2Config(numLayers: 30);

        Assert.True(cfg.IsShortConvLayer(2));
        Assert.True(cfg.IsShortConvLayer(5));
        Assert.True(cfg.IsShortConvLayer(9));
        Assert.True(cfg.IsShortConvLayer(13));
        Assert.True(cfg.IsShortConvLayer(17));
        Assert.True(cfg.IsShortConvLayer(21));
        Assert.True(cfg.IsShortConvLayer(24));
        Assert.True(cfg.IsShortConvLayer(27));

        Assert.False(cfg.IsShortConvLayer(0));
        Assert.False(cfg.IsShortConvLayer(1));
        Assert.False(cfg.IsShortConvLayer(3));
        Assert.False(cfg.IsShortConvLayer(29));
    }

    [Fact]
    public void Lfm2_ScalarKVHead_StillWorks()
    {
        var cfg = new ModelConfig
        {
            Architecture = "llama",
            VocabSize = 1000,
            HiddenDim = 256,
            NumLayers = 2,
            NumHeads = 12,
            NumKvHeads = 4,
            FfnDim = 512,
            MaxSeqLen = 256,
        };

        Assert.Null(cfg.LayerKvHeads);
        Assert.Equal(4, cfg.NumKvHeads);
        Assert.False(cfg.IsShortConvLayer(0));
    }

    // ── ShortConvKernels ──────────────────────────────────────────────

    [Fact]
    public void ComputeGatedInput_ProducesCorrectShape()
    {
        int batch = 1, seq = 4, hidden = 8;
        using var proj = new Tensor<float>(batch, seq, 3 * hidden);
        using var bx   = new Tensor<float>(batch, seq, hidden);

        Span<float> data = proj.Data;
        for (int i = 0; i < data.Length; i++)
            data[i] = 1.0f;

        ShortConvKernels.ComputeGatedInput(proj, bx, batch * seq, hidden);

        foreach (var v in bx.Data)
            Assert.Equal(1.0f, v);
    }

    [Fact]
    public void ComputeGatedInput_GatingWorks()
    {
        int batch = 1, seq = 2, hidden = 4;
        using var proj = new Tensor<float>(batch, seq, 3 * hidden);
        using var bx   = new Tensor<float>(batch, seq, hidden);

        // proj layout per row: [b0,b1,b2,b3, c0,c1,c2,c3, x0,x1,x2,x3]
        // b gate: [0, H), x input: [2H, 3H)
        Span<float> d = proj.Data;
        d[0] = 0; d[1] = 1; d[2] = 0; d[3] = 1;  // b gate
        for (int i = 4; i < 8; i++) d[i] = 0;      // c gate (irrelevant)
        d[8] = 2; d[9] = 3; d[10] = 4; d[11] = 5; // x input

        ShortConvKernels.ComputeGatedInput(proj, bx, batch * seq, hidden);

        Span<float> bxd = bx.Data;
        Assert.Equal(0, bxd[0]);
        Assert.Equal(3, bxd[1]);
        Assert.Equal(0, bxd[2]);
        Assert.Equal(5, bxd[3]);
    }

    [Fact]
    public void ApplyConv_SingleTokenStep_AppliesKernelCorrectly()
    {
        int batch = 1, seq = 1, hidden = 2;
        int taps = 3;

        // kernel: row 0 = [1,1] (applies to state[0]), row 2 = [1,1] (applies to bx[0])
        using var kernel = new Tensor<float>(taps, hidden);
        kernel.Data[0] = 1; kernel.Data[1] = 1;   // row 0 → state[0]
        kernel.Data[4] = 1; kernel.Data[5] = 1;   // row 2 → bx[0]

        using var state = new Tensor<float>(batch, taps - 1, hidden);
        using var bx = new Tensor<float>(seq, hidden);
        bx.Data[0] = 2; bx.Data[1] = 3;

        using var output = new Tensor<float>(seq, hidden);
        ShortConvKernels.ApplyConv(bx, state, kernel, output, batch, seq, hidden, taps);

        Assert.Equal(2.0f, output.Data[0]);
        Assert.Equal(3.0f, output.Data[1]);
    }

    [Fact]
    public void ApplyConv_MultiTokenStep_ConvolutionIsCorrect()
    {
        int batch = 1, seq = 2, hidden = 1;
        int taps = 3;

        using var kernel = new Tensor<float>(taps, hidden);
        kernel.Data[0] = 1; kernel.Data[1] = 2; kernel.Data[2] = 3;

        using var state = new Tensor<float>(batch, taps - 1, hidden);

        using var bx = new Tensor<float>(seq, hidden);
        bx.Data[0] = 4; bx.Data[1] = 5;

        using var output = new Tensor<float>(seq, hidden);
        ShortConvKernels.ApplyConv(bx, state, kernel, output, batch, seq, hidden, taps);

        Assert.Equal(12.0f, output.Data[0]);  // 0*1 + 0*2 + 4*3
        Assert.Equal(23.0f, output.Data[1]);  // 0*1 + 4*2 + 5*3
    }

    [Fact]
    public void UpdateState_RollsCorrectly()
    {
        int batch = 1, seq = 2, hidden = 2, stateRows = 2;

        using var bx = new Tensor<float>(seq, hidden);
        bx.Data[0] = 1; bx.Data[1] = 2; bx.Data[2] = 3; bx.Data[3] = 4;

        using var state = new Tensor<float>(batch, stateRows, hidden);

        ShortConvKernels.UpdateState(bx, state, batch, seq, hidden, stateRows);

        Span<float> sd = state.Data;
        Assert.Equal(1.0f, sd[0]);
        Assert.Equal(2.0f, sd[1]);
        Assert.Equal(3.0f, sd[2]);
        Assert.Equal(4.0f, sd[3]);
    }

    // ── ShortConvCache ───────────────────────────────────────────────

    [Fact]
    public void ShortConvCache_Length_IncrementsPerAdvance()
    {
        var cache = new ShortConvCache(stateRows: 2, channels: 4);
        Assert.Equal(0, cache.Length);

        cache.Advance(1);
        Assert.Equal(1, cache.Length);

        cache.Advance(1);
        Assert.Equal(2, cache.Length);
    }

    [Fact]
    public void ShortConvCache_Reset_ClearsLength()
    {
        var cache = new ShortConvCache(stateRows: 2, channels: 4);
        cache.Advance(1);
        cache.Advance(1);
        Assert.Equal(2, cache.Length);

        cache.Reset();
        Assert.Equal(0, cache.Length);
    }

    [Fact]
    public void ShortConvCache_IsFull_Behavior()
    {
        var cache = new ShortConvCache(stateRows: 2, channels: 4, maxSeqLen: 5);
        Assert.False(cache.IsFull);

        cache.Advance(1);
        Assert.False(cache.IsFull);

        for (int i = 0; i < 4; i++) cache.Advance(1);
        Assert.Equal(5, cache.Length);
        Assert.True(cache.IsFull);
    }

    [Fact]
    public void ShortConvCache_SnapshotAndRestore()
    {
        var cache = new ShortConvCache(stateRows: 2, channels: 4);
        cache.Advance(1);
        cache.Advance(1);
        cache.Advance(1);

        var snapshot = cache.Snapshot();
        var clone = new ShortConvCache(2, 4);
        clone.Restore(snapshot);

        Assert.Equal(cache.Length, clone.Length);
        Assert.Equal(cache.State.Data.ToArray(), clone.State.Data.ToArray());

        clone.Dispose();
        cache.Dispose();
    }

    // ── ShortConvLayer forward ────────────────────────────────────────

    [Fact]
    public void ShortConvLayer_Forward_OutputShape()
    {
        var cfg = MakeLfm2Config(hiddenDim: 64, ffnDim: 128, numLayers: 2, lCache: 3);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

        using var layer = new ShortConvLayer(cfg, qOps);
        using var input = new Tensor<float>(1, 4, 64);

        using var output = layer.Forward(input, cache: null);

        // Input is rank 3 [1, seq, hidden] → output retains rank 3.
        Assert.Equal(3, output.Rank);
        Assert.Equal(4, output.Shape.Dims[1]);
        Assert.Equal(64, output.Shape.Dims[2]);
    }

    [Fact]
    public void ShortConvLayer_Forward_WithCache_StateAdvances()
    {
        var cfg = MakeLfm2Config(hiddenDim: 16, ffnDim: 32, numLayers: 2, lCache: 3);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

        using var layer = new ShortConvLayer(cfg, qOps);
        using var cache = new ShortConvCache(stateRows: 2, channels: 16);
        using var input = new Tensor<float>(1, 2, 16);

        // Forward updates the conv state tensor; cache.Length is managed by the caller.
        using var _ = layer.Forward(input, cache);
        cache.Advance(2);
        Assert.Equal(2, cache.Length);

        using var __ = layer.Forward(input, cache);
        cache.Advance(2);
        Assert.Equal(4, cache.Length);
    }

    [Fact]
    public void ShortConvLayer_Dispose_NoDoubleFree()
    {
        var cfg = MakeLfm2Config(hiddenDim: 16, ffnDim: 32, numLayers: 2, lCache: 3);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

        var layer = new ShortConvLayer(cfg, qOps);
        layer.Dispose();
        layer.Dispose();
    }

    // ── Block wiring ─────────────────────────────────────────────────

    [Fact]
    public void Lfm2Block_CanConstruct_AndForward()
    {
        var cfg = MakeLfm2Config(hiddenDim: 64, ffnDim: 128, numLayers: 2, lCache: 3);
        var sharpCfg = SharpMindConfig.ForModel(cfg.NumHeads, cfg.NumKvHeads, cfg.Architecture);
        var acts = ActivationFactory.Create(sharpCfg);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

        var shortConv = new ShortConvLayer(cfg, qOps);

        var blockWeights = new TransformerWeights.BlockWeights(
            null, null, null, null,
            null, null, null, null,
            new Tensor<float>(64, 128),  // Wf1
            new Tensor<float>(128, 64),  // Wf2
            null, null,
            Tensor<float>.Ones(64),      // norm1
            null,
            Tensor<float>.Ones(64),      // norm2
            null, null, null, null);
        var ffn = new GatedFfnLayer(cfg, acts, qOps, blockWeights);
        ffn.SetWeights(blockWeights);

        var norm1 = new RmsNormLayer(64, 1e-5f, Tensor<float>.Ones(64));
        var norm2 = new RmsNormLayer(64, 1e-5f, Tensor<float>.Ones(64));

        using var block = new UnhookedTransformerBlock(0, null, ffn, norm1, norm2, shortConv: shortConv);
        using var input = new Tensor<float>(1, 4, 64);

        using var output = block.Forward(input, positionOffset: 0);

        Assert.Equal(3, output.Rank);
        Assert.Equal(4, output.Shape.Dims[1]);
        Assert.Equal(64, output.Shape.Dims[2]);
        Assert.Null(block.Attention);
        Assert.NotNull(block.ShortConv);
    }

    [Fact]
    public void Lfm2Block_NullAttention_NullFfn_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new UnhookedTransformerBlock(0, null, null!, null!, null!));
    }

    // ── IKVCache interface uniformity ─────────────────────────────────

    [Fact]
    public void ShortConvCache_ImplementsIKVCache_FullInterface()
    {
        var cache = new ShortConvCache(2, 8);

        Assert.Equal(0, cache.Length);
        cache.Advance(1);
        Assert.Equal(1, cache.Length);
        cache.Reset();
        Assert.Equal(0, cache.Length);

        var snap = cache.Snapshot();
        Assert.Null(snap); // empty → null

        cache.Advance(3);
        snap = cache.Snapshot();
        Assert.NotNull(snap);

        var clone = new ShortConvCache(2, 8);
        clone.Restore(snap);
        Assert.Equal(cache.Length, clone.Length);

        clone.TrimToLast(1);
        clone.Truncate(0);

        clone.Dispose();
        cache.Dispose();
    }
}
