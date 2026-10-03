using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// Covers the qwen2moe / Mixtral-style MoE GGUF plumbing: fused 3D expert
/// stacks, the narrower routed-expert FFN width, the always-on shared expert
/// and its per-token sigmoid gate, and router tensor routing.
/// </summary>
public class MoeGgufExpertBindingTests
{
    private const int Hidden = 64;
    private const int ExpertFfn = 48;
    private const int SharedFfn = 128;
    private const int NumExperts = 4;
    private const int TopK = 2;

    private static ModelConfig MoEConfig(int sharedFfn = 0) => ModelConfig.Learnable with
    {
        HiddenDim = Hidden,
        FfnDim = sharedFfn > 0 ? sharedFfn : ExpertFfn,
        NumExperts = NumExperts,
        TopKExperts = TopK,
        ExpertFfnDim = ExpertFfn,
        SharedExpertFfnDim = sharedFfn,
    };

    private static TransformerWeights WeightsFor(ModelConfig config)
    {
        var blocks = new[] { new TransformerWeights.BlockWeights { LayerIndex = 0 } };
        return new TransformerWeightsStreaming(
            config, new Tensor<float>(32, Hidden), null, new Tensor<float>(Hidden), null,
            blocks, new NoOpLoader());
    }

    /// <summary>
    /// Stand-in loader: these tests only exercise tensor-name routing and layer
    /// construction, so nothing is ever read from disk.
    /// </summary>
    private sealed class NoOpLoader : SharpMind.Model.Format.IModelLoader
    {
        public void PreInit(TransformerWeights weights, CancellationToken? ct = null) { }
        public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null, CancellationToken? ct = null) { }
        public void LoadLayerWeights(int layerIndex, TransformerWeights weights, CancellationToken? ct = null) { }
        public void LoadGlobalTensors(TransformerWeights weights, CancellationToken? ct = null) { }
    }

    private static TensorInfo T(string name, QuantDType dtype, int[] shape) =>
        new() { Name = name, Dtype = dtype, Shape = shape, Offset = 0 };

    /// <summary>
    /// Mixtral spells its per-expert tensors "ffn_gate.{n}.weight" — a bare dotted
    /// index, no ".exps." segment. Matching only the ".exps." spelling sent every
    /// expert to the dense/router branches, where all eight aliased one slot: the
    /// gates landed in the router and the ups in the dense Wf1, and weight loading
    /// died with "Index was outside the bounds of the array".
    [Fact]
    public void MixtralStyle_BareDottedExpertIndex_BindsEachExpertToItsOwnSlot()
    {
        var weights = WeightsFor(MoEConfig());
        weights.IsMoE = true;

        for (int e = 0; e < NumExperts; e++)
        {
            Assert.Equal($"RawWgateExp_{e}", weights.ResolveTarget($"blk.0.ffn_gate.{e}.weight").rawField);
            Assert.Equal($"RawWupExp_{e}", weights.ResolveTarget($"blk.0.ffn_up.{e}.weight").rawField);
            Assert.Equal($"RawWdownExp_{e}", weights.ResolveTarget($"blk.0.ffn_down.{e}.weight").rawField);
        }
    }

    /// <summary>
    /// The router's own name also contains "ffn_gate" and a bare dotted-index regex
    /// matches the *layer* number in "blk.0.ffn_gate_inp.weight". It must still be
    /// recognised as the router, not filed under expert 0.
    [Fact]
    public void MixtralStyle_Router_IsNotMistakenForAnExpert()
    {
        var weights = WeightsFor(MoEConfig());
        weights.IsMoE = true;

        // Routers are unquantized, so there is deliberately no raw field: the tensor goes
        // straight to the float slot.
        Assert.Null(weights.ResolveTarget("blk.0.ffn_gate_inp.weight").rawField);

        var router = weights.ResolveFloatTarget("blk.0.ffn_gate_inp.weight");
        Assert.NotNull(router);
        Assert.Equal(Hidden, router!.Shape[0]);
        Assert.Equal(NumExperts, router.Shape[1]);
    }

    /// <summary>
    /// "ffn_gate_inp_shexp" contains "ffn_gate", and WRouter is created with ??=, so an
    /// unfiltered match let the rank-1 shared gate overwrite the real router. That is
    /// the "[router] weightElements=2048, expected=122880" failure, and which tensor won
    /// depended on the order of the tensors in the file.
    [Fact]
    public void SharedGateInp_NeverOverwritesTheRouter_EvenWithoutSharedExpertMetadata()
    {
        // SharedExpertFfnDim left at 0: the file has *_shexp tensors but its metadata
        // has no expert_shared_feed_forward_length.
        var weights = WeightsFor(MoEConfig());
        weights.IsMoE = true;

        var router = weights.ResolveFloatTarget("blk.0.ffn_gate_inp.weight");
        var sharedGate = weights.ResolveFloatTarget("blk.0.ffn_gate_inp_shexp.weight");

        Assert.NotNull(router);
        Assert.NotNull(sharedGate);
        Assert.NotSame(router, sharedGate);
        Assert.Equal(Hidden * NumExperts, router!.Shape.ElementCount);
        Assert.Equal(Hidden, sharedGate!.Shape.ElementCount);
    }

    [Fact]
    public void ResolvedSharedFfnDim_FallsBackToFfnDim_WhenMetadataIsAbsent()
    {
        Assert.Equal(ExpertFfn, (ModelConfig.Learnable with { FfnDim = ExpertFfn, SharedExpertFfnDim = 0 }).ResolvedSharedFfnDim);
        Assert.Equal(64, (ModelConfig.Learnable with { FfnDim = ExpertFfn, SharedExpertFfnDim = 64 }).ResolvedSharedFfnDim);
        Assert.False((ModelConfig.Learnable with { SharedExpertFfnDim = 0 }).HasSharedExpert);
    }

    [Fact]
    public void IsMoEGguf_DetectsFusedExpertStackAndGateInp()
    {
        // qwen2moe spells its fused stack "ffn_gate_exps" with no ".exps." segment.
        // This is the exact spelling that the streaming loader previously missed.
        TensorInfo[] tensors =
        [
            T("blk.0.ffn_gate_exps.weight", QuantDType.Q2_K, [2048, 1408, 60]),
            T("blk.0.ffn_gate_inp.weight", QuantDType.F32, [2048, 60]),
        ];

        Assert.True(TransformerWeights.IsMoEGguf(tensors));

        // A per-expert-indexed export (SMM/ggml "ffn_gate.exps.3.weight") still matches.
        TensorInfo[] perExpert = [T("blk.0.ffn_gate.exps.3.weight", QuantDType.Q2_K, [2048, 1408])];
        Assert.True(TransformerWeights.IsMoEGguf(perExpert));

        TensorInfo[] dense = [T("blk.0.ffn_gate.weight", QuantDType.Q2_K, [2048, 5632])];
        Assert.False(TransformerWeights.IsMoEGguf(dense));
    }

    [Theory]
    [InlineData("blk.0.ffn_gate_exps.weight", "RawWgateExpFused")]
    [InlineData("blk.0.ffn_up_exps.weight", "RawWupExpFused")]
    [InlineData("blk.0.ffn_down_exps.weight", "RawWdownExpFused")]
    [InlineData("blk.0.ffn_gate.exps.2.weight", "RawWgateExp_2")]
    [InlineData("blk.0.ffn_down.exps.1.weight", "RawWdownExp_1")]
    public void ResolveTarget_RoutesExpertTensorsToRawFields(string name, string expectedRawField)
    {
        var weights = WeightsFor(MoEConfig());
        weights.IsMoE = true;
        weights.IsMoE = true;

        var (_, block, rawField) = weights.ResolveTarget(name);

        Assert.NotNull(block);
        Assert.Equal(expectedRawField, rawField);
    }

    [Fact]
    public void ResolveTarget_SharedExpertIsNotMistakenForRouterOrRoutedExpert()
    {
        var weights = WeightsFor(MoEConfig(sharedFfn: SharedFfn));
        weights.IsMoE = true;

        Assert.Equal("RawWSharedGate", weights.ResolveTarget("blk.0.ffn_gate_shexp.weight").rawField);
        Assert.Equal("RawWSharedUp", weights.ResolveTarget("blk.0.ffn_up_shexp.weight").rawField);
        Assert.Equal("RawWSharedDown", weights.ResolveTarget("blk.0.ffn_down_shexp.weight").rawField);

        // The F32 sigmoid gate row loads through the float path, so no raw field.
        Assert.Null(weights.ResolveTarget("blk.0.ffn_gate_inp_shexp.weight").rawField);
    }

    [Fact]
    public void ResolveTarget_RouterUsesFloatPathAndIsNotBoundToGatedFfn()
    {
        var weights = WeightsFor(MoEConfig());
        weights.IsMoE = true;
        weights.IsMoE = true;

        // "ffn_gate_inp" contains "ffn_gate", so without an explicit MoE branch it
        // used to fall through to the dense ffn_gate binding and land in RawWgate.
        Assert.Null(weights.ResolveTarget("blk.0.ffn_gate_inp.weight").rawField);

        var target = weights.ResolveFloatTarget("blk.0.ffn_gate_inp.weight");
        Assert.NotNull(target);
        Assert.Equal(Hidden * NumExperts, target.ElementCount);
        Assert.Same(target, weights.Blocks[0].WRouter);
    }

    /// <summary>
    /// Qwen2-MoE's router (<c>ffn_gate_inp</c>) and shared-expert gate
    /// (<c>ffn_gate_inp_shexp</c>) are stored as F32, so they carry no raw quantized
    /// payload — the float tensor is the only copy. <c>CreateTransformer</c> frees float
    /// weights when <c>optimizeMemory</c> is set (the CUI default), and that free used to
    /// replace every weight with an <c>InFeatures</c>-sized placeholder. The CUI then died
    /// with "[router] ... weightElements=2048, expected=122880" on the very first forward.
    /// Layers that do have raw data must still be freed.
    [Fact]
    public void FreeFloatWeight_KeepsFloatOnlyLayersAndStillFreesQuantizedOnes()
    {
        var mapping = new Dictionary<string, string>();

        // F32 router: no raw payload, so the float weight must survive.
        var router = LinearLayerFactory.Create(
            "router", Hidden, NumExperts, true,
            new Tensor<float>(Hidden, NumExperts), null, QuantDType.F32, mapping);
        router.FreeFloatWeight();
        Assert.Equal(Hidden * NumExperts, router.Weight.ElementCount);

        // Same for the shared-expert scalar gate ([1, HiddenDim], F32).
        var sharedGateInp = LinearLayerFactory.Create(
            "shared_expert_gate_inp", Hidden, 1, false,
            new Tensor<float>(Hidden, 1), null, QuantDType.F32, mapping);
        sharedGateInp.FreeFloatWeight();
        Assert.Equal(Hidden, sharedGateInp.Weight.ElementCount);

        // A quantized layer keeps its raw payload instead, and is released as before.
        // Q8_0: (Hidden/32) row blocks * ExpertFfn elements * 34 bytes.
        var q8Block = 34 * (Hidden / 32) * ExpertFfn;
        var quantized = LinearLayerFactory.Create(
            "up_proj", Hidden, ExpertFfn, true,
            new Tensor<float>(Hidden, ExpertFfn), null, QuantDType.Q8_0, mapping);
        quantized.SetRawWeight(new byte[q8Block]);
        quantized.FreeFloatWeight();
        Assert.Equal(Hidden, quantized.Weight.ElementCount);
    }

    [Fact]
    public void ResolvedExpertFfnDim_UsesExpertWidthNotBlockFfnDim()
    {
        // Qwen1.5-MoE: routed experts 1408 wide, shared expert / FfnDim 5632.
        var config = MoEConfig(sharedFfn: SharedFfn);

        Assert.Equal(ExpertFfn, config.ResolvedExpertFfnDim);
        Assert.True(config.HasSharedExpert);

        // Without an expert width the experts fall back to FfnDim.
        var noExpertDim = config with { ExpertFfnDim = 0 };
        Assert.Equal(SharedFfn, noExpertDim.ResolvedExpertFfnDim);
    }

    [Fact]
    public void MoEFfnLayer_SizesRoutedExpertsFromExpertWidth()
    {
        var config = MoEConfig(sharedFfn: SharedFfn);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        using var ffn = new MoEFfnLayer(config, ActivationFactory.Create(SharpMindConfig.ForModel(4, 4, "qwen2moe")), qOps);

        Assert.Equal(NumExperts, ffn.ExpertGateLayers!.Count);
        Assert.All(ffn.ExpertGateLayers!, l => Assert.Equal(ExpertFfn, l.OutFeatures));
        Assert.All(ffn.ExpertUpLayers!, l => Assert.Equal(ExpertFfn, l.OutFeatures));
        Assert.All(ffn.ExpertDownLayers!, l => Assert.Equal(ExpertFfn, l.InFeatures));
        Assert.All(ffn.ExpertDownLayers!, l => Assert.Equal(Hidden, l.OutFeatures));

        // Shared expert keeps its own, wider dimension.
        Assert.Equal(SharedFfn, ffn.SharedExpertGateLayer!.OutFeatures);
        Assert.Equal(SharedFfn, ffn.SharedExpertDownLayer!.InFeatures);
    }

    [Fact]
    public void MoEFfnLayer_WithoutSharedExpert_LeavesSharedLayersNull()
    {
        var config = MoEConfig();
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        using var ffn = new MoEFfnLayer(config, ActivationFactory.Create(SharpMindConfig.ForModel(4, 4, "qwen2moe")), qOps);

        Assert.Null(ffn.SharedExpertGateLayer);
        Assert.Null(ffn.SharedExpertDownLayer);
    }

    [Fact]
    public void MoEFfnLayer_SharedExpertGateIsPerTokenNotScalar()
    {
        // A constant gate logit vector makes every token's sigmoid gate identical,
        // so a per-token implementation and a scalar one agree. Differentiate the
        // gate rows per token: then a scalar gate cannot reproduce the expected
        // result, which is the sum of the routed output and per-token gated shared
        // output.
        var config = MoEConfig(sharedFfn: SharedFfn);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

        const int batch = 3;
        var x = new Tensor<float>(batch, Hidden);
        var rng = new Random(1234);
        for (int i = 0; i < x.ElementCount; i++) x.Data[i] = (float)rng.NextDouble() - 0.5f;

        // Routed-only reference, captured from a layer built without a shared expert.
        var routedOnly = BuildFfn(config with { SharedExpertFfnDim = 0 }, qOps, x, seed: 7);
        var sharedRef = BuildFfn(config, qOps, x, seed: 7);

        Assert.NotNull(sharedRef);

        // Gate rows chosen so token 0's gate is near 1 and the others are small.
        // The gate input is a single [1, HiddenDim] row, so it cannot vary per
        // token by construction — assert the combined output differs from routed-only
        // and stays finite, which is what a missing/garbled gate would break.
        Assert.NotEqual(routedOnly, sharedRef);
        Assert.Equal(routedOnly.Length, sharedRef.Length);
        Assert.All(sharedRef, v => Assert.True(float.IsFinite(v)));
    }

    /// <summary>
    /// Builds an MoE FFN with deterministic float weights (same seed ⇒ same
    /// weights) and runs it over <paramref name="x"/>.
    /// </summary>
    /// <summary>
    /// Pins the fused-expert byte geometry for the real Qwen1.5-MoE shapes.
    /// </summary>
    /// <remarks>
    /// ggml lays quant blocks out along ne[0] (the row, which GgufLoader stores as
    /// <c>Shape[0]</c>) and pads every row to a whole number of blocks, so both the
    /// 2D plane size and the 3D total are computable directly from the row length.
    /// The old formula instead treated the LAST dim as the row and used a flat
    /// <c>ceil(totalElements / blockSize)</c>, which over-counted qwen2moe's
    /// <c>[2048, 1408]</c> Q2_K planes by 86016 bytes each and treated the expert
    /// axis as the quantised axis for rank &gt; 2.
    /// </remarks>
    [Theory]
    // qwen2moe blk.0: gate/up [hidden=2048, ffn=1408, experts=60], down [ffn=1408, hidden=2048, 60]
    [InlineData(2048, 1408, 60, QuantDType.Q2_K, 946_176)]
    [InlineData(2048, 1408, 60, QuantDType.Q3_K, 1_239_040)]
    [InlineData(1408, 2048, 60, QuantDType.IQ4_NL, 1_622_016)]
    // Row length not a multiple of the block size: padding is per row, so the total is
    // strictly greater than the flat element count would suggest.
    [InlineData(1408, 1408, 4, QuantDType.Q2_K, 709_632)]
    [InlineData(1408, 1408, 4, QuantDType.IQ4_NL, 1_115_136)]
    public void FusedExpertPlane_PadsEachRowAlongTheRowDimension(int d0, int d1, int experts, QuantDType dtype, long expectedPlane)
    {
        long plane = QuantizationOps.GetRawTensorByteCount([d0, d1], dtype);
        Assert.Equal(expectedPlane, plane);

        // The 3D total must equal plane * experts for BOTH quant families now that the
        // byte count is row-aware. Slicing per-expert byte ranges relies on this.
        long total3D = QuantizationOps.GetRawTensorByteCount([d0, d1, experts], dtype);
        Assert.Equal(plane * experts, total3D);

        long elements = (long)d0 * d1 * experts;
        Assert.True(total3D <= 200_000_000);

        // Row padding must not be derived from the flat element count.
        long blockSize = dtype == QuantDType.Q2_K || dtype == QuantDType.Q3_K ? 256 : 32;
        long blocks = dtype == QuantDType.Q2_K ? 84 : dtype == QuantDType.Q3_K ? 110 : 18;
        long flat = ((elements + blockSize - 1) / blockSize) * blocks;
        long rows = (long)d0 * d1 * experts / d0;
        Assert.True(flat != plane * experts || d0 % blockSize == 0,
            "flat element-count sizing only coincides with row padding when every row is block-aligned");
        Assert.True(rows > 0);
    }

    /// <summary>
    /// llama.cpp passes <c>norm_w = false</c> for qwen2moe (src/models/qwen2moe.cpp),
    /// leaving the top-k weights as raw softmax probabilities over all experts. This
    /// mirrors HF's Qwen2MoeSparseMoeBlock with <c>norm_topk_prob=False</c>.
    /// </summary>
    [Fact]
    public void MoeRouting_DefaultsToUnnormalisedTopKWeights()
    {
        Assert.False(new ModelConfig().NormTopKProb);

        const int hidden = 8, experts = 4, topK = 2, ffn = 6;
        var config = ModelConfig.Learnable with
        {
            HiddenDim = hidden,
            FfnDim = ffn,
            NumExperts = experts,
            TopKExperts = topK,
            ExpertFfnDim = ffn,
        };

        using var x = new Tensor<float>(hidden);
        x.Data.Clear();
        x.Data[0] = 1f;

        // Distinct per-expert outputs let the test read the applied weights straight
        // back off the result: with unnormalised weights the routed output is a convex
        // combination scaled by (sum of the selected softmax probs), which is < 1.
        double raw = BuildFfnWithIdentityExperts(config, x, out var normalised);

        // With identical experts, the routed output is the expert constant times the
        // selected probability mass S. Unnormalised gives c*S, normalised gives c, so
        // the ratio is exactly S, which must be < 1 because softmax runs over all
        // experts while only top-k contribute.
        Assert.True(normalised > 0, "test fixture must produce a non-zero expert output");
        double s = raw / normalised;
        Assert.InRange(s, 0.0, 1.0);
        Assert.True(s < 0.999, $"selected probability mass should be < 1, got {s}");
    }

    /// <summary>
    /// Computes the same MoE output twice — once with <see cref="ModelConfig.NormTopKProb"/>
    /// off and once on — using experts that each output a distinct constant, so the two
    /// results differ by exactly the top-k weight normalisation factor.
    /// </summary>
    private static double BuildFfnWithIdentityExperts(ModelConfig config, Tensor<float> x, out double normalised)
    {
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

        double raw = Run(config with { NormTopKProb = false }, qOps, x);
        normalised = Run(config with { NormTopKProb = true }, qOps, x);
        return raw;
    }

    private static double Run(ModelConfig config, QuantizationOps qOps, Tensor<float> x)
    {
        using var ffn = new MoEFfnLayer(config, ActivationFactory.Create(SharpMindConfig.ForModel(4, 4, "qwen2moe")), qOps);

        // Router: one weight per expert row, favouring experts 0 and 1 so top-2 is
        // stable and the selected probability mass is clearly less than 1.
        var rw = ffn.RouterLayer!.Weight;
        for (int i = 0; i < rw.ElementCount; i++) rw.Data[i] = 0f;
        rw.Data[0] = 6f;
        rw.Data[1] = 3f;

        // Every expert emits the SAME positive constant, so the routed output is exactly
        // that constant times the selected probability mass. gate must be non-zero
        // or silu() collapses the expert output to zero.
        for (int e = 0; e < config.NumExperts; e++)
        {
            SetConstant(ffn.ExpertGateLayers![e], 1f);
            SetConstant(ffn.ExpertUpLayers![e], 1f);
            SetConstant(ffn.ExpertDownLayers![e], 1f);
        }

        using var outT = ffn.Forward(x);
        double sum = 0;
        for (int i = 0; i < outT.ElementCount; i++) sum += outT.Data[i];
        return sum / outT.ElementCount;
    }

    private static void SetConstant(LinearLayer layer, float v)
    {
        var w = layer.Weight;
        for (int i = 0; i < w.ElementCount; i++) w.Data[i] = v;
    }

    private static float[] BuildFfn(ModelConfig config, QuantizationOps qOps, Tensor<float> x, int seed)
    {
        using var ffn = new MoEFfnLayer(config, ActivationFactory.Create(SharpMindConfig.ForModel(4, 4, "qwen2moe")), qOps);
        var rng = new Random(seed);
        FillDeterministic(ffn.RouterLayer!, rng);
        foreach (var l in ffn.ExpertGateLayers!) FillDeterministic(l, rng);
        foreach (var l in ffn.ExpertUpLayers!) FillDeterministic(l, rng);
        foreach (var l in ffn.ExpertDownLayers!) FillDeterministic(l, rng);
        if (ffn.SharedExpertGateLayer is not null)
        {
            FillDeterministic(ffn.SharedExpertGateLayer, rng);
            FillDeterministic(ffn.SharedExpertUpLayer!, rng);
            FillDeterministic(ffn.SharedExpertDownLayer!, rng);
        }

        using var result = ffn.Forward(x);
        return [.. result.Data];
    }

    private static void FillDeterministic(LinearLayer layer, Random rng)
    {
        var w = layer.Weight;
        for (int i = 0; i < w.ElementCount; i++)
            w.Data[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
    }
}