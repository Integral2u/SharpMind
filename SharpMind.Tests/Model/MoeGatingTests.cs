using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// DeepSeek V3 / Moonlight routing: sigmoid gating instead of softmax, a
/// selection bias that moves the top-k choice without touching the weights, and
/// the routed-branch multiplier. All three are metadata-only knobs
/// (<c>expert_gating_func</c>, <c>ffn_exp_probs_b</c>,
/// <c>expert_weights_scale</c>) whose absence silently degrades the model rather
/// than failing, so each is pinned against llama.cpp's build_moe_ffn order:
/// gate -> gating func -> top-k on (probs + bias) -> weight from probs ->
/// renormalise -> scale.
/// </summary>
public class MoeGatingTests
{
    private const int Hidden = 4;
    private const int Experts = 4;
    private const int TopK = 2;
    private const int Ffn = 4;

    private static ModelConfig Config => ModelConfig.Learnable with
    {
        HiddenDim = Hidden,
        FfnDim = Ffn,
        NumExperts = Experts,
        TopKExperts = TopK,
        ExpertFfnDim = Ffn,
    };

    /// <summary>
    /// Router logits 2, 1, -1, -3 for the test input; every expert emits the same
    /// constant, so the routed output is that constant times the summed weights.
    /// </summary>
    private static (LinearLayer router, LinearLayer[] gate, LinearLayer[] up, LinearLayer[] down) Build()
    {
        var mapping = new Dictionary<string, string>();
        var router = LinearLayerFactory.Create(
            "router", Hidden, Experts, true, new Tensor<float>(Hidden, Experts), null, QuantDType.F32, mapping);
        var rw = router.Weight;
        for (int i = 0; i < rw.ElementCount; i++) rw.Data[i] = 0f;
        // Weights are [in, out]: only the first input row is non-zero, so the
        // logits are exactly 2, 1, -1, -3 for x = e0.
        rw.Data[0 * Experts + 0] = 2f;
        rw.Data[0 * Experts + 1] = 1f;
        rw.Data[0 * Experts + 2] = -1f;
        rw.Data[0 * Experts + 3] = -3f;

        var gate = new LinearLayer[Experts];
        var up = new LinearLayer[Experts];
        var down = new LinearLayer[Experts];
        for (int e = 0; e < Experts; e++)
        {
            gate[e] = LinearLayerFactory.Create($"g{e}", Hidden, Ffn, true, new Tensor<float>(Hidden, Ffn), null, QuantDType.F32, mapping);
            up[e] = LinearLayerFactory.Create($"u{e}", Hidden, Ffn, true, new Tensor<float>(Hidden, Ffn), null, QuantDType.F32, mapping);
            down[e] = LinearLayerFactory.Create($"d{e}", Ffn, Hidden, true, new Tensor<float>(Ffn, Hidden), null, QuantDType.F32, mapping);
            Fill(gate[e].Weight, 1f);
            Fill(up[e].Weight, 1f);
            Fill(down[e].Weight, 1f);
        }
        return (router, gate, up, down);
    }

    private static void Fill(Tensor<float> t, float v)
    {
        for (int i = 0; i < t.ElementCount; i++) t.Data[i] = v;
    }

    /// <summary>Runs the routing and returns the scalar every output element holds.</summary>
    private static float Route(
        Tensor<float>? bias = null, MoEGatingFunc gating = MoEGatingFunc.Softmax,
        bool norm = false, float scale = 0f)
    {
        var (router, gate, up, down) = Build();
        using var x = new Tensor<float>(1, Hidden);
        x.Data.Clear();
        x.Data[0] = 1f;

        using var result = FfnKernels.MoE(
            x, router, gate, up, down, TopK, ActivationFactory.Create(SharpMindConfig.ForModel(4, 4, "deepseek2")),
            workspace: null, normTopKProb: norm, residency: null, routerBias: bias, gating: gating, weightScale: scale);

        // Every element receives the same constant; check that and return it.
        float first = result.Data[0];
        for (int i = 1; i < result.ElementCount; i++) Assert.Equal(first, result.Data[i], 5);
        return first;
    }

    /// <summary>silu(1) * Ffn: what one expert emits for every output element.</summary>
    private static float ExpertConstant => (1f / (1f + MathF.Exp(-1f))) * 1f * Ffn;

    private static float Softmax0 => MathF.Exp(2f) / (MathF.Exp(2f) + MathF.Exp(1f) + MathF.Exp(-1f) + MathF.Exp(-3f));
    private static float Softmax1 => MathF.Exp(1f) / (MathF.Exp(2f) + MathF.Exp(1f) + MathF.Exp(-1f) + MathF.Exp(-3f));

    private static float Sigmoid(float z) => 1f / (1f + MathF.Exp(-z));

    [Fact]
    public void SoftmaxGating_Unnormalised_KeepsRawProbabilities()
    {
        float expected = ExpertConstant * (Softmax0 + Softmax1);
        float actual = Route();
        Assert.True(Math.Abs(expected - actual) < 1e-3,
            $"expected {expected} got {actual} (expert {ExpertConstant}, p0 {Softmax0}, p1 {Softmax1})");
    }

    /// <summary>
    /// expert_gating_func = 2: per-expert sigmoid, not a softmax over the four
    /// logits. The two differ by a factor of the partition function, which is
    /// exactly the kind of quiet 2x error a wrong gate produces.
    /// </summary>
    [Fact]
    public void SigmoidGating_UsesPerExpertSigmoid()
    {
        float expected = ExpertConstant * (Sigmoid(2f) + Sigmoid(1f));
        Assert.Equal(expected, Route(gating: MoEGatingFunc.Sigmoid), 4);

        // And it is not the softmax result.
        Assert.NotEqual(Route(), Route(gating: MoEGatingFunc.Sigmoid), 2);
    }

    /// <summary>
    /// expert_weights_norm: DeepSeek V3 renormalises the selected top-k so the
    /// weights sum to 1, then scales the branch.
    /// </summary>
    [Fact]
    public void SigmoidGating_RenormalisedAndScaled()
    {
        float p0 = Sigmoid(2f), p1 = Sigmoid(1f);
        float expected = ExpertConstant * (p0 / (p0 + p1) + p1 / (p0 + p1)) * 2.446f;
        Assert.Equal(expected, Route(gating: MoEGatingFunc.Sigmoid, norm: true, scale: 2.446f), 4);
    }

    /// <summary>
    /// The router bias is a <em>selection</em> bias (DeepSeek V3's
    /// ffn_exp_probs_b): it moves the top-k choice, but the weight still comes
    /// from the unbiased probability, so a biased expert is not amplified.
    /// llama.cpp carries a separate selection_probs tensor for exactly this.
    /// </summary>
    [Fact]
    public void RouterBias_ShiftsSelectionOnly()
    {
        // +1 to expert 2 (probability sigmoid(-1) = 0.269) puts it ahead of
        // expert 1 (0.731), so the selected pair becomes {0, 2}.
        using var bias = new Tensor<float>(Experts);
        bias.Data[0] = 0f;
        bias.Data[1] = 0f;
        bias.Data[2] = 1f;
        bias.Data[3] = 0f;

        float expected = ExpertConstant * (Sigmoid(2f) + Sigmoid(-1f));
        Assert.Equal(expected, Route(bias: bias, gating: MoEGatingFunc.Sigmoid), 4);

        // Selection-only: the biased weight is sigmoid(-1), NOT sigmoid(-1 + 1).
        float biasedWeight = ExpertConstant * (Sigmoid(2f) + Sigmoid(0f));
        Assert.NotEqual(biasedWeight, Route(bias: bias, gating: MoEGatingFunc.Sigmoid), 3);
    }

    [Fact]
    public void RouterBias_IsAllocatedForFilesThatCarryIt()
    {
        // Without the tensor, HasRouterBias is false and no bias is allocated.
        Assert.False(Config.HasRouterBias);
        Assert.Null(BiasFor(Config with { HasRouterBias = false }));
        Assert.NotNull(BiasFor(Config with { HasRouterBias = true }));
    }

    private static Tensor<float>? BiasFor(ModelConfig config)
    {
        var block = new TransformerWeights.BlockWeights { LayerIndex = 0 };
        using var ffn = new MoEFfnLayer(
            config, ActivationFactory.Create(SharpMindConfig.ForModel(4, 4, "deepseek2")),
            QuantizationFactory.Create(HardwareTier.Scalar), block);
        return ffn.RouterBiasWeights;
    }

    /// <summary>
    /// expert_weights_scale is skipped for both 0 and 1 (DeepSeek V2 writes 1),
    /// so a V2 file must route exactly as before this knob existed.
    /// </summary>
    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    public void WeightScale_IsSkippedForZeroAndOne(float scale)
    {
        Assert.Equal(Route(), Route(scale: scale), 6);
    }

    /// <summary>Builds a MoE layer with a block the router bias can be allocated on.</summary>
    private sealed class NoOpLoader : SharpMind.Model.Format.IModelLoader
    {
        public void PreInit(TransformerWeights weights, CancellationToken? ct = null) { }
        public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null, CancellationToken? ct = null) { }
        public void LoadLayerWeights(int layerIndex, TransformerWeights weights, CancellationToken? ct = null) { }
        public void LoadGlobalTensors(TransformerWeights weights, CancellationToken? ct = null) { }
    }
}
