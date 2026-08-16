using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training.LoRA;

namespace SharpMind.Tests.Training;

/// <summary>
/// Regression tests for the LoRA adapters. LoRALayer previously swapped the
/// constructor arguments (A became <c>[rank, in]</c> and B <c>[out, rank]</c>,
/// so <see cref="LoRALayer.InFeatures"/>/<see cref="LoRALayer.OutFeatures"/>
/// both reported the rank) and the matmul kernels were invoked with
/// InFeatures/rank in place of the layers' output dims. These tests pin the
/// corrected contracts: A is <c>[in, rank]</c>, B is <c>[rank, out]</c>, the
/// forward produces <c>[batch, out]</c>, and the deltas reproduce the
/// reference <c>x @ W + scale * (x @ A @ B)</c> maths.
/// </summary>
public sealed class LoRATests
{
    private const int Batch = 3;

    [Fact]
    public void LoRALayer_ExposesInOutAndRank()
    {
        using var layer = new LoRALayer(inFeatures: 6, outFeatures: 4, rank: 2);
        Assert.Equal(6, layer.InFeatures);
        Assert.Equal(4, layer.OutFeatures);
        Assert.Equal(2, layer.Rank);
    }

    [Fact]
    public void LoRALayer_ClampsRankToSmallestDimension()
    {
        using var layer = new LoRALayer(inFeatures: 4, outFeatures: 8, rank: 32);
        Assert.Equal(4, layer.Rank);
    }

    [Fact]
    public void LoRALayer_Parameters_HaveRankSizedShapes()
    {
        using var layer = new LoRALayer(inFeatures: 6, outFeatures: 4, rank: 2);
        var ps = layer.Parameters().ToArray();

        // A = [in, rank], B = [rank, out]
        Assert.Equal(new[] { 6, 2 }, ps[0].Data.Shape.Dims);
        Assert.Equal(new[] { 2, 4 }, ps[1].Data.Shape.Dims);

        long total = ps.Sum(p => p.Data.ElementCount);
        Assert.Equal(6L * 2 + 2 * 4, total);
    }

    [Fact]
    public void LoRALayer_Forward_ProducesOutColumnsAndMatchesReference()
    {
        using var layer = new LoRALayer(inFeatures: 6, outFeatures: 4, rank: 2, scale: 1.5f);
        var ps = layer.Parameters().ToArray();
        var aRaw = ps[0].Data; // [in, rank]
        var bRaw = ps[1].Data; // [rank, out]

        using var input = Tensor<float>.From(
            [1f, 2f, 3f, 4f, 5f, 6f,
             2f, 1f, 4f, 3f, 6f, 5f,
             0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f], Batch, 6);

        using var frozenWeights = Tensor<float>.From(
            [0.1f, -0.2f, 0.3f, 0.4f,
             -0.5f, 0.6f, -0.7f, 0.8f,
             0.9f, -1.0f, 0.11f, -0.12f,
             0.13f, 0.14f, -0.15f, 0.16f,
             -0.17f, 0.18f, 0.19f, -0.2f,
             0.21f, -0.22f, 0.23f, 0.24f], 6, 4);

        using var actual = layer.Forward(input, frozenWeights);

        Assert.Equal(new[] { Batch, 4 }, actual.Shape.Dims);

        // Reference: result = x @ W + scale * (x @ A @ B)
        for (int b = 0; b < Batch; b++)
        {
            for (int o = 0; o < 4; o++)
            {
                float frozen = 0f;
                for (int i = 0; i < 6; i++) frozen += input[b, i] * frozenWeights[i, o];

                float delta = 0f;
                for (int r = 0; r < 2; r++)
                {
                    float projected = 0f;
                    for (int i = 0; i < 6; i++) projected += input[b, i] * aRaw[i * 2 + r];
                    delta += projected * bRaw[r * 4 + o];
                }

                float expected = frozen + 1.5f * delta;
                Assert.Equal(expected, actual[b, o], precision: 4);
            }
        }
    }

    [Fact]
    public void LoRALayer_ForwardEmbedding_ProducesInByOutDelta()
    {
        using var layer = new LoRALayer(inFeatures: 6, outFeatures: 4, rank: 2, scale: 2f);
        var ps = layer.Parameters().ToArray();
        var aRaw = ps[0].Data; // [in, rank]
        var bRaw = ps[1].Data; // [rank, out]

        using var lora = layer.ForwardEmbedding();

        Assert.Equal(new[] { 6, 4 }, lora.Shape.Dims);

        // Reference: delta = scale * (A @ B)
        for (int i = 0; i < 6; i++)
        for (int o = 0; o < 4; o++)
        {
            float dot = 0f;
            for (int r = 0; r < 2; r++) dot += aRaw[i * 2 + r] * bRaw[r * 4 + o];
            Assert.Equal(2f * dot, lora[i, o], precision: 4);
        }
    }

    [Fact]
    public void LoRAModel_TrainableRatio_CountsActualElements()
    {
        var cfg = new ModelConfig
        {
            VocabSize = 16,
            HiddenDim = 8,
            NumLayers = 1,
            NumHeads = 2,
            NumKvHeads = 2,
            FfnDim = 16,
            MaxSeqLen = 16,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        using var loraModel = new LoRAModel(model, new LoRAConfig { Rank = 2 });

        long loraParams = loraModel.LoRAParameters().Sum(p => p.Data.ElementCount);
        double ratio = loraModel.TrainableRatio();

        Assert.Equal((double)loraParams / model.ParameterCount, ratio, precision: 12);

        // The old estimate (paramCount * Rank) must not equal the true count here.
        long brokenEstimate = loraModel.LoRAParameters().Count() * 2L;
        Assert.NotEqual(brokenEstimate, loraParams);
    }

    [Fact]
    public void LoRAAttention_DefaultTargetsAllFourProjections()
    {
        using var attention = new LoRAAttention(hiddenDim: 8, numHeads: 2, headDim: 4, new LoRAConfig { Rank = 2 });

        var ps = attention.Parameters().ToArray();
        Assert.Equal(8, ps.Length);
        Assert.Equal(128, ps.Sum(p => p.Data.ElementCount));
    }

    [Fact]
    public void LoRAAttention_RespectsTargetModules()
    {
        using var qv = new LoRAAttention(
            hiddenDim: 8, numHeads: 2, headDim: 4,
            new LoRAConfig { Rank = 2, TargetModules = ["q_proj", "v_proj"] });

        var ps = qv.Parameters().ToArray();
        Assert.Equal(4, ps.Length);
        Assert.Equal(64, ps.Sum(p => p.Data.ElementCount));
    }

    [Fact]
    public void LoRAAttention_EmptyTargetModules_AddsNoParameters()
    {
        using var attention = new LoRAAttention(
            hiddenDim: 8, numHeads: 2, headDim: 4,
            new LoRAConfig { Rank = 2, TargetModules = [] });

        Assert.Empty(attention.Parameters());
    }

    [Fact]
    public void LoRAAttention_UntargetedApplyFallsBackToFrozen()
    {
        var config = new LoRAConfig { Rank = 2, TargetModules = ["o_proj"] };
        using var attention = new LoRAAttention(hiddenDim: 8, numHeads: 2, headDim: 4, config);

        using var x = InputTensor();
        using var wq = WeightTensor();
        using var actual = attention.ApplyToQ(x, wq);

        Assert.Equal(new[] { 3, 8 }, actual.Shape.Dims);
        for (int b = 0; b < 3; b++)
        for (int o = 0; o < 8; o++)
        {
            float expected = 0f;
            for (int i = 0; i < 8; i++) expected += x[b, i] * wq[i, o];
            Assert.Equal(expected, actual[b, o], precision: 4);
        }
    }

    [Fact]
    public void LoRAAttention_TargetedApplyAddsScaleTimesADelta()
    {
        var config = new LoRAConfig { Rank = 2, Alpha = 4f, TargetModules = ["q_proj"] };
        using var attention = new LoRAAttention(hiddenDim: 8, numHeads: 2, headDim: 4, config);

        var ps = attention.Parameters().ToArray();
        var aRaw = ps[0].Data; // [in, rank]
        var bRaw = ps[1].Data; // [rank, out]
        float scale = config.Scale;

        using var x = InputTensor();
        using var wq = WeightTensor();
        using var actual = attention.ApplyToQ(x, wq);

        for (int b = 0; b < 3; b++)
        for (int o = 0; o < 8; o++)
        {
            float frozen = 0f;
            for (int i = 0; i < 8; i++) frozen += x[b, i] * wq[i, o];

            float delta = 0f;
            for (int r = 0; r < 2; r++)
            {
                float projected = 0f;
                for (int i = 0; i < 8; i++) projected += x[b, i] * aRaw[i * 2 + r];
                delta += projected * bRaw[r * 8 + o];
            }

            Assert.Equal(frozen + scale * delta, actual[b, o], precision: 3);
        }
    }

    private static Tensor<float> InputTensor()
    {
        var data = new float[24];
        for (int i = 0; i < data.Length; i++) data[i] = (i % 8) * 0.25f + i % 3;
        return Tensor<float>.From(data, 3, 8);
    }

    private static Tensor<float> WeightTensor()
    {
        var data = new float[64];
        for (int i = 0; i < 8; i++)
        for (int o = 0; o < 8; o++)
            data[i * 8 + o] = (i + 1) * 0.1f + (o + 1) * 0.01f;
        return Tensor<float>.From(data, 8, 8);
    }
}