using System.Runtime.Intrinsics.X86;
using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// LayerNorm kernels are resolved per layer from the model's own hardware tier. They used to
/// come from a process-wide default that every <see cref="ModelFactory"/> build overwrote with
/// its own tier, so a <see cref="HardwareTier.Scalar"/> model's output changed bit-for-bit
/// whenever an Auto-tier model was constructed anywhere in the process (bit-exact tests flaked
/// in full-suite runs and never in isolation).
/// </summary>
public sealed class LayerNormLayerTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 128,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 64,
        MaxSeqLen = 64,
    };

    private static Transformer BuildModel(SharpMindConfig sharpConfig)
    {
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static float[] Logits(Transformer model, Tensor<int> input)
    {
        using var logits = model.Forward(input);
        return logits.Data.ToArray();
    }

    [Fact]
    public void BuildingAnotherModel_DoesNotChangeThisModelsLogits()
    {
        // Without AVX2 the Auto tier resolves the same scalar kernel and nothing can flip.
        if (!Avx2.IsSupported) return;

        using var model = BuildModel(SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar });
        int[] ids = [.. Enumerable.Range(0, 16).Select(i => (i * 7) % Cfg.VocabSize)];
        using var input = Tensor<int>.From(ids, 1, ids.Length);

        float[] before = Logits(model, input);
        using var other = BuildModel(SharpMindConfig.Gpt); // Hardware = Auto
        float[] after = Logits(model, input);

        Assert.Equal(before, after);
    }

    [Fact]
    public void ApplyRow_UsesTheOpsItWasGiven()
    {
        if (!Avx2.IsSupported) return;

        // dim >= 8 so the AVX2 kernel's vector loops run; with smaller rows (or exact
        // inputs) both kernels are bit-identical and the test would prove nothing.
        const int dim = 32;
        const float eps = 1e-5f;
        var rng = new Random(42);
        float[] src = [.. Enumerable.Range(0, dim).Select(_ => (float)(rng.NextDouble() * 2 - 1))];
        float[] ones = [.. Enumerable.Repeat(1f, dim)];
        float[] zeros = new float[dim];

        float[] expectedScalar = new float[dim];
        float[] expectedAvx2 = new float[dim];
        NormKernels.LayerNormRowScalar(src, ones, zeros, expectedScalar, eps);
        NormKernels.LayerNormRowAVX2(src, ones, zeros, expectedAvx2, eps);
        Assert.NotEqual(expectedScalar, expectedAvx2); // the input must separate the kernels

        var scalarOps = NormOpsFactory.Create(new SharpMindConfig { Hardware = HardwareTier.Scalar });
        var avx2Ops = NormOpsFactory.Create(new SharpMindConfig { Hardware = HardwareTier.AVX2 });
        using var scalarLayer = new LayerNormLayer(dim, eps, ops: scalarOps);
        using var avx2Layer = new LayerNormLayer(dim, eps, ops: avx2Ops);

        float[] actualScalar = new float[dim];
        float[] actualAvx2 = new float[dim];
        scalarLayer.ApplyRow(src, ones, actualScalar, 0f);
        avx2Layer.ApplyRow(src, ones, actualAvx2, 0f);

        Assert.Equal(expectedScalar, actualScalar);
        Assert.Equal(expectedAvx2, actualAvx2);
    }

    [Fact]
    public void ModelFactory_GivesEachModelTheKernelsOfItsOwnTier()
    {
        if (!Avx2.IsSupported) return;

        using var scalarModel = BuildModel(SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar });
        using var autoModel = BuildModel(SharpMindConfig.Gpt);

        int dim = Cfg.HiddenDim;
        var rng = new Random(42);
        float[] src = [.. Enumerable.Range(0, dim).Select(_ => (float)(rng.NextDouble() * 2 - 1))];
        float[] ones = [.. Enumerable.Repeat(1f, dim)];

        float[] expectedScalar = new float[dim];
        float[] expectedAvx2 = new float[dim];
        NormKernels.LayerNormRowScalar(src, ones, scalarModel.FinalNorm.NormBias!.Data, expectedScalar, scalarModel.FinalNorm.Eps);
        NormKernels.LayerNormRowAVX2(src, ones, autoModel.FinalNorm.NormBias!.Data, expectedAvx2, autoModel.FinalNorm.Eps);
        Assert.NotEqual(expectedScalar, expectedAvx2);

        float[] actualScalar = new float[dim];
        float[] actualAvx2 = new float[dim];
        scalarModel.FinalNorm.ApplyRow(src, ones, actualScalar, 0f);
        autoModel.FinalNorm.ApplyRow(src, ones, actualAvx2, 0f);

        Assert.Equal(expectedScalar, actualScalar);
        Assert.Equal(expectedAvx2, actualAvx2);

        // Block norms are built on a different path than the final norm; same tier expected.
        float[] blockScalar = new float[dim];
        scalarModel.GetBlock(0)!.Norm1.ApplyRow(src, ones, blockScalar, 0f);
        Assert.Equal(expectedScalar, blockScalar);
    }

    [Fact]
    public void CreateTransformer_AcceptsAMappingWithoutTheNormKey()
    {
        // Callers may pass a quantization-only mapping (the WebAssembly host does); the norm
        // kernel then comes from the config's tier instead of failing to assemble.
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var quantOnly = new MappingBuilder(HardwareTier.Auto).ApplyQuantPreset(sharpConfig).Build();
        Assert.False(quantOnly.ContainsKey(SharpMindConfig.KeyLayerNormRow));

        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, quantOnly, optimizeMemory: false);

        int dim = Cfg.HiddenDim;
        var rng = new Random(42);
        float[] src = [.. Enumerable.Range(0, dim).Select(_ => (float)(rng.NextDouble() * 2 - 1))];
        float[] ones = [.. Enumerable.Repeat(1f, dim)];
        float[] expected = new float[dim];
        float[] actual = new float[dim];
        NormKernels.LayerNormRowScalar(src, ones, model.FinalNorm.NormBias!.Data, expected, model.FinalNorm.Eps);
        model.FinalNorm.ApplyRow(src, ones, actual, 0f);
        Assert.Equal(expected, actual);
    }
}
