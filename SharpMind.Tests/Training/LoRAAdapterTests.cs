using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.LoRA;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// C3: a trained adapter can be saved on its own and restored onto an identical
/// base, and is rejected when the base it is paired with is not the one it was
/// trained against. The base weights are never written to the adapter file.
/// </summary>
public sealed class LoRAAdapterTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static readonly ModelConfig SmallConfig = new()
    {
        VocabSize = 32,
        HiddenDim = 8,
        NumLayers = 2,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 512,
    };

    private const int Batch = 2;
    private const int Seq = 4;

    private static Tensor<int> DeterministicBatch(out Tensor<int> labels)
    {
        labels = Tensor<int>.From([0, 1, 2, 3, 4, 5, 6, 7], Batch, Seq);
        return Tensor<int>.From([3, 9, 12, 5, 17, 2, 8, 31], Batch, Seq);
    }

    private static Transformer Fixture(int seed = 9001)
    {
        var sc = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(SmallConfig, sc);
        WeightInitializer.InitializeRandomly(weights, seed);
        return ModelFactory.CreateTrainingTransformer(weights, sc);
    }

    private static void RandomiseAdapters(LoRAModel lora, int seed)
    {
        var rng = new Random(seed);
        foreach (var p in lora.LoRAParameters())
            for (int i = 0; i < p.Data.ElementCount; i++)
                p.Data.Data[i] = (float)(rng.NextDouble() - 0.5) * 0.3f;
    }

    [Fact]
    public void SaveAdapter_ThenLoadAdapter_ReproducesTheAdaptedForward()
    {
        using var ids = DeterministicBatch(out var labels);
        using var _l = labels;

        using var source = Fixture();
        using var lora = new LoRAModel(source, new LoRAConfig { Rank = 2, Alpha = 4f }, seed: 5);
        RandomiseAdapters(lora, 13);
        using var expected = source.Forward(ids);

        string path = Path.Combine(_dir.Path, "adapter.smla");
        lora.SaveAdapter(path);
        Assert.True(File.Exists(path));

        using var target = Fixture(); // same seed => same base weights
        using var restored = LoRAModel.LoadAdapter(target, path);
        Assert.Equal(lora.AdapterCount, restored.AdapterCount);

        using var actual = target.Forward(ids);
        for (int i = 0; i < expected.ElementCount; i++)
            Assert.True(Math.Abs(expected.Data[i] - actual.Data[i]) <= 1e-5f * (1f + Math.Abs(expected.Data[i])),
                $"logit {i}: saved={expected.Data[i]} restored={actual.Data[i]}");
    }

    [Fact]
    public void LoadAdapter_RejectsADifferentBaseModel()
    {
        using var source = Fixture(seed: 9001);
        using var lora = new LoRAModel(source, new LoRAConfig { Rank = 2, Alpha = 4f }, seed: 5);
        RandomiseAdapters(lora, 13);
        string path = Path.Combine(_dir.Path, "adapter.smla");
        lora.SaveAdapter(path);

        using var other = Fixture(seed: 1234); // different base weights
        Assert.NotEqual(LoRAModel.BaseFingerprint(source), LoRAModel.BaseFingerprint(other));

        var ex = Assert.Throws<InvalidDataException>(() => LoRAModel.LoadAdapter(other, path));
        Assert.Contains("different base model", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("fingerprint", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BaseFingerprint_IsStableForTheSameWeights_AndDiffersForAnotherSeed()
    {
        using var a = Fixture(seed: 42);
        using var b = Fixture(seed: 42);
        using var c = Fixture(seed: 43);

        Assert.Equal(LoRAModel.BaseFingerprint(a), LoRAModel.BaseFingerprint(b));
        Assert.NotEqual(LoRAModel.BaseFingerprint(a), LoRAModel.BaseFingerprint(c));
    }

    [Fact]
    public void SaveAdapter_AfterMerge_HasNothingToSave()
    {
        using var model = Fixture();
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 2 }, seed: 5);
        lora.Merge();

        string path = Path.Combine(_dir.Path, "adapter.smla");
        Assert.Throws<InvalidOperationException>(() => lora.SaveAdapter(path));
    }
}
