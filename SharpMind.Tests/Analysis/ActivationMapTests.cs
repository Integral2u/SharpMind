using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Analysis;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Analysis;

public sealed class ActivationMapTests
{
    private static Transformer BuildTinyModel()
    {
        var config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 64,
            NumLayers = 2,
            NumHeads = 4,
            NumKvHeads = 2,
            FfnDim = 128,
            MaxSeqLen = 64,
            Architecture = "qwen2",
        };
        var sharp = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(config, sharp);
        WeightInitializer.InitializeRandomly(weights, seed: 424242);
        return ModelFactory.CreateTransformer(weights, sharp, optimizeMemory: false);
    }

    [Fact]
    public void Capture_ReturnsALayerByTokenMap()
    {
        using var model = BuildTinyModel();
        int[] ids = [1, 7, 13, 29, 3];

        ActivationMap map = ActivationMapAnalyzer.Capture(model, ids, maxTokens: ids.Length);

        Assert.Equal(2, map.Layers);
        Assert.Equal(ids.Length, map.Tokens);
        Assert.Equal(2 * ids.Length, map.AttentionRms.Length);
        Assert.Equal(2 * ids.Length, map.FeedForwardRms.Length);
        Assert.All(map.AttentionRms, value => Assert.True(float.IsFinite(value) && value >= 0));
        Assert.All(map.FeedForwardRms, value => Assert.True(float.IsFinite(value) && value >= 0));
        Assert.True(map.MaxAttention >= 0);
        Assert.True(map.MaxFeedForward >= 0);
    }

    [Fact]
    public void Capture_RejectsAnEmptyPrompt()
    {
        using var model = BuildTinyModel();
        Assert.Throws<ArgumentException>(() => ActivationMapAnalyzer.Capture(model, []));
    }
}
