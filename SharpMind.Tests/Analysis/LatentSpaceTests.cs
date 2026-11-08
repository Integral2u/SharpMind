using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Analysis;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Analysis;

public sealed class LatentSpaceTests
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
    public void Capture_ProjectsEveryPromptToken()
    {
        using var model = BuildTinyModel();
        int[] ids = [1, 7, 13, 29, 3, 11, 2];

        LatentSpaceMap map = LatentSpaceAnalyzer.Capture(model, ids, id => $"t{id}", maxTokens: ids.Length);

        Assert.Equal(ids.Length, map.Tokens);
        Assert.Equal(64, map.Dimensions);
        Assert.Equal(ids.Length, map.Points.Count);
        Assert.InRange(map.ExplainedVariance, 0, 1);
        Assert.All(map.Points, point =>
        {
            Assert.True(double.IsFinite(point.X));
            Assert.True(double.IsFinite(point.Y));
            Assert.StartsWith("t", point.Text, StringComparison.Ordinal);
        });
    }

    [Fact]
    public void Capture_RejectsAnEmptyPrompt()
    {
        using var model = BuildTinyModel();
        Assert.Throws<ArgumentException>(() => LatentSpaceAnalyzer.Capture(model, []));
    }
}
