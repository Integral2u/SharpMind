using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Analysis;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Analysis;

public sealed class LayerImportanceAnalyzerTests
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

    [Theory]
    [InlineData(LayerAblation.Attention)]
    [InlineData(LayerAblation.FeedForward)]
    [InlineData(LayerAblation.Both)]
    public void Analyze_ReportsEveryLayerAndAFiniteLossDelta(LayerAblation mode)
    {
        using var model = BuildTinyModel();
        var analyzer = new LayerImportanceAnalyzer(model);
        int[] ids = [1, 7, 13, 29, 3, 11, 5, 2];

        LayerImportanceReport report = analyzer.Analyze(ids, mode, maxTokens: ids.Length);

        Assert.Equal(2, report.Layers.Count);
        Assert.Equal(ids.Length - 1, report.Tokens);
        Assert.Equal(mode, report.Mode);
        Assert.True(double.IsFinite(report.BaselineLoss));
        Assert.All(report.Layers, layer =>
        {
            Assert.True(double.IsFinite(layer.AblatedLoss));
            Assert.True(double.IsFinite(layer.DeltaLoss));
            Assert.True(layer.AttentionRms >= 0);
            Assert.True(layer.FeedForwardRms >= 0);
        });
    }

    [Fact]
    public void Analyze_CanRestrictTheLayerListAndReportsProgress()
    {
        using var model = BuildTinyModel();
        var analyzer = new LayerImportanceAnalyzer(model);
        int[] ids = [1, 7, 13, 29, 3, 11];
        var progress = new List<(int Done, int Total)>();

        LayerImportanceReport report = analyzer.Analyze(
            ids, LayerAblation.Both, layers: [1], maxTokens: ids.Length,
            progress: (done, total) => progress.Add((done, total)));

        LayerImportance layer = Assert.Single(report.Layers);
        Assert.Equal(1, layer.Layer);
        Assert.Equal([(1, 1)], progress);
    }

    [Fact]
    public void Analyze_ProducesASurgeryCurveWhenRequested()
    {
        using var model = BuildTinyModel();
        var analyzer = new LayerImportanceAnalyzer(model);
        int[] ids = [1, 7, 13, 29, 3, 11];

        LayerImportanceReport report = analyzer.Analyze(
            ids, LayerAblation.Both, maxTokens: ids.Length, maxSkipLayers: 2);

        Assert.Equal(2, report.Surgery.Count);
        Assert.Single(report.Surgery[0].Layers);
        Assert.Equal(2, report.Surgery[1].Layers.Length);
        Assert.True(double.IsFinite(report.Surgery[0].Loss));
        Assert.Equal(
            report.Surgery[0].Layers.Length, report.Surgery[0].Layers.Distinct().Count());
    }

    [Fact]
    public void Analyze_RejectsASequenceTooShortToPredictAToken()
    {
        using var model = BuildTinyModel();
        var analyzer = new LayerImportanceAnalyzer(model);

        Assert.Throws<ArgumentException>(() => analyzer.Analyze([1]));
    }

    [Fact]
    public void Analyze_LeastAndMostImportantComeFromTheRankedDeltas()
    {
        using var model = BuildTinyModel();
        var analyzer = new LayerImportanceAnalyzer(model);
        LayerImportanceReport report = analyzer.Analyze([1, 7, 13, 29, 3, 11], maxTokens: 6);

        Assert.NotNull(report.LeastImportant);
        Assert.NotNull(report.MostImportant);
        Assert.True(report.LeastImportant!.DeltaLoss <= report.MostImportant!.DeltaLoss);
    }
}
