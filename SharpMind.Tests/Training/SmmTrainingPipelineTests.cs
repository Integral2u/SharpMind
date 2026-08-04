using SharpMind.Core;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Training;

namespace SharpMind.Tests.Training;

/// <summary>
/// End-to-end round trip of the <see cref="SmmTrainingPipeline"/>: export a
/// tiny trained-ish model, reload it for inference, and generate greedily
/// without leaving the model's vocab.
/// </summary>
public class SmmTrainingPipelineTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    [Fact]
    public void LoadForInference_RoundTripsAndGenerates()
    {
        var cfg = new ModelConfig
        {
            VocabSize = 64,
            HiddenDim = 8,
            NumLayers = 1,
            NumHeads = 1,
            NumKvHeads = 1,
            FfnDim = 16,
            MaxSeqLen = 512,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);

        var generator = new LearnableGenerator(new LearnableConfig(), new Random(1234));
        var tokenizer = TrainingTokenizerBuilder.BuildForVocab(generator, cfg.VocabSize);

        string path = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(weights, tokenizer, path, new SmmWriteOptions
        {
            Compression = CompressionMode.Auto,
            Source = "training",
        });

        using var reloaded = SmmTrainingPipeline.LoadForInference(path, out var reloadedTokenizer, out var reloadedConfig);

        Assert.Equal("gpt2", reloadedConfig.Architecture);
        Assert.Equal(64, reloadedConfig.VocabSize);
        Assert.NotNull(reloadedTokenizer);
        Assert.Equal(26, reloadedTokenizer.VocabSize);

        // Greedy generation grows the prompt and stays inside the vocab.
        var ids = SmmTrainingPipeline.GenerateGreedy(reloaded, [3, 14, 20], reloadedConfig.VocabSize, steps: 5);
        Assert.Equal(8, ids.Count);
        foreach (int id in ids)
            Assert.InRange(id, 0, 63);

        // Decode is well-defined for the whole sequence.
        string text = reloadedTokenizer.Decode(ids.ToArray());
        Assert.False(string.IsNullOrWhiteSpace(text));
    }
}
