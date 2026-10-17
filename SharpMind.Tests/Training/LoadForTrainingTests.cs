using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.LoRA;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// C1: an .SMM shipped by the export path can be reloaded as a <em>trainable</em>
/// model (<see cref="SmmTrainingPipeline.LoadForTraining"/>), fine-tuned with LoRA,
/// merged, and re-exported so the result still loads through the ordinary
/// inference path. This is the round trip a "continue training from a base
/// checkpoint" workflow depends on.
/// </summary>
public sealed class LoadForTrainingTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private const int Batch = 2;
    private const int Seq = 8;

    private static Tensor<int> DeterministicBatch(out Tensor<int> labels)
    {
        labels = Tensor<int>.From([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], Batch, Seq);
        return Tensor<int>.From([3, 9, 12, 5, 17, 2, 8, 31, 1, 6, 10, 4, 19, 7, 22, 13], Batch, Seq);
    }

    private string WriteBaseModel(out Tokenizer tokenizer, out ModelConfig config)
    {
        config = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        using var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);

        using var generator = new LearnableGenerator(
            new LearnableConfig { IncludeNouns = true, IncludeVerbs = true, IncludeObjects = true },
            new Random(1234));
        tokenizer = TokenizationPipeline.TrainCharacterAsync(new InMemorySource(generator.Vocabulary))
            .GetAwaiter().GetResult();

        string path = Path.Combine(_dir.Path, "base.smm");
        SmmTrainingExporter.Export(weights, tokenizer, path, new SmmWriteOptions
        {
            QuantizationLevel = QuantDType.F32,
            Source = "training",
        });
        return path;
    }

    [Fact]
    public void LoadForTraining_RebuildsATrainableModelMatchingTheFile()
    {
        string path = WriteBaseModel(out var sourceTokenizer, out var sourceConfig);

        using var model = SmmTrainingPipeline.LoadForTraining(
            path, new TrainingLoadOptions { Hardware = HardwareTier.Scalar },
            out var tokenizer, out var config);

        Assert.Equal(sourceConfig.VocabSize, config.VocabSize);
        Assert.Equal(sourceConfig.HiddenDim, config.HiddenDim);
        Assert.Equal(sourceConfig.NumLayers, config.NumLayers);
        Assert.Equal(sourceTokenizer.VocabSize, tokenizer.VocabSize);

        // The linear layers must be the float training layers, not inference layers:
        // that is what makes LoRA attachable and what the backprop engine walks.
        for (int l = 0; l < config.NumLayers; l++)
        {
            var block = model.GetBlock(l)!;
            Assert.IsType<TrainingLinearLayer>(block.Attention.Wq);
            Assert.IsType<TrainingLinearLayer>(block.Attention.Wo);
        }
    }

    [Fact]
    public void LoadForTraining_FineTuneWithLoRaLowersLoss_AndMergedExportLoadsInInference()
    {
        string path = WriteBaseModel(out _, out var sourceConfig);

        using var model = SmmTrainingPipeline.LoadForTraining(
            path, new TrainingLoadOptions { Hardware = HardwareTier.Scalar },
            out var tokenizer, out _);

        var sharpConfig = model.Config.ForModel(HardwareTier.Scalar);
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 4, Alpha = 8f }, seed: 3);
        var parameters = lora.LoRAParameters();
        Assert.NotEmpty(parameters);

        // Snapshot the frozen base: LoRA training must not move it.
        var baseSnapshot = model.Parameters().Select(p => (p.Name, Data: p.Data.Data.ToArray())).ToList();

        var mapping = GradientMappingFactory.Create(sharpConfig);
        using var engine = new BackpropEngine(model, mapping, parameters, sharpConfig);
        using var optimizer = new AdamW(parameters, lr: 5e-2f, weightDecay: 0f);
        var loss = new CrossEntropyLoss();

        using var ids = DeterministicBatch(out var labels);
        using var _l = labels;
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = ids.Reshape(Batch * Seq);

        float first = -1f, last = -1f;
        for (int step = 0; step < 50; step++)
        {
            optimizer.ZeroGrad();
            using var ctx = new ForwardContext();
            var logits = engine.ForwardAndRecord(ctx, ids);
            using var logitsFlat = logits.Reshape(Batch * Seq, sourceConfig.VocabSize);
            float l = loss.Compute(logitsFlat, flatLabels);
            if (first < 0) first = l;
            last = l;
            using var dLogits = loss.Backward(logitsFlat, flatLabels);
            engine.Backward(ctx, dLogits, flatIds);
            optimizer.Update();
        }

        Assert.True(last < first, $"loss did not descend over 50 LoRA steps: {first} -> {last}");

        var afterSnapshot = model.Parameters().Select(p => p.Data.Data.ToArray()).ToList();
        for (int i = 0; i < baseSnapshot.Count; i++)
            Assert.True(baseSnapshot[i].Data.AsSpan().SequenceEqual(afterSnapshot[i]),
                $"frozen base parameter changed during LoRA training: {baseSnapshot[i].Name}");

        // Fold the adapters back in and ship the result through the normal export path.
        lora.Merge();
        string merged = Path.Combine(_dir.Path, "merged.smm");
        SmmTrainingExporter.Export(model.Weights, tokenizer, merged, new SmmWriteOptions
        {
            QuantizationLevel = QuantDType.Q8_0,
            Source = "training",
        });

        using var reloaded = SmmTrainingPipeline.LoadForInference(merged, out var loadedTokenizer, out var loadedConfig);
        Assert.Equal(sourceConfig.VocabSize, loadedConfig.VocabSize);

        var prompt = new List<int> { 1, 2, 3 };
        var generated = SmmTrainingPipeline.GenerateGreedy(reloaded, prompt, loadedConfig.VocabSize, steps: 4);
        Assert.Equal(prompt.Count + 4, generated.Count);
        Assert.All(generated, id => Assert.InRange(id, 0, loadedConfig.VocabSize - 1));
        Assert.Equal(tokenizer.VocabSize, loadedTokenizer.VocabSize);
    }
}
