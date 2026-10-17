using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Data.Batching;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Loss;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// Verifies a quantized model loaded through the inference path
/// (<see cref="SmmTrainingPipeline.LoadForInference"/>) can be scored by the
/// training-side <see cref="Evaluator"/>: evaluating an F32 export and a Q8_0
/// export of the same weights must both produce finite loss/perplexity, with
/// the quantized loss staying close to the exact one.
/// </summary>
public sealed class EvaluatorQuantizedTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    [Fact]
    public async Task Evaluator_ScoresQuantizedInferenceLoadedModel()
    {
        var modelConfig = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        using var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);

        using var generator = new LearnableGenerator(
            new LearnableConfig { IncludeNouns = true, IncludeVerbs = true, IncludeObjects = true },
            new Random(1234));

        var tokenizer = await TokenizationPipeline.TrainCharacterAsync(
            new InMemorySource(generator.Vocabulary));

        float f32 = EvalLoss(weights, tokenizer, generator, modelConfig, QuantDType.F32, "f32");
        float q8 = EvalLoss(weights, tokenizer, generator, modelConfig, QuantDType.Q8_0, "q8");

        Assert.True(float.IsFinite(f32) && f32 > 0, $"f32 loss not finite: {f32}");
        Assert.True(float.IsFinite(q8) && q8 > 0, $"q8 loss not finite: {q8}");
        Assert.True(MathF.Abs(f32 - q8) < 0.5f,
            $"quantized loss {q8} strayed from the exact loss {f32}");
    }

    private float EvalLoss(
        TransformerWeights weights,
        Tokenizer tokenizer,
        LearnableGenerator generator,
        ModelConfig config,
        QuantDType dtype,
        string tag)
    {
        string path = Path.Combine(_dir.Path, $"{tag}.smm");
        SmmTrainingExporter.Export(weights, tokenizer, path, new SmmWriteOptions
        {
            QuantizationLevel = dtype,
            Source = "training",
        });

        // The inference loader is what the runtime uses; proving the evaluator
        // works here means the same container that drives generation can also be
        // scored during or after training.
        using var model = SmmTrainingPipeline.LoadForInference(path, out _, out var loadedConfig);
        Assert.Equal(config.VocabSize, loadedConfig.VocabSize);
        Assert.Equal(config.HiddenDim, loadedConfig.HiddenDim);

        using var batch = MakeBatch(generator.GenerateTrainingSample().TokenIds);
        var evaluator = new Evaluator(model, new CrossEntropyLoss(ignoreId: -100));
        var (loss, perplexity) = evaluator.Evaluate([batch]);

        Assert.True(float.IsFinite(perplexity), $"perplexity not finite: {perplexity}");
        return loss;
    }

    private static TrainingBatch MakeBatch(int[] ids)
    {
        int seq = ids.Length;
        var labels = new int[seq];
        for (int s = 0; s < seq; s++)
            labels[s] = s < seq - 1 ? ids[s + 1] : -100;
        var attention = new float[seq];
        Array.Fill(attention, 1f);

        return new TrainingBatch(
            Tensor<int>.From(ids, 1, seq),
            Tensor<int>.From(labels, 1, seq),
            Tensor<float>.From(attention, 1, seq),
            seq);
    }
}
