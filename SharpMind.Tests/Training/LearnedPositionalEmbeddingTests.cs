using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;

namespace SharpMind.Tests.Training;

/// <summary>
/// Covers GPT-2 style learned positional embeddings
/// (<see cref="PositionalEncoding.Learned"/>): the [MaxSeqLen, HiddenDim]
/// position table is allocated and initialised by the factory, surfaced as the
/// <c>position_embedding</c> parameter, trained through the backprop engine, and
/// written/read as the <c>position_embd.weight</c> .SMM global tensor.
/// </summary>
public sealed class LearnedPositionalEmbeddingTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    private static readonly ModelConfig Config = new()
    {
        VocabSize = 32,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 16,
        PositionalEncoding = PositionalEncoding.Learned,
    };

    private static SharpMindConfig Sharp => SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

    private const int Batch = 2;
    private const int Seq = 4;

    private static Tensor<int> DeterministicBatch(out Tensor<int> labels)
    {
        labels = Tensor<int>.From(
            [0, 1, 2, 3, 4, 5, 6, 7], Batch, Seq);
        return Tensor<int>.From(
            [3, 9, 12, 5, 17, 2, 8, 31], Batch, Seq);
    }

    [Fact]
    public void Weights_ExposePositionTable_OfMaxSeqByHidden()
    {
        using var weights = ModelFactory.CreateForTraining(Config, Sharp);
        Assert.NotNull(weights.PositionEmbedding);
        Assert.Equal(Config.MaxSeqLen, weights.PositionEmbedding!.Shape.Rows);
        Assert.Equal(Config.HiddenDim, weights.PositionEmbedding.Shape.Cols);
    }

    [Fact]
    public void Parameters_IncludePositionEmbedding()
    {
        using var weights = ModelFactory.CreateForTraining(Config, Sharp);
        WeightInitializer.InitializeRandomly(weights, 5);
        using var model = ModelFactory.CreateTrainingTransformer(weights, Sharp);
        Assert.Contains("position_embedding", model.Parameters().Select(p => p.Name));
    }

    [Fact]
    public void Backward_AccumulatesPositionGradient()
    {
        using var weights = ModelFactory.CreateForTraining(Config, Sharp);
        WeightInitializer.InitializeRandomly(weights, 9001);
        using var model = ModelFactory.CreateTrainingTransformer(weights, Sharp);

        var parameters = model.Parameters().ToList();
        var posParam = parameters.First(p => p.Name == "position_embedding");
        var mapping = GradientMappingFactory.Create(Sharp);
        using var engine = new BackpropEngine(model, mapping, parameters, Sharp);

        using var tokenIds = DeterministicBatch(out var labels);
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = tokenIds.Reshape(Batch * Seq);
        var loss = new CrossEntropyLoss();

        using var ctx = new ForwardContext();
        var logits = engine.ForwardAndRecord(ctx, tokenIds);
        using var logitsFlat = logits.Reshape(Batch * Seq, Config.VocabSize);
        loss.Compute(logitsFlat, flatLabels);
        using var dLogits = loss.Backward(logitsFlat, flatLabels);
        engine.Backward(ctx, dLogits, flatIds);

        Assert.NotNull(posParam.Grad);
        bool nonzero = false;
        for (int i = 0; i < posParam.Grad.ElementCount; i++)
            if (!float.IsNaN(posParam.Grad.Data[i]) && MathF.Abs(posParam.Grad.Data[i]) > 1e-6f)
                nonzero = true;
        Assert.True(nonzero);
    }

    [Fact]
    public void Backprop_LossDescends_WithLearnedPositions()
    {
        using var weights = ModelFactory.CreateForTraining(Config, Sharp);
        WeightInitializer.InitializeRandomly(weights, 9001);
        using var model = ModelFactory.CreateTrainingTransformer(weights, Sharp);

        var parameters = model.Parameters().ToList();
        var mapping = GradientMappingFactory.Create(Sharp);
        using var engine = new BackpropEngine(model, mapping, parameters, Sharp);
        using var optimizer = new AdamW(parameters, lr: 0.05f, weightDecay: 0f);

        using var tokenIds = DeterministicBatch(out var labels);
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = tokenIds.Reshape(Batch * Seq);
        var loss = new CrossEntropyLoss();

        var losses = new List<float>();
        for (int step = 0; step < 15; step++)
        {
            optimizer.ZeroGrad();
            using var ctx = new ForwardContext();
            var logits = engine.ForwardAndRecord(ctx, tokenIds);
            using var logitsFlat = logits.Reshape(Batch * Seq, Config.VocabSize);
            losses.Add(loss.Compute(logitsFlat, flatLabels));
            using var dLogits = loss.Backward(logitsFlat, flatLabels);
            engine.Backward(ctx, dLogits, flatIds);
            optimizer.Update();
        }

        Assert.True(float.IsFinite(losses[^1]));
        Assert.True(losses[^1] < losses[0], $"backprop loss did not descend: {losses[0]:F4} → {losses[^1]:F4}");
    }

    [Fact]
    public void Export_Reload_RoundTripsPositionTable()
    {
        using var weights = ModelFactory.CreateForTraining(Config, Sharp);
        WeightInitializer.InitializeRandomly(weights, 4242);

        string smm = Path.Combine(_temp.Path, "learned.smm");
        SmmTrainingExporter.Export(weights, tokenizer: null, smm, new SmmWriteOptions { Source = "training" });

        var qOps = QuantizationFactory.Create(Sharp.ResolvedHardware);
        using var reloaded = ModelFactory.CreateWeights(Config, Sharp, qOps, smm, LoadMode.Full);
        reloaded.InitializeWeights();

        Assert.NotNull(reloaded.PositionEmbedding);
        Assert.Equal(weights.PositionEmbedding!.Shape, reloaded.PositionEmbedding!.Shape);
        for (int i = 0; i < weights.PositionEmbedding.ElementCount; i++)
        {
            float a = weights.PositionEmbedding.Data[i];
            float b = reloaded.PositionEmbedding.Data[i];
            float bound = 1e-6f * MathF.Max(1f, MathF.Abs(a));
            Assert.True(MathF.Abs(a - b) <= bound, $"position_embd[{i}] diff {MathF.Abs(a - b)} exceeds {bound}");
        }
    }
}