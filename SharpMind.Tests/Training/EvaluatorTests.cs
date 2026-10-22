using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Loss;
using Xunit;

namespace SharpMind.Tests.Training;

public sealed class EvaluatorTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 16,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 16,
    };

    [Fact]
    public void Evaluate_WeightsBatchLossesByRealTokenCount()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        var loss = new CrossEntropyLoss(ignoreId: -100);

        using var small = MakeBatch([3, 4, 5, 6], [4, 5, -100, -100], realTokens: 2);
        using var large = MakeBatch([7, 8, 9, 10], [8, 9, 10, -100], realTokens: 3);

        float smallLoss = BatchLoss(model, loss, small);
        float largeLoss = BatchLoss(model, loss, large);
        float expected = (smallLoss * 2 + largeLoss * 3) / 5f;

        var evaluator = new Evaluator(model, loss);
        var (actual, _) = evaluator.Evaluate([small, large]);

        Assert.Equal(expected, actual, precision: 5);
    }

    private static float BatchLoss(Transformer model, ILoss<int> loss, TrainingBatch batch)
    {
        using var logits = model.Forward(batch.TokenIds);
        int rows = batch.TokenIds.Shape[0] * batch.TokenIds.Shape[1];
        using var flatLogits = logits.Reshape(rows, model.Config.VocabSize);
        using var flatLabels = batch.Labels.Reshape(rows);
        return loss.Compute(flatLogits, flatLabels);
    }

    private static TrainingBatch MakeBatch(int[] ids, int[] labels, int realTokens)
    {
        int seq = ids.Length;
        var attention = new float[seq];
        Array.Fill(attention, 1f);
        return new TrainingBatch(
            Tensor<int>.From(ids, 1, seq),
            Tensor<int>.From(labels, 1, seq),
            Tensor<float>.From(attention, 1, seq),
            realTokens);
    }
}
