using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;

namespace SharpMind.Tests.Training;

/// <summary>
/// Verifies the full backprop path (<see cref="BackpropEngine"/>) against a
/// numeric finite-difference reference and that optimizer steps driven by
/// backprop gradients actually reduce the loss.
/// </summary>
public sealed class BackpropEngineTests
{
    private static readonly ModelConfig SmallConfig = new()
    {
        VocabSize = 32,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 512,
    };

    private static readonly ModelConfig MultiLayerConfig = SmallConfig with { NumLayers = 2 };

    private const int Batch = 2;
    private const int Seq = 4;

    private static Tensor<int> DeterministicBatch(out Tensor<int> labels)
    {
        labels = Tensor<int>.From(
            [0, 1, 2, 3, 4, 5, 6, 7], Batch, Seq);
        return Tensor<int>.From(
            [3, 9, 12, 5, 17, 2, 8, 31], Batch, Seq);
    }

    private static (Transformer Model, IReadOnlyList<Parameter> Params, SharpMindConfig Config) Fixture(ModelConfig modelConfig, SharpMindConfig sharpConfig, int seed = 9001)
    {
        var sc = sharpConfig with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(modelConfig, sc);
        WeightInitializer.InitializeRandomly(weights, seed);
        var model = ModelFactory.CreateTrainingTransformer(weights, sc);
        var parameters = model.Parameters().ToList();
        return (model, parameters, sc);
    }

    [Theory]
    [InlineData(0, nameof(SharpMindConfig.Gpt))]
    [InlineData(0, nameof(SharpMindConfig.Llama))]
    [InlineData(1, nameof(SharpMindConfig.Gpt))]
    [InlineData(1, nameof(SharpMindConfig.Llama))]
    public void Backward_GradientsMatchFiniteDifference(int configIdx, string configName)
    {
        var modelConfig = configIdx == 0 ? SmallConfig : MultiLayerConfig;
        var sharpConfig = configName == "Gpt" ? SharpMindConfig.Gpt : SharpMindConfig.Llama;
        var (model, parameters, config) = Fixture(modelConfig, sharpConfig);
        using var _m = model;

        var mapping = GradientMappingFactory.Create(config);
        using var engine = new BackpropEngine(model, mapping, parameters, config);

        using var tokenIds = DeterministicBatch(out var labels);
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = tokenIds.Reshape(Batch * Seq);

        using var ctx = new ForwardContext();
        var logits = engine.ForwardAndRecord(ctx, tokenIds);
        using var logitsFlat = logits.Reshape(Batch * Seq, modelConfig.VocabSize);

        var loss = new CrossEntropyLoss();
        loss.Compute(logitsFlat, flatLabels);
        using var dLogits = loss.Backward(logitsFlat, flatLabels);

        engine.Backward(ctx, dLogits, flatIds);

        var targets = SelectTargets(model);
        Assert.NotEmpty(targets);
        foreach (var target in targets)
        {
            var param = parameters.First(p => ReferenceEquals(p.Data, target.Data));
            AssertGradientsMatch(model, param, tokenIds, labels, loss, tolerance: 2e-2f);
        }
    }

    [Fact]
    public void Backprop_LossDescends()
    {
        var (model, parameters, config) = Fixture(SmallConfig, SharpMindConfig.Gpt);
        using var _m = model;

        var mapping = GradientMappingFactory.Create(config);
        using var engine = new BackpropEngine(model, mapping, parameters, config);
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
            using var logitsFlat = logits.Reshape(Batch * Seq, SmallConfig.VocabSize);
            float l = loss.Compute(logitsFlat, flatLabels);
            using var dLogits = loss.Backward(logitsFlat, flatLabels);
            engine.Backward(ctx, dLogits, flatIds);
            optimizer.Update();
            losses.Add(l);
        }

        Assert.True(float.IsFinite(losses[^1]));
        Assert.True(losses[^1] < losses[0], $"backprop loss did not descend: {losses[0]:F4} → {losses[^1]:F4}");
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    private static List<Parameter> SelectTargets(Transformer model)
    {
        var all = model.Parameters().ToList();
        var targets = new List<Parameter>();
        // embedding (weight-tied head)
        targets.Add(all[0]);
        // final norm
        targets.AddRange(model.FinalNorm.Parameters());
        // every block: both norms, all attention projections, and both ffn paths
        for (int i = 0; i < model.Config.NumLayers; i++)
        {
            var block = model.GetBlock(i)!;
            targets.AddRange(block.Norm1.Parameters());
            targets.AddRange(block.Norm2.Parameters());
            targets.AddRange(block.Attention.Wq.Parameters());
            targets.AddRange(block.Attention.Wk.Parameters());
            targets.AddRange(block.Attention.Wv.Parameters());
            targets.AddRange(block.Attention.Wo.Parameters());
            if (block.Ffn.W1Layer is not null)
            {
                targets.AddRange(block.Ffn.W1Layer.Parameters());
                targets.AddRange(block.Ffn.W2Layer!.Parameters());
            }
            else
            {
                targets.AddRange(block.Ffn.WGated!.Parameters());
                targets.AddRange(block.Ffn.WDown!.Parameters());
            }
        }
        return targets.Distinct().ToList();
    }

    private static void AssertGradientsMatch(
        Transformer model, Parameter param, Tensor<int> tokenIds, Tensor<int> labels, ILoss<int> loss, float tolerance)
    {
        const float h = 1e-3f;
        var data = param.Data.Data;
        var engineGrad = param.Grad.Data;

        int max = Math.Min(data.Length, 24); // spot-check a subset to keep the test fast
        for (int i = 0; i < max; i++)
        {
            float original = data[i];
            data[i] = original + h;
            float plus = LossFor(model, tokenIds, labels, loss);
            data[i] = original - h;
            float minus = LossFor(model, tokenIds, labels, loss);
            data[i] = original;
            float fd = (plus - minus) / (2 * h);
            float diff = Math.Abs(engineGrad[i] - fd);
            Assert.True(diff <= tolerance * (1f + Math.Abs(fd)),
                $"{param.Name}[{i}] backprop={engineGrad[i]:E3} fd={fd:E3} diff={diff:E3}");
        }
    }

    private static float LossFor(Transformer model, Tensor<int> tokenIds, Tensor<int> labels, ILoss<int> loss)
    {
        using var logits = model.Forward(tokenIds);
        using var flat = logits.Reshape(tokenIds.ElementCount, model.Config.VocabSize);
        using var flatLabels = labels.Reshape(tokenIds.ElementCount);
        return loss.Compute(flat, flatLabels);
    }
}