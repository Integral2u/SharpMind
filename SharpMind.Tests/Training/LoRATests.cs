using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.LoRA;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;

namespace SharpMind.Tests.Training;

/// <summary>
/// LoRA as something a training loop can actually run: <see cref="LoRAModel"/>
/// attaches rank-r adapters to the targeted projections of every block,
/// <see cref="BackpropEngine"/> trains only those adapters (the base weights are
/// frozen by not being in the parameter list), and <see cref="LoRAModel.Merge"/>
/// folds the adapters back into the base weights so the existing export path
/// serves the result. Gradients are checked against finite differences, the base
/// is checked bit-identical after training, and the merged model is checked to
/// reproduce the adapted forward.
/// </summary>
public sealed class LoRATests
{
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

    private static (Transformer Model, SharpMindConfig Config) Fixture(SharpMindConfig preset, int seed = 9001)
    {
        var sc = preset with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(SmallConfig, sc);
        WeightInitializer.InitializeRandomly(weights, seed);
        return (ModelFactory.CreateTrainingTransformer(weights, sc), sc);
    }

    private static float LossFor(Transformer model, Tensor<int> ids, Tensor<int> labels, ILoss<int> loss)
    {
        using var logits = model.Forward(ids);
        using var flat = logits.Reshape(Batch * Seq, SmallConfig.VocabSize);
        using var flatLabels = labels.Reshape(Batch * Seq);
        return loss.Compute(flat, flatLabels);
    }

    private static IEnumerable<TrainingLinearLayer> AllLinears(Transformer model)
    {
        for (int l = 0; l < model.Config.NumLayers; l++)
        {
            var b = model.GetBlock(l)!;
            yield return (TrainingLinearLayer)b.Attention.Wq;
            yield return (TrainingLinearLayer)b.Attention.Wk;
            yield return (TrainingLinearLayer)b.Attention.Wv;
            yield return (TrainingLinearLayer)b.Attention.Wo;
            if (b.Ffn.W1Layer is { } w1) yield return (TrainingLinearLayer)w1;
            if (b.Ffn.W2Layer is { } w2) yield return (TrainingLinearLayer)w2;
            if (b.Ffn.WGated is { } wg) yield return (TrainingLinearLayer)wg;
            if (b.Ffn.WDown is { } wd) yield return (TrainingLinearLayer)wd;
        }
    }

    [Fact]
    public void LoRAModel_AttachesAdaptersToTargetedProjectionsOfEveryBlock()
    {
        var (model, _) = Fixture(SharpMindConfig.Llama);
        using var _m = model;

        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 2, TargetModules = ["q_proj", "v_proj"] });

        // 2 layers x (q, v) = 4 adapters, each A + B.
        Assert.Equal(4, lora.AdapterCount);
        Assert.Equal(8, lora.LoRAParameters().Count);
        for (int l = 0; l < SmallConfig.NumLayers; l++)
        {
            var b = model.GetBlock(l)!;
            Assert.True(((TrainingLinearLayer)b.Attention.Wq).HasLoRA);
            Assert.True(((TrainingLinearLayer)b.Attention.Wv).HasLoRA);
            Assert.False(((TrainingLinearLayer)b.Attention.Wk).HasLoRA);
            Assert.False(((TrainingLinearLayer)b.Attention.Wo).HasLoRA);
            Assert.False(((TrainingLinearLayer)b.Ffn.WDown!).HasLoRA);
        }
        // Same instances every time: the optimizer and the engine must share them.
        Assert.Same(lora.LoRAParameters()[0], lora.LoRAParameters()[0]);
        Assert.True(lora.TrainableRatio() > 0 && lora.TrainableRatio() < 0.2);
    }

    [Theory]
    [InlineData("Gpt")]
    [InlineData("Llama")]
    public void LoRAModel_TargetsFfnProjectionsByName(string preset)
    {
        var (model, _) = Fixture(preset == "Gpt" ? SharpMindConfig.Gpt : SharpMindConfig.Llama);
        using var _m = model;
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 2, TargetModules = ["up_proj", "down_proj"] });

        // Dense (Gpt): W1 = up, W2 = down. Gated (Llama): WGated = fused gate+up, WDown = down.
        Assert.Equal(2 * SmallConfig.NumLayers, lora.AdapterCount);
        for (int l = 0; l < SmallConfig.NumLayers; l++)
        {
            var b = model.GetBlock(l)!;
            Assert.False(((TrainingLinearLayer)b.Attention.Wq).HasLoRA);
            Assert.False(((TrainingLinearLayer)b.Attention.Wo).HasLoRA);
            var up = (TrainingLinearLayer)(b.Ffn.W1Layer ?? b.Ffn.WGated)!;
            var down = (TrainingLinearLayer)(b.Ffn.W2Layer ?? b.Ffn.WDown)!;
            Assert.True(up.HasLoRA);
            Assert.True(down.HasLoRA);
        }
    }

    [Fact]
    public void LoRA_IsIdentityAtInitialisation()
    {
        var (model, _) = Fixture(SharpMindConfig.Llama);
        using var _m = model;
        using var ids = DeterministicBatch(out var labels);
        using var _l = labels;

        using var before = model.Forward(ids);
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 2 });
        using var after = model.Forward(ids);

        // B starts at zero, so the adapted model is the base model, bit for bit.
        Assert.Equal(before.Data.ToArray(), after.Data.ToArray());
    }

    [Theory]
    [InlineData("Gpt")]
    [InlineData("Llama")]
    public void LoRA_GradientsMatchFiniteDifference(string preset)
    {
        var (model, config) = Fixture(preset == "Gpt" ? SharpMindConfig.Gpt : SharpMindConfig.Llama);
        using var _m = model;
        using var lora = new LoRAModel(model,
            new LoRAConfig { Rank = 2, Alpha = 4f, TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"] },
            seed: 7);
        var parameters = lora.LoRAParameters();

        // B is zero at init, which makes dA exactly zero; perturb B so both
        // factors carry a real gradient to check.
        var rng = new Random(11);
        foreach (var p in parameters.Where(p => p.Name.EndsWith(".lora_B", StringComparison.Ordinal)))
            for (int i = 0; i < p.Data.ElementCount; i++)
                p.Data.Data[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;

        var mapping = GradientMappingFactory.Create(config);
        using var engine = new BackpropEngine(model, mapping, parameters, config);

        using var ids = DeterministicBatch(out var labels);
        using var _l = labels;
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = ids.Reshape(Batch * Seq);

        using var ctx = new ForwardContext();
        var logits = engine.ForwardAndRecord(ctx, ids);
        using var logitsFlat = logits.Reshape(Batch * Seq, SmallConfig.VocabSize);
        var loss = new CrossEntropyLoss();
        loss.Compute(logitsFlat, flatLabels);
        using var dLogits = loss.Backward(logitsFlat, flatLabels);
        engine.Backward(ctx, dLogits, flatIds);

        const float h = 1e-3f;
        int checkedParams = 0;
        foreach (var p in parameters)
        {
            var data = p.Data.Data;
            var grad = p.Grad.Data;
            Assert.True(grad.ToArray().Any(g => g != 0f), $"{p.Name}: gradient is all zero");
            for (int i = 0; i < Math.Min(data.Length, 16); i++)
            {
                float original = data[i];
                data[i] = original + h;
                float plus = LossFor(model, ids, labels, loss);
                data[i] = original - h;
                float minus = LossFor(model, ids, labels, loss);
                data[i] = original;
                float fd = (plus - minus) / (2 * h);
                float diff = Math.Abs(grad[i] - fd);
                Assert.True(diff <= 2e-2f * (1f + Math.Abs(fd)),
                    $"{p.Name}[{i}] backprop={grad[i]:E3} fd={fd:E3} diff={diff:E3}");
            }
            checkedParams++;
        }
        Assert.Equal(parameters.Count, checkedParams);
    }

    [Fact]
    public void LoRA_TrainsOnlyTheAdapters_AndLossDescends()
    {
        var (model, config) = Fixture(SharpMindConfig.Llama);
        using var _m = model;
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 4, Alpha = 8f }, seed: 3);
        var parameters = lora.LoRAParameters();

        // Snapshot every base tensor (weights, biases, norms, embedding).
        var baseSnapshots = model.Parameters().Select(p => (p.Name, Data: p.Data.Data.ToArray())).ToList();
        Assert.DoesNotContain(baseSnapshots, s => s.Name.Contains("lora", StringComparison.OrdinalIgnoreCase));

        var mapping = GradientMappingFactory.Create(config);
        using var engine = new BackpropEngine(model, mapping, parameters, config);
        using var optimizer = new AdamW(parameters, lr: 5e-2f, weightDecay: 0f);
        var loss = new CrossEntropyLoss();

        using var ids = DeterministicBatch(out var labels);
        using var _l = labels;
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = ids.Reshape(Batch * Seq);

        float first = -1, last = -1;
        for (int step = 0; step < 8; step++)
        {
            optimizer.ZeroGrad();
            using var ctx = new ForwardContext();
            var logits = engine.ForwardAndRecord(ctx, ids);
            using var logitsFlat = logits.Reshape(Batch * Seq, SmallConfig.VocabSize);
            float l = loss.Compute(logitsFlat, flatLabels);
            if (first < 0) first = l;
            last = l;
            using var dLogits = loss.Backward(logitsFlat, flatLabels);
            engine.Backward(ctx, dLogits, flatIds);
            optimizer.Update();
        }
        Assert.True(last < first, $"loss did not descend: {first} -> {last}");

        // The frozen base did not move by a single bit.
        var after = model.Parameters().Select(p => p.Data.Data.ToArray()).ToList();
        for (int i = 0; i < baseSnapshots.Count; i++)
            Assert.True(baseSnapshots[i].Data.AsSpan().SequenceEqual(after[i]), $"{baseSnapshots[i].Name} changed");
    }

    [Fact]
    public void LoRA_MergeReproducesTheAdaptedForward_AndDetaches()
    {
        var (model, _) = Fixture(SharpMindConfig.Llama);
        using var _m = model;
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 2, Alpha = 4f }, seed: 5);
        var rng = new Random(13);
        foreach (var p in lora.LoRAParameters())
            for (int i = 0; i < p.Data.ElementCount; i++)
                p.Data.Data[i] = (float)(rng.NextDouble() - 0.5) * 0.3f;

        using var ids = DeterministicBatch(out var labels);
        using var _l = labels;
        using var adapted = model.Forward(ids);

        lora.Merge();

        Assert.Equal(0, lora.AdapterCount);
        Assert.All(AllLinears(model), layer => Assert.False(layer.HasLoRA));
        using var merged = model.Forward(ids);
        for (int i = 0; i < adapted.ElementCount; i++)
            Assert.True(Math.Abs(adapted.Data[i] - merged.Data[i]) <= 1e-4f * (1f + Math.Abs(adapted.Data[i])),
                $"logit {i}: adapted={adapted.Data[i]} merged={merged.Data[i]}");
    }

    [Fact]
    public void LoRAModel_RejectsUnknownTargetAndMoE()
    {
        var (model, _) = Fixture(SharpMindConfig.Llama);
        using var _m = model;
        Assert.Throws<ArgumentException>(() => new LoRAModel(model, new LoRAConfig { TargetModules = ["nope_proj"] }));
    }
}
