using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.LoRA;
using SharpMind.Training.Loss;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// C2: pins the CPU training support matrix and the clarity of the
/// <see cref="NotSupportedException"/>s raised outside it.
///
/// Supported (full forward+backward): the three decoder FFN families the autograd
/// implements — dense (GPT-2), gated (LLaMA) and Mixture-of-Experts (Mixtral) —
/// across their usual attention/norm variants.
///
/// Not supported, and required to say exactly which feature is missing rather than
/// failing on some downstream shape check: Gemma-3 post-attention/post-FFN norms,
/// and LoRA adapters on non-dense/gated FFN projections.
/// </summary>
public sealed class TrainingSupportMatrixTests
{
    private const int Batch = 2;
    private const int Seq = 4;

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

    private static Tensor<int> DeterministicBatch(out Tensor<int> labels)
    {
        labels = Tensor<int>.From([0, 1, 2, 3, 4, 5, 6, 7], Batch, Seq);
        return Tensor<int>.From([3, 9, 12, 5, 17, 2, 8, 31], Batch, Seq);
    }

    private static (Transformer Model, ModelConfig Config, SharpMindConfig Sharp) Build(string preset, int seed = 9001)
    {
        var modelConfig = preset == "mixtral" ? SmallConfig with { NumExperts = 4, TopKExperts = 2 } : SmallConfig;
        var sharp = (preset switch
        {
            "gpt" => SharpMindConfig.Gpt,
            "llama" => SharpMindConfig.Llama,
            _ => SharpMindConfig.ForModel(modelConfig.NumHeads, modelConfig.NumKvHeads, "mixtral"),
        }) with { Hardware = HardwareTier.Scalar };

        var weights = ModelFactory.CreateForTraining(modelConfig, sharp);
        WeightInitializer.InitializeRandomly(weights, seed);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharp);
        if (preset == "mixtral")
            WeightInitializer.InitializeModelMoE(model, seed + 1013);
        return (model, modelConfig, sharp);
    }

    [Theory]
    [InlineData("gpt", "dense FFN / MHA / LayerNorm")]
    [InlineData("llama", "gated FFN / GQA / RMSNorm")]
    [InlineData("mixtral", "Mixture-of-Experts FFN / router")]
    public void SupportedPresets_RunAFullForwardBackwardStep(string preset, string _)
    {
        var (model, modelConfig, sharp) = Build(preset);
        using var _m = model;
        var parameters = model.Parameters().ToList();

        var mapping = GradientMappingFactory.Create(sharp);
        using var engine = new BackpropEngine(model, mapping, parameters, sharp);
        var loss = new CrossEntropyLoss();
        using var tokenIds = DeterministicBatch(out var labels);
        using var flatLabels = labels.Reshape(Batch * Seq);
        using var flatIds = tokenIds.Reshape(Batch * Seq);

        using var ctx = new ForwardContext();
        var logits = engine.ForwardAndRecord(ctx, tokenIds);
        using var logitsFlat = logits.Reshape(Batch * Seq, modelConfig.VocabSize);
        float l = loss.Compute(logitsFlat, flatLabels);
        using var dLogits = loss.Backward(logitsFlat, flatLabels);
        engine.Backward(ctx, dLogits, flatIds);

        Assert.True(float.IsFinite(l), $"{preset}: loss not finite");
        Assert.Contains(parameters, p => p.Grad.Data.ToArray().Any(g => g != 0f));
    }

    /// <summary>
    /// Gemma-3's extra post-attention / post-FFN norms have no backward. The engine
    /// must name the feature, not die on a shape mismatch somewhere deeper.
    /// </summary>
    [Fact]
    public void Gemma3PostNorms_ThrowAClearNotSupported()
    {
        var sharp = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(SmallConfig, sharp);
        WeightInitializer.InitializeRandomly(weights, 9001);
        // A Gemma-3 weight set carries this tensor; its presence builds a PostAttnNorm.
        weights.Blocks[0].PostNorm1W = Tensor<float>.Ones(SmallConfig.HiddenDim);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharp);

        var mapping = GradientMappingFactory.Create(sharp);
        using var engine = new BackpropEngine(model, mapping, model.Parameters().ToList(), sharp);
        using var tokenIds = DeterministicBatch(out _);
        using var ctx = new ForwardContext();

        var ex = Assert.Throws<NotSupportedException>(() => engine.ForwardAndRecord(ctx, tokenIds));
        Assert.Contains("Gemma-3 post-attention/post-FFN norms", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// LoRA on an MoE FFN projection has no implementation, but the adapter is still
    /// allowed on the attention projections of the same model; both must be explicit.
    /// </summary>
    [Fact]
    public void LoRA_OnMoEFfn_ThrowsAClearNotSupported_ButAttentionTargetsStillWork()
    {
        var (model, _, _) = Build("mixtral");
        using var _m = model;

        var ex = Assert.Throws<NotSupportedException>(
            () => new LoRAModel(model, new LoRAConfig { Rank = 2, TargetModules = ["down_proj"] }));
        Assert.Contains("LoRA on FFN kind", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not supported", ex.Message, StringComparison.OrdinalIgnoreCase);

        using var attentionOnly = new LoRAModel(model, new LoRAConfig { Rank = 2, TargetModules = ["q_proj", "v_proj"] });
        Assert.Equal(2 * SmallConfig.NumLayers, attentionOnly.AdapterCount);
    }
}
