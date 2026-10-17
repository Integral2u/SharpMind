using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.LoRA;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// C5: the training memory estimator next to <see cref="ModelSizer"/> reports a
/// coherent budget — the parts sum to the total, Adam adds two moments per
/// trainable element, and a LoRA run's trainable/gradient/optimizer terms scale
/// with the ranked adapters rather than the whole model.
/// </summary>
public sealed class TrainingMemoryEstimatorTests
{
    private static readonly ModelConfig Config = new()
    {
        VocabSize = 512,
        HiddenDim = 256,
        NumLayers = 4,
        NumHeads = 8,
        NumKvHeads = 8,
        FfnDim = 1024,
        MaxSeqLen = 128,
    };

    [Fact]
    public void FullFineTune_SplitsIntoWeightsGradientsAndTwoAdamMoments()
    {
        var e = TrainingMemoryEstimator.EstimateFull(Config, batchSize: 2, sequenceLength: 64);

        long paramBytes = ModelSizer.CountParameters(Config) * 4;
        Assert.Equal(paramBytes, e.BaseWeightBytes);
        Assert.Equal(0, e.TrainableWeightBytes);
        Assert.Equal(paramBytes, e.GradientBytes);
        Assert.Equal(paramBytes * 2, e.OptimizerStateBytes);
        Assert.True(e.ActivationBytes > 0);
        Assert.Equal(
            e.BaseWeightBytes + e.TrainableWeightBytes + e.GradientBytes + e.OptimizerStateBytes + e.ActivationBytes,
            e.TotalBytes);
    }

    [Fact]
    public void FullFineTune_WithoutAdam_HasNoOptimizerState()
    {
        var e = TrainingMemoryEstimator.EstimateFull(Config, adam: false);
        Assert.Equal(0, e.OptimizerStateBytes);
    }

    [Fact]
    public void Lora_TrainsOnlyTheAdapters_SoGradientsAndOptimizerScaleWithRank()
    {
        var lora = new LoRAConfig { Rank = 8, TargetModules = ["q_proj", "v_proj"] };
        var e = TrainingMemoryEstimator.EstimateLora(Config, lora, batchSize: 2, sequenceLength: 64);

        long paramBytes = ModelSizer.CountParameters(Config) * 4;
        long adapterParams = TrainingMemoryEstimator.CountAdapterParameters(Config, lora);
        long adapterBytes = adapterParams * 4;

        // Base is resident but frozen: no gradient or optimizer state for it.
        Assert.Equal(paramBytes, e.BaseWeightBytes);
        Assert.Equal(adapterBytes, e.TrainableWeightBytes);
        Assert.Equal(adapterBytes, e.GradientBytes);
        Assert.Equal(adapterBytes * 2, e.OptimizerStateBytes);
        Assert.True(adapterBytes < paramBytes, "LoRA trainable bytes should be far below the full model");

        var full = TrainingMemoryEstimator.EstimateFull(Config, batchSize: 2, sequenceLength: 64);
        Assert.True(e.TotalBytes < full.TotalBytes, $"LoRA total {e.TotalMiB:F1} MiB should beat full {full.TotalMiB:F1} MiB");
    }

    [Fact]
    public void Lora_HigherRankCostsMore()
    {
        var small = TrainingMemoryEstimator.EstimateLora(Config, new LoRAConfig { Rank = 4 });
        var large = TrainingMemoryEstimator.EstimateLora(Config, new LoRAConfig { Rank = 32 });
        Assert.True(large.TrainableWeightBytes > small.TrainableWeightBytes);
    }

    [Fact]
    public void Lora_FusedGateAndUp_AreCountedOnce()
    {
        var both = TrainingMemoryEstimator.CountAdapterParameters(Config, new LoRAConfig { Rank = 8, TargetModules = ["gate_proj", "up_proj"] });
        var upOnly = TrainingMemoryEstimator.CountAdapterParameters(Config, new LoRAConfig { Rank = 8, TargetModules = ["up_proj"] });
        // A gated FFN has one fused gate+up adapter, so requesting both must not double it.
        Assert.Equal(upOnly, both);
    }

    [Fact]
    public void Activations_ScaleWithBatchAndSequence()
    {
        var b1 = TrainingMemoryEstimator.EstimateFull(Config, batchSize: 1, sequenceLength: 64);
        var b2 = TrainingMemoryEstimator.EstimateFull(Config, batchSize: 2, sequenceLength: 64);
        var s128 = TrainingMemoryEstimator.EstimateFull(Config, batchSize: 1, sequenceLength: 128);

        Assert.True(b2.ActivationBytes > b1.ActivationBytes);
        Assert.True(s128.ActivationBytes > b1.ActivationBytes);
    }

    [Fact]
    public void LargerModel_NeedsMoreMemory()
    {
        var small = TrainingMemoryEstimator.EstimateFull(Config);
        var big = TrainingMemoryEstimator.EstimateFull(Config with { NumLayers = Config.NumLayers * 2 });
        Assert.True(big.TotalBytes > small.TotalBytes);
    }

    [Fact]
    public void NonPositiveBatch_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TrainingMemoryEstimator.EstimateFull(Config, batchSize: 0));
    }
}
