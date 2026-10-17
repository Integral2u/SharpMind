using SharpMind.Model.Config;
using SharpMind.Training.LoRA;

namespace SharpMind.Training;

/// <summary>
/// A rough training-memory budget for a model configuration, split so the dominant
/// term is obvious. Training holds the F32 master weights plus, per trainable
/// element, a gradient and (with Adam) two optimizer moments; activations add a
/// per-token, per-layer allowance.
/// </summary>
public sealed record TrainingMemoryEstimate(
    long BaseWeightBytes,
    long TrainableWeightBytes,
    long GradientBytes,
    long OptimizerStateBytes,
    long ActivationBytes)
{
    /// <summary>Resident weights, gradients, optimizer state and activations, summed.</summary>
    public long TotalBytes => BaseWeightBytes + TrainableWeightBytes + GradientBytes + OptimizerStateBytes + ActivationBytes;

    public double TotalMiB => TotalBytes / 1048576.0;
    public double TotalGiB => TotalBytes / 1073741824.0;
}

/// <summary>
/// Estimates how much memory a training run of a given <see cref="ModelConfig"/>
/// needs, for a full fine-tune or a LoRA fine-tune, so a config can be chosen
/// before a run OOMs. Sits next to <see cref="ModelSizer"/>: that picks a config,
/// this says whether the chosen one fits. It is an estimate — the activation term
/// in particular is a heuristic, not a measured peak.
/// </summary>
public static class TrainingMemoryEstimator
{
    private const int BytesPerFloat = 4;

    // Saved-activation slots per layer per token (normed inputs, attention/FFN
    // intermediates the backward pass reads). A heuristic; real peaks also include
    // transient buffers, which is why the estimate is documented as approximate.
    private const int ActivationSlotsPerLayerPerToken = 8;

    /// <summary>Estimates a full fine-tune: every parameter is trainable.</summary>
    public static TrainingMemoryEstimate EstimateFull(
        ModelConfig config, int batchSize = 1, int sequenceLength = 0, bool adam = true)
    {
        ArgumentNullException.ThrowIfNull(config);
        int seq = ResolveSequenceLength(config, sequenceLength);
        ValidateBatch(batchSize);

        long parameterBytes = ModelSizer.CountParameters(config) * BytesPerFloat;
        return new TrainingMemoryEstimate(
            BaseWeightBytes: parameterBytes,
            TrainableWeightBytes: 0,
            GradientBytes: parameterBytes,
            OptimizerStateBytes: adam ? parameterBytes * 2 : 0,
            ActivationBytes: ActivationBytes(config, batchSize, seq));
    }

    /// <summary>Estimates a LoRA fine-tune: the base is resident but frozen, only adapters train.</summary>
    public static TrainingMemoryEstimate EstimateLora(
        ModelConfig config, LoRAConfig lora, int batchSize = 1, int sequenceLength = 0, bool adam = true)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(lora);
        int seq = ResolveSequenceLength(config, sequenceLength);
        ValidateBatch(batchSize);

        long baseBytes = ModelSizer.CountParameters(config) * BytesPerFloat;
        long adapterBytes = CountAdapterParameters(config, lora) * BytesPerFloat;
        return new TrainingMemoryEstimate(
            BaseWeightBytes: baseBytes,
            TrainableWeightBytes: adapterBytes,
            GradientBytes: adapterBytes,
            OptimizerStateBytes: adam ? adapterBytes * 2 : 0,
            ActivationBytes: ActivationBytes(config, batchSize, seq));
    }

    /// <summary>Trainable parameter count a LoRA config adds across every targeted projection.</summary>
    public static long CountAdapterParameters(ModelConfig config, LoRAConfig lora)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(lora);

        var targets = new HashSet<string>(lora.TargetModules ?? [], StringComparer.OrdinalIgnoreCase);
        int hidden = config.HiddenDim;
        int headDim = hidden / Math.Max(1, config.NumHeads);
        int qDim = hidden;                                   // NumHeads * headDim
        int kvDim = Math.Max(1, config.NumKvHeads) * headDim;
        int ffn = config.FfnDim;
        long r = lora.Rank;

        (string Name, int In, int Out)[] projections =
        [
            ("q_proj", hidden, qDim),
            ("k_proj", hidden, kvDim),
            ("v_proj", hidden, kvDim),
            ("o_proj", qDim, hidden),
            ("up_proj", hidden, ffn),
            ("gate_proj", hidden, ffn),
            ("down_proj", ffn, hidden),
        ];

        long perLayer = 0;
        foreach (var (name, inFeatures, outFeatures) in projections)
            if (targets.Contains(name))
                perLayer += r * (inFeatures + outFeatures);

        // gate_proj and up_proj are one fused adapter on a gated FFN: count it once.
        if (targets.Contains("gate_proj") && targets.Contains("up_proj"))
            perLayer -= r * (hidden + ffn);

        return perLayer * config.NumLayers;
    }

    private static long ActivationBytes(ModelConfig config, int batchSize, int sequenceLength)
    {
        long perToken = (long)config.HiddenDim * ActivationSlotsPerLayerPerToken * config.NumLayers;
        long hiddenActivations = (long)batchSize * sequenceLength * perToken;
        long attentionScores = (long)batchSize * config.NumHeads * sequenceLength * sequenceLength;
        return (hiddenActivations + attentionScores) * BytesPerFloat;
    }

    private static int ResolveSequenceLength(ModelConfig config, int sequenceLength)
        => sequenceLength > 0 ? sequenceLength : config.MaxSeqLen;

    private static void ValidateBatch(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
    }
}
