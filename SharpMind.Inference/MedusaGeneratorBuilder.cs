using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Inference;

public class MedusaGeneratorBuilder<T> : IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
{
    public const int DefaultNumHeads = 3;

    /// <summary>Number of draft heads. More heads propose a longer draft per round.</summary>
    public int NumHeads { get; set; } = DefaultNumHeads;

    /// <summary>Number of greedy samples to collect for head calibration. 0 = skip calibration (start with random heads).</summary>
    public int CalibrationSamples { get; set; } = 0;

    /// <summary>
    /// Optional prompt to start calibration from. A real prompt makes the heads
    /// match the text they will draft; null starts at token 0.
    /// </summary>
    public string? CalibrationPrompt { get; set; }

    /// <summary>SGD training steps over the calibration data.</summary>
    public int CalibrationSteps { get; set; } = 20;

    /// <summary>SGD learning rate for head training.</summary>
    public float CalibrationLearningRate { get; set; } = 0.01f;

    /// <summary>Result of the last calibration in <see cref="CreateGenerator"/>, or null when calibration was skipped.</summary>
    public MedusaCalibrationResult? LastCalibration { get; private set; }

    public IGenerator<T> CreateGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]? caches,
        int? seed = null,
        int? maxCacheLen = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(NumHeads);
        ArgumentOutOfRangeException.ThrowIfNegative(CalibrationSamples);
        if (CalibrationSamples > 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(CalibrationSteps);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(CalibrationLearningRate);
        }

        var lmHeadWeight = model.LmHead ?? model.EmbeddingWeight;
        var medusaHeads = new MedusaHeads(
            numHeads: NumHeads,
            hiddenDim: model.Config.HiddenDim,
            vocabSize: model.Config.VocabSize,
            lmHeadWeight: lmHeadWeight,
            rawEmbedding: model.RawEmbedding,
            rawDtype: model.RawEmbeddingDtype,
            qOps: model.QOps);

        // Self-calibrate: train heads on the model's own greedy outputs.
        LastCalibration = null;
        if (CalibrationSamples > 0)
        {
            IReadOnlyList<int>? seedTokens = null;
            if (!string.IsNullOrWhiteSpace(CalibrationPrompt))
                seedTokens = tokenizer.Encode(CalibrationPrompt, addBos: true, addEos: false);

            LastCalibration = medusaHeads.SelfCalibrate(model,
                numSamples: CalibrationSamples,
                trainingSteps: CalibrationSteps,
                learningRate: CalibrationLearningRate,
                seedTokens: seedTokens);
        }

        return new MedusaGenerator<T>(
            model, tokenizer, addBos, addEos,
            medusaHeads, caches, seed, maxCacheLen);
    }
}
