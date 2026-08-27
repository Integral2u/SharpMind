using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Inference;

public class MedusaGeneratorBuilder<T> : IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
{
    private const int DefaultNumHeads = 3;

    /// <summary>Number of greedy samples to collect for head calibration. 0 = skip calibration (start with random heads).</summary>
    public int CalibrationSamples { get; set; } = 0;

    /// <summary>SGD training steps over the calibration data.</summary>
    public int CalibrationSteps { get; set; } = 20;

    /// <summary>SGD learning rate for head training.</summary>
    public float CalibrationLearningRate { get; set; } = 0.01f;

    public IGenerator<T> CreateGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]? caches,
        int? seed = null,
        int? maxCacheLen = null)
    {
        var lmHeadWeight = model.LmHead ?? model.EmbeddingWeight;
        var medusaHeads = new MedusaHeads(
            numHeads: DefaultNumHeads,
            hiddenDim: model.Config.HiddenDim,
            vocabSize: model.Config.VocabSize,
            lmHeadWeight: lmHeadWeight,
            rawEmbedding: model.RawEmbedding,
            rawDtype: model.RawEmbeddingDtype,
            qOps: model.QOps);

        // Self-calibrate: train heads on the model's own greedy outputs.
        if (CalibrationSamples > 0)
        {
            medusaHeads.SelfCalibrate(model,
                numSamples: CalibrationSamples,
                trainingSteps: CalibrationSteps,
                learningRate: CalibrationLearningRate);
        }

        return new MedusaGenerator<T>(
            model, tokenizer, addBos, addEos,
            medusaHeads, caches, seed, maxCacheLen);
    }
}
