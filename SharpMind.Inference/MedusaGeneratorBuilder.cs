using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Inference;

/// <summary>Where the builder got its draft heads from.</summary>
public enum MedusaHeadLoadResult
{
    /// <summary>No head file was configured; the heads are random and uncalibrated.</summary>
    Skipped,

    /// <summary>Heads were read from <see cref="MedusaGeneratorBuilder{T}.MedusaHeadsPath"/>.</summary>
    Loaded,

    /// <summary>Heads were calibrated in this call, and saved when a path was configured.</summary>
    Calibrated,
}

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

    /// <summary>
    /// Optional sidecar file for calibrated heads. If it exists the heads are
    /// loaded and calibration is skipped; if it is missing and
    /// <see cref="CalibrationSamples"/> is set, calibration runs and the result is
    /// saved there. A file that does not match the model is recalibrated when
    /// samples are configured, and otherwise throws.
    /// </summary>
    public string? MedusaHeadsPath { get; set; }

    /// <summary>Write calibrated heads to <see cref="MedusaHeadsPath"/>. Default true.</summary>
    public bool SaveCalibratedHeads { get; set; } = true;

    /// <summary>
    /// Optional caller-supplied model identity (for example a file path or hash).
    /// When set, loading a head file with a different fingerprint fails loudly.
    /// </summary>
    public string? CalibrationFingerprint { get; set; }

    /// <summary>Cancels collection or training between steps.</summary>
    public CancellationToken CalibrationCancellationToken { get; set; }

    /// <summary>Optional sink for calibration progress; invoked on the calling thread.</summary>
    public IProgress<MedusaCalibrationProgress>? CalibrationProgress { get; set; }

    /// <summary>Result of the last calibration in <see cref="CreateGenerator"/>, or null when calibration was skipped.</summary>
    public MedusaCalibrationResult? LastCalibration { get; private set; }

    /// <summary>Where the last <see cref="CreateGenerator"/> call got its heads.</summary>
    public MedusaHeadLoadResult? LastHeadLoad { get; private set; }

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

        try
        {
            LastCalibration = null;
            LastHeadLoad = MedusaHeadLoadResult.Skipped;

            // A calibrated sidecar avoids retraining on every generator. A file
            // that no longer matches the model is only ignored when the caller
            // also allowed calibration.
            bool loadedHeads = false;
            if (!string.IsNullOrWhiteSpace(MedusaHeadsPath))
            {
                if (File.Exists(MedusaHeadsPath))
                {
                    try
                    {
                        var snapshot = MedusaHeadFile.Load(MedusaHeadsPath);
                        medusaHeads.LoadWeights(snapshot, CalibrationFingerprint);
                        LastCalibration = snapshot.Calibration;
                        LastHeadLoad = MedusaHeadLoadResult.Loaded;
                        loadedHeads = true;
                    }
                    catch (Exception ex)
                        when ((ex is InvalidDataException or EndOfStreamException)
                              && CalibrationSamples > 0)
                    {
                        // Stale, truncated, or mismatched heads: recalibrate.
                    }
                }
                else if (CalibrationSamples <= 0)
                {
                    throw new FileNotFoundException(
                        $"Medusa head file not found and CalibrationSamples is 0: {MedusaHeadsPath}",
                        MedusaHeadsPath);
                }
            }

            // Self-calibrate: train heads on the model's own greedy outputs.
            if (!loadedHeads && CalibrationSamples > 0)
            {
                IReadOnlyList<int>? seedTokens = null;
                if (!string.IsNullOrWhiteSpace(CalibrationPrompt))
                    seedTokens = tokenizer.Encode(CalibrationPrompt, addBos: true, addEos: false);

                LastCalibration = medusaHeads.SelfCalibrate(model,
                    numSamples: CalibrationSamples,
                    trainingSteps: CalibrationSteps,
                    learningRate: CalibrationLearningRate,
                    seedTokens: seedTokens,
                    cancellationToken: CalibrationCancellationToken,
                    progress: CalibrationProgress);
                LastHeadLoad = MedusaHeadLoadResult.Calibrated;

                if (SaveCalibratedHeads && !string.IsNullOrWhiteSpace(MedusaHeadsPath))
                {
                    MedusaHeadFile.Save(
                        MedusaHeadsPath, medusaHeads, LastCalibration, CalibrationFingerprint);
                }
            }

            return new MedusaGenerator<T>(
                model, tokenizer, addBos, addEos,
                medusaHeads, caches, seed, maxCacheLen);
        }
        catch
        {
            // The generator owns and disposes its heads; until it exists, a throw
            // must not leak them.
            medusaHeads.Dispose();
            throw;
        }
    }
}
