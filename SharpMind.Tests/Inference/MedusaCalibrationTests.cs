using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Medusa only changes throughput when its draft heads are trained. These tests
/// pin the self-calibration contract (samples, steps, finite loss, IsTrained) and
/// the guarantee that calibration never changes the accepted token stream.
/// </summary>
public sealed class MedusaCalibrationTests
{
    private static ModelConfig Cfg => new()
    {
        Architecture = "lfm2",
        VocabSize = 256,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        LayerKvHeads = [2, 0],
        FfnDim = 64,
        ShortConvCacheLength = 3,
        MaxSeqLen = 1024,
    };

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string>();
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);
    }

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.ForModel(
            Cfg.NumHeads, Cfg.NumKvHeads, Cfg.Architecture, HardwareTier.Scalar);
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static async Task<int[]> Generate<T>(IGenerator<T> generator, int[] prompt, int maxNew)
        where T : IKVCacheBuilder, new()
    {
        var generation = new GenerationConfig
        {
            MaxNewTokens = maxNew,
            RepetitionPenalty = 1f,
            Stream = false,
        };

        await foreach (var _ in generator.GenerateFromTokensAsync(prompt, SamplingConfig.Greedy, generation))
        {
        }

        return generator.CurrentGeneratedIds!.ToArray();
    }

    [Fact]
    public void Calibrate_OverfitsOneSample()
    {
        // A single (hidden, target) pair gives the optimiser an unambiguous
        // objective: a correct gradient must drive this loss toward zero.
        using var lmHead = new Tensor<float>(8, 4);
        for (int i = 0; i < lmHead.Data.Length; i++)
            lmHead.Data[i] = ((i % 7) - 3) * 0.1f;

        using var heads = new MedusaHeads(
            numHeads: 1, hiddenDim: 4, vocabSize: 8, lmHeadWeight: lmHead);
        using var hidden = new Tensor<float>(1, 4);
        hidden.Data[0] = 1.0f;
        hidden.Data[1] = 0.5f;
        hidden.Data[2] = -0.25f;
        hidden.Data[3] = 0.75f;
        int[][] targets = [[3]];

        float loss = heads.Calibrate(hidden, targets, learningRate: 0.1f, steps: 200);

        Assert.True(heads.IsTrained);
        Assert.True(loss < 1.0f, $"calibration did not overfit: loss={loss:F4}");
    }

    [Fact]
    public void SelfCalibrate_TrainsHeadsAndReturnsFiniteLoss()
    {
        using var model = BuildModel();
        using var heads = new MedusaHeads(
            numHeads: 2,
            hiddenDim: Cfg.HiddenDim,
            vocabSize: Cfg.VocabSize,
            lmHeadWeight: model.EmbeddingWeight,
            rawEmbedding: model.RawEmbedding,
            rawDtype: model.RawEmbeddingDtype,
            qOps: model.QOps);

        Assert.False(heads.IsTrained);
        Assert.Null(heads.LastCalibration);

        MedusaCalibrationResult result = heads.SelfCalibrate(
            model, numSamples: 8, trainingSteps: 3, learningRate: 0.02f);

        Assert.True(heads.IsTrained);
        Assert.Equal(8, result.Samples);
        Assert.Equal(3, result.Steps);
        Assert.True(float.IsFinite(result.FinalLoss));
        Assert.True(result.FinalLoss >= 0f);
        Assert.True(heads.LastCalibration.HasValue);
        Assert.Equal(result, heads.LastCalibration.Value);
    }

    [Fact]
    public async Task CalibratedGenerator_MatchesGreedyAndReportsStats()
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();
        int[] prompt = [100, 101, 102, 103];

        var builder = new MedusaGeneratorBuilder<KVCacherBuilder>
        {
            NumHeads = 2,
            CalibrationSamples = 8,
            CalibrationPrompt = "abc",
            CalibrationSteps = 3,
            CalibrationLearningRate = 0.02f,
        };
        using var medusaGenerator = builder.CreateGenerator(
            model, tokenizer, addBos: false, addEos: false, caches: null, seed: 7);

        Assert.True(builder.LastCalibration.HasValue);
        Assert.Equal(8, builder.LastCalibration.Value.Samples);

        int[] medusaIds = await Generate(medusaGenerator, prompt, maxNew: 8);
        var medusa = Assert.IsType<MedusaGenerator<KVCacherBuilder>>(medusaGenerator);

        Assert.Equal(medusaIds.Length, medusa.Stats.GeneratedTokens);
        Assert.True(medusa.Stats.Rounds > 0);
        Assert.True(medusa.Stats.DraftTokensProposed > 0);
        Assert.InRange(medusa.Stats.DraftAcceptanceRate, 0.0, 1.0);
        Assert.True(medusa.Stats.ForwardsPerToken > 0);
        Assert.True(medusa.Calibration.HasValue);

        using var standard = new StandardGeneratorBuilder<KVCacherBuilder>().CreateGenerator(
            model, tokenizer, addBos: false, addEos: false, caches: null, seed: 7);
        int[] standardIds = await Generate(standard, prompt, maxNew: 8);

        // Medusa can only change throughput. Calibration must not change the
        // tokens the model actually accepts.
        Assert.Equal(standardIds, medusaIds);
    }

    [Fact]
    public void SelfCalibrate_ReportsProgressAndHonoursCancellation()
    {
        using var model = BuildModel();
        using var heads = new MedusaHeads(
            numHeads: 2,
            hiddenDim: Cfg.HiddenDim,
            vocabSize: Cfg.VocabSize,
            lmHeadWeight: model.EmbeddingWeight,
            rawEmbedding: model.RawEmbedding,
            rawDtype: model.RawEmbeddingDtype,
            qOps: model.QOps);

        var reports = new List<MedusaCalibrationProgress>();
        using var cts = new CancellationTokenSource();
        var progress = new InlineProgress<MedusaCalibrationProgress>(p =>
        {
            reports.Add(p);
            if (p.Phase == MedusaCalibrationPhase.Training && p.Completed == 1)
                cts.Cancel();
        });

        Assert.Throws<OperationCanceledException>(() => heads.SelfCalibrate(
            model, numSamples: 8, trainingSteps: 5, learningRate: 0.02f,
            cancellationToken: cts.Token, progress: progress));

        Assert.False(heads.IsTrained);
        Assert.Null(heads.LastCalibration);

        Assert.NotEmpty(reports);
        Assert.Equal(MedusaCalibrationPhase.Collecting, reports[0].Phase);
        Assert.True(reports[0].Total > 0);
        Assert.Null(reports[0].Loss);
        Assert.Contains(reports, r => r.Phase == MedusaCalibrationPhase.Training && r.Loss.HasValue);
        Assert.Equal(
            reports.Count(r => r.Phase == MedusaCalibrationPhase.Training),
            reports.Where(r => r.Phase == MedusaCalibrationPhase.Training)
                   .Select(r => r.Completed)
                   .Distinct()
                   .Count());
    }

    [Fact]
    public void HeadFile_RoundTripsWeightsAndCalibration()
    {
        string path = TempHeadPath();
        try
        {
            using var model = BuildModel();
            using var heads = new MedusaHeads(
                numHeads: 2,
                hiddenDim: Cfg.HiddenDim,
                vocabSize: Cfg.VocabSize,
                lmHeadWeight: model.EmbeddingWeight,
                rawEmbedding: model.RawEmbedding,
                rawDtype: model.RawEmbeddingDtype,
                qOps: model.QOps);

            var calibration = heads.SelfCalibrate(
                model, numSamples: 4, trainingSteps: 2, learningRate: 0.02f);
            MedusaHeadFile.Save(path, heads, calibration, fingerprint: "model-a");

            var snapshot = MedusaHeadFile.Load(path);

            Assert.Equal(heads.NumHeads, snapshot.NumHeads);
            Assert.Equal(heads.HiddenDim, snapshot.HiddenDim);
            Assert.Equal(heads.VocabSize, snapshot.VocabSize);
            Assert.True(snapshot.IsTrained);
            Assert.Equal(calibration, snapshot.Calibration);
            Assert.Equal("model-a", snapshot.Fingerprint);

            using var restored = new MedusaHeads(
                numHeads: 2,
                hiddenDim: Cfg.HiddenDim,
                vocabSize: Cfg.VocabSize,
                lmHeadWeight: model.EmbeddingWeight);
            restored.LoadWeights(snapshot, expectedFingerprint: "model-a");

            Assert.True(restored.IsTrained);
            Assert.Equal(calibration, restored.LastCalibration);
            for (int h = 0; h < heads.NumHeads; h++)
            {
                Assert.Equal(
                    heads.GetHeadWeight(h).Data.ToArray(),
                    restored.GetHeadWeight(h).Data.ToArray());
                Assert.Equal(
                    heads.GetHeadBias(h).Data.ToArray(),
                    restored.GetHeadBias(h).Data.ToArray());
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void HeadFile_RejectsMismatchedModel()
    {
        string path = TempHeadPath();
        try
        {
            using var model = BuildModel();
            using var heads = new MedusaHeads(
                numHeads: 2,
                hiddenDim: Cfg.HiddenDim,
                vocabSize: Cfg.VocabSize,
                lmHeadWeight: model.EmbeddingWeight);
            MedusaHeadFile.Save(path, heads, calibration: null, fingerprint: "model-a");
            var snapshot = MedusaHeadFile.Load(path);

            using var otherModel = new MedusaHeads(
                numHeads: 2,
                hiddenDim: Cfg.HiddenDim + 8,
                vocabSize: Cfg.VocabSize,
                lmHeadWeight: model.EmbeddingWeight);
            using var sameShape = new MedusaHeads(
                numHeads: 2,
                hiddenDim: Cfg.HiddenDim,
                vocabSize: Cfg.VocabSize,
                lmHeadWeight: model.EmbeddingWeight);

            Assert.Throws<InvalidDataException>(() => otherModel.LoadWeights(snapshot));
            Assert.Throws<InvalidDataException>(
                () => sameShape.LoadWeights(snapshot, expectedFingerprint: "model-b"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task Builder_LoadsSavedHeadsWithoutRecalibrating()
    {
        string path = TempHeadPath();
        try
        {
            using var model = BuildModel();
            var tokenizer = BuildTokenizer();
            int[] prompt = [100, 101, 102, 103];

            var saving = new MedusaGeneratorBuilder<KVCacherBuilder>
            {
                NumHeads = 2,
                CalibrationSamples = 8,
                CalibrationPrompt = "abc",
                CalibrationSteps = 3,
                CalibrationLearningRate = 0.02f,
                MedusaHeadsPath = path,
                CalibrationFingerprint = "model-a",
            };
            using var first = saving.CreateGenerator(
                model, tokenizer, addBos: false, addEos: false, caches: null, seed: 7);

            Assert.Equal(MedusaHeadLoadResult.Calibrated, saving.LastHeadLoad);
            Assert.True(File.Exists(path));
            int[] firstIds = await Generate(first, prompt, maxNew: 4);

            var loading = new MedusaGeneratorBuilder<KVCacherBuilder>
            {
                NumHeads = 2,
                CalibrationSamples = 0,
                MedusaHeadsPath = path,
                CalibrationFingerprint = "model-a",
            };
            using var second = loading.CreateGenerator(
                model, tokenizer, addBos: false, addEos: false, caches: null, seed: 7);

            Assert.Equal(MedusaHeadLoadResult.Loaded, loading.LastHeadLoad);
            Assert.Equal(saving.LastCalibration, loading.LastCalibration);
            Assert.Equal(firstIds, await Generate(second, prompt, maxNew: 4));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Builder_MissingHeadFileFailsLoudly()
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        var builder = new MedusaGeneratorBuilder<KVCacherBuilder>
        {
            NumHeads = 2,
            CalibrationSamples = 0,
            MedusaHeadsPath = TempHeadPath(),
        };

        Assert.Throws<FileNotFoundException>(() => builder.CreateGenerator(
            model, tokenizer, addBos: false, addEos: false, caches: null, seed: 7));
    }

    private static string TempHeadPath() =>
        Path.Combine(Path.GetTempPath(), $"sharpmedusa-{Guid.NewGuid():N}.mdh");

    private sealed class InlineProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}
