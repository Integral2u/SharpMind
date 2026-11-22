using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// SpeculativeGenerator used to sample a draft from the trunk's own logits and
/// keep the greedy token on rejection, which silently disabled temperature and
/// top-p. It now delegates to StandardGenerator, so the two must agree on every
/// sampling path and MTP drafting must actually run.
/// </summary>
public sealed class SpeculativeGeneratorTests
{
    private static readonly Tokenizer Tokenizer = BuildTokenizer();

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null,
            bosId: 1, eosId: 2, tokenizerModel: "gpt2");
    }

    /// <summary>A tiny qwen35-shaped model with one MTP block.</summary>
    private static Transformer BuildModel(int seed = 1234)
    {
        var config = new ModelConfig
        {
            VocabSize = 259,
            HiddenDim = 16,
            NumLayers = 2,
            NumHeads = 4,
            NumKvHeads = 2,
            FfnDim = 32,
            MaxSeqLen = 180,
            NextnPredictLayers = 1,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Ffn = FfnKind.Gated };
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);

        var mtp = weights.Blocks[config.NumLayers];
        mtp.MtpEhProj = new Tensor<float>(2 * config.HiddenDim, config.HiddenDim);
        mtp.MtpHnorm = Tensor<float>.Ones(config.HiddenDim);
        mtp.MtpEnorm = Tensor<float>.Ones(config.HiddenDim);
        mtp.MtpSharedHeadNorm = Tensor<float>.Ones(config.HiddenDim);

        WeightInitializer.InitializeRandomly(weights, seed);
        var rng = new Random(seed + 1);
        foreach (var w in new[] { mtp.MtpEhProj, mtp.MtpHnorm, mtp.MtpEnorm, mtp.MtpSharedHeadNorm })
            for (int i = 0; i < w.ElementCount; i++)
                w.Data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.02f;

        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private sealed record Run(int[] Ids, (int Drafted, int Accepted) Mtp);

    private static async Task<Run> Generate(
        IGenerator<KVCacherBuilder> generator,
        int[] prompt,
        SamplingConfig sampling,
        GenerationConfig generation)
    {
        await foreach (var _ in generator.GenerateFromTokensAsync(prompt, sampling, generation))
        {
        }

        var mtp = generator is SpeculativeGenerator<KVCacherBuilder> speculative
            ? speculative.MtpStats
            : ((StandardGenerator<KVCacherBuilder>)generator).MtpStats;
        return new Run([.. generator.CurrentGeneratedIds!], mtp);
    }

    [Fact]
    public async Task Greedy_MatchesStandardGenerator()
    {
        using var model = BuildModel();
        int[] prompt = [3, 4, 5, 6, 7, 8];
        var sampling = new SamplingConfig { Temperature = 0f, TopK = 1, TopP = 1f };
        var generation = new GenerationConfig
        {
            MaxNewTokens = 20,
            RepetitionPenalty = 1f,
            Stream = true,
        };

        using var standard = new StandardGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 7, maxCacheLen: 180);
        using var speculative = new SpeculativeGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 7, maxCacheLen: 180);

        var expected = await Generate(standard, prompt, sampling, generation);
        var actual = await Generate(speculative, prompt, sampling, generation);

        Assert.Equal(expected.Ids, actual.Ids);
    }

    [Fact]
    public async Task Sampled_MatchesStandardGenerator()
    {
        using var model = BuildModel();
        int[] prompt = [9, 10, 11, 12, 13];
        var sampling = new SamplingConfig { Temperature = 0.8f, TopK = 5, TopP = 0.95f, Seed = 11 };
        var generation = new GenerationConfig
        {
            MaxNewTokens = 20,
            RepetitionPenalty = 1f,
            Stream = true,
        };

        using var standard = new StandardGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 3, maxCacheLen: 180);
        using var speculative = new SpeculativeGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 3, maxCacheLen: 180);

        var expected = await Generate(standard, prompt, sampling, generation);
        var actual = await Generate(speculative, prompt, sampling, generation);

        // The old implementation always emitted the greedy token here.
        var greedy = new SamplingConfig { Temperature = 0f, TopK = 1, TopP = 1f };
        using var greedyGenerator = new StandardGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 7, maxCacheLen: 180);
        var greedyRun = await Generate(greedyGenerator, prompt, greedy, generation);

        Assert.Equal(expected.Ids, actual.Ids);
        Assert.NotEqual(greedyRun.Ids, actual.Ids);
    }

    [Fact]
    public async Task MtpDraft_IsHonoured()
    {
        using var model = BuildModel();
        int[] prompt = [3, 4, 5, 6, 7, 8, 9, 10];
        var sampling = new SamplingConfig { Temperature = 0f, TopK = 1, TopP = 1f };
        var baselineConfig = new GenerationConfig
        {
            MaxNewTokens = 20,
            RepetitionPenalty = 1f,
            Stream = true,
        };
        var mtpConfig = baselineConfig with { MtpDraft = true, MtpDraftTokens = 4 };

        using var baseline = new SpeculativeGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 5, maxCacheLen: 180);
        using var drafted = new SpeculativeGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 5, maxCacheLen: 180);

        var expected = await Generate(baseline, prompt, sampling, baselineConfig);
        var actual = await Generate(drafted, prompt, sampling, mtpConfig);

        Assert.Equal(expected.Ids, actual.Ids);
        Assert.True(actual.Mtp.Drafted > 0, "the MTP head never proposed a draft");
    }

    [Fact]
    public async Task TracksResidentCacheTokens()
    {
        using var model = BuildModel();
        using var generator = new SpeculativeGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 1, maxCacheLen: 180);
        var seen = new List<int>();
        generator.OnTokenGenerated += seen.Add;

        int[] prompt = [3, 4, 5];
        await Generate(
            generator, prompt,
            new SamplingConfig { Temperature = 0.7f, TopK = 4, TopP = 0.95f },
            new GenerationConfig { MaxNewTokens = 8, RepetitionPenalty = 1f, Stream = true });

        int[] generated = generator.CurrentGeneratedIds!.ToArray();
        int[] resident = generator.CacheTokens!.ToArray();
        Assert.Equal(generated, seen);
        Assert.Equal([.. prompt, .. generated], resident);
        Assert.Equal(generator.Caches[0].Length, resident.Length);
    }
}
