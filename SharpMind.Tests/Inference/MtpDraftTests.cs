using System.Text;
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
/// MTP drafting must change speed, never output. A multi-token chain feeds each
/// draft step's head-normed hidden to the next, one verification forward covers
/// the whole chain, and a rejected chain truncates the head's cache back to the
/// verified prefix. These pin the two paths that are easy to get wrong: a
/// fully-accepted run (the head's last draft needs an explicit feed) and a run
/// that the trunk rejects (cache and rows must rewind exactly).
/// </summary>
public sealed class MtpDraftTests
{
    private static readonly Tokenizer Tokenizer = BuildTokenizer();

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null,
            bosId: 1, eosId: 2, tokenizerModel: "gpt2");
    }

    /// <summary>
    /// A tiny qwen35-shaped model with one MTP block. <paramref name="zero"/>
    /// leaves every weight at zero, so the trunk and the head both pick token 0
    /// and every draft is accepted; the random variant mostly rejects, exercising
    /// the rollback/truncation path.
    /// </summary>
    private static Transformer BuildModel(bool zero, int seed = 1234)
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

        if (!zero)
        {
            WeightInitializer.InitializeRandomly(weights, seed);
            var rng = new Random(seed + 1);
            foreach (var w in new[] { mtp.MtpEhProj, mtp.MtpHnorm, mtp.MtpEnorm, mtp.MtpSharedHeadNorm })
                for (int i = 0; i < w.ElementCount; i++)
                    w.Data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.02f;
        }

        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private sealed record Run(int[] Ids, (int Drafted, int Accepted) Mtp);

    private static async Task<Run> Generate(
        Transformer model, int[] prompt, bool mtp, int mtpTokens = 1, int maxNewTokens = 24)
    {
        var generator = new StandardGenerator<KVCacherBuilder>(
            model, Tokenizer, addBos: false, addEos: false, seed: 1, maxCacheLen: 180);
        var sampling = new SamplingConfig { Temperature = 0f, TopK = 1, TopP = 1f };
        var generation = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            RepetitionPenalty = 1f,
            MtpDraft = mtp,
            MtpDraftTokens = mtpTokens,
            Stream = true,
        };
        var text = new StringBuilder();
        await foreach (var fragment in generator.GenerateFromTokensAsync(prompt, sampling, generation))
            text.Append(fragment);
        return new Run([.. generator.CurrentGeneratedIds!], generator.MtpStats);
    }

    [Fact]
    public async Task FullyAcceptedChain_MatchesTheNonMtpRun()
    {
        using var model = BuildModel(zero: true);
        int[] prompt = [3, 4, 5, 6, 7, 8];

        var baseline = await Generate(model, prompt, mtp: false);
        var drafted = await Generate(model, prompt, mtp: true, mtpTokens: 4);

        Assert.Equal(baseline.Ids, drafted.Ids);
        Assert.True(drafted.Mtp.Drafted > 0, "the head never proposed a draft");
        Assert.Equal(drafted.Mtp.Drafted, drafted.Mtp.Accepted);
    }

    [Fact]
    public async Task RejectedChains_StillMatchTheNonMtpRun_ForEveryDraftLength()
    {
        using var model = BuildModel(zero: false);
        int[] prompt = [3, 4, 5, 6, 7, 8, 9, 10];

        var baseline = await Generate(model, prompt, mtp: false);
        for (int m = 1; m <= GenerationConfig.MaxMtpDraftTokens; m++)
        {
            var drafted = await Generate(model, prompt, mtp: true, mtpTokens: m);
            Assert.Equal(baseline.Ids, drafted.Ids);
            Assert.True(drafted.Mtp.Drafted > 0, $"m={m}: the head never proposed a draft");
        }
    }

    [Fact]
    public async Task DraftLength_IsClampedToTheConfiguredMaximum()
    {
        using var model = BuildModel(zero: true);
        int[] prompt = [3, 4, 5];

        var run = await Generate(model, prompt, mtp: true, mtpTokens: 10_000);
        Assert.Equal(24, run.Ids.Length);
        Assert.True(run.Mtp.Drafted > 0);
    }

    [Fact]
    public async Task PrefillFillsTheDraftHead_FirstSampledTokenHasADraft()
    {
        using var model = BuildModel(zero: true);
        int[] prompt = [3, 4, 5];

        // Two tokens is enough: the first is sampled from the prefill logits,
        // and a ready head can draft for the second without an intermediate
        // normal forward.
        var run = await Generate(model, prompt, mtp: true, mtpTokens: 2, maxNewTokens: 2);
        Assert.True(run.Mtp.Drafted > 0, "the prefill did not fill the draft head");
    }

    [Fact]
    public async Task MultiChunkPrefill_FillsTheDraftHeadAndKeepsTheOutput()
    {
        using var model = BuildModel(zero: true);
        int[] prompt = [.. Enumerable.Range(0, 70).Select(i => 3 + i % 20)];

        var baseline = await Generate(model, prompt, mtp: false, maxNewTokens: 4);
        var drafted = await Generate(model, prompt, mtp: true, mtpTokens: 2, maxNewTokens: 4);

        Assert.Equal(baseline.Ids, drafted.Ids);
        Assert.True(drafted.Mtp.Drafted > 0);
    }

    [Fact]
    public async Task ChatSession_ReplyUsesTheSessionsMtpSetting()
    {
        using var model = BuildModel(zero: true);
        var session = new SharpMind.Inference.Chat.ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
            model, Tokenizer, formatter: new SharpMind.Inference.Chat.PromptFormatters.SimpleFormatter())
        {
            Temperature = 0f,
        };
        session.InitializeChat();
        // The setting arrives the way the CUI and server pass it: through SetGenerationConfig.
        session.SetGenerationConfig(new GenerationConfig
        {
            MaxNewTokens = 12,
            RepetitionPenalty = 1f,
            MtpDraft = true,
            MtpDraftTokens = 2,
        });

        await foreach (var _ in session.GetResponseStreamAsync("abab abab abab abab abab")) { }

        var stats = Assert.IsType<StandardGenerator<KVCacherBuilder>>(session.Generator).MtpStats;
        Assert.True(stats.Drafted > 0, "the session did not enable MTP drafting");
    }

    private static async Task<Run[]> GenerateBatched(
        Transformer model, int[][] prompts, bool mtp, int mtpTokens = 1, int maxNewTokens = 20)
    {
        await using var generator = new BatchedGenerator(model, Tokenizer, new BatchedGeneratorOptions
        {
            MaxBatch = prompts.Length,
            MaxSeqLen = 64,
            Seed = 1,
        });
        var sampling = new SamplingConfig { Temperature = 0f, TopK = 1, TopP = 1f };
        var generation = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            RepetitionPenalty = 1f,
            MtpDraft = mtp,
            MtpDraftTokens = mtpTokens,
            Stream = false,
        };

        var seqs = new List<BatchedSequence>();
        foreach (var prompt in prompts)
            seqs.Add(await generator.EnqueueTokensAsync(prompt, sampling, generation));

        await Task.WhenAll(seqs.Select(s => s.WhenCompleted));
        return [.. seqs.Select(s => new Run([.. s.GeneratedIds], s.MtpStats))];
    }

    [Fact]
    public async Task BatchedMtp_FullyAcceptedChainsMatchTheNonMtpBatch()
    {
        using var model = BuildModel(zero: true);
        int[][] prompts = [[3, 4, 5], [6, 7, 8, 9], [10, 11]];

        var baseline = await GenerateBatched(model, prompts, mtp: false);
        var drafted = await GenerateBatched(model, prompts, mtp: true, mtpTokens: 4);

        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(baseline[i].Ids, drafted[i].Ids);
            Assert.True(drafted[i].Mtp.Drafted > 0, $"row {i} never drafted");
            Assert.Equal(drafted[i].Mtp.Drafted, drafted[i].Mtp.Accepted);
        }
    }

    [Fact]
    public async Task BatchedMtp_RejectedChainsMatchTheNonMtpBatch()
    {
        using var model = BuildModel(zero: false);
        int[][] prompts = [[3, 4, 5], [6, 7, 8, 9, 10], [11, 12]];

        var baseline = await GenerateBatched(model, prompts, mtp: false);
        var drafted = await GenerateBatched(model, prompts, mtp: true, mtpTokens: 4);

        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(baseline[i].Ids, drafted[i].Ids);
            Assert.True(drafted[i].Mtp.Drafted > 0, $"row {i} never drafted");
        }
    }
}
