using System.Text;
using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Inference.Grammar;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Prompt-lookup drafting in <see cref="StandardGenerator{T}"/> must change speed, never output:
/// every run with drafting on is compared token for token with the same run with drafting off.
/// A random tiny model almost never agrees with a draft on its own, so most tests constrain the
/// output to the letters a/b: the model's own logits still choose between them (so wrong logits
/// change the output), and prompts made of a/b give the drafter plenty to propose, about half of
/// which the model accepts.
/// </summary>
public sealed class PromptLookupTests
{
    private const int Draft = 6;

    private static readonly Tokenizer Tokenizer = BuildTokenizer();
    private static readonly TokenByteTable ByteTable = TokenByteTable.Get(Tokenizer);

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }

    private static Transformer BuildModel(int maxSeqLen = 256, string? architecture = null)
    {
        var config = architecture == "lfm2"
            ? new ModelConfig
            {
                Architecture = "lfm2", VocabSize = 259, HiddenDim = 32, NumLayers = 4, NumHeads = 4,
                NumKvHeads = 2, LayerKvHeads = [2, 0, 2, 0], FfnDim = 64, ShortConvCacheLength = 3,
                MaxSeqLen = maxSeqLen,
            }
            : new ModelConfig
            {
                VocabSize = 259, HiddenDim = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2,
                FfnDim = 64, MaxSeqLen = maxSeqLen,
            };
        var sharpConfig = SharpMindConfig.ForModel(config.NumHeads, config.NumKvHeads, config.Architecture, HardwareTier.Scalar);
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 4321);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static int Id(char c) => 3 + c;

    /// <summary>A deterministic a/b string: repetitive enough for the drafter, varied enough for partial accepts.</summary>
    private static int[] AbPrompt(int length, int seed)
    {
        var rng = new Random(seed);
        var ids = new int[length];
        for (int i = 0; i < length; i++)
            ids[i] = Id(rng.Next(3) == 0 ? 'b' : 'a');
        return ids;
    }

    private static IGrammarConstraint AbConstraint(int length)
        => GbnfGrammar.Parse("root ::= " + string.Join(' ', Enumerable.Repeat("[ab]", length)))
                      .CreateConstraint(ByteTable, [Tokenizer.EosId]);

    private static IGrammarConstraint LiteralConstraint(string text)
        => GbnfGrammar.Parse($"root ::= \"{text}\"").CreateConstraint(ByteTable, [Tokenizer.EosId]);

    private sealed record Run(int[] Ids, string Text, PromptLookupStats Stats, int CacheLen);

    private static async Task<Run> Generate(
        StandardGenerator<KVCacherBuilder> generator, int[] prompt, int draft,
        int maxNewTokens = 40, IGrammarConstraint? constraint = null, float temperature = 0f, int? seed = null,
        float repetitionPenalty = 1f, IReadOnlyList<string>? stopStrings = null, bool reset = true)
    {
        if (reset) generator.ResetCache();
        var sampling = new SamplingConfig { Temperature = temperature, TopK = 0, TopP = 1f, Seed = seed, Constraint = constraint };
        var generation = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            StopTokenIds = [Tokenizer.EosId],
            StopStrings = stopStrings ?? [],
            RepetitionPenalty = repetitionPenalty,
            PromptLookupDraft = draft,
            Stream = true,
        };
        var text = new StringBuilder();
        await foreach (var fragment in generator.GenerateFromTokensAsync(prompt, sampling, generation))
            text.Append(fragment);
        return new Run([.. generator.CurrentGeneratedIds!], text.ToString(), generator.PromptLookupStats, generator.Caches[0].Length);
    }

    private static StandardGenerator<KVCacherBuilder> NewGenerator(Transformer model, int? maxCacheLen = null)
        => new(model, Tokenizer, addBos: false, addEos: false, seed: 1, maxCacheLen: maxCacheLen);

    // ── Drafter ────────────────────────────────────────────────────────────

    [Fact]
    public void Drafter_PrefersTheLongestKey_ThenTheLatestMatch()
    {
        var drafter = new PromptLookupDrafter();
        Span<int> draft = stackalloc int[4];
        // "x y z w 1 2 | ... y z w 3 4 | ... z w" — the 4-key cannot match, the 3-key "y z w" matches
        // twice: the latest occurrence wins (followed by 3 4), not the 2-key's later "z w".
        int[] ctx = [9, 1, 2, 3, 50, 51, 7, 1, 2, 3, 60, 61, 8, 2, 3, 70, 1, 2, 3];
        int m = drafter.Propose(ctx, draft);
        Assert.Equal(4, m);
        Assert.Equal(new[] { 60, 61, 8, 2 }, draft[..m].ToArray());
    }

    [Fact]
    public void Drafter_RepeatsAnOverlappingMatch_AndClampsToTheSpan()
    {
        var drafter = new PromptLookupDrafter();
        Span<int> draft = stackalloc int[6];
        // "7 5 6" follows the earlier "5 6" and runs into the tail: a period-3 pattern, which repeats.
        Assert.Equal(6, drafter.Propose(new[] { 5, 6, 7, 5, 6 }, draft));
        Assert.Equal(new[] { 7, 5, 6, 7, 5, 6 }, draft.ToArray());
        Assert.Equal(2, drafter.Propose(new[] { 5, 6, 7, 5, 6 }, draft[..2])); // clamped to the span
        // A run of one token drafts a full run of it, not a single token.
        Assert.Equal(6, drafter.Propose(new[] { 4, 9, 9, 9, 9 }, draft));
        Assert.Equal(new[] { 9, 9, 9, 9, 9, 9 }, draft.ToArray());
        Assert.Equal(0, drafter.Propose(new[] { 5, 6, 7, 8 }, draft)); // no earlier match
        Assert.Equal(0, drafter.Propose(new[] { 5 }, draft));          // shorter than any key
        Assert.Equal(0, drafter.Propose(new[] { 5, 6, 5, 6 }, Span<int>.Empty));
    }

    [Fact]
    public void Drafter_GateTripsOnlyAfterEnoughDrafts()
    {
        var drafter = new PromptLookupDrafter();
        drafter.Record(PromptLookupDrafter.GateMinDrafted - 1, 0);
        Assert.True(drafter.Enabled);
        drafter.Record(1, 0);
        Assert.False(drafter.Enabled);
        Assert.Equal(0, drafter.Propose(new[] { 5, 6, 7, 5, 6 }, new int[4]));

        var good = new PromptLookupDrafter();
        good.Record(30, 6); // exactly 20%: keeps drafting
        Assert.True(good.Enabled);
    }

    // ── Output identical to drafting off ──────────────────────────────────

    [Fact]
    public async Task Greedy_SameTokensAsWithoutDrafting_WithAcceptedAndRejectedDrafts()
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        var total = new PromptLookupStats();
        int tokens = 0;

        for (int seed = 1; seed <= 6; seed++)
        {
            int[] prompt = AbPrompt(120, seed);
            var off = await Generate(gen, prompt, 0, constraint: AbConstraint(60), maxNewTokens: 60);
            var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(60), maxNewTokens: 60);

            Assert.Equal(off.Ids, on.Ids);
            Assert.Equal(off.Text, on.Text);
            Assert.Equal(new PromptLookupStats(), off.Stats);
            total = new(total.Forwards + on.Stats.Forwards, total.Drafted + on.Stats.Drafted, total.Accepted + on.Stats.Accepted);
            tokens += on.Ids.Length;
        }

        // Across the prompts both paths ran: drafts the model kept and drafts it replaced.
        Assert.True(total.Accepted > 0, $"no draft accepted: {total}");
        Assert.True(total.Accepted < total.Drafted, $"no draft rejected: {total}");
        Assert.True(total.Forwards < tokens, $"drafting saved no forward pass: {total} for {tokens} tokens");
    }

    [Fact]
    public async Task Copy_AcceptsTheDrafts_AndMatchesWithoutDrafting()
    {
        // The output is forced to repeat a stretch of the prompt whose 4-grams are unique, so every
        // draft holds until the copy ends; only the last run (drafting past the end) can be cut short.
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        const string text = "the quick brown fox jumps over a lazy dog";
        int[] prompt = [.. text.Select(Id), Id('|'), .. text[..4].Select(Id)];
        string copy = text[4..];

        var off = await Generate(gen, prompt, 0, constraint: LiteralConstraint(copy));
        var on = await Generate(gen, prompt, Draft, constraint: LiteralConstraint(copy));

        Assert.Equal(off.Ids, on.Ids);
        Assert.Equal(copy, on.Text);
        Assert.True(on.Stats.Drafted - on.Stats.Accepted <= Draft, $"rejections before the copy ended: {on.Stats}");
        Assert.True(on.Stats.Accepted >= copy.Length - Draft, $"{on.Stats}");
        Assert.True(on.Stats.Forwards * 4 <= copy.Length + 4, $"expected several tokens per forward: {on.Stats}");
    }

    [Theory]
    [InlineData(0.9f, 5, 1f)]
    [InlineData(0.7f, 11, 1f)]
    [InlineData(0f, null, 1.3f)]
    [InlineData(0.8f, 3, 1.2f)]
    public async Task SampledAndPenalised_SameTokensAsWithoutDrafting(float temperature, int? seed, float penalty)
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 7);

        var off = await Generate(gen, prompt, 0, constraint: AbConstraint(60), maxNewTokens: 60,
                                 temperature: temperature, seed: seed, repetitionPenalty: penalty);
        var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(60), maxNewTokens: 60,
                                temperature: temperature, seed: seed, repetitionPenalty: penalty);

        Assert.Equal(off.Ids, on.Ids);
        Assert.True(on.Stats.Drafted > 0, $"{on.Stats}");
    }

    [Fact]
    public async Task MaxNewTokens_EndingInsideARun_SameTokens()
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 2);

        for (int max = 1; max <= 30; max++)
        {
            var off = await Generate(gen, prompt, 0, constraint: AbConstraint(max), maxNewTokens: max);
            var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(max), maxNewTokens: max);
            Assert.Equal(off.Ids, on.Ids);
            AssertCacheConsistent(gen, prompt, on.Ids);
        }
    }

    [Fact]
    public async Task StopString_InsideARun_SameTextAndConsistentCache()
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 3);
        string[] stops = ["bab", "abba"];

        var off = await Generate(gen, prompt, 0, constraint: AbConstraint(60), maxNewTokens: 60, stopStrings: stops);
        var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(60), maxNewTokens: 60, stopStrings: stops);

        Assert.Equal(off.Ids, on.Ids);
        Assert.Equal(off.Text, on.Text);
        AssertCacheConsistent(gen, prompt, on.Ids);
    }

    [Fact]
    public async Task PoorAcceptance_GateStopsDrafting_SameTokens()
    {
        // Eight letters, and a prompt holding every pair of them: every step finds a match and drafts.
        // Sampled hot, the random model picks among the eight almost uniformly, so a drafted letter
        // holds about one time in eight. The gate must switch drafting off once GateMinDrafted is
        // reached, and output stays identical (same seed, one draw per token either way).
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        const string letters = "abcdefgh";
        int[] prompt = [.. letters.SelectMany(x => letters.SelectMany(y => new[] { Id(x), Id(y) }))];
        var eight = GbnfGrammar.Parse("root ::= " + string.Join(' ', Enumerable.Repeat("[a-h]", 80)));

        var off = await Generate(gen, prompt, 0, maxNewTokens: 80, temperature: 2f, seed: 3,
                                 constraint: eight.CreateConstraint(ByteTable, [Tokenizer.EosId]));
        var on = await Generate(gen, prompt, Draft, maxNewTokens: 80, temperature: 2f, seed: 3,
                                constraint: eight.CreateConstraint(ByteTable, [Tokenizer.EosId]));

        Assert.Equal(off.Ids, on.Ids);
        Assert.True(on.Stats.Accepted * 5 < on.Stats.Drafted, $"acceptance too high for this test: {on.Stats}");
        Assert.InRange(on.Stats.Drafted, PromptLookupDrafter.GateMinDrafted, PromptLookupDrafter.GateMinDrafted + Draft);
    }

    [Fact]
    public async Task OversizedDraft_IsClamped_SameTokens()
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 4);

        var off = await Generate(gen, prompt, 0, constraint: AbConstraint(60), maxNewTokens: 60);
        var on = await Generate(gen, prompt, 1000, constraint: AbConstraint(60), maxNewTokens: 60);

        Assert.Equal(off.Ids, on.Ids);
    }

    // ── Cache limits ───────────────────────────────────────────────────────

    [Fact]
    public async Task SlidingWindowTrim_DuringGeneration_SameTokens()
    {
        // Caches of 40-56 rows under a 256-position model: generation passes the cache size and trims.
        // The sweep makes some run end exactly at the cache's last free row.
        using var model = BuildModel();
        for (int cacheLen = 40; cacheLen <= 56; cacheLen++)
        {
            using var gen = NewGenerator(model, maxCacheLen: cacheLen);
            int[] prompt = AbPrompt(30, 5);

            var off = await Generate(gen, prompt, 0, constraint: AbConstraint(90), maxNewTokens: 90);
            var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(90), maxNewTokens: 90);

            Assert.True(off.Ids.SequenceEqual(on.Ids), $"cache {cacheLen}: output differs");
            // Same tokens and the same trims must leave the same rows: a trim inside a run would
            // keep unverified draft rows that output alone barely shows.
            Assert.Equal(off.CacheLen, on.CacheLen);
            Assert.True(on.Stats.Drafted > 0, $"cache {cacheLen}: {on.Stats}");
        }
    }

    [Fact]
    public async Task SlidingWindowTrim_IntoTheContextEnd_SameTokens()
    {
        // Cache smaller than the 64-position context: positions run past the cache size (trims) and
        // up to the context end, where only the context-end clamp keeps a run inside the context.
        using var model = BuildModel(maxSeqLen: 64);
        for (int cacheLen = 20; cacheLen <= 36; cacheLen++)
        {
            using var gen = NewGenerator(model, maxCacheLen: cacheLen);
            int[] prompt = AbPrompt(16, 9);

            var off = await Generate(gen, prompt, 0, constraint: AbConstraint(60), maxNewTokens: 60);
            var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(60), maxNewTokens: 60);

            Assert.True(off.Ids.SequenceEqual(on.Ids), $"cache {cacheLen}: output differs");
            Assert.Equal(off.CacheLen, on.CacheLen);
        }
    }

    [Fact]
    public async Task ContextEnd_SameTokens()
    {
        // MaxPosition 64: the prompt plus generation runs into the end of the context. The sweep of
        // prompt lengths makes some run reach the very last position.
        using var model = BuildModel(maxSeqLen: 64);
        using var gen = NewGenerator(model);
        for (int promptLen = 30; promptLen <= 50; promptLen++)
        {
            int[] prompt = AbPrompt(promptLen, 6);

            var off = await Generate(gen, prompt, 0, constraint: AbConstraint(60), maxNewTokens: 60);
            var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(60), maxNewTokens: 60);

            Assert.True(off.Ids.SequenceEqual(on.Ids), $"prompt {promptLen}: output differs");
        }
    }

    [Fact]
    public async Task RecurrentCaches_DisableDrafting_SameTokens()
    {
        using var model = BuildModel(architecture: "lfm2");
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(40, 1);

        var off = await Generate(gen, prompt, 0, constraint: AbConstraint(20), maxNewTokens: 20);
        var on = await Generate(gen, prompt, Draft, constraint: AbConstraint(20), maxNewTokens: 20);

        Assert.Equal(off.Ids, on.Ids);
        Assert.Equal(0, on.Stats.Drafted);
    }

    // ── Leaving a run early, and the next turn ─────────────────────────────

    [Theory]
    [InlineData(3, false)]
    [InlineData(7, false)]
    [InlineData(12, false)]
    [InlineData(5, true)]
    [InlineData(11, true)]
    public async Task LeavingMidRun_KeepsCacheConsistent_AndTheNextTurnMatchesAFreshPrefill(int afterTokens, bool cancel)
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 1);

        // Turn 1: stop consuming (break, like a tool call) or cancel (Ctrl+C) after a few tokens.
        gen.ResetCache();
        using var cts = new CancellationTokenSource();
        var sampling = new SamplingConfig { Temperature = 0f, Constraint = AbConstraint(60) };
        var generation = new GenerationConfig
        {
            MaxNewTokens = 60, RepetitionPenalty = 1f, PromptLookupDraft = Draft, Stream = true,
            StopTokenIds = [Tokenizer.EosId],
        };
        int seen = 0;
        try
        {
            await foreach (var _ in gen.GenerateFromTokensAsync(prompt, sampling, generation, cts.Token))
            {
                if (++seen < afterTokens) continue;
                if (cancel) cts.Cancel(); else break;
            }
        }
        catch (OperationCanceledException) when (cancel) { }
        Assert.True(gen.PromptLookupStats.Drafted > 0 || seen < afterTokens, $"{gen.PromptLookupStats}");
        int[] turn1 = [.. gen.CurrentGeneratedIds!];
        AssertCacheConsistent(gen, prompt, turn1);

        // Turn 2 the way ChatSession does it: keep the common prefix, prefill only the rest.
        int[] prompt2 = [.. prompt, .. turn1, .. AbPrompt(10, 99)];
        int keep = CommonPrefix(gen.CacheTokens!, prompt2);
        gen.TruncateCache(keep);
        var reused = await Generate(gen, prompt2[keep..], Draft, constraint: AbConstraint(20), maxNewTokens: 20, reset: false);

        using var fresh = NewGenerator(model);
        var expected = await Generate(fresh, prompt2, 0, constraint: AbConstraint(20), maxNewTokens: 20);

        Assert.Equal(expected.Ids, reused.Ids);
    }

    [Fact]
    public async Task SecondTurn_ReusesTheDraftedTurnsCache_SameAsFreshPrefill()
    {
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 8);

        var turn1 = await Generate(gen, prompt, Draft, constraint: AbConstraint(40), maxNewTokens: 40);
        Assert.True(turn1.Stats.Accepted > 0, $"{turn1.Stats}");
        AssertCacheConsistent(gen, prompt, turn1.Ids);

        int[] prompt2 = [.. prompt, .. turn1.Ids, .. AbPrompt(10, 77)];
        int keep = CommonPrefix(gen.CacheTokens!, prompt2);
        Assert.True(keep >= prompt.Length + turn1.Ids.Length - 1, $"prefix reuse lost: kept {keep}");
        gen.TruncateCache(keep);
        var reused = await Generate(gen, prompt2[keep..], Draft, constraint: AbConstraint(20), maxNewTokens: 20, reset: false);

        using var fresh = NewGenerator(model);
        var expected = await Generate(fresh, prompt2, 0, constraint: AbConstraint(20), maxNewTokens: 20);

        Assert.Equal(expected.Ids, reused.Ids);
    }

    [Fact]
    public async Task SecondTurn_DraftsFromEarlierTurnsInTheCache()
    {
        // Turn 2 prefills a single new token and may draft only once (two tokens allowed): a draft
        // can only come from turn 1's prompt and reply, which reach the drafter through the cache.
        using var model = BuildModel();
        using var gen = NewGenerator(model);
        int[] prompt = AbPrompt(120, 8);
        var turn1 = await Generate(gen, prompt, Draft, constraint: AbConstraint(20), maxNewTokens: 20);

        int[] prompt2 = [.. gen.CacheTokens!, Id('a')];
        var turn2 = await Generate(gen, prompt2[gen.CacheTokens!.Count..], Draft,
                                   constraint: AbConstraint(2), maxNewTokens: 2, reset: false);

        Assert.Equal(1, turn2.Stats.Drafted);
    }

    // ── ChatSession wiring ─────────────────────────────────────────────────

    [Theory]
    [InlineData(Draft, true)]
    [InlineData(0, false)]
    public async Task ChatSession_ReplyUsesTheSessionsPromptLookupSetting(int draft, bool expectLookup)
    {
        using var model = BuildModel();
        var session = new SharpMind.Inference.Chat.ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
            model, Tokenizer, formatter: new SharpMind.Inference.Chat.PromptFormatters.SimpleFormatter())
        {
            Temperature = 0f,
        };
        session.InitializeChat();
        // The setting arrives the way the CUI and server pass it: through SetGenerationConfig.
        session.SetGenerationConfig(new GenerationConfig { MaxNewTokens = 12, RepetitionPenalty = 1f, PromptLookupDraft = draft });

        await foreach (var _ in session.GetResponseStreamAsync("abab abab abab abab abab")) { }

        var stats = Assert.IsType<StandardGenerator<KVCacherBuilder>>(session.Generator).PromptLookupStats;
        Assert.Equal(expectLookup, stats.Forwards > 0);
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /// <summary>
    /// After a generation the vouched cache ids must describe the cache exactly (count = rows) and be
    /// the prompt followed by the emitted tokens (all of them, or all but the last one when generation
    /// stopped before feeding it) — what ChatSession's prefix reuse relies on.
    /// </summary>
    private static void AssertCacheConsistent(StandardGenerator<KVCacherBuilder> gen, int[] prompt, int[] generated)
    {
        var tokens = gen.CacheTokens;
        Assert.NotNull(tokens);
        Assert.Equal(gen.Caches[0].Length, tokens!.Count);
        int[] expected = [.. prompt, .. generated];
        Assert.True(tokens.Count == expected.Length || tokens.Count == expected.Length - 1,
            $"cache holds {tokens.Count} ids, prompt+generated is {expected.Length}");
        Assert.Equal(expected[..tokens.Count], tokens.ToArray());
    }

    private static int CommonPrefix(IReadOnlyList<int> a, int[] b)
    {
        int n = 0;
        while (n < a.Count && n < b.Length && a[n] == b[n]) n++;
        return n;
    }
}
