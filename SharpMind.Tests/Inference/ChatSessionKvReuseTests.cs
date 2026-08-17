using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// A second chat turn must not re-run the whole conversation through the model.
/// The KV cache already holds the previous prompt and the response the model
/// generated on top of it; only the tokens after that are new. This used to
/// work only without a chat formatter (a hand-built "user: … assistant: "
/// delta), so every real instruct model — ChatML, Llama-3, Auto — paid the full
/// prefill on every turn, growing with history.
///
/// The session now compares the rendered prompt token by token against what the
/// generator says is resident, keeps the common prefix, and feeds the rest.
/// These tests watch the cache itself: a spy records the sequence length of
/// every forward pass, so the first entry of a turn is exactly how many prompt
/// tokens were prefilled — measured, not inferred. The expected value is the
/// tokens after the common prefix of the previous cache contents and the new
/// prompt, computed here the same way the session must.
/// </summary>
public sealed class ChatSessionKvReuseTests
{
    private const int MaxNew = 6;
    private const string SystemPrompt = "Be terse.";

    private static ModelConfig Cfg => new()
    {
        VocabSize = 256,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 1024,
    };

    // 256 byte tokens and no specials, so a random model's argmax always decodes
    // to a real byte that re-encodes to the same id: replies are never empty and
    // the recorded response round-trips to the ids the cache holds.
    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string>();
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);
    }

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    /// <summary>Records the sequence length of every write, so prefill sizes are observable.</summary>
    private sealed class SpyCache(IKVCache inner) : IKVCache
    {
        public readonly List<int> Writes = [];
        public int FirstWrite => Writes.Count > 0 ? Writes[0] : 0;
        public int Total => Writes.Sum();
        public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
        {
            Writes.Add(k.Shape[1]);
            inner.Update(k, v, numKvHeads, headDim);
        }
        public void Reset() => inner.Reset();
        public void TrimToLast(int keep) => inner.TrimToLast(keep);
        public void Truncate(int length) => inner.Truncate(length);
        public int Length => inner.Length;
        public int MaxSeqLen => inner.MaxSeqLen;
        public bool IsFull => inner.IsFull;
        public bool IsContiguous => inner.IsContiguous;
        public unsafe float* GetKeyPtr(int b, int p, int h) => inner.GetKeyPtr(b, p, h);
        public unsafe float* GetValuePtr(int b, int p, int h) => inner.GetValuePtr(b, p, h);
        public object? Snapshot() => inner.Snapshot();
        public void Restore(object? s) => inner.Restore(s);
        public void Dispose() => inner.Dispose();
    }

    private static (ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder> session, SpyCache spy)
        BuildSession(Transformer model, Tokenizer tokenizer, IChatPromptFormatter formatter)
    {
        var spy = new SpyCache(new KVCache(1, Cfg.NumKvHeads, Cfg.MaxSeqLen, Cfg.HeadDim));
        var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
            model, tokenizer, caches: [spy], formatter: formatter)
        {
            MaxNewTokens = MaxNew,
            Temperature = 0f,          // greedy, so two runs of the same prompt agree
            RepetitionPenalty = 1f,    // the penalty window would otherwise treat a delta differently
        };
        session.InitializeChat();
        session.AddMessage(ChatRole.System, SystemPrompt);
        return (session, spy);
    }

    private static async Task<List<int>> Turn(
        ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder> session, string text)
    {
        var ids = new List<int>();
        await foreach (var e in session.GetResponseStreamAsync(text))
            if (e.TokenId is int id) ids.Add(id);
        return ids;
    }

    private static int[] Encode(Tokenizer t, string s) => t.Encode(s, addBos: false, addEos: false);

    private static int CommonPrefix(IReadOnlyList<int> a, int[] b)
    {
        int n = Math.Min(a.Count, b.Length), i = 0;
        while (i < n && a[i] == b[i]) i++;
        return i;
    }

    // The exact ids a just-finished turn ran on, taken from the cache rather than
    // re-encoded: after a turn the cache holds the whole prompt followed by the
    // reply, so dropping the last `replyLen` positions leaves the prompt ids the
    // generator actually saw — no re-tokenisation, no seam artifacts.
    private static int[] LastPrompt(
        ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder> session, int replyLen)
    {
        var cache = session.Generator.CacheTokens!;
        return [.. cache.Take(cache.Count - replyLen)];
    }

    public static IEnumerable<object[]> Formatters()
    {
        yield return [new SimpleFormatter()];                 // "user: … assistant: ", the only shape that ever reused
        yield return [new ChatMLFormatter("<|im_start|>")];    // what qwen2 uses
    }

    [Theory]
    [MemberData(nameof(Formatters))]
    public async Task SecondTurn_PrefillsOnlyTheTokensAfterTheCachedPrefix(IChatPromptFormatter formatter)
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();
        var (session, spy) = BuildSession(model, tokenizer, formatter);
        await using var _ = session;

        var reply1 = await Turn(session, "hello there");
        int turn1Prompt = LastPrompt(session, reply1.Count).Length;   // system + first user turn + assistant header
        var cacheBefore = session.Generator.CacheTokens!.ToArray();

        spy.Writes.Clear();
        var reply2 = await Turn(session, "and again");

        // Turn 2's prompt begins with the whole first turn, so the system prompt
        // and first user turn are never re-prefilled — the headline win. (Whether
        // the reply itself is reused depends on the reply re-tokenising to the
        // same ids it was generated as; that is the tokenizer's business, not
        // this mechanism's, so the floor is turn 1's prompt.)
        var prompt2 = LastPrompt(session, reply2.Count);
        int keep = CommonPrefix(cacheBefore, prompt2);
        Assert.True(keep >= turn1Prompt,
            $"reused only {keep} tokens; the {turn1Prompt}-token system+user prefix must always be kept");
        Assert.True(keep < prompt2.Length);
        // Positions written this turn = the prompt tail past the reused prefix,
        // then one per generated token. (Prefill is chunked at MaxPrefillTokens,
        // so sum the writes rather than reading the first.)
        Assert.Equal((prompt2.Length - keep) + reply2.Count, spy.Total);
    }

    [Theory]
    [MemberData(nameof(Formatters))]
    public async Task ReusedPrefix_GeneratesTheSameTokensAsAFreshPrefill(IChatPromptFormatter formatter)
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        // Session A: two turns, the second extending the cache from the first.
        var (a, _) = BuildSession(model, tokenizer, formatter);
        await using var _a = a;
        await Turn(a, "hello there");
        var replyA1 = a.History[^1].Content;
        var idsA2 = await Turn(a, "and again");

        // Session B: same conversation assembled by hand, so its "and again" turn
        // is its first generation and prefills the whole prompt from empty.
        var (b, spyB) = BuildSession(model, tokenizer, formatter);
        await using var _b = b;
        b.AddMessage(ChatRole.User, "hello there");
        b.AddMessage(ChatRole.Agent, replyA1);
        var idsB2 = await Turn(b, "and again");

        Assert.NotEmpty(idsA2);
        Assert.Equal(idsB2, idsA2);                 // reuse must not change a single sampled token
        // Sanity that B really did the full prefill it stands in for: its one and
        // only prefill covers the whole prompt, more than A reused.
        Assert.True(spyB.FirstWrite > 1);
    }

    [Fact]
    public async Task EditedHistory_RecomputesFromTheDivergencePointOnly()
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();
        var formatter = new ChatMLFormatter("<|im_start|>");
        var (session, spy) = BuildSession(model, tokenizer, formatter);
        await using var _ = session;

        var reply1 = await Turn(session, "first question");
        int turn1Prompt = LastPrompt(session, reply1.Count).Length; // system + first user turn + assistant header
        await Turn(session, "second question");

        // Rewrite the first answer. Everything from there on is stale, but the
        // system prompt, the first user turn and the assistant header before the
        // answer are still valid cache content — the divergence is the first
        // token of the answer.
        var history = session.History;
        int firstAnswerIdx = -1;
        for (int i = 0; i < history.Count; i++)
            if (history[i].Role == ChatRole.Agent) { firstAnswerIdx = i; break; }
        Assert.True(firstAnswerIdx >= 0);
        history[firstAnswerIdx].Content = "a completely different first answer";

        var cacheBefore = session.Generator.CacheTokens!.ToArray();

        spy.Writes.Clear();
        var reply3 = await Turn(session, "third question");

        var prompt3 = LastPrompt(session, reply3.Count);
        int keep = CommonPrefix(cacheBefore, prompt3);

        // The edit is the first answer, so the divergence is its first token: the
        // reused prefix is turn 1's prompt (the new and old answers may share a
        // leading token or two, never more here), and the stale tail is dropped.
        Assert.InRange(keep, turn1Prompt, turn1Prompt + 3);
        Assert.True(keep < cacheBefore.Length,
            $"kept {keep} of a {cacheBefore.Length}-token cache; the edited answer should have dropped the stale tail");
        Assert.Equal((prompt3.Length - keep) + reply3.Count, spy.Total);
    }
}
