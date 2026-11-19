using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The batched generator exists so N callers share one forward pass. The risk
/// is that sharing leaks between rows: a token sampled from another sequence's
/// logits, a cache row written by the wrong sequence, a stop that ends everyone.
/// These pin that every request gets exactly the tokens it would have got on its
/// own, while the aggregate decode really is batched.
/// </summary>
public sealed class BatchedGeneratorTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 64,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 64,
        MaxSeqLen = 64,
    };

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 90210);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);
    }

    private static int[] Prompt(int length, int seed)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, length).Select(_ => rng.Next(Cfg.VocabSize))];
    }

    private static GenerationConfig Gen(int maxNew) => new()
    {
        MaxNewTokens = maxNew,
        RepetitionPenalty = 1f,
        Stream = false,
    };

    /// <summary>Greedy reference: the same prompt through a single-sequence generator.</summary>
    private static async Task<int[]> ReferenceAsync(Transformer model, Tokenizer tokenizer, int[] prompt, int maxNew)
    {
        using var gen = new StandardGenerator<KVCacherBuilder>(model, tokenizer, addBos: false, addEos: false, seed: 1);
        var ids = new List<int>();
        gen.OnTokenGenerated += ids.Add;

        await foreach (var _ in gen.GenerateFromTokensAsync(prompt, SamplingConfig.Greedy, Gen(maxNew))) { }
        return [.. ids];
    }

    [Fact]
    public async Task ConcurrentSequences_EachGetTheirOwnGreedyAnswer()
    {
        const int maxNew = 6;
        int[][] prompts = [.. new[] { 5, 8, 3, 6, 4 }.Select((len, i) => Prompt(len, 600 + i))];

        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        var expected = new int[prompts.Length][];
        for (int i = 0; i < prompts.Length; i++)
            expected[i] = await ReferenceAsync(model, tokenizer, prompts[i], maxNew);

        await using var batch = new BatchedGenerator(model, tokenizer, new BatchedGeneratorOptions
        {
            MaxBatch = 3,
            MaxSeqLen = Cfg.MaxSeqLen,
        });

        var handles = new List<BatchedSequence>();
        for (int i = 0; i < prompts.Length; i++)
            handles.Add(await batch.EnqueueTokensAsync(prompts[i], SamplingConfig.Greedy, Gen(maxNew)));

        await Task.WhenAll(handles.Select(h => h.WhenCompleted)).WaitAsync(TimeSpan.FromMinutes(2));

        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(BatchFinishReason.MaxTokens, handles[i].FinishReason);
            Assert.Equal(expected[i], handles[i].GeneratedIds);
        }

        // Five requests over rows of three: the decode steps must really carry
        // more than one sequence, or batching is not happening at all.
        var stats = batch.Stats;
        Assert.True(stats.Steps > 0);
        Assert.True(stats.AverageBatchSize > 1.5,
            $"expected a batched mean, got {stats.AverageBatchSize:0.00}");
        Assert.True(stats.GeneratedTokens > 0 && stats.TokensPerSecond > 0);
    }

    [Fact]
    public async Task Streaming_FragmentsConcatenateToTheFullText()
    {
        const int maxNew = 5;
        int[] prompt = Prompt(6, 701);

        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        await using var batch = new BatchedGenerator(model, tokenizer, new BatchedGeneratorOptions { MaxBatch = 2 });
        var seq = await batch.EnqueueTokensAsync(prompt, SamplingConfig.Greedy,
            new GenerationConfig { MaxNewTokens = maxNew, RepetitionPenalty = 1f, Stream = true });

        var parts = new List<string>();
        await foreach (string fragment in seq.StreamAsync())
            parts.Add(fragment);

        Assert.Equal(string.Concat(parts), seq.Text);
        Assert.Equal(maxNew, seq.GeneratedIds.Count);
        Assert.StartsWith(seq.Text, string.Concat(parts));
    }

    /// <summary>
    /// Abandoning one sequence must free its row for a request queued behind
    /// it, without disturbing the answers of sequences still running.
    /// </summary>
    [Fact]
    public async Task AbandonedSequence_FreesItsRowForTheNextRequest()
    {
        const int maxNew = 30;
        int[] longPrompt = Prompt(4, 801);
        int[] shortPrompt = Prompt(7, 802);

        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        int[] expectedShort = await ReferenceAsync(model, tokenizer, shortPrompt, 4);

        await using var batch = new BatchedGenerator(model, tokenizer, new BatchedGeneratorOptions { MaxBatch = 1 });

        var slow = await batch.EnqueueTokensAsync(longPrompt, SamplingConfig.Greedy,
            new GenerationConfig { MaxNewTokens = maxNew, RepetitionPenalty = 1f, Stream = true });

        // Read the first fragment, then walk away: a real client closing the tab.
        await using (var walker = slow.StreamAsync().GetAsyncEnumerator())
        {
            Assert.True(await walker.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromMinutes(2)));
        }

        var next = await batch.EnqueueTokensAsync(shortPrompt, SamplingConfig.Greedy, Gen(4));
        await next.WhenCompleted.WaitAsync(TimeSpan.FromMinutes(2));

        Assert.Equal(expectedShort, next.GeneratedIds);
        Assert.True(next.Text.Length > 0 || next.GeneratedIds.Count == expectedShort.Length);
    }

    /// <summary>
    /// A row that a finished sequence leaves behind must be empty for the next
    /// sequence. Reaping the last active row has nobody to move over it, so a
    /// missing reset used to leave the row at the old length and the next
    /// sequence appended after stale positions — plausible logits, wrong answer,
    /// and eventually a spurious context-full stop.
    /// </summary>
    [Fact]
    public async Task ReusedRow_StartsFromAnEmptyCache()
    {
        const int maxNew = 4;
        int[] firstPrompt = Prompt(9, 1001);
        int[] secondPrompt = Prompt(6, 1002);

        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        int[] expected = await ReferenceAsync(model, tokenizer, secondPrompt, maxNew);

        await using var batch = new BatchedGenerator(model, tokenizer, new BatchedGeneratorOptions { MaxBatch = 1 });

        var first = await batch.EnqueueTokensAsync(firstPrompt, SamplingConfig.Greedy, Gen(maxNew));
        await first.WhenCompleted.WaitAsync(TimeSpan.FromMinutes(2));
        Assert.Equal(maxNew, first.GeneratedIds.Count);

        var second = await batch.EnqueueTokensAsync(secondPrompt, SamplingConfig.Greedy, Gen(maxNew));
        await second.WhenCompleted.WaitAsync(TimeSpan.FromMinutes(2));

        Assert.Equal(expected, second.GeneratedIds);
        Assert.Equal(BatchFinishReason.MaxTokens, second.FinishReason);
    }

    [Fact]
    public async Task StopTokenIds_StopOnlyTheirOwnSequence()
    {
        const int maxNew = 20;
        int[] promptA = Prompt(5, 901);
        int[] promptB = Prompt(5, 902);

        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        int[] answerA = await ReferenceAsync(model, tokenizer, promptA, maxNew);

        // Make the first token A would produce a stop for A only.
        int stopId = answerA[0];

        await using var batch = new BatchedGenerator(model, tokenizer, new BatchedGeneratorOptions { MaxBatch = 2 });

        var a = await batch.EnqueueTokensAsync(promptA, SamplingConfig.Greedy,
            new GenerationConfig { MaxNewTokens = maxNew, RepetitionPenalty = 1f, StopTokenIds = [stopId], Stream = false });
        var b = await batch.EnqueueTokensAsync(promptB, SamplingConfig.Greedy, Gen(maxNew));

        await Task.WhenAll(a.WhenCompleted, b.WhenCompleted).WaitAsync(TimeSpan.FromMinutes(2));

        Assert.Equal(BatchFinishReason.StopToken, a.FinishReason);
        Assert.Single(a.GeneratedIds);
        Assert.Equal(BatchFinishReason.MaxTokens, b.FinishReason);
        Assert.Equal(maxNew, b.GeneratedIds.Count);
    }
}
