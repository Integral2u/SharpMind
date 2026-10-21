using SharpMind.Core;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

public sealed class AbsolutePositionTrackingTests
{
    private static ModelConfig Config => new()
    {
        VocabSize = 128,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 64,
        MaxSeqLen = 128,
    };

    private static Transformer BuildModel(ModelConfig? cfg = null)
    {
        cfg ??= Config;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static IKVCache[] BuildCaches(ModelConfig cfg, int capacity) =>
        [.. Enumerable.Range(0, cfg.NumLayers)
            .Select(_ => (IKVCache)new KVCache(1, cfg.NumKvHeads, capacity, cfg.HeadDim))];

    private static int[] Prompt(int length, int seed = 7)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, length).Select(_ => rng.Next(Config.VocabSize))];
    }

    [Fact]
    public void ChunkedPrefill_StartPosition_ShiftsPromptAndNextTokenTogether()
    {
        int[] promptIds = Prompt(5);
        int[] next = [3];

        using var model = BuildModel();
        var cfg = Config;
        var atZero = BuildCaches(cfg, 64);
        var atOffset = BuildCaches(cfg, 64);
        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(cfg.HiddenDim, cfg.FfnDim, cfg.VocabSize, cfg.NumLayers, cfg.MaxSeqLen));

        float[] expected;
        using (var _ = Prefill.ForwardLastLogitsChunked(model, atZero, promptIds, workspace, startPosition: 0)) { }
        using (var logits = Prefill.ForwardLastLogitsChunked(model, atZero, next, workspace, startPosition: promptIds.Length))
            expected = logits.Data[..cfg.VocabSize].ToArray();

        float[] actual;
        using (var _ = Prefill.ForwardLastLogitsChunked(model, atOffset, promptIds, workspace, startPosition: 35)) { }
        using (var logits = Prefill.ForwardLastLogitsChunked(model, atOffset, next, workspace, startPosition: 40))
            actual = logits.Data[..cfg.VocabSize].ToArray();

        // RoPE depends on relative distance: the same tokens at positions
        // 35..40 must produce the same logits as at 0..5.
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-3f,
                $"logit[{i}] zero={expected[i]} offset={actual[i]}");

        // Leaving the next token at the cache length (the old bug) rotates it
        // to position 5 instead of 40 and must change the logits.
        float[] stale;
        using (var logits = Prefill.ForwardLastLogitsChunked(
            model, atOffset, next, workspace, startPosition: atOffset[0].Length))
            stale = logits.Data[..cfg.VocabSize].ToArray();

        Assert.Contains(Enumerable.Range(0, expected.Length),
            i => MathF.Abs(expected[i] - stale[i]) > 1e-4f);

        foreach (var c in atZero) c.Dispose();
        foreach (var c in atOffset) c.Dispose();
    }

    [Fact]
    public async Task StandardGenerator_KeepsAbsolutePositionAcrossSlidingWindowTrim()
    {
        var cfg = Config with { SlidingWindowSize = 16 };
        using var model = BuildModel(cfg);
        var tokenizer = BuildTokenizer();

        const int maxNewTokens = 40;
        int[] prompt = Prompt(3);
        int[] expected = ReferenceGreedy(model, cfg, prompt, maxNewTokens, keep: 8);

        using var gen = new StandardGenerator<KVCacherBuilder>(model, tokenizer,
            addBos: false, addEos: false, seed: 1);
        var actual = new List<int>();
        gen.OnTokenGenerated += actual.Add;

        var generation = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            RepetitionPenalty = 1f,
            SlidingWindowSize = 8,
            Stream = false,
        };

        await foreach (var _ in gen.GenerateFromTokensAsync(prompt, SamplingConfig.Greedy, generation))
        {
        }

        Assert.Equal(expected, actual);
        Assert.True(gen.Caches[0].Length < prompt.Length + maxNewTokens,
            "The sliding window should have trimmed the cache.");
    }

    [Fact]
    public async Task SpeculativeGenerator_SmallSlidingWindow_CompletesWithoutCacheOverflow()
    {
        var cfg = Config with { SlidingWindowSize = 16 };
        using var model = BuildModel(cfg);
        var tokenizer = BuildTokenizer();
        using var gen = new SpeculativeGenerator<KVCacherBuilder>(model, tokenizer,
            addBos: false, addEos: false, seed: 1);

        var generation = new GenerationConfig
        {
            MaxNewTokens = 40,
            RepetitionPenalty = 1f,
            SlidingWindowSize = 8,
            Stream = false,
        };

        int chars = 0;
        await foreach (var fragment in gen.GenerateFromTokensAsync(
            Prompt(3), SamplingConfig.Greedy, generation))
        {
            chars += fragment.Length;
        }

        Assert.True(chars > 0);
        Assert.True(gen.Caches[0].Length > 3);
        Assert.All(gen.Caches, c => Assert.True(c.Length <= c.MaxSeqLen,
            $"cache length {c.Length} exceeded capacity {c.MaxSeqLen}"));
    }

    [Fact]
    public async Task MedusaGenerator_SmallSlidingWindow_CompletesWithoutCacheOverflow()
    {
        var cfg = Config with { SlidingWindowSize = 16 };
        using var model = BuildModel(cfg);
        var tokenizer = BuildTokenizer();
        using var gen = new MedusaGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, tokenizer, addBos: false, addEos: false, caches: null, seed: 1);

        var generation = new GenerationConfig
        {
            MaxNewTokens = 40,
            RepetitionPenalty = 1f,
            SlidingWindowSize = 8,
            Stream = false,
        };

        int chars = 0;
        await foreach (var fragment in gen.GenerateFromTokensAsync(
            Prompt(3), SamplingConfig.Greedy, generation))
        {
            chars += fragment.Length;
        }

        Assert.True(chars > 0);
        Assert.True(gen.Caches[0].Length > 3);
        Assert.All(gen.Caches, c => Assert.True(c.Length <= c.MaxSeqLen,
            $"cache length {c.Length} exceeded capacity {c.MaxSeqLen}"));
    }

    private static int[] ReferenceGreedy(
        Transformer model, ModelConfig cfg, int[] prompt, int maxNewTokens, int keep)
    {
        var caches = BuildCaches(cfg, ModelConfig.ComputeMaxCacheLength(cfg));
        try
        {
            using var promptInput = Tensor<int>.From(prompt, 1, prompt.Length);
            float[] logits;
            using (var initial = model.ForwardLastLogits(promptInput, caches, 0, null))
                logits = initial.Data[..cfg.VocabSize].ToArray();

            int position = prompt.Length;
            var ids = new List<int>(maxNewTokens);
            for (int step = 0; step < maxNewTokens; step++)
            {
                int next = ArgMax(logits);
                ids.Add(next);

                if (step + 1 == maxNewTokens)
                    break;

                if (caches[0].IsFull)
                {
                    for (int i = 0; i < caches.Length; i++)
                        caches[i].TrimToLast(keep);
                }

                using var stepInput = Tensor<int>.From([next], 1, 1);
                using var stepLogits = model.ForwardLastLogits(stepInput, caches, position, null);
                logits = stepLogits.Data[..cfg.VocabSize].ToArray();
                position++;
            }

            return [.. ids];
        }
        finally
        {
            foreach (var c in caches) c.Dispose();
        }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }
}
