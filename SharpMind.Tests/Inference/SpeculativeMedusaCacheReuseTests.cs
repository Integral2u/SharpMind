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
/// Speculative and Medusa now report what their caches hold, so a caller (and
/// ChatSession's prefix matcher) can extend a warm cache instead of resetting.
/// The short-conv layer is in the mix: a real prefix extension must still only
/// prefill the new tail.
/// </summary>
public sealed class SpeculativeMedusaCacheReuseTests
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

    private sealed class SpyCache(IKVCache inner) : IKVCache
    {
        public readonly List<int> Writes = [];
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
        public byte[]? SnapshotBytes() => inner.SnapshotBytes();
        public void RestoreBytes(byte[] blob) => inner.RestoreBytes(blob);
        public void Dispose() => inner.Dispose();
    }

    private static (IKVCache[] caches, SpyCache spy) BuildCaches()
    {
        var spy = new SpyCache(new KVCache(1, Cfg.NumKvHeads, Cfg.MaxSeqLen, Cfg.HeadDimForLayer(0)));
        var shortConv = new ShortConvCache(Cfg.ShortConvCacheLength - 1, Cfg.HiddenDim, 1, Cfg.MaxSeqLen);
        return ([spy, shortConv], spy);
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

    private static async Task AssertExtendsWarmCache<T>(IGenerator<T> generator, SpyCache spy)
        where T : IKVCacheBuilder, new()
    {
        int[] prompt1 = [100, 101, 102]; // "abc"
        int[] first = await Generate(generator, prompt1, maxNew: 4);

        int[] resident = generator.CacheTokens!.ToArray();
        Assert.Equal([.. prompt1, .. first], resident);
        Assert.Equal(generator.Caches[0].Length, resident.Length);

        // ChatSession matches the resident ids against the new prompt and feeds
        // only the tail. The generator must extend from its tracked position.
        int[] extra = [10, 11, 12];
        spy.Writes.Clear();
        int[] second = await Generate(generator, extra, maxNew: 3);

        // Medusa writes whole draft batches (including the rejected suffix), so
        // only the first write is the prefill size: exactly the new tail.
        Assert.Equal(extra.Length, spy.Writes[0]);
        Assert.Equal([.. resident, .. extra, .. second], generator.CacheTokens!.ToArray());
    }

    [Fact]
    public async Task Speculative_ExtendsWarmCacheWithoutReprefilling()
    {
        using var model = BuildModel();
        var (caches, spy) = BuildCaches();
        using var generator = new SpeculativeGenerator<KVCacherBuilder>(
            model, BuildTokenizer(), addBos: false, addEos: false, caches: caches, seed: 1);

        await AssertExtendsWarmCache(generator, spy);
    }

    [Fact]
    public async Task Medusa_ExtendsWarmCacheWithoutReprefilling()
    {
        using var model = BuildModel();
        var (caches, spy) = BuildCaches();
        using var generator = new MedusaGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, BuildTokenizer(), addBos: false, addEos: false, caches: caches, seed: 1);

        await AssertExtendsWarmCache(generator, spy);
    }
}
