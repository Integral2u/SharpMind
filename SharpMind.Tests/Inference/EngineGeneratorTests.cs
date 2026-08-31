using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The decode-loop control logic of <see cref="EngineGenerator{T}"/>, exercised against a stub
/// engine so no model/GPU is needed. Backend correctness (logits/caches) is covered by the GPU
/// engine tests that compare EngineGenerator against StandardGenerator end-to-end.
/// </summary>
public sealed class EngineGeneratorTests
{
    /// <summary>A stub engine that always returns the highest logit on a fixed token id.</summary>
    private sealed class StubEngine(int fixedId, int maxCache = 32) : IInferenceEngine
    {
        private int _cached;
        private readonly float[] _logits = new float[300];

        public int CachedLength => _cached;
        public int MaxCacheLength => maxCache;
        public bool IsCacheFull => _cached >= maxCache;
        public string Description => "CPU";

        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress = null, CancellationToken ct = default)
        {
            _cached += tokenIds.Length;
            return Logits();
        }

        public ReadOnlyMemory<float> DecodeStep(int tokenId, CancellationToken ct = default)
        {
            _cached++;
            return Logits();
        }

        private ReadOnlyMemory<float> Logits()
        {
            Array.Clear(_logits);
            _logits[fixedId] = 10f;
            return _logits;
        }

        public void TruncateCache(int length) => _cached = length;
        public void TrimToLast(int keep) => _cached = keep;
        public void ResetCache() => _cached = 0;
        public KVCacheSnapshot ExportCache(int[] promptTokenIds) => throw new NotImplementedException();
        public void ImportCache(KVCacheSnapshot snapshot) => throw new NotImplementedException();
        public void Dispose() { }
    }

    private static Tokenizer MakeTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }

    [Fact]
    public async Task HonoursMaxNewTokens()
    {
        using var engine = new StubEngine(fixedId: 70); // "F"
        using var gen = new EngineGenerator<KVCacherBuilder>(engine, MakeTokenizer(), addBos: false, addEos: false, numLayers: 2);

        int fragments = 0;
        await foreach (var f in gen.GenerateFromTokensAsync([65], generation: new GenerationConfig { MaxNewTokens = 5, Stream = true }))
            if (f.Length > 0) fragments++;

        Assert.Equal(5, fragments);
        Assert.Equal(5, gen.CurrentGeneratedIds!.Count);
    }

    [Fact]
    public async Task HaltsOnStopTokenId()
    {
        int stopId = 71; // "G" - also the fixed emitted id, so decode halts on the first step
        using var engine = new StubEngine(fixedId: stopId);
        using var gen = new EngineGenerator<KVCacherBuilder>(engine, MakeTokenizer(), addBos: false, addEos: false, numLayers: 2);

        await foreach (var _ in gen.GenerateFromTokensAsync([65],
            generation: new GenerationConfig { MaxNewTokens = 20, Stream = false, StopTokenIds = [stopId] }))
        {
        }

        Assert.Single(gen.CurrentGeneratedIds!);
    }

    [Fact]
    public async Task StreamsAndReportsRateStats()
    {
        using var engine = new StubEngine(fixedId: 70);
        using var gen = new EngineGenerator<KVCacherBuilder>(engine, MakeTokenizer(), addBos: false, addEos: false, numLayers: 2);

        await foreach (var _ in gen.GenerateFromTokensAsync([65],
            generation: new GenerationConfig { MaxNewTokens = 8, Stream = true }))
        {
        }

        Assert.NotNull(gen.TokensPerSecond);
        Assert.NotNull(gen.CumulativeTokensPerSecond);
        Assert.Equal(8, gen.CacheTokens!.Count - 1); // 8 generated after the 1-token prompt
    }

    [Fact]
    public void ResetCache_ForwardsToEngine()
    {
        using var engine = new StubEngine(fixedId: 70);
        using var gen = new EngineGenerator<KVCacherBuilder>(engine, MakeTokenizer(), addBos: false, addEos: false, numLayers: 2);

        Assert.Equal(0, engine.CachedLength);
        gen.SetCacheTokens([1, 2, 3]);
        gen.ResetCache();
        Assert.Equal(0, engine.CachedLength);
        Assert.Empty(gen.CacheTokens!);
    }
}
