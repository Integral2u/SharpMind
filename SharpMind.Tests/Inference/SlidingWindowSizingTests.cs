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
/// Tests for KV-cache allocation capping, trim keep-size clamping, and
/// prefill mid-loop trimming when the model declares a sliding window.
/// </summary>
public sealed class SlidingWindowSizingTests : IClassFixture<SlidingWindowSizingTests.ModelFixture>
{
    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;

    public SlidingWindowSizingTests(ModelFixture fixture)
    {
        _model = fixture.Model;
        _tokenizer = fixture.Tokenizer;
    }

    public sealed class ModelFixture : IDisposable
    {
        public Transformer Model { get; }
        public Tokenizer Tokenizer { get; }

        public ModelFixture()
        {
            var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
            var weights = ModelFactory.CreateForTraining(Config, sharpConfig);
            WeightInitializer.InitializeRandomly(weights, 42);
            Model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
            Tokenizer = BuildTokenizer();
        }

        public void Dispose() => Model.Dispose();
    }

    /// <summary>
    /// A config with a sliding window smaller than MaxSeqLen to exercise
    /// the caching and trimming paths.
    /// </summary>
    private static ModelConfig Config => new()
    {
        VocabSize = 300,
        HiddenDim = 512,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 4,
        FfnDim = 1536,
        MaxSeqLen = 2048,
        SlidingWindowSize = 64,
    };

    private static ModelConfig FullContextConfig => new()
    {
        VocabSize = 300,
        HiddenDim = 512,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 4,
        FfnDim = 1536,
        MaxSeqLen = 2048,
        SlidingWindowSize = 0,
    };

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }

    // ── ModelConfig.EffectiveInferenceCacheLength ──

    [Fact]
    public void EffectiveCacheLength_SlidingWindow_ReturnsMinOfMaxSeqLenAndWindow()
    {
        Assert.Equal(64, Config.EffectiveInferenceCacheLength);
    }

    [Fact]
    public void EffectiveCacheLength_NoWindow_ReturnsMaxSeqLen()
    {
        Assert.Equal(2048, FullContextConfig.EffectiveInferenceCacheLength);
    }

    [Fact]
    public void EffectiveCacheLength_WindowLargerThanMaxSeqLen_ReturnsMaxSeqLen()
    {
        var cfg = new ModelConfig
        {
            VocabSize = 100,
            HiddenDim = 32,
            NumLayers = 1,
            NumHeads = 4,
            NumKvHeads = 4,
            FfnDim = 64,
            MaxSeqLen = 128,
            SlidingWindowSize = 4096,
        };
        Assert.Equal(128, cfg.EffectiveInferenceCacheLength);
    }

    // ── Generator cache capping ──

    [Fact]
    public void StandardGenerator_CacheCapacity_CappedToSlidingWindow()
    {
        using var gen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer,
            addBos: false, addEos: false, seed: 1);
        // Cache capacity should equal effective cache length (64)
        Assert.Equal(64, gen.Caches[0].MaxSeqLen);
    }

    [Fact]
    public async Task StandardGenerator_WindowCapacity_CompletesGeneration()
    {
        using var gen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer,
            addBos: false, addEos: false, seed: 1);

        // Generate with a prompt shorter than the window — should work fine.
        var prompt = new int[] { 65, 66, 67, 68, 69 };
        var genCfg = new GenerationConfig { MaxNewTokens = 10 };
        int count = 0;
        await foreach (var _ in gen.GenerateFromTokensAsync(prompt, generation: genCfg))
            count++;

        Assert.True(count > 0, "Generation should produce tokens.");
    }

    [Fact]
    public async Task StandardGenerator_LongPrompt_PrefillTrimsCacheWithoutThrow()
    {
        using var gen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer,
            addBos: false, addEos: false, seed: 1);

        // Prompt longer than the 64-token window — exercises the mid-loop trim.
        var prompt = new int[200];
        for (int i = 0; i < prompt.Length; i++)
            prompt[i] = 65 + (i % 60);

        var genCfg = new GenerationConfig { MaxNewTokens = 5 };
        int count = 0;
        await foreach (var _ in gen.GenerateFromTokensAsync(prompt, generation: genCfg))
            count++;

        Assert.True(count > 0, "Long prompt with small window should complete after mid-loop trim.");
    }

    // ── Trim keep-size clamp ──

    [Fact]
    public void KVCache_TrimToLast_KeepEqualsCapacity_Progresses()
    {
        // When keep >= CurrentPosition, TrimToLast is a no-op.
        // When keep < CurrentPosition, TrimToLast retains exactly 'keep' entries.
        var cache = new KVCache(1, 4, 16, 32);
        var k = new Tensor<float>(1, 4, 32);
        var v = new Tensor<float>(1, 4, 32);
        cache.Update(k, v, 4, 32);
        Assert.Equal(4, cache.Length);

        // TrimToLast with keep >= CurrentPosition should be a no-op.
        cache.TrimToLast(100);
        Assert.Equal(4, cache.Length);

        // TrimToLast with keep < CurrentPosition should work.
        cache.TrimToLast(2);
        Assert.Equal(2, cache.Length);
    }
}
