using System.Runtime.CompilerServices;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Compatibility name for SharpMind's multi-token decoder.
///
/// This class used to run a self-draft loop that sampled a draft from the same
/// logits as the trunk, accepted it only when it matched the greedy token, and
/// emitted the greedy token otherwise. That made it standard decoding under
/// greedy sampling (one forward per token, plus snapshot overhead) and silently
/// forced greedy decoding under temperature/top-p.
///
/// It now delegates to <see cref="StandardGenerator{T}"/>, which performs real
/// sampling and does multi-token drafting through the model's MTP head or
/// prompt-lookup drafting when <see cref="GenerationConfig"/> enables them.
/// Kept as a separate type so existing builders, samples, and saved profiles
/// keep working; new code should use <see cref="StandardGenerator{T}"/>.
/// </summary>
public sealed class SpeculativeGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new()
{
    private readonly StandardGenerator<T> _inner;
    private bool _disposed;

    public string Name { get; init; } = "Speculative";

    public SpeculativeGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]? caches = null,
        int? seed = null,
        int? maxCacheLen = null)
    {
        _inner = new StandardGenerator<T>(
            model, tokenizer, addBos, addEos, caches, seed, maxCacheLen);
    }

    /// <summary>Fired once for every generated token id.</summary>
    public Action<int>? OnTokenGenerated
    {
        get => _inner.OnTokenGenerated;
        set => _inner.OnTokenGenerated = value;
    }

    /// <summary>MTP drafts proposed/accepted in the last generation.</summary>
    public (int Drafted, int Accepted) MtpStats => _inner.MtpStats;

    /// <summary>Prompt-lookup drafts proposed/accepted in the last generation.</summary>
    public PromptLookupStats PromptLookupStats => _inner.PromptLookupStats;

    public IReadOnlyList<IKVCache> Caches => _inner.Caches;

    public Action<double>? PrefillProgress
    {
        get => _inner.PrefillProgress;
        set => _inner.PrefillProgress = value;
    }

    public IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default) =>
        _inner.GenerateAsync(prompt, sampling, generation, cancellationToken);

    public IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[] promptIds,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default) =>
        _inner.GenerateFromTokensAsync(promptIds, sampling, generation, cancellationToken);

    public void ResetCache() => _inner.ResetCache();

    public void TruncateCache(int length) => _inner.TruncateCache(length);

    public void SetCacheTokens(IReadOnlyList<int> tokens) => _inner.SetCacheTokens(tokens);

    public IReadOnlyList<int>? CacheTokens => _inner.CacheTokens;

    public IReadOnlyList<int>? CurrentGeneratedIds => _inner.CurrentGeneratedIds;

    public float CacheFillRatio => _inner.CacheFillRatio;

    public float? TokensPerSecond => _inner.TokensPerSecond;

    public float? CumulativeTokensPerSecond => _inner.CumulativeTokensPerSecond;

    public float? TimeToFirstToken => _inner.TimeToFirstToken;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _inner.Dispose();
        GC.SuppressFinalize(this);
    }
}
