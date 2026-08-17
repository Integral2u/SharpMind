using SharpMind.Model;

namespace SharpMind.Inference;
public interface IGenerator<T> : IDisposable where T : IKVCacheBuilder, new()
{
    string Name { get; }
    IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[] promptIds,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Optional callback fired once per prefill chunk with the overall fraction
    /// (0..1) of the prompt prefilled so far. Lets a host surface "Prefilling
    /// NN.NN%" during the (potentially slow) first turn instead of appearing
    /// stuck. Set to <see langword="null"/> to suppress.
    /// </summary>
    Action<double>? PrefillProgress { get; set; }

    void ResetCache();
    float CacheFillRatio { get; }
    float? TokensPerSecond { get; }
    float? CumulativeTokensPerSecond { get; }
    float? TimeToFirstToken { get; }
    IReadOnlyList<int>? CurrentGeneratedIds => null;

    /// <summary>
    /// The token ids whose keys and values are resident in the KV cache, in
    /// position order — every prompt token that was prefilled plus every
    /// generated token that was fed back through the model — or
    /// <see langword="null"/> when the generator cannot vouch for the cache's
    /// contents (it never tracked them, or a sliding-window trim has moved
    /// entries away from their original positions). A caller that wants to
    /// continue a conversation compares its new prompt against this, keeps the
    /// common prefix with <see cref="TruncateCache"/>, and feeds only the rest.
    /// </summary>
    IReadOnlyList<int>? CacheTokens => null;

    /// <summary>
    /// Drops every cached position at or beyond <paramref name="length"/>, so the
    /// next generation continues from position <paramref name="length"/>. Valid
    /// because position <c>i</c> depends only on tokens before it. Generators
    /// that do not track their cache may simply reset it.
    /// </summary>
    void TruncateCache(int length) => ResetCache();
}
