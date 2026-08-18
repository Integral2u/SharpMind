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
}