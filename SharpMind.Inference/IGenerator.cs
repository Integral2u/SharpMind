using System.Runtime.CompilerServices;

namespace SharpMind.Inference;
//public interface IGenerator<T> : IDisposable where T: IKVCache
public interface IGenerator : IDisposable
{
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

    void ResetCache();
    float CacheFillRatio { get; }
    float? TokensPerSecond { get; }
    float? CumulativeTokensPerSecond { get; }
}
