using System.Runtime.CompilerServices;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Presents a shared <see cref="BatchedGenerator"/> as an ordinary
/// <see cref="IGenerator{T}"/>, so a <c>ChatSession</c> can generate through the
/// batch instead of owning a private generator.
///
/// The adapter deliberately owns no conversation state: every generation is a
/// fresh batch sequence carrying the whole prompt. That is the price of sharing
/// one forward pass between callers — a per-session KV cache and the
/// cross-request prefix cache are single-sequence features, so they are disabled
/// here (<see cref="CacheTokens"/> is null, <see cref="ResetCache"/> is a no-op,
/// and the exposed caches are placeholders that hold nothing). Nothing may
/// restore a cache snapshot into <see cref="Caches"/>: those rows belong to the
/// batcher.
/// </summary>
public sealed class BatchedGeneratorAdapter<K> : IGenerator<K> where K : IKVCacheBuilder, new()
{
    private readonly BatchedGenerator _batcher;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly bool _addBos;
    private readonly bool _addEos;
    private readonly IKVCache[] _placeholder;
    private BatchedSequence? _current;
    private BatchedSequence? _last;

    /// <param name="numLayers">
    /// Placeholder count for <see cref="Caches"/>. The caches hold nothing; hosts
    /// only read their count and <see cref="IKVCache.SupportsTruncate"/>.
    /// </param>
    public BatchedGeneratorAdapter(
        BatchedGenerator batcher,
        Tokenization.Tokenizer tokenizer,
        int numLayers,
        bool addBos = false,
        bool addEos = false)
    {
        ArgumentNullException.ThrowIfNull(batcher);
        ArgumentNullException.ThrowIfNull(tokenizer);

        _batcher = batcher;
        _tokenizer = tokenizer;
        _addBos = addBos;
        _addEos = addEos;
        _placeholder = new IKVCache[Math.Max(1, numLayers)];
        for (int i = 0; i < _placeholder.Length; i++)
            _placeholder[i] = new KVCache(1, 1, 1, 1);
    }

    public string Name => "Batched";

    public Action<double>? PrefillProgress { get; set; }

    /// <summary>Ids of the last sequence this adapter ran (the current one while it streams).</summary>
    public IReadOnlyList<int>? CurrentGeneratedIds => (_current ?? _last)?.GeneratedIds;

    /// <summary>
    /// Single-sequence KV state does not exist here; every request prefills its
    /// own prompt. Returned so hosts can ask about truncatability; never written.
    /// </summary>
    public IReadOnlyList<IKVCache> Caches => _placeholder;

    /// <summary>Always null: the batch owns the rows, not this session.</summary>
    public IReadOnlyList<int>? CacheTokens => null;

    public float CacheFillRatio => 0f;
    public float? TokensPerSecond => null;
    public float? CumulativeTokensPerSecond => null;
    public float? TimeToFirstToken => null;

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int[] promptIds = _tokenizer.Encode(prompt, _addBos, _addEos);
        await foreach (string fragment in GenerateFromTokensAsync(promptIds, sampling, generation, cancellationToken)
                           .ConfigureAwait(false))
            yield return fragment;
    }

    public async IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[] promptIds,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var sequence = await _batcher.EnqueueTokensAsync(promptIds, sampling, generation, cancellationToken)
            .ConfigureAwait(false);

        sequence.PrefillProgress = PrefillProgress;
        _current = sequence;
        _last = sequence;

        try
        {
            await foreach (string fragment in sequence.StreamAsync(cancellationToken).ConfigureAwait(false))
                yield return fragment;
        }
        finally
        {
            _current = null;
            await sequence.DisposeAsync().ConfigureAwait(false);
            // The row must be free before the session can start another
            // generation, or a slow client could pin rows it no longer reads.
            await sequence.WhenReleased.ConfigureAwait(false);
        }
    }

    /// <summary>No per-session cache exists to reset.</summary>
    public void ResetCache() { }

    /// <summary>No per-session cache exists to truncate.</summary>
    public void TruncateCache(int length) { }

    /// <summary>No per-session cache exists to vouch for.</summary>
    public void SetCacheTokens(IReadOnlyList<int> tokens) { }

    /// <summary>The batcher is shared and owned by the host; this adapter owns nothing.</summary>
    public void Dispose()
    {
        foreach (var cache in _placeholder) cache.Dispose();
    }
}
