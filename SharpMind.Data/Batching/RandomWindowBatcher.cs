using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Data.Batching;

/// <summary>
/// Random contiguous-window batcher — the nanoGPT-style data feed. Consumes the
/// document stream once, concatenates every token into a single flat buffer,
/// then produces each batch of <see cref="BatchSize"/> rows as a random
/// contiguous <see cref="SeqLen"/>-token window sliced from that buffer
/// (labels are the window shifted by one token, exactly matching
/// <c>x = idx[t:t+block], y = idx[t+1:t+block+1]</c>).
///
/// Contrast with <see cref="PackingBatcher"/>, which packs/filters documents.
/// Random windows over one long body of text (a single-file character corpus,
/// for example) expose the model to every position of the corpus over time and
/// are the standard feed for small-character GPTs.
///
/// The flat buffer is cached on first use, so the epoch-style re-enumeration in
/// <see cref="DataLoader"/> never re-reads or re-tokenises the source. Pass a
/// <paramref name="seed"/> for reproducible runs; the default is a fresh
/// time-seeded <see cref="Random"/> (matching the Python example's per-step
/// <c>torch.randint</c>).
/// </summary>
public sealed class RandomWindowBatcher : IBatchStrategy
{
    private readonly int _batchSize;
    private readonly int _seqLen;
    private readonly Random _random;
    private int[]? _buffer;

    public RandomWindowBatcher(int batchSize, int seqLen, int? seed = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);

        _batchSize = batchSize;
        _seqLen = seqLen;
        _random = seed is { } s ? new Random(s) : new Random();
    }

    public async IAsyncEnumerable<TrainingBatch> BatchAsync(
        IAsyncEnumerable<string> documents,
        Func<string, int[]> tokenise,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(tokenise);

        // Buffer the whole corpus once; later DataLoader passes reuse it and
        // never re-read the (possibly expensive) source documents.
        if (_buffer is null)
        {
            var tokens = new List<int>();
            await foreach (string doc in documents.WithCancellation(cancellationToken))
            {
                if (string.IsNullOrEmpty(doc)) continue;
                var ids = tokenise(doc);
                if (ids.Length > 0)
                    tokens.AddRange(ids);
            }

            if (tokens.Count < _seqLen + 1)
                throw new InvalidOperationException(
                    $"RandomWindowBatcher needs at least {_seqLen + 1} tokens in the corpus " +
                    $"to form a shifted {_seqLen}-token window, but only {tokens.Count} were produced.");

            _buffer = [.. tokens];
        }

        // Windows can be produced indefinitely; DataLoader's maxBatches budget
        // terminates the consumer. Each row samples an independent start offset.
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return BuildBatch();
        }
    }

    private TrainingBatch BuildBatch()
    {
        var tokenIds = new Tensor<int>(_batchSize, _seqLen);
        var labels = new Tensor<int>(_batchSize, _seqLen);
        var attentionMask = new Tensor<float>(_batchSize, _seqLen);

        // Need seqLen+1 consecutive tokens so labels can be the next-token shift.
        int maxStart = _buffer!.Length - _seqLen - 1;

        for (int b = 0; b < _batchSize; b++)
        {
            int start = _random.Next(0, maxStart + 1);
            var tokRow = tokenIds.RowSpan(b);
            var lblRow = labels.RowSpan(b);
            var mskRow = attentionMask.RowSpan(b);

            for (int i = 0; i < _seqLen; i++)
            {
                tokRow[i] = _buffer[start + i];
                lblRow[i] = _buffer[start + i + 1];
                mskRow[i] = 1f;
            }
        }

        return new TrainingBatch(tokenIds, labels, attentionMask, _batchSize * _seqLen);
    }
}