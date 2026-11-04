using System.Runtime.CompilerServices;
using SharpMind.Data.Batching;
using SharpMind.Data.Records;

namespace SharpMind.Data;

/// <summary>
/// Streams batches from a record-aware batch strategy. This is the counterpart
/// of <see cref="DataLoader"/> for SFT and other record-shaped data: the source
/// re-enumerates for each epoch, and the same budget/skip semantics apply so a
/// checkpoint's batch cursor still resumes correctly.
/// </summary>
public sealed class RecordDataLoader : ITrainingBatchSource
{
    private readonly IRecordSource _source;
    private readonly IRecordBatchStrategy _batcher;
    private readonly Func<DataRecord, TokenizedExample> _tokenise;
    private readonly int? _maxBatches;
    private readonly Action<int, int>? _skipProgress;

    public RecordDataLoader(
        IRecordSource source,
        IRecordBatchStrategy batcher,
        Func<DataRecord, TokenizedExample> tokenise,
        int? maxBatches = null,
        Action<int, int>? skipProgress = null)
    {
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _batcher = batcher ?? throw new ArgumentNullException(nameof(batcher));
        _tokenise = tokenise ?? throw new ArgumentNullException(nameof(tokenise));
        _maxBatches = maxBatches;
        _skipProgress = skipProgress;
    }

    public string Description => $"{_source.Description} → {_batcher.GetType().Name}";

    public async IAsyncEnumerable<TrainingBatch> LoadAsync(
        int? maxBatches = null,
        int skipBatches = 0,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int? totalBudget = maxBatches ?? _maxBatches;
        int skipTotal = Math.Max(0, skipBatches);
        int skipRemaining = skipTotal;
        int? budget = totalBudget is { } total ? Math.Max(0, total - skipRemaining) : null;
        if (budget is 0)
        {
            yield break;
        }

        int produced = 0;
        bool firstPass = true;
        while (firstPass || budget is { } remaining && produced < remaining)
        {
            firstPass = false;
            int passBatches = 0;
            int passSkipped = 0;

            await foreach (TrainingBatch batch in _batcher.BatchAsync(
                _source.ReadRecordsAsync(cancellationToken), _tokenise, cancellationToken))
            {
                if (skipRemaining > 0)
                {
                    skipRemaining--;
                    passSkipped++;
                    batch.Dispose();
                    if (_skipProgress is not null
                        && (skipRemaining % 100 == 0 || skipRemaining == 0))
                        _skipProgress(skipTotal - skipRemaining, skipTotal);
                    continue;
                }

                if (budget is { } limit && produced >= limit)
                {
                    batch.Dispose();
                    yield break;
                }

                produced++;
                passBatches++;
                yield return batch;
            }

            // An empty pass would otherwise spin forever against a budget. A
            // pass that only satisfied the resume skip is not dead.
            if (passBatches == 0 && passSkipped == 0) break;
            if (budget is null) break;
        }
    }
}
