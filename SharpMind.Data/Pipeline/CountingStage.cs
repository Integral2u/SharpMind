using System.Text;

namespace SharpMind.Data.Pipeline;

/// <summary>
/// B1: a decorator that wraps any <see cref="ICleaningStage"/> and reports how
/// many documents (and UTF-8 bytes) went in and came out, to an
/// <see cref="IStageCountSink"/>. It changes nothing about the document stream —
/// the inner stage's result is passed through verbatim — so it can be dropped
/// into any pipeline to power a per-stage funnel without typed records.
///
/// Unlike ordinary stages it is stateful (it accumulates counters), which is the
/// whole point; the counters are updated with interlocked operations so a
/// pipeline read from one thread is safe, and a concurrent read would still not
/// tear a counter.
/// </summary>
public sealed class CountingStage : ICleaningStage
{
    private readonly ICleaningStage _inner;
    private readonly IStageCountSink _sink;

    private long _in;
    private long _out;
    private long _bytesIn;
    private long _bytesOut;

    /// <param name="inner">The stage to wrap.</param>
    /// <param name="sink">Receives a fresh snapshot after every document.</param>
    /// <param name="name">Overrides <see cref="ICleaningStage.Name"/>; defaults to the inner stage's.</param>
    public CountingStage(ICleaningStage inner, IStageCountSink sink, string? name = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _sink = sink ?? throw new ArgumentNullException(nameof(sink));
        Name = string.IsNullOrWhiteSpace(name) ? inner.Name : name;
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <summary>Latest cumulative counters for this stage.</summary>
    public StageCounters Counters => new(
        Name,
        Interlocked.Read(ref _in),
        Interlocked.Read(ref _out),
        Interlocked.Read(ref _bytesIn),
        Interlocked.Read(ref _bytesOut));

    /// <inheritdoc/>
    public string? Process(string document)
    {
        string? result = _inner.Process(document);

        Interlocked.Increment(ref _in);
        Interlocked.Add(ref _bytesIn, Encoding.UTF8.GetByteCount(document));
        if (result is not null)
        {
            Interlocked.Increment(ref _out);
            Interlocked.Add(ref _bytesOut, Encoding.UTF8.GetByteCount(result));
        }

        _sink.Report(Counters);
        return result;
    }
}
