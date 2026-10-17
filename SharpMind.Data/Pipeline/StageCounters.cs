namespace SharpMind.Data.Pipeline;

/// <summary>
/// Cumulative counters for one stage of a cleaning pipeline, as observed by a
/// <see cref="CountingStage"/>. <see cref="In"/> counts the documents the stage
/// was offered; <see cref="Out"/> those it kept; the rest were dropped
/// (<see cref="Dropped"/>). Bytes are UTF-8 sizes of the document text before
/// and after the stage.
/// </summary>
/// <param name="Stage">Name of the stage these counters belong to.</param>
/// <param name="In">Documents offered to the stage so far.</param>
/// <param name="Out">Documents the stage kept so far.</param>
/// <param name="BytesIn">UTF-8 bytes offered to the stage so far.</param>
/// <param name="BytesOut">UTF-8 bytes emitted by the stage so far.</param>
public readonly record struct StageCounters(
    string Stage,
    long In,
    long Out,
    long BytesIn,
    long BytesOut)
{
    /// <summary>Documents the stage discarded (<see cref="In"/> − <see cref="Out"/>).</summary>
    public long Dropped => In - Out;

    /// <summary>Change in document bytes across the stage (<see cref="BytesOut"/> − <see cref="BytesIn"/>).</summary>
    public long BytesDelta => BytesOut - BytesIn;
}

/// <summary>
/// Receives live per-stage counters from a <see cref="CountingStage"/>. Implement
/// it to aggregate a funnel, log progress, or drive a UI. Implementations must be
/// cheap — a stage reports after every document.
/// </summary>
public interface IStageCountSink
{
    /// <summary>Called with the stage's latest cumulative counters after each document.</summary>
    void Report(StageCounters counters);
}

/// <summary>
/// An <see cref="IStageCountSink"/> that keeps the newest snapshot per stage,
/// keyed by <see cref="StageCounters.Stage"/>. Each report replaces the previous
/// snapshot for that stage (the counters are cumulative, not deltas).
/// </summary>
public sealed class StageCountCollector : IStageCountSink
{
    private readonly Dictionary<string, StageCounters> _byStage = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public void Report(StageCounters counters)
        => _byStage[counters.Stage] = counters;

    /// <summary>Latest snapshot for each stage seen, keyed by stage name.</summary>
    public IReadOnlyDictionary<string, StageCounters> Counters => _byStage;

    /// <summary>Latest counters for <paramref name="stage"/>, or default when never reported.</summary>
    public StageCounters this[string stage]
        => _byStage.TryGetValue(stage, out var c) ? c : default;
}

/// <summary>An <see cref="IStageCountSink"/> backed by a delegate.</summary>
public sealed class DelegateStageCountSink(Action<StageCounters> onReport) : IStageCountSink
{
    private readonly Action<StageCounters> _onReport =
        onReport ?? throw new ArgumentNullException(nameof(onReport));

    /// <inheritdoc/>
    public void Report(StageCounters counters) => _onReport(counters);
}
