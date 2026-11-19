using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using SharpMind.Model.Config;

namespace SharpMind.Inference;

/// <summary>
/// One caller's view of a sequence inside <see cref="BatchedGenerator"/>: a
/// stream of decoded text fragments plus the ids, finish reason and full text
/// once it ends. The scheduler owns the row; the caller only reads.
///
/// The stream is single-reader. A caller that stops reading should dispose the
/// sequence so its cache row is freed for the next request instead of running
/// to completion for nobody.
/// </summary>
public sealed class BatchedSequence : IAsyncDisposable
{
    private readonly Channel<string> _out;
    private readonly TaskCompletionSource _completed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _released = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly CancellationTokenSource _abandon = new();
    private readonly CancellationTokenRegistration _cancellationRegistration;
    private readonly int[] _promptTokens;

    internal BatchedSequence(int id, int[] promptIds, SamplingConfig sampling, GenerationConfig generation, CancellationToken cancellation)
    {
        Id = id;
        _promptTokens = promptIds;
        Sampling = sampling;
        Generation = generation;
        MaxStopLength = generation.StopStrings.Count == 0 ? 0 : generation.StopStrings.Max(s => s.Length);
        _out = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
            AllowSynchronousContinuations = false,
        });

        _cancellationRegistration = cancellation.Register(
            static state => ((CancellationTokenSource)state!).Cancel(), _abandon);
        AbandonToken = _abandon.Token;
    }

    /// <summary>Monotonic id within the generator (for logs and diagnostics).</summary>
    public int Id { get; }

    /// <summary>The prompt tokens this sequence was created with.</summary>
    public IReadOnlyList<int> PromptIds => _promptTokens;

    internal int[] PromptTokens => _promptTokens;

    /// <summary>Tokens generated so far (valid while streaming, frozen at the end).</summary>
    public IReadOnlyList<int> GeneratedIds => Generated;

    public BatchFinishReason FinishReason { get; internal set; }
    public Exception? Error { get; internal set; }

    /// <summary>
    /// The whole completion. For a streaming request this is the concatenation
    /// of the fragments yielded by <see cref="StreamAsync"/>; for a
    /// non-streaming request it is the single fragment. Available once the
    /// sequence has completed.
    /// </summary>
    public string Text { get; internal set; } = "";

    /// <summary>Completes when the sequence ends, whatever the reason.</summary>
    public Task WhenCompleted => _completed.Task;

    /// <summary>
    /// Completes once the scheduler has dropped this sequence and its cache row
    /// is free. A host that holds the model for the request's duration can await
    /// it before releasing the model, so an unload cannot race the batcher's
    /// last forward pass.
    /// </summary>
    public Task WhenReleased => _released.Task;

    public bool IsCompleted => _completed.Task.IsCompleted;

    /// <summary>Marks the cache row free. Called by the scheduler exactly once per removal.</summary>
    internal void MarkReleased() => _released.TrySetResult();

    /// <summary>Fired once per prefill chunk with the fraction of the prompt fed so far (0..1).</summary>
    public Action<double>? PrefillProgress { get; set; }

    // ── Scheduler-owned state ────────────────────────────────────────────────
    // Kept on the sequence so a step can work row-by-row without a side table.

    internal int Slot { get; set; }
    internal SequenceState State { get; set; }
    internal int PromptFed { get; set; }
    internal int PendingToken { get; set; }
    internal bool Stream => Generation.Stream;
    internal bool EndedOnStop { get; set; }
    internal readonly List<int> Generated = [];
    internal readonly RepetitionPenaltyState Penalty = new();
    internal readonly StringBuilder Decoded = new();
    internal int DetokPrefix;
    internal int DetokRead;
    internal float[]? LogitScratch;
    internal char[]? StopScratch;
    internal int MaxStopLength { get; }
    internal Random Rng = Random.Shared;
    internal SamplingConfig Sampling { get; }
    internal GenerationConfig Generation { get; }

    // ── MTP draft state ──────────────────────────────────────────────────────
    // The head keeps its own cache row; these buffers hold the trunk hidden of
    // the last verified token and the chain hidden that feeds the last draft.
    internal float[]? MtpHidden;
    internal float[]? MtpChainHidden;
    internal float[]? MtpRunHidden;
    internal int[]? MtpDrafts;
    internal int MtpDraftCount;
    internal int MtpTrunkLenBeforeRun;
    internal int MtpLenBeforeRun;
    internal bool MtpReady;
    internal int MtpDrafted;
    internal int MtpAccepted;

    /// <summary>MTP drafts proposed/accepted for this sequence (zero when it was off).</summary>
    public (int Drafted, int Accepted) MtpStats => (MtpDrafted, MtpAccepted);

    /// <summary>Cancels when the caller cancels the enqueue token or disposes this handle.</summary>
    internal CancellationToken AbandonToken { get; }

    internal void Emit(string fragment) => _out.Writer.TryWrite(fragment);

    internal void Complete(BatchFinishReason reason, string text)
    {
        if (IsCompleted) return;
        FinishReason = reason;
        Text = text;
        _out.Writer.TryComplete();
        _completed.TrySetResult();
        Cleanup();
    }

    internal void Fail(Exception error)
    {
        if (IsCompleted) return;
        FinishReason = BatchFinishReason.Error;
        Error = error;
        Text = Decoded.ToString();
        _out.Writer.TryComplete(error);
        _completed.TrySetException(error);
        Cleanup();
    }

    private void Cleanup()
    {
        _cancellationRegistration.Dispose();
        _abandon.Dispose();
    }

    /// <summary>
    /// Reads the decoded fragments as they are produced. Completes when the
    /// sequence ends; throws the generation error if one occurred.
    /// </summary>
    public async IAsyncEnumerable<string> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (string fragment in _out.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            yield return fragment;
    }

    /// <summary>
    /// Abandons the sequence: the scheduler drops it at the next step and frees
    /// its cache row. Safe to call after completion.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        if (!_abandon.IsCancellationRequested)
        {
            try { _abandon.Cancel(); } catch (ObjectDisposedException) { }
        }
        return ValueTask.CompletedTask;
    }
}

internal enum SequenceState
{
    Pending,
    Prefilling,
    Decoding,
    Finished,
}
