namespace SharpMind.Inference.Chat;

/// <summary>
/// What the session should do with a tool call after handing it to an
/// external <see cref="IChatSession.ProcessToolRequest"/> handler.
/// </summary>
public enum ToolRequestOutcome
{
    /// <summary>
    /// The external handler dispatched the tool and produced a result; the
    /// session feeds that result back and continues its agentic loop.
    /// </summary>
    Handled,

    /// <summary>
    /// The external handler did not recognise the tool (or chose not to run
    /// it); the session dispatches it natively through its agent builder,
    /// including the File/Network permission gate.
    /// </summary>
    Defer,

    /// <summary>
    /// The session stops dispatching, hands the entire tool call back to the
    /// caller (via a <see cref="ChatStreamEntry"/> with status
    /// <see cref="ChatStatus.ToolCall"/>), and ends the turn without running
    /// the tool or feeding a result back.
    /// </summary>
    ReturnToCaller
}

/// <summary>
/// The outcome of an external tool-request handler, as returned from
/// <see cref="IChatSession.ProcessToolRequest"/>. Use the static helpers to
/// construct well-formed values:
/// <code>
///   ToolRequestResult.Handled("42")
///   ToolRequestResult.Defer()
///   ToolRequestResult.ReturnToCaller()
/// </code>
/// </summary>
public sealed class ToolRequestResult
{
    public ToolRequestOutcome Outcome { get; }
    public string? Result { get; }

    private ToolRequestResult(ToolRequestOutcome outcome, string? result)
    {
        Outcome = outcome;
        Result = result;
    }

    /// <summary>Report that the handler ran the tool. <paramref name="result"/> is fed back to the model.</summary>
    public static ToolRequestResult Handled(string? result) => new(ToolRequestOutcome.Handled, result);

    /// <summary>Report that the handler did not run the tool; the session dispatches it natively.</summary>
    public static ToolRequestResult Defer() => new(ToolRequestOutcome.Defer, null);

    /// <summary>Report that the tool call should be returned to the caller and the turn ended.</summary>
    public static ToolRequestResult ReturnToCaller() => new(ToolRequestOutcome.ReturnToCaller, null);
}
