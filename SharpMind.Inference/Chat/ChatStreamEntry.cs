using System.Text.Json.Nodes;

namespace SharpMind.Inference.Chat;

/// <summary>
/// Streaming response entry for real-time updates.
/// </summary>
public sealed class ChatStreamEntry
{
    public required ChatStatus Status { get; init; }
    public string? Token { get; init; }
    public ChatArtifact? Artifact { get; init; }
    public bool IsComplete { get; init; }
    public int? TokenId { get; init; }

    /// <summary>
    /// The parsed tool call <c>{"tool": "...", "arguments": { ... }}</c> when
    /// <see cref="Status"/> is <see cref="ChatStatus.ToolCall"/> — i.e. when an
    /// external <see cref="IChatSession.ProcessToolRequest"/> handler asked the
    /// session to hand the call back to the caller. Null otherwise.
    /// </summary>
    public JsonObject? ToolCall { get; init; }

    /// <summary>
    /// Live tokens-per-second at this point in the stream.
    /// Rolling average over the last N tokens during generation;
    /// final cumulative rate on the <see cref="ChatStatus.Complete"/> entry.
    /// Null before generation begins.
    /// </summary>
    public float? TokensPerSecond { get; init; }

    /// <summary>Seconds from start to first output token (includes prefill).</summary>
    public float? TimeToFirstToken { get; init; }

    /// <summary>
    /// Set when generation ended because of an unexpected exception rather than
    /// a user interruption. Without this an internal failure was indistinguishable
    /// from a cancelled turn — it surfaced as an empty reply and nothing else.
    /// Null for normal streaming and for genuine cancellation.
    /// </summary>
    public string? Error { get; init; }
}
