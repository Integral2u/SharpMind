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

    /// <summary>
    /// Live tokens-per-second at this point in the stream.
    /// Rolling average over the last N tokens during generation;
    /// final cumulative rate on the <see cref="ChatStatus.Complete"/> entry.
    /// Null before generation begins.
    /// </summary>
    public float? TokensPerSecond { get; init; }

    /// <summary>Seconds from start to first output token (includes prefill).</summary>
    public float? TimeToFirstToken { get; init; }
}
