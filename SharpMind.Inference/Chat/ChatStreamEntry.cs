namespace SharpMind.Inference.Chat;

/// <summary>
/// Streaming response entry for real-time updates.
/// </summary>
public sealed class ChatStreamEntry
{
    public required ChatStatus Status { get; init; }
    public string? TextDelta { get; init; }
    public ChatArtifact? Artifact { get; init; }
    public bool IsComplete { get; init; }
}
