namespace SharpMind.Inference.Chat;

/// <summary>
/// Result from chat response generation.
/// Can be streamed as it's being generated.
/// </summary>
public sealed class ChatResult
{
    public ChatStatus Status { get; internal init; }
    public string? Content { get; internal init; }
    public List<ChatArtifact>? Artifacts { get; internal init; }
    public bool IsStreaming { get; internal init; }
    public bool IsComplete { get; internal init; }
    public string? Error { get; internal init; }
}
