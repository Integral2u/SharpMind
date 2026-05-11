namespace SharpMind.Inference.Chat;

/// <summary>
/// Artifact attached to a chat response (images, code blocks, etc.).
/// </summary>
public sealed class ChatArtifact
{
    public required string Type { get; init; }  // "text", "image", "code", "json"
    public required string Content { get; init; }
    public string? Language { get; init; }
    public string? FileName { get; init; }
}
