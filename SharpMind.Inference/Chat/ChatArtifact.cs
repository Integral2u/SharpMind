namespace SharpMind.Inference.Chat;

/// <summary>
/// Artifact attached to a chat response (images, code blocks, etc.).
/// </summary>
public sealed class ChatArtifact
{
    public required string Type { get; init; }  // "text", "image", "code", "json"
    public required byte[] Content { get; init; }
    public string? Language { get; init; }
    public string? FileName { get; init; }
    /// <summary>
    /// Original media type, needed to build a data URL when an attachment is
    /// sent to a remote provider. <see cref="Type"/> alone is too coarse.
    /// </summary>
    public string? MimeType { get; init; }
    /// <summary>Original source path of the file, preserved for tool access.</summary>
    public string? SourcePath { get; init; }
}
