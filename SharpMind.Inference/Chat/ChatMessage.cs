namespace SharpMind.Inference.Chat;

/// <summary>
/// A single message in the chat conversation.
/// </summary>
public sealed class ChatMessage
{
    public required ChatRole Role { get; init; }
    public required string Content { get; set; }
    public string? Name { get; init; }
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;
    /// <summary>Optional metadata for context management, tagging, and interrupt markers.</summary>
    public Dictionary<string, string>? Metadata { get; init; }
    /// <summary>When true, this message is exempted from truncation/eviction.</summary>
    public bool IsPinned { get; set; }
    public bool Ignore { get; set; }
    public static ChatMessage User(string content)
        => new() { Role = ChatRole.User, Content = content };

    public static ChatMessage System(string content)
        => new() { Role = ChatRole.System, Content = content };

    public static ChatMessage Agent(string content, string? name = null)
        => new() { Role = ChatRole.Agent, Content = content, Name = name };
}
