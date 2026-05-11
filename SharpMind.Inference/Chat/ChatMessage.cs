namespace SharpMind.Inference.Chat;

/// <summary>
/// A single message in the chat conversation.
/// </summary>
public sealed class ChatMessage
{
    public required ChatRole Role { get; init; }
    public required string Content { get; init; }
    public string? Name { get; init; }
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;

    public static ChatMessage User(string content)
        => new() { Role = ChatRole.User, Content = content };

    public static ChatMessage System(string content)
        => new() { Role = ChatRole.System, Content = content };

    public static ChatMessage Agent(string content, string? name = null)
        => new() { Role = ChatRole.Agent, Content = content, Name = name };
}
