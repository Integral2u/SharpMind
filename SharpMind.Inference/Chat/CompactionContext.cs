namespace SharpMind.Inference.Chat;

public sealed class CompactionContext
{
    public required List<ChatMessage> History { get; init; }
    public required int CurrentTokenCount { get; init; }
    public required int MaxTokens { get; init; }
}
