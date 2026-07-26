using SharpMind.Model;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;

public sealed class CompactionContext
{
    public required List<ChatMessage> History { get; init; }
    public required int CurrentTokenCount { get; init; }
    public required int MaxTokens { get; init; }
    /// <summary>The session's transformer model, available for compactors that need to run inference.</summary>
    public Transformer? Model { get; init; }
    /// <summary>The session's tokenizer, available for compactors that need to count or encode tokens.</summary>
    public Tokenizer? Tokenizer { get; init; }
    /// <summary>Delegate that summarizes a text segment using the session's generator.</summary>
    public Func<string, Task<string>>? SummarizeAsync { get; init; }
}
