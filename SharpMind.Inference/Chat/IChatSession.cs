using SharpMind.Model;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;
public interface IChatSession : IAsyncDisposable
{
    int MaxTokens { get; set; }
    float Temperature { get; set; }
    int TopK { get; set; }
    float TopP { get; set; }
    float RepetitionPenalty { get; set; }
    int RepetitionWindow { get; set; }
    float? TokensPerSecond { get; }
    float? TimeToFirstToken { get; }
    Tokenizer Tokenizer { get; }
    Transformer Model { get; }
    IReadOnlyList<ChatMessage> History { get; }

    void AddMessage(ChatRole role, string content);
    void AddMessage(ChatMessage message);
    string GetFormattedPrompt();
    void ClearHistory();
    void ResetCaches();

    Task<ChatMessage[]> StartChatAsync(Func<Task<ChatMessage>> prompt, Action<ChatStreamEntry> response, CancellationToken token = default);
    Task<ChatMessage[]> StartChatAsync(Func<ChatMessage> prompt, Action<ChatStreamEntry> response, CancellationToken token = default);
    Task<ChatMessage[]> StartChatAsync(Func<Task<string>> prompt, Action<string> response, CancellationToken token = default);
    Task<ChatMessage[]> StartChatAsync(Func<string> prompt, Action<string> response, CancellationToken token = default);
}