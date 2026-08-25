using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;
public interface IChatSession : IAsyncDisposable
{
    /// <summary>Context window the prompt is trimmed to fit, in tokens.</summary>
    public int MaxTokens { get; set; }

    /// <summary>
    /// Maximum tokens generated per turn. Distinct from <see cref="MaxTokens"/>,
    /// which is the context budget — setting the generation length on MaxTokens
    /// instead leaves generation at its default and starves the context.
    /// </summary>
    public int MaxNewTokens { get; set; }
    public float Temperature { get; set; }
    public int TopK { get; set; }
    public float TopP { get; set; }
    public float RepetitionPenalty { get; set; }
    public int RepetitionWindow { get; set; }
    public IReadOnlyList<int>? StopTokenIds { get; set; }
    public IReadOnlyList<string>? StopStrings { get; set; }
    public bool ShowThinking { get; set; }
    public bool EnableThinking { get; set; }
    public string UserName { get; set; }
    public float? TokensPerSecond { get; }
    public float? TimeToFirstToken { get; }
    public Tokenizer Tokenizer { get; }
    public Transformer Model { get; }
    public IReadOnlyList<ChatMessage> History { get; }

    /// <summary>
    /// The resolved chat prompt formatter after <see cref="InitializeChat"/>
    /// runs. Null until the session is initialized. Use
    /// <c>Formatter?.GetType().Name</c> to display the concrete formatter name
    /// (e.g. "Llama3Formatter", "ChatMLFormatter") in the UI.
    /// </summary>
    IChatPromptFormatter? Formatter { get; }

    /// <summary>
    /// Number of tokens fed to the generator as prefill on the most recent turn:
    /// the full prompt when the KV cache could not be extended, or just the delta
    /// when the previous turn's cache was verified and extended incrementally.
    /// </summary>
    internal int LastPrefillTokenCount { get; }

    public void AddMessage(ChatRole role, string content);
    public void AddMessage(ChatMessage message);
    public string GetFormattedPrompt();
    public void ClearHistory();
    public void ResetCaches();
    public void Interrupt();
    public void InitializeChat(IProgress<float>? progress = null);
    public ChatSessionSnapshot GetSnapshot();
    public void LoadSnapshot(ChatSessionSnapshot snapshot);

    /// <summary>
    /// Generate a single response to <paramref name="userInput"/>.
    /// The user message is added to history automatically.
    /// </summary>
    public IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
        string userInput,
        ChatArtifact[]? artifacts = null,
        CancellationToken ct = default);

    public Task<ChatMessage[]> StartChatAsync(Func<Task<ChatMessage>> prompt, Action<ChatStreamEntry> response, CancellationToken token = default);
    public Task<ChatMessage[]> StartChatAsync(Func<ChatMessage> prompt, Action<ChatStreamEntry> response, CancellationToken token = default);
    public Task<ChatMessage[]> StartChatAsync(Func<Task<string>> prompt, Action<string> response, CancellationToken token = default);
    public Task<ChatMessage[]> StartChatAsync(Func<string> prompt, Action<string> response, CancellationToken token = default);
}