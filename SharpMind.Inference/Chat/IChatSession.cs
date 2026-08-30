using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Tokenization;
using System.Text.Json.Nodes;

namespace SharpMind.Inference.Chat;
public interface IChatSession : IAsyncDisposable
{
    /// <summary>
    /// Optional external tool-call interceptor. When set, every tool call the
    /// model makes is first routed through this delegate before the session's
    /// native agent dispatches it. Return:
    /// <list type="bullet">
    ///   <item><see cref="ToolRequestResult.Handled(string?)"/> — you ran the tool; the session
    ///         feeds your result back and continues its loop.</item>
    ///   <item><see cref="ToolRequestResult.Defer"/> — you didn't handle it; the session
    ///         dispatches it natively (with its File/Network permission gate).</item>
    ///   <item><see cref="ToolRequestResult.ReturnToCaller"/> — the session stops, surfaces the
    ///         call via a <see cref="ChatStatus.ToolCall"/> entry, and ends the turn so the
    ///         caller can dispatch it and resume by adding the result.</item>
    /// </list>
    /// Null (the default) keeps the current behaviour: the built-in agent loop
    /// dispatches every tool call itself.
    /// </summary>
    public Func<string, JsonObject, CancellationToken, Task<ToolRequestResult>>? ProcessToolRequest { get; set; }

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