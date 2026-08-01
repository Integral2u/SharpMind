using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

public interface IChatPromptFormatter
{
    /// <param name="enableThinking">
    /// Passed as the <c>enable_thinking</c> template variable. Some chat
    /// templates (e.g. Qwen3) check this to decide whether to leave their
    /// reasoning block open or emit an empty <c>&lt;think&gt;&lt;/think&gt;</c>
    /// pair. Formatters that don't reference this variable simply ignore it.
    /// </param>
    /// <param name="toolsJson">
    /// Optional JSON array of tool/function definitions (from AgentBuilder.ToolDefinitions).
    /// Passed to Jinja templates as the <c>tools</c> and <c>custom_tools</c> variables.
    /// </param>
    string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null);

    /// <summary>
    /// Stop strings that naturally terminate generation for this format.
    ///
    /// Formats that emit a dedicated end-of-turn token (ChatML, Llama-3)
    /// return that token's text; formats with no special tokens at all (Raw,
    /// Q&amp;A, Alpaca) return the plain-text markers that separate turns, so
    /// the model has something concrete to stop on besides EOS.
    ///
    /// Callers should merge these into <see cref="GenerationConfig.StopStrings"/>
    /// (the model's own EOS token is always handled separately via StopTokenIds).
    /// A formatter may return an empty list when its only reliable terminator is
    /// the EOS token, e.g. Jinja templates, which embed the model's own
    /// end-of-turn token and are already covered by EOS handling.
    /// </summary>
    IReadOnlyList<string> DefaultStopStrings { get; }
}
