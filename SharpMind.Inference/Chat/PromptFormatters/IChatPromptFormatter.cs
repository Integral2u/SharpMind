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
}
