using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

    public sealed class RawTemplateFormatter : IChatPromptFormatter
{
    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history) sb.AppendLine(msg.Content);
        
        return sb.ToString();
    }
}
