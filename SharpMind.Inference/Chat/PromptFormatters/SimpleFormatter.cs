using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

public sealed class SimpleFormatter : IChatPromptFormatter
{
    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            var prefix = msg.Role switch
            {
                ChatRole.System => "user: ",
                ChatRole.Agent => "assistant: ",
                ChatRole.User => "user: ",
                _ => ""
            };
            sb.Append(prefix);
            sb.Append(msg.Content);
            sb.Append('\n');
        }

        sb.Append("assistant: ");
        return sb.ToString();
    }
}
