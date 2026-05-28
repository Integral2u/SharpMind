using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

public sealed class PassThroughFormatter : IChatPromptFormatter
{
    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            sb.Append(msg.Content);
            sb.Append('\n');
        }

        return sb.ToString();
    }
}
