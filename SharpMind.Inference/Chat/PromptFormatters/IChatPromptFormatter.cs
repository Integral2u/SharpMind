using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

public interface IChatPromptFormatter
{
    string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos);
}
