using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Formats prompts for DeepSeek models using the DeepSeek-specific
/// special tokens: <｜User｜> for user messages, <｜Assistant｜> for
/// system/assistant messages.
///
/// Faithful to the Jinja template:
/// {% for message in messages %}
///   {{'<｜User｜>' if message['role'] == 'user' else '<｜Assistant｜>'}}
///   {{message['content']}}
/// {% endfor %}
/// <｜Assistant｜>\n
///
/// System messages are mapped to <｜Assistant｜> (same as assistant),
/// exactly as the original Jinja template specifies.
/// </summary>
public sealed class DeepSeekFormatter : IChatPromptFormatter
{
    private const string UserToken = "<｜User｜>";
    private const string NonUserToken = "<｜Assistant｜>";
    private const string Suffix = "<｜Assistant｜>\n";

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            sb.Append(msg.Role == ChatRole.User ? UserToken : NonUserToken);
            sb.Append(msg.Content);
        }

        sb.Append(Suffix);
        return sb.ToString();
    }
}
