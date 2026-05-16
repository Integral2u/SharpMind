using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Formats prompts for Llama 3.x models using the official Llama chat format:
/// <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
///
/// Template:
/// {% for message in messages %}<|start_header_id|>{{role}}<|end_header_id|>\n\n{{content}}<|eot_id|>{% endfor %}
/// {% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}
/// </summary>
public sealed class LlamaFormatter : IChatPromptFormatter
{
    private const string HeaderStart = "<|start_header_id|>";
    private const string HeaderEnd = "<|end_header_id|>";
    private const string EotId = "<|eot_id|>";

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            var role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.Agent => "assistant",
                ChatRole.User => "user",
                _ => "unknown"
            };

            sb.Append(HeaderStart);
            sb.Append(role);
            sb.Append(HeaderEnd);
            sb.Append("\n\n");
            sb.Append(msg.Content);
            sb.Append(EotId);
        }

        sb.Append(HeaderStart);
        sb.Append("assistant");
        sb.Append(HeaderEnd);
        sb.Append("\n\n");
        return sb.ToString();
    }
}
