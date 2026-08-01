using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Llama-3 "header" chat format:
///
///   &lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;
///
///   {system message}&lt;|eot_id|&gt;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;
///
///   {user message}&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;
///
/// This is the fixed structural skeleton shared by every Llama-3.x chat
/// checkpoint (3, 3.1, 3.2, 3.3), independent of the fancier bits their
/// official chat_template adds on top (tool-calling blocks, the
/// Cutting-Knowledge-Date/Today-Date system preamble, strftime_now, etc.).
/// Those extras aren't part of the model's actual learned format — they're
/// prompt content Meta's template happens to inject — so a model missing its
/// chat_template but carrying &lt;|start_header_id|&gt;/&lt;|end_header_id|&gt;/
/// &lt;|eot_id|&gt; in its vocab will respond correctly to this skeleton alone.
///
/// No system message is added by default; pass one explicitly via a leading
/// ChatRole.System entry if wanted. BOS is <|begin_of_text|>, handled like
/// every other formatter via tokenizer.BosId rather than hardcoded here, so
/// it stays correct even if a given GGUF's BOS token differs.
/// </summary>
public sealed class Llama3Formatter : IChatPromptFormatter
{
    private const string HeaderStart = "<|start_header_id|>";
    private const string HeaderEnd = "<|end_header_id|>\n\n";
    private const string Eot = "<|eot_id|>";

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            string role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Agent => "assistant",
                _ => "user"
            };

            sb.Append(HeaderStart);
            sb.Append(role);
            sb.Append(HeaderEnd);
            sb.Append(msg.Content.Trim());
            sb.Append(Eot);
        }

        sb.Append(HeaderStart);
        sb.Append("assistant");
        sb.Append(HeaderEnd);
        return sb.ToString();
    }
}
