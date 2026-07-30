using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Plain "Q: ... A:" completion-style formatter for base (non-instruct) models.
///
/// Base checkpoints are usually shipped with a generic ChatML chat_template
/// in tokenizer_config.json (copied verbatim from the same family's Instruct
/// repo, or just a llama.cpp converter default) even though the weights were
/// never fine-tuned to follow <|im_start|>/<|im_end|> turn-taking. Driving
/// such a model through JinjaTemplateFormatter/ChatMLFormatter tends to
/// produce fluent-looking but semantically-drifting output that never hits
/// a stop token, because the model has no learned notion of "answer, then
/// stop" — it's just doing free continuation on tokens it barely saw.
///
/// This formatter instead renders history the way base models actually see
/// this pattern at scale in pretraining data (scraped trivia/QA/exam-style
/// text), which in practice gets clean, short, self-terminating answers out
/// of otherwise "broken-looking" base checkpoints. No <|im_start|>/<|im_end|>
/// or any other special tokens are emitted — only ordinary text tokens, so
/// StopStrings (e.g. "\n\n", "\nQ:") should be set on GenerationConfig for
/// this formatter, since the model won't reliably emit EOS on its own.
///
/// Multi-turn history is rendered as a flat Q/A transcript. A leading
/// System message, if present, is emitted as a plain instruction line
/// before the first Q — some base models pick up on it weakly as ambient
/// context, but don't rely on it being followed strictly.
/// </summary>
public sealed class QuestionAnswerFormatter : IChatPromptFormatter
{
    private const string QPrefix = "Q: ";
    private const string APrefix = "A:";

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            switch (msg.Role)
            {
                case ChatRole.System:
                    // No special framing — base models don't know what a
                    // "system message" is. Just surface it as plain text.
                    sb.Append(msg.Content.Trim());
                    sb.Append('\n');
                    break;

                case ChatRole.User:
                    sb.Append(QPrefix);
                    sb.Append(msg.Content.Trim());
                    sb.Append('\n');
                    break;

                case ChatRole.Agent:
                    sb.Append(APrefix);
                    sb.Append(' ');
                    sb.Append(msg.Content.Trim());
                    sb.Append('\n');
                    break;
            }
        }

        // Open the trailing answer for the model to complete.
        sb.Append(APrefix);
        return sb.ToString();
    }
}
