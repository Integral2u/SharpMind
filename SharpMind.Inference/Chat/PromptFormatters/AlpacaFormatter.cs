using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Classic Alpaca / "legacy instruct" format:
///
///   Below is an instruction that describes a task...
///
///   ### Instruction:
///   {user message}
///
///   ### Response:
///   {assistant reply}
///
/// This predates ChatML/Llama-3-style role tokens entirely — it's plain text,
/// no special tokens involved, which is exactly why it's the right guess for
/// older or lesser-known instruct-tuned checkpoints that ship no
/// tokenizer.chat_template and have no <|...|>-style added tokens in their
/// vocab either (see ChatPromptFormatterFactory.Detect below). Many
/// early/mid-2023 instruction-tuned Llama/Llama-2 derivatives (the original
/// Alpaca, plus a large fraction of its finetuned descendants) use this
/// format verbatim or with only minor wording changes to the preamble.
///
/// No special tokens are emitted, so — same caveat as QuestionAnswerFormatter —
/// pair this with explicit StopStrings (e.g. "\n### Instruction:", "\n###")
/// on GenerationConfig, since these models won't reliably emit EOS at the
/// right point on their own.
/// </summary>
public sealed class AlpacaFormatter : IChatPromptFormatter
{
    private const string Preamble =
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";

    private const string PreambleWithInput =
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n";

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        // Alpaca has no native concept of a system message or multi-turn
        // history — fold a leading System message into the preamble instead
        // of trying to force it into ### Instruction:/### Response: pairs.
        string? systemPrefix = null;
        int startIndex = 0;
        if (history.Count > 0 && history[0].Role == ChatRole.System)
        {
            systemPrefix = history[0].Content.Trim();
            startIndex = 1;
        }

        sb.Append(systemPrefix != null ? PreambleWithInput : Preamble);
        if (systemPrefix != null)
        {
            sb.Append(systemPrefix);
            sb.Append("\n\n");
        }

        for (int i = startIndex; i < history.Count; i++)
        {
            var msg = history[i];
            switch (msg.Role)
            {
                case ChatRole.User:
                    sb.Append("### Instruction:\n");
                    sb.Append(msg.Content.Trim());
                    sb.Append("\n\n");
                    break;

                case ChatRole.Agent:
                    sb.Append("### Response:\n");
                    sb.Append(msg.Content.Trim());
                    sb.Append("\n\n");
                    break;

                // Alpaca has no system-role turns beyond the leading one
                // folded into the preamble above; treat any later system
                // message as an additional instruction rather than dropping it.
                case ChatRole.System:
                    sb.Append("### Instruction:\n");
                    sb.Append(msg.Content.Trim());
                    sb.Append("\n\n");
                    break;
            }
        }

        sb.Append("### Response:\n");
        return sb.ToString();
    }
}