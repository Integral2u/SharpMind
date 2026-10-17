using SharpMind.Data.Batching;
using SharpMind.Data.Records;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Tokenization;

namespace SharpMind.Training.Sft;

/// <summary>
/// B6: role-aware tokenisation for supervised fine-tuning. Renders a
/// conversation with the model's own <see cref="IChatPromptFormatter"/> and
/// masks every token the assistant actually produced (its content plus the
/// end-of-turn marker), leaving the prompt, the injected system block, the
/// template scaffolding, and the trailing generation prompt unmasked.
///
/// <para>
/// The assistant spans are found by rendering the history turn by turn and
/// diffing the successive renderings: the increment that appears between the
/// prefix that ends at the assistant's opening marker and the prefix that ends
/// at the turn's end-of-turn is exactly the assistant output. This keeps the
/// mask aligned with whatever skeletal structure the template emits, without
/// the tokenizer having to understand any particular chat format.
/// </para>
///
/// <para>
/// It relies on the convention that rendering a growing history produces
/// prefix-stable text (each message only appends, the formatter only strips the
/// trailing generation prompt to move on to the next turn). Formats that
/// rewrite earlier text when later turns arrive (folding, thinking-block
/// collapsing) must be tokenised some other way.
/// </para>
/// </summary>
public sealed class ConversationTokenizer
{
    private readonly IChatPromptFormatter _formatter;
    private readonly Tokenizer _tokenizer;
    private readonly bool _addBos;
    private readonly bool _addEos;
    private readonly bool _enableThinking;

    public ConversationTokenizer(
        IChatPromptFormatter formatter,
        Tokenizer tokenizer,
        bool addBos = true,
        bool addEos = false,
        bool enableThinking = false)
    {
        _formatter = formatter ?? throw new ArgumentNullException(nameof(formatter));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _addBos = addBos;
        _addEos = addEos;
        _enableThinking = enableThinking;
    }

    /// <summary>Encodes a chat history, marking assistant spans as targets.</summary>
    public TokenizedExample Encode(IReadOnlyList<ChatMessage> history)
    {
        ArgumentNullException.ThrowIfNull(history);

        string full = _formatter.Format(history, _tokenizer, _addBos, _enableThinking);
        int[] tokens = _tokenizer.Encode(full, addBos: false, addEos: _addEos);

        if (history.Count == 0 || !history.Any(m => m.Role == ChatRole.Agent))
            return new TokenizedExample(tokens);

        // An empty history renders as just the generation prompt the formatter
        // appends to solicits the next turn - the exact text that separates one
        // assistant turn's end-of-turn from the start of the following turn.
        string generationPrompt = _formatter.Format([], _tokenizer, addBos: false, _enableThinking);

        var mask = new bool[tokens.Length];
        bool any = false;

        for (int j = 0; j < history.Count; j++)
        {
            if (history[j].Role != ChatRole.Agent)
                continue;

            string before = _formatter.Format(Range(history, j), _tokenizer, _addBos, _enableThinking);
            string through = _formatter.Format(Range(history, j + 1), _tokenizer, _addBos, _enableThinking);

            int startChar = before.Length;
            int endChar = through.Length;

            // The trailing generation prompt opens the *next* turn; it is not
            // something the assistant said, so leave it unlabelled.
            if (generationPrompt.Length > 0 && through.EndsWith(generationPrompt, StringComparison.Ordinal))
                endChar -= generationPrompt.Length;

            if (endChar <= startChar)
                continue;

            int startToken = _tokenizer.Encode(full[..startChar], addBos: false).Length;
            int endToken = _tokenizer.Encode(full[..endChar], addBos: false).Length;

            for (int t = startToken; t < Math.Min(endToken, tokens.Length); t++)
            {
                mask[t] = true;
                any = true;
            }
        }

        return new TokenizedExample(tokens, any ? mask : null);
    }

    /// <summary>
    /// Encodes a typed record: a record with turns is tokenised with its
    /// assistant turns masked; a record that is only raw text is treated as a
    /// plain language-modelling example (every token is a target).
    /// </summary>
    public TokenizedExample Encode(DataRecord record)
    {
        ArgumentNullException.ThrowIfNull(record);

        if (record.Turns is { Count: > 0 })
            return Encode(ToMessages(record.Turns));

        return new TokenizedExample(_tokenizer.Encode(record.Text, _addBos, _addEos));
    }

    /// <summary>Adapts a chat role string ("assistant", "model", "user", ...).</summary>
    public static ChatRole ParseRole(string role) => role.ToLowerInvariant() switch
    {
        "system" => ChatRole.System,
        "assistant" or "model" or "agent" or "gpt" => ChatRole.Agent,
        "tool" or "function" => ChatRole.Tool,
        _ => ChatRole.User,
    };

    public static ChatMessage[] ToMessages(IReadOnlyList<ConversationTurn> turns)
    {
        var messages = new ChatMessage[turns.Count];
        for (int i = 0; i < turns.Count; i++)
        {
            messages[i] = new ChatMessage
            {
                Role = ParseRole(turns[i].Role),
                Content = turns[i].Content,
            };
        }
        return messages;
    }

    private static ChatMessage[] Range(IReadOnlyList<ChatMessage> history, int count)
    {
        var slice = new ChatMessage[count];
        for (int i = 0; i < count; i++)
            slice[i] = history[i];
        return slice;
    }
}
