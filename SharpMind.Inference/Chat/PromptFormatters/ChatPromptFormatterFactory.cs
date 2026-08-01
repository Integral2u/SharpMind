using SharpMind.Model.Format;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Creates an <see cref="IChatPromptFormatter"/> from a <see cref="ModelMetaData"/>.
///
/// When a <c>tokenizer.chat_template</c> key is present the template is executed
/// by <see cref="JinjaTemplateFormatter"/>, which implements the Jinja2 subset
/// used by all known GGUF models (ChatML, Llama-3, TinyLlama/Zephyr, DeepSeek, Qwen…).
///
/// When no template is present a plain <see cref="SimpleFormatter"/> is returned
/// as a safe fallback.
/// </summary>
public static class ChatPromptFormatterFactory
{

    /// <summary>
    /// Overload retained for callers that already hold a raw template string
    /// (e.g. <c>QuickDiagnostic</c>).
    /// </summary>
    public static IChatPromptFormatter Create(string? chatTemplate)
    {
        if (!string.IsNullOrWhiteSpace(chatTemplate))
            return new JinjaTemplateFormatter(chatTemplate);

        return new SimpleFormatter();
    }

    /// <summary>
    /// Best-effort formatter selection for the no-chat_template case.
    /// Order of confidence, highest first:
    ///   1. tokenizer.chat_template present → JinjaTemplateFormatter (exact).
    ///   2. Vocab contains role-turn special tokens (&lt;|im_start|&gt;,
    ///      &lt;|user|&gt;, etc.) → ChatMLFormatter can construct itself
    ///      directly from a synthesized token list, no template text needed.
    ///   3. general.name/general.basename/general.finetune metadata mentions
    ///      a known legacy format by name → that formatter.
    ///   4. Vocab has no chat-style added tokens at all, which is itself a
    ///      signal: models with genuine chat fine-tuning almost always add
    ///      *some* turn-delimiter token, even an unlabelled one. Its total
    ///      absence points at either a base/completion model or a legacy
    ///      plain-text instruct format — Alpaca is the more useful guess of
    ///      the two, since a base model produces garbage under any wrapper.
    ///   5. No signal at all → SimpleFormatter (safest, most neutral default).
    ///
    /// Requires the tokenizer (not just meta) to inspect the vocab, so this
    /// is a separate overload from <see cref="Create(ModelMetaData?)"/> rather
    /// than a silent behavior change to the existing call sites.
    /// </summary>
    public static IChatPromptFormatter Create(ModelMetaData? meta, Tokenization.Tokenizer? tokenizer = null)
    {
        string? tmpl = meta?.GetChatTemplate();
        if (!string.IsNullOrWhiteSpace(tmpl))
            return new JinjaTemplateFormatter(tmpl);

        var vocab = tokenizer?.Vocab;
        bool hasChatML = vocab != null && vocab.Contains("<|im_start|>");
        bool hasZephyr = vocab != null && (vocab.Contains("<|user|>") || vocab.Contains("<|assistant|>"));
        bool hasLlama3 = vocab != null && vocab.Contains("<|start_header_id|>");

        if (hasChatML || hasZephyr)
        {
            // ChatMLFormatter's constructor scans a template STRING for these
            // markers via regex — hand it the actual tokens we found in the
            // vocab (not a fabricated placeholder) so its own ChatML-vs-Zephyr
            // detection runs against real data.
            var found = new System.Text.StringBuilder();
            if (vocab!.Contains("<|im_start|>")) found.Append("<|im_start|>");
            if (vocab.Contains("<|im_end|>")) found.Append("<|im_end|>");
            if (vocab.Contains("<|system|>")) found.Append("<|system|>");
            if (vocab.Contains("<|user|>")) found.Append("<|user|>");
            if (vocab.Contains("<|assistant|>")) found.Append("<|assistant|>");
            return new ChatMLFormatter(found.ToString());
        }

        if (hasLlama3)
            return new Llama3Formatter();

        string name = meta?.GetString("general.name") ?? "";
        string basename = meta?.GetString("general.basename") ?? "";
        string finetune = meta?.GetString("general.finetune") ?? "";
        string haystack = $"{name} {basename} {finetune}".ToLowerInvariant();

        if (haystack.Contains("alpaca"))
            return new AlpacaFormatter();

        // No chat_template, no chat-style added tokens, no name hint: most
        // likely a base/completion checkpoint or an unlabelled legacy
        // instruct model. Alpaca is the better default of the two — a
        // genuine base model will produce weak output under any wrapper,
        // while a real legacy instruct model will only behave correctly
        // under something resembling its actual training format.
        if (vocab != null && !hasChatML && !hasZephyr && !hasLlama3)
            return new AlpacaFormatter();

        return new SimpleFormatter();
    }
}