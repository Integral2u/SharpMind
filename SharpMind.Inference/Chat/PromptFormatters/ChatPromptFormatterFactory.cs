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
    /// Creates the best available formatter from the model's GGUF metadata.
    /// Reads <c>tokenizer.chat_template</c> directly from the KV pairs; no
    /// heuristic string matching is required.
    /// </summary>
    public static IChatPromptFormatter Create(ModelMetaData? meta)
    {
        string? tmpl = meta?.GetChatTemplate();

        if (!string.IsNullOrWhiteSpace(tmpl))
            return new JinjaTemplateFormatter(tmpl);

        return new SimpleFormatter();
    }

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
}