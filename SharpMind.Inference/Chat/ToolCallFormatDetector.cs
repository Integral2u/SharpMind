using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;

/// <summary>
/// Resolves which <see cref="ToolCallFormat"/> a model is expected to use,
/// from the same metadata the prompt formatter is chosen from.
/// Mirrors <c>ChatPromptFormatterFactory</c>: best-effort heuristics with a
/// safe default.
/// </summary>
public static class ToolCallFormatDetector
{
    /// <summary>
    /// Resolution order, highest confidence first — same shape as
    /// <c>ChatPromptFormatterFactory.Create</c>:
    /// <list type="number">
    /// <item>No metadata at all → <see cref="ToolCallFormat.None"/>. The
    /// session cannot know the model, so tools stay off.</item>
    /// <item>Chat template contains <c>[TOOL_CALLS]</c> →
    /// <see cref="ToolCallFormat.Mistral"/>.</item>
    /// <item>Chat template contains <c>tool_call</c>/<c>tool_calls</c>
    /// (but not as part of <c>[TOOL_CALLS]</c>) →
    /// <see cref="ToolCallFormat.Qwen"/>.</item>
    /// <item>Chat template contains <c>&lt;|python_tag|&gt;</c> →
    /// <see cref="ToolCallFormat.Llama3"/>.</item>
    /// <item>Chat template contains <c>functionCall</c> →
    /// <see cref="ToolCallFormat.Gemma"/>.</item>
    /// <item>Model name/finetune mentions Qwen →
    /// <see cref="ToolCallFormat.Qwen"/>.</item>
    /// <item>Model name/finetune mentions Mistral/Ministral →
    /// <see cref="ToolCallFormat.Mistral"/>.</item>
    /// <item>Model name/finetune mentions Llama →
    /// <see cref="ToolCallFormat.Llama3"/>.</item>
    /// <item>Model name/finetune mentions Gemma/FunctionGemma →
    /// <see cref="ToolCallFormat.Gemma"/>.</item>
    /// <item>Anything else → <see cref="ToolCallFormat.SharpMind"/>, the native
    /// tagged contract SharpMind trains its own models on.</item>
    /// </list>
    /// </summary>
    public static ToolCallFormat Resolve(ModelMetaData? meta, Tokenizer? tokenizer = null)
    {
        if (meta is null) return ToolCallFormat.None;

        string? template = meta.GetChatTemplate();
        if (template is not null)
        {
            // Mistral uses [TOOL_CALLS] — check before the generic tool_call
            // substring since a Mistral template may also contain "tool_call"
            // as part of other variable names.
            if (template.Contains("[TOOL_CALLS]", StringComparison.Ordinal))
                return ToolCallFormat.Mistral;

            if (template.Contains("tool_call", StringComparison.Ordinal)
                || template.Contains("tool_calls", StringComparison.Ordinal))
                return ToolCallFormat.Qwen;

            if (template.Contains("<|python_tag|>", StringComparison.Ordinal))
                return ToolCallFormat.Llama3;
            if (template.Contains("functionCall", StringComparison.Ordinal)
                || template.Contains("function_call", StringComparison.Ordinal))
                return ToolCallFormat.Gemma;
        }

        string name = meta.GetString("general.name");
        string basename = meta.GetString("general.basename");
        string finetune = meta.GetString("general.finetune");
        string haystack = $"{name} {basename} {finetune}".ToLowerInvariant();

        if (haystack.Contains("qwen", StringComparison.Ordinal))
            return ToolCallFormat.Qwen;

        if (haystack.Contains("mistral", StringComparison.Ordinal)
            || haystack.Contains("ministral", StringComparison.Ordinal))
            return ToolCallFormat.Mistral;

        if (haystack.Contains("llama", StringComparison.Ordinal))
            return ToolCallFormat.Llama3;

        if (haystack.Contains("gemma", StringComparison.Ordinal))
            return ToolCallFormat.Gemma;

        return ToolCallFormat.SharpMind;
    }
}