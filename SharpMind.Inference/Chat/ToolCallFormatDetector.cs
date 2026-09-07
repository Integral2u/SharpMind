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
    /// <item>The chat template embeds function-calling machinery (Qwen2.5/3
    /// templates contain <c>tool_call</c>/<c>tool_calls</c> blocks) →
    /// <see cref="ToolCallFormat.Qwen"/>.</item>
    /// <item>The model's own name/finetune mentions Qwen (catches Qwen2-0.5B,
    /// whose template has no tool-call block) → <see cref="ToolCallFormat.Qwen"/>.</item>
    /// <item>Anything else → <see cref="ToolCallFormat.SharpMind"/>, the native
    /// tagged contract SharpMind trains its own models on.</item>
    /// </list>
    /// </summary>
    public static ToolCallFormat Resolve(ModelMetaData? meta, Tokenizer? tokenizer = null)
    {
        if (meta is null) return ToolCallFormat.None;

        string? template = meta.GetChatTemplate();
        if (template is not null
            && (template.Contains("tool_call", StringComparison.Ordinal)
                || template.Contains("tool_calls", StringComparison.Ordinal)))
            return ToolCallFormat.Qwen;

        string name = meta.GetString("general.name");
        string basename = meta.GetString("general.basename");
        string finetune = meta.GetString("general.finetune");
        string haystack = $"{name} {basename} {finetune}".ToLowerInvariant();
        if (haystack.Contains("qwen", StringComparison.Ordinal))
            return ToolCallFormat.Qwen;

        return ToolCallFormat.SharpMind;
    }
}