namespace SharpMind.Inference.Chat;

/// <summary>
/// The tool-calling contract a session expects from the model, and therefore
/// the one the agent prompt teaches. One value must be shared by the prompt
/// (which toolRules and call-format example <c>BuildAgentPrompt</c> emits) and
/// the capture side (how <c>ChatSession</c> recognizes a call in model output).
/// Resolved from the model's metadata via <see cref="ToolCallFormatDetector"/>.
/// </summary>
public enum ToolCallFormat
{
    /// <summary>No model metadata / unknown model — do not advertise tools and
    /// never attempt to capture a call. The safe default.</summary>
    None,

    /// <summary>SharpMind's native contract: narration in prose plus a call as
    /// a single JSON object inside &lt;tool_call&gt;...&lt;/tool_call&gt; tags.
    /// The format SharpMind will train its own models to emit.</summary>
    SharpMind,

    /// <summary>Qwen-instruct's native function-calling shape: a raw
    /// <c>{"name":...,"arguments":{...}}</c> JSON object with no wrapper.</summary>
    Qwen,
}