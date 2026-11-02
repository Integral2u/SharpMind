namespace SharpMind.Inference.Chat;

/// <summary>
/// Controls how a session handles a tool call that the strict format parser
/// cannot read. Strict keeps the historical behaviour; Lenient runs a
/// structural recovery pass after strict parsing fails. A call that already
/// parses strictly is never modified.
/// </summary>
public enum ToolCallRecoveryMode
{
    /// <summary>Only the model's native tool-call format is accepted.</summary>
    Strict = 0,

    /// <summary>
    /// After strict parsing fails, extract a JSON call from prose or a fenced
    /// block, normalize the tool/arguments keys, resolve the tool name against
    /// the registered tools, and infer a tool from an unambiguous argument
    /// shape. Never invents arguments or parses free-form prose into a call.
    /// </summary>
    Lenient = 1,
}
