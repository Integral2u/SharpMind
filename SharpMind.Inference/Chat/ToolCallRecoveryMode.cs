namespace SharpMind.Inference.Chat;

/// <summary>
/// Controls how a session handles a tool call that the strict format parser
/// cannot read. Strict keeps the historical behaviour; Lenient runs a
/// structural recovery pass after strict parsing fails. A call that already
/// parses strictly is never modified beyond argument-name normalization.
/// </summary>
public enum ToolCallRecoveryMode
{
    /// <summary>
    /// Only the model's native tool-call format is accepted. Argument names
    /// still match across case and separators (<c>action_type</c> binds to
    /// <c>actionType</c>); no structural recovery or fuzzy matching runs.
    /// </summary>
    Strict = 0,

    /// <summary>
    /// After strict parsing fails, extract a JSON call from prose or a fenced
    /// block, normalize the tool/arguments keys, resolve the tool name against
    /// the registered tools, and infer a tool from an unambiguous argument
    /// shape. Never invents arguments or parses free-form prose into a call.
    /// </summary>
    Lenient = 1,

    /// <summary>
    /// Everything Lenient does, plus unique-best fuzzy matching for tool names
    /// and argument names (Damerau-Levenshtein, at most one edit for short
    /// names and two for longer ones). A call whose arguments name a filesystem
    /// path is never recovered by guessing. Opt-in: Lenient remains the default.
    /// </summary>
    Loose = 2,
}
