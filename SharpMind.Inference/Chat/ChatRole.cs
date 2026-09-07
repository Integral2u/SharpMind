namespace SharpMind.Inference.Chat;

/// <summary>
/// Chat roles for conversation participants.
/// </summary>
public enum ChatRole
{
    /// <summary>System prompt - sets behavior/instructions.</summary>
    System,
    /// <summary>AI assistant/agent responses.</summary>
    Agent,
    /// <summary>Human user input.</summary>
    User,
    /// <summary>Result of a tool the assistant called. Native to function-calling
    /// templates (e.g. Qwen's <c>&lt;|im_start|&gt;tool</c> turn), where it must
    /// NOT be a second system block or the model will not recognize the result
    /// as its own.</summary>
    Tool
}
