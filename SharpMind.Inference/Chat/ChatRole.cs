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
    User
}
