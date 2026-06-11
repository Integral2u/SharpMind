namespace SharpMind.Inference.Agent;

/// <summary>
/// Permission level the host application grants for a particular IO access attempt.
/// </summary>
public enum ToolPermission
{
    /// <summary>Block this access unconditionally.</summary>
    Never,
    /// <summary>
    /// Prompt the user and wait for their decision before proceeding.
    /// The callback itself is responsible for surfacing UI and resolving to
    /// <see cref="Always"/> or <see cref="Never"/> before returning.
    /// </summary>
    Ask,
    /// <summary>Permit the access immediately without further prompting.</summary>
    Always
}