namespace SharpMind.Inference.Chat;

/// <summary>
/// Status values during chat response generation.
/// </summary>
public enum ChatStatus
{
    /// <summary>Analyzing request, planning response.</summary>
    Thinking,
    /// <summary>Updating context/history.</summary>
    Updating,
    /// <summary>Executing tools/skills.</summary>
    Executing,
    /// <summary>Generating text response.</summary>
    Responding,
    /// <summary>Waiting for input or tool results.</summary>
    Waiting,
    /// <summary>That chat was interrupted, cancelled or failed.</summary>
    Interrupted,
    /// <summary>Completed.</summary>
    Complete
}
