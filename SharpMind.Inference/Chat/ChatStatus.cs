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
    /// <summary>A parsed tool call is being handed back to the caller to dispatch (never executed in-session).</summary>
    ToolCall,
    /// <summary>Generating text response.</summary>
    Responding,
    /// <summary>Waiting for input.</summary>
    Waiting,
    /// <summary>Sub-agent is generating a response (streamed outside main history/KV cache).</summary>
    Researching,
    /// <summary>That chat was interrupted, cancelled or failed.</summary>
    Interrupted,
    /// <summary>Completed.</summary>
    Complete
}
