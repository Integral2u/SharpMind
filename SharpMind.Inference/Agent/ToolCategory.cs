namespace SharpMind.Inference.Agent;

/// <summary>
/// Broad IO category of a tool access attempt, set at runtime by the
/// <see cref="Chat.InterceptingFileSystem"/> or
/// <see cref="Chat.InterceptingNetworkService"/> interceptors
/// when a tool actually touches the file system or network during
/// <see cref="IAgentBuilder.CallToolAsync"/>.
/// </summary>
public enum ToolCategory
{
    /// <summary>Pure computation — no IO observed.</summary>
    General,
    /// <summary>The tool attempted to read or write the local file system.</summary>
    File,
    /// <summary>The tool attempted to make an outbound network call.</summary>
    Network
}
