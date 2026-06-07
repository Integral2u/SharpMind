using System.Text.Json.Nodes;

namespace SharpMind.Inference.Agent;

/// <summary>
/// Broad IO category of a tool access attempt, set at runtime by the
/// <see cref="SharpMind.Inference.Chat.InterceptingFileSystem"/> or
/// <see cref="SharpMind.Inference.Chat.InterceptingNetworkService"/> interceptors
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

/// <summary>
/// Passed to <see cref="SharpMind.Inference.Chat.ChatSession{T,K}.PermissionCallback"/>
/// when a tool call attempts file system or network IO.
/// </summary>
public sealed class ToolPermissionContext
{
    /// <summary>Registered tool name that triggered the access attempt.</summary>
    public required string ToolName { get; init; }

    /// <summary>
    /// IO category detected at runtime by the active interceptor —
    /// <see cref="ToolCategory.File"/> or <see cref="ToolCategory.Network"/>.
    /// Never <see cref="ToolCategory.General"/> (general tool calls bypass the gate).
    /// </summary>
    public required ToolCategory Category { get; init; }

    /// <summary>
    /// The specific path or URL the tool is attempting to access.
    /// Provided for display and audit; access is blocked until the callback
    /// returns <see cref="ToolPermission.Always"/>.
    /// </summary>
    public required string Resource { get; init; }

    /// <summary>
    /// Raw arguments the model supplied for this tool call, for display / audit.
    /// </summary>
    public required JsonObject Arguments { get; init; }
}
