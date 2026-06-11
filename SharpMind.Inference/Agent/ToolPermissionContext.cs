using System.Text.Json.Nodes;

namespace SharpMind.Inference.Agent;

/// <summary>
/// Passed to <see cref="Chat.ChatSession{T,K}.PermissionCallback"/>
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
