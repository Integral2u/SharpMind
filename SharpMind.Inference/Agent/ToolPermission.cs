using System.Text.Json.Nodes;

namespace SharpMind.Inference.Agent;

/// <summary>
/// Broad category of a tool, used to decide which permission gate to apply.
/// Determined by whether the tool host class implements
/// <see cref="IFileToolService"/> or <see cref="INetworkToolService"/>.
/// </summary>
public enum ToolCategory
{
    /// <summary>Pure computation — no IO involved.</summary>
    General,
    /// <summary>Reads or writes the local file system.</summary>
    File,
    /// <summary>Makes outbound network calls.</summary>
    Network
}

/// <summary>
/// Permission level the host application grants for a particular tool invocation.
/// </summary>
public enum ToolPermission
{
    /// <summary>Block this call unconditionally.</summary>
    Never,
    /// <summary>Prompt the user and wait for their decision before proceeding.</summary>
    Ask,
    /// <summary>Execute immediately without prompting.</summary>
    Always
}

/// <summary>
/// Passed to the <see cref="ChatSession{T,K}.PermissionCallback"/> so the host
/// application has everything it needs to make an informed allow/deny decision.
/// </summary>
public sealed class ToolPermissionContext
{
    /// <summary>Registered tool name (matches <c>AgentBuilder.ToolMethods</c> key).</summary>
    public required string ToolName { get; init; }

    /// <summary>Broad IO category derived from the tool host's marker interface.</summary>
    public required ToolCategory Category { get; init; }

    /// <summary>
    /// Raw arguments the model produced. Provided for display / audit;
    /// the session does not execute the call until <see cref="ToolPermission.Always"/> is returned.
    /// </summary>
    public required JsonObject Arguments { get; init; }
}
