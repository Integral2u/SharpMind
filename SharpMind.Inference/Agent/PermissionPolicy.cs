namespace SharpMind.Inference.Agent;

/// <summary>
/// Pure decision logic for the session permission gate. Split out of the CUI's
/// <c>PermissionGate</c> so the Ask/Always/Never rules, the path-aware file rule,
/// and the embedded-plugin Ask-forcing are unit-testable with no UI dependency.
/// </summary>
public static class PermissionPolicy
{
    /// <summary>
    /// Returns the permission a host should grant for a single access attempt.
    /// A result of <see cref="ToolPermission.Ask"/> means "block until the user
    /// answers" — the host is responsible for surfacing the request.
    /// </summary>
    /// <param name="toolName">Name of the tool making the attempt (tool method name).</param>
    /// <param name="category">Whether the resource is file-system or network.</param>
    /// <param name="resource">The path or URL being accessed.</param>
    /// <param name="fileAccess">Configured default for file access.</param>
    /// <param name="networkAccess">Configured default for network access.</param>
    /// <param name="approvedRoots">Directories implicitly trusted for file access.</param>
    /// <param name="restrictedToolNames">
    /// Tool names (embedded plugins) that are forced through the Ask flow even when the
    /// configured default is <see cref="ToolPermission.Always"/>. <see cref="ToolPermission.Never"/>
    /// still blocks outright.
    /// </param>
    public static ToolPermission Resolve(
        string? toolName,
        ToolCategory category,
        string resource,
        ToolPermission fileAccess,
        ToolPermission networkAccess,
        IReadOnlyCollection<string> approvedRoots,
        IReadOnlySet<string>? restrictedToolNames)
    {
        var configured = category == ToolCategory.Network ? networkAccess : fileAccess;

        // Embedded-plugin tools are forced through the Ask flow so the first access
        // attempt always calls the permission callback. Never still blocks outright.
        if (toolName is not null && restrictedToolNames?.Contains(toolName) == true)
            return configured == ToolPermission.Never ? ToolPermission.Never : ToolPermission.Ask;

        // Path-aware rule refines Ask mode only: resources inside an approved root
        // are assumed accessible; escaping to a parent directory triggers a prompt.
        if (category == ToolCategory.File
            && configured == ToolPermission.Ask
            && approvedRoots.Count > 0
            && PermissionPathPolicy.IsResourceInsideRoots(resource, approvedRoots))
        {
            return ToolPermission.Always;
        }

        return configured;
    }
}
