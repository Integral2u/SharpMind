using SharpMind.Inference.Agent;

namespace SharpMind.CUI.App;

/// <summary>
/// A pending yes/no permission decision, posted from the background
/// chat-loop thread (inside ChatSession's permission callback) and resolved
/// by the UI thread once the person answers a confirmation dialog. Same
/// shape and same reason as <see cref="ChoiceRequest"/> — a tool call is
/// synchronously blocked waiting on this, so there's never more than one in
/// flight at a time.
/// </summary>
public sealed class PermissionRequest
{
    public required ToolPermissionContext Context { get; init; }
    private readonly TaskCompletionSource<ToolPermission> _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public Task<ToolPermission> Result => _completion.Task;
    public void Resolve(ToolPermission decision) => _completion.TrySetResult(decision);
}

/// <summary>
/// Builds the <c>Func&lt;ToolPermissionContext, Task&lt;ToolPermission&gt;&gt;</c>
/// callback ChatSession calls before any gated file/network operation.
/// Always/Never modes answer immediately with no UI involved; Ask mode posts
/// a <see cref="PermissionRequest"/> the render thread picks up (see
/// MainWindow's poll loop) and shows as a real confirmation dialog,
/// blocking the calling tool until it's answered — exactly the same pattern
/// already used for UIShowOptionSelection's choice dialog, since both are
/// "a background thread needs an answer only the UI thread can actually
/// provide" problems.
/// </summary>
public sealed class PermissionGate
{
    private readonly object _gate = new();
    private PermissionRequest? _pending;

    /// <summary>
    /// Builds the session's permission callback.
    /// </summary>
    /// <param name="options">Session options carrying the FileAccess/NetworkAccess defaults and approved roots.</param>
    /// <param name="restrictedToolNames">
    /// Tool names that are forced through the Ask flow even when the configured default is
    /// <see cref="ToolPermission.Always"/> — used for tools originating from plugins embedded
    /// in an SMM model. <see cref="ToolPermission.Never"/> still blocks outright.
    /// </param>
    public Func<ToolPermissionContext, Task<ToolPermission>> BuildCallback(
        SessionOptions options,
        IReadOnlySet<string>? restrictedToolNames = null)
    {
        var approvedRoots = BuildApprovedRoots(options);
        return context =>
        {
            var decision = PermissionPolicy.Resolve(
                context.ToolName,
                context.Category,
                context.Resource,
                options.FileAccess,
                options.NetworkAccess,
                approvedRoots,
                restrictedToolNames);

            if (decision != ToolPermission.Ask)
                return Task.FromResult(decision);

            var request = new PermissionRequest { Context = context };
            lock (_gate)
            {
                if (_pending is not null)
                    throw new InvalidOperationException("A permission request is already awaiting a response.");
                _pending = request;
            }
            return request.Result;
        };
    }

    /// <summary>
    /// Directories treated as implicitly trusted for file access: the project path,
    /// the model file's own directory, the tools folder, and the plugins folder.
    /// Resources inside these are allowed without prompting in Ask mode; anything
    /// that escapes to a parent directory triggers a request.
    /// </summary>
    private static List<string> BuildApprovedRoots(SessionOptions options)
    {
        var roots = new List<string>();
        if (!string.IsNullOrWhiteSpace(options.ProjectPath)) roots.Add(options.ProjectPath);
        if (!string.IsNullOrWhiteSpace(options.ModelPath))
        {
            string? dir = Path.GetDirectoryName(options.ModelPath);
            if (!string.IsNullOrWhiteSpace(dir)) roots.Add(dir);
        }
        if (!string.IsNullOrWhiteSpace(options.ToolsFolder)) roots.Add(options.ToolsFolder);
        roots.Add(Path.Combine(AppContext.BaseDirectory, "plugins"));
        return roots;
    }

    /// <summary>Polled once per frame by the UI thread. Returns and clears the pending request, if any.</summary>
    public PermissionRequest? TakePending()
    {
        lock (_gate)
        {
            var p = _pending;
            _pending = null;
            return p;
        }
    }
}
