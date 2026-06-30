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

    public Func<ToolPermissionContext, Task<ToolPermission>> BuildCallback(SessionOptions options) => context =>
    {
        var configured = context.Category == ToolCategory.Network ? options.NetworkAccess : options.FileAccess;

        if (configured != ToolPermission.Ask)
            return Task.FromResult(configured);

        var request = new PermissionRequest { Context = context };
        lock (_gate)
        {
            if (_pending is not null)
                throw new InvalidOperationException("A permission request is already awaiting a response.");
            _pending = request;
        }
        return request.Result;
    };

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
