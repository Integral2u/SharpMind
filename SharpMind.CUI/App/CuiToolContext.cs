namespace SharpMind.CUI.App;

/// <summary>
/// Shared state between <see cref="CuiTools"/> (whose methods run on the
/// background chat-loop thread, invoked via reflection from
/// <c>AgentBuilder.CallToolAsync</c>) and <see cref="App"/> (running on the
/// render thread, the only thing that can actually draw a dialog and read
/// keystrokes).
///
/// There's deliberately only one mutable field here. A tool call is a
/// synchronous round trip from the model's point of view — it cannot issue
/// a second <c>UIShowOptionSelection</c> call before the first one's result
/// comes back — so there is never more than one request in flight, and a
/// single nullable field is the entire amount of coordination needed. If
/// that ever stops being true (e.g. concurrent sub-agents each wanting their
/// own dialog at the same time), this would need to become a queue instead;
/// it isn't one yet because nothing in the engine today can actually produce
/// that situation.
/// </summary>
public sealed class CuiToolContext
{
    private readonly object _gate = new();
    private ChoiceRequest? _pending;

    /// <summary>
    /// Called by a tool method to post a request and (asynchronously) wait
    /// for the render thread to resolve it. Throws if a request is already
    /// pending — surfaced back to the model as a tool error rather than
    /// silently overwriting or queueing, since that situation should not be
    /// reachable given the one-call-at-a-time nature of tool dispatch, and
    /// if it ever is reached, that's a real bug worth knowing about loudly
    /// rather than a dialog quietly being replaced out from under the user.
    /// </summary>
    public Task<string> RequestChoiceAsync(string prompt, IReadOnlyList<string> options, bool allowFreeText)
    {
        var request = new ChoiceRequest { Prompt = prompt, Options = options, AllowFreeText = allowFreeText };
        lock (_gate)
        {
            if (_pending is not null)
                throw new InvalidOperationException("A choice request is already awaiting a response.");
            _pending = request;
        }
        return request.Result;
    }

    /// <summary>Polled by the render thread once per frame. Returns and clears the pending request, if any.</summary>
    public ChoiceRequest? TakePending()
    {
        lock (_gate)
        {
            var p = _pending;
            _pending = null;
            return p;
        }
    }
}
