namespace SharpMind.CUI.App;

/// <summary>
/// A request for the user to pick one of several options, optionally
/// allowing a free-typed answer instead. Raised from a tool call running on
/// the background chat loop (see <see cref="ChatSessionBridge"/> for why
/// that's a different thread from the one that can actually draw a dialog),
/// resolved once the render thread has shown <see cref="ChoiceDialog"/> and
/// the user has picked something.
///
/// This is intentionally a plain data+TaskCompletionSource pair rather than
/// something fancier — there is exactly one of these in flight at a time
/// (the model is synchronously awaiting the tool call's result, so it
/// cannot issue a second choice request before this one resolves), so there
/// is nothing to queue or coordinate beyond "set the result, wake up the
/// waiting tool call."
/// </summary>
public sealed class ChoiceRequest
{
    public required string Prompt { get; init; }
    public required IReadOnlyList<string> Options { get; init; }

    /// <summary>If true, the dialog also offers a free-text entry as an alternative to picking a listed option.</summary>
    public bool AllowFreeText { get; init; }

    private readonly TaskCompletionSource<string> _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);

    /// <summary>Awaited by the tool call; completes once the render thread calls <see cref="Resolve"/>.</summary>
    public Task<string> Result => _completion.Task;

    /// <summary>Called by the render thread once the user has made a choice — either a listed option's text, or their own typed answer.</summary>
    public void Resolve(string chosenText) => _completion.TrySetResult(chosenText);

    /// <summary>Called if the dialog is dismissed without a choice (e.g. the session ends mid-prompt) so the waiting tool call doesn't hang forever.</summary>
    public void Cancel() => _completion.TrySetCanceled();
}
