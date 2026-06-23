using System.Collections.Concurrent;
using SharpMind.Inference.Chat;

namespace SharpMind.CUI.App;

/// <summary>
/// Implements <see cref="IChatBridge"/> with no real model behind it at all
/// — no <c>Transformer</c>, no <c>Tokenizer</c>, no GGUF file. This exists
/// purely to test the CUI's own plumbing (rendering, the chat screen's
/// status sidebar, the choice dialog, sub-agent name display) against known,
/// repeatable scripted output instead of needing a real model loaded and
/// genuinely inferring every time something in the UI layer needs checking.
///
/// Recognises a small set of literal commands typed into the chat input;
/// anything else just gets echoed back with a fixed-pace simulated token
/// stream so the transcript/status-sidebar/tokens-per-second display all
/// have something realistic to render.
/// </summary>
public sealed class DebugChatBridge(CuiToolContext cuiContext) : IChatBridge
{
    private readonly ConcurrentQueue<ChatStreamEntry> _incoming = new();
    private readonly CancellationTokenSource _cts = new();
    private Task? _runningTurn;

    public bool Faulted { get; private set; }
    public Exception? Fault { get; private set; }

    /// <summary>
    /// The simulated speaker name for whatever is currently in flight, or
    /// null for the ordinary top-level-agent case. App polls this once per
    /// frame, right alongside <see cref="DrainEntries"/>, and forwards it to
    /// <see cref="ChatScreen.SetDebugSpeakerOverride"/> — this only exists on
    /// the debug bridge because only TestAgent ever needs it; the real
    /// bridge never sets this to anything but null because the real engine
    /// has no per-entry speaker identity to report in the first place.
    /// </summary>
    public string? CurrentSpeakerOverride { get; private set; }

    public void SubmitUserInput(string text)
    {
        // Each submission runs as its own short-lived task rather than a single
        // long-running loop like the real bridge's — there's no model generation
        // to serialise against here, and letting each turn be independent makes
        // it trivial to fire off a TestAgent turn without blocking on whatever
        // the previous scripted turn was doing.
        _runningTurn = Task.Run(() => RunScriptedTurnAsync(text, _cts.Token));
    }

    private async Task RunScriptedTurnAsync(string input, CancellationToken token)
    {
        try
        {
            string command = input.Trim();

            if (string.Equals(command, "TestOptions", StringComparison.OrdinalIgnoreCase))
            {
                await RunTestOptionsAsync(token);
                return;
            }

            if (string.Equals(command, "TestAgent", StringComparison.OrdinalIgnoreCase))
            {
                await RunTestAgentAsync(token);
                return;
            }

            await RunEchoAsync(input, token);
        }
        catch (OperationCanceledException)
        {
            // expected on shutdown
        }
        catch (Exception ex)
        {
            Faulted = true;
            Fault = ex;
        }
        finally
        {
            CurrentSpeakerOverride = null;
        }
    }

    /// <summary>
    /// Exercises the same UIShowOptionSelection path a real model would call
    /// through CuiTools — drives it through CuiToolContext directly rather
    /// than going via reflection/JSON tool-call parsing, since there's no
    /// model output to parse here; the point is testing the dialog and the
    /// render-thread handoff, not re-testing the JSON tool-call dispatcher
    /// (which has nothing to do with this bridge being scripted or real).
    /// </summary>
    private async Task RunTestOptionsAsync(CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(200, token);

        Emit(ChatStatus.Executing, "UIShowOptionSelection");
        string chosen = await cuiContext.RequestChoiceAsync(
            "This is a scripted UIShowOptionSelection call from TestOptions — pick anything.",
            ["Option A", "Option B", "Option C"],
            allowFreeText: true);

        Emit(ChatStatus.Responding, null);
        await StreamTextAsync($"You chose: \"{chosen}\". That round trip — tool call to dialog to result — is the exact path a real model's UIShowOptionSelection call takes.", token);
        Complete();
    }

    /// <summary>
    /// Simulates a sub-agent responding under its own name, to verify the
    /// transcript actually distinguishes "Delta" from a named sub-agent
    /// rather than labelling every response with the top-level agent's name
    /// regardless of which agent produced it. See
    /// <see cref="CurrentSpeakerOverride"/> for the honest caveat on how far
    /// this actually reaches — it's a CUI-side simulation, not a real
    /// engine capability.
    /// </summary>
    private async Task RunTestAgentAsync(CancellationToken token)
    {
        CurrentSpeakerOverride = "Athena-Alpha";

        Emit(ChatStatus.Thinking, null);
        await Task.Delay(150, token);

        Emit(ChatStatus.Responding, null);
        await StreamTextAsync("This response is tagged as coming from a sub-agent named Athena-Alpha, not from the top-level agent — if the transcript shows that name instead of the agent's, sub-agent visibility is wired correctly in the CUI layer. The real engine does not yet report sub-agent identity on its own, so a genuine model's sub-agent calls won't show this until that's added upstream.", token);
        Complete();
    }

    private async Task RunEchoAsync(string input, CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(150, token);
        Emit(ChatStatus.Responding, null);
        await StreamTextAsync($"[debug echo] {input}", token);
        Complete();
    }

    /// <summary>Splits text into word-sized fake "tokens" with a small delay between each, so the streaming UI has something realistic to animate.</summary>
    private async Task StreamTextAsync(string text, CancellationToken token)
    {
        var words = text.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            token.ThrowIfCancellationRequested();
            string piece = i == 0 ? words[i] : " " + words[i];
            Emit(ChatStatus.Responding, piece, tokensPerSecond: 999f);
            await Task.Delay(20, token);
        }
    }

    private void Emit(ChatStatus status, string? token, float? tokensPerSecond = null)
    {
        _incoming.Enqueue(new ChatStreamEntry
        {
            Status = status,
            Token = token,
            TokensPerSecond = tokensPerSecond,
            IsComplete = false
        });
    }

    private void Complete()
    {
        _incoming.Enqueue(new ChatStreamEntry { Status = ChatStatus.Complete, IsComplete = true, TokensPerSecond = 999f });
    }

    public IEnumerable<ChatStreamEntry> DrainEntries()
    {
        while (_incoming.TryDequeue(out var entry))
            yield return entry;
    }

    public async ValueTask DisposeAsync()
    {
        await _cts.CancelAsync();
        if (_runningTurn is not null)
        {
            try { await _runningTurn; } catch { /* already surfaced via Fault if relevant */ }
        }
        _cts.Dispose();
    }
}
