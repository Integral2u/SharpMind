using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;

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
public sealed class DebugChatBridge(CuiToolContext cuiContext, Func<ToolPermissionContext, Task<ToolPermission>> permissionCallback) : IChatBridge
{
    public string UserName { get; set; } = "User";
    private readonly ConcurrentQueue<ChatStreamEntry> _incoming = new();
    private readonly List<ChatMessage> _history = [];
    private CancellationTokenSource _cts = new();
    private Task? _runningTurn;
    private readonly Func<ToolPermissionContext, Task<ToolPermission>> _permissionCallback = permissionCallback;

    public bool Faulted { get; private set; }
    public Exception? Fault { get; private set; }
    public bool ShowThinking { get; set; } = true;
    public bool EnableThinking { get; set; }
    public ChatArtifact[]? LastArtifacts { get; private set; }

    public IChatPromptFormatter? Formatter => null;

    public void SubmitUserInput(string text, ChatArtifact[]? artifacts = null)
    {
        var msg = ChatMessage.User(text, UserName);
        if (artifacts is { Length: > 0 })
            msg.Artifacts = artifacts;
        _history.Add(msg);
        // Each submission runs as its own short-lived task rather than a single
        // long-running loop like the real bridge's — there's no model generation
        // to serialise against here, and letting each turn be independent makes
        // it trivial to fire off a TestAgent turn without blocking on whatever
        // the previous scripted turn was doing.
        _runningTurn = Task.Run(() => RunScriptedTurnAsync(text, _cts.Token));
    }

    /// <summary>
    /// Single source of truth for both dispatch and the help listing, so the
    /// two can't drift out of sync — adding a new TestX command here makes it
    /// both runnable and discoverable in one place.
    /// </summary>
    private static readonly (string Command, string Description)[] KnownCommands =
    [
        ("TestOptions", "Exercises a real UIShowOptionSelection round trip through the choice dialog."),
        ("TestAgent", "Simulates a sub-agent delegation (Executing/Researching/Responding) to test transcript attribution."),
        ("TestWeather", "Simulates a weather tool call to test Network permission interception."),
        ("TestFile", "Simulates a file system tool call to test File permission interception."),
        ("TestFileArtifact", "Sends a .txt file artifact with the user message and simulates 2 returned code artifacts."),
    ];

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

        if (string.Equals(command, "TestWeather", StringComparison.OrdinalIgnoreCase))
        {
            await RunTestWeatherAsync(token);
            return;
        }

        if (string.Equals(command, "TestFile", StringComparison.OrdinalIgnoreCase))
        {
            await RunTestFileAsync(token);
            return;
        }

        if (string.Equals(command, "TestFileArtifact", StringComparison.OrdinalIgnoreCase))
        {
            await RunTestFileArtifactAsync(token);
            return;
        }

        await RunHelpAsync(command, token);
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
    }

    /// <summary>
    /// UIDebug mode has no real model to interpret arbitrary input, so
    /// anything that isn't one of <see cref="KnownCommands"/> gets this
    /// instead of a generic echo — the point of the mode is exercising
    /// specific scripted scenarios, and a person poking at it for the first
    /// time should be told what's actually runnable rather than left
    /// guessing from an echoed-back message that doesn't explain anything.
    /// </summary>
    private async Task RunHelpAsync(string unrecognizedInput, CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(80, token);
        Emit(ChatStatus.Responding, null);

        var lines = new List<string> { $"\"{unrecognizedInput}\" isn't a recognized test command. Available commands:" };
        lines.AddRange(KnownCommands.Select(c => $"  {c.Command} — {c.Description}"));
        string message = string.Join('\n', lines);

        _history.Add(ChatMessage.Agent(message));
        await StreamTextAsync(message, token);
        Complete();
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
        string message = $"You chose: \"{chosen}\". That round trip — tool call to dialog to result — is the exact path a real model's UIShowOptionSelection call takes.";
        _history.Add(ChatMessage.Agent(message));
        await StreamTextAsync(message, token);
        Complete();
    }

    /// <summary>
    /// Reproduces the actual entry sequence ChatSession emits for a real
    /// {{agent:name:query}} delegation: Executing(name) to announce the
    /// sub-agent, Researching(fragment) for each piece of its streamed
    /// output, then — because ChatSession's loop continues and generates
    /// again after a sub-agent call rather than ending the turn there — a
    /// further Responding stream from the "top-level" agent picking back up,
    /// before the one Complete that ends the whole turn. This exercises
    /// ChatScreen's real flush-on-phase-transition attribution logic
    /// directly, rather than a parallel shortcut, since that logic is the
    /// part actually worth testing here.
    /// </summary>
    private async Task RunTestAgentAsync(CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(150, token);

        Emit(ChatStatus.Executing, "Athena-Alpha");
        await Task.Delay(100, token);

        string subAgentText = "This text is streamed as Researching fragments, exactly like a real sub-agent delegation — if it shows up under \"Athena-Alpha\" rather than the top-level agent's name, attribution is working correctly.";
        var subAgentWords = subAgentText.Split(' ');
        for (int i = 0; i < subAgentWords.Length; i++)
        {
            token.ThrowIfCancellationRequested();
            string piece = i == 0 ? subAgentWords[i] : " " + subAgentWords[i];
            Emit(ChatStatus.Researching, piece);
            await Task.Delay(20, token);
        }

        // ChatSession loops back to the top-level agent after a sub-agent call
        // completes, rather than ending the turn — this Responding burst is
        // what should land under the ordinary agent name, separately from the
        // sub-agent text above, in the same transcript.
        Emit(ChatStatus.Responding, null);
        await StreamTextAsync("And this part streams afterward as ordinary Responding tokens from the top-level agent picking back up — it should land as a separate transcript entry under the regular agent name, not get merged into the sub-agent's text above.", token);
        Complete();
    }

    private async Task RunTestWeatherAsync(CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(100, token);

        Emit(ChatStatus.Executing, "GetCurrentWeather");
        
        // Trigger the real permission gate
        var permission = await _permissionCallback(new ToolPermissionContext
        {
            ToolName = "GetCurrentWeather",
            Category = ToolCategory.Network,
            Resource = "api.open-meteo.com",
            Arguments = []
        });

        if (permission == ToolPermission.Never)
        {
            Emit(ChatStatus.Responding, null);
            string message = "Error: Network access was denied by the user.";
            _history.Add(ChatMessage.Agent(message));
            await StreamTextAsync(message, token);
            Complete();
            return;
        }

        await Task.Delay(500, token);

        Emit(ChatStatus.Responding, null);
        string weatherMsg = "The current weather in London is 12°C with light rain and a wind speed of 15km/h.";
        _history.Add(ChatMessage.Agent(weatherMsg));
        await StreamTextAsync(weatherMsg, token);
        Complete();
    }

    private async Task RunTestFileAsync(CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(100, token);

        Emit(ChatStatus.Executing, "ListFiles");
        
        // Trigger the real permission gate
        var permission = await _permissionCallback(new ToolPermissionContext
        {
            ToolName = "ListFiles",
            Category = ToolCategory.File,
            Resource = ".",
            Arguments = []
        });

        if (permission == ToolPermission.Never)
        {
            Emit(ChatStatus.Responding, null);
            string message = "Error: File system access was denied by the user.";
            _history.Add(ChatMessage.Agent(message));
            await StreamTextAsync(message, token);
            Complete();
            return;
        }

        await Task.Delay(500, token);

        Emit(ChatStatus.Responding, null);
        string fileMsg = "[DIR] src\n[FILE] README.md\n[FILE] LICENSE";
        _history.Add(ChatMessage.Agent(fileMsg));
        await StreamTextAsync(fileMsg, token);
        Complete();
    }

    /// <summary>
    /// Simulates sending a .txt file artifact with the user message and receiving
    /// 2 code artifacts back — one yielded via ChatStreamEntry.Artifact (streamed)
    /// and one attached to the final ChatMessage.Artifacts. Exercises both
    /// artifact data paths so the bridge's LastArtifacts merge works correctly.
    /// </summary>
    private async Task RunTestFileArtifactAsync(CancellationToken token)
    {
        Emit(ChatStatus.Thinking, null);
        await Task.Delay(100, token);

        Emit(ChatStatus.Responding, null);

        // Stream some simulated response text
        string responseText = "I've analyzed the file and generated 2 code artifacts.";
        _history.Add(ChatMessage.Agent(responseText));
        await StreamTextAsync(responseText, token);

        // Yield first artifact via ChatStreamEntry.Artifact (streamed path)
        var streamedArtifact = new ChatArtifact
        {
            Type = "code",
            Content = "def hello():\n    print('Hello from streamed artifact')"u8.ToArray(),
            Language = "py",
            FileName = "hello.py"
        };
        _incoming.Enqueue(new ChatStreamEntry
        {
            Status = ChatStatus.Responding,
            Token = null,
            Artifact = streamedArtifact,
            IsComplete = false,
            TokensPerSecond = 999f
        });

        // Attach second artifact to the agent message (final message path)
        var msgArtifact = new ChatArtifact
        {
            Type = "code",
            Content = "console.log('Hello from message artifact');"u8.ToArray(),
            Language = "js",
            FileName = "hello.js"
        };
        _history[^1].Artifacts = [msgArtifact];

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
        if (_history.Count > 0 && _history[^1].Role == ChatRole.Agent)
            LastArtifacts = _history[^1].Artifacts;
        _incoming.Enqueue(new ChatStreamEntry { Status = ChatStatus.Complete, IsComplete = true, TokensPerSecond = 999f });
    }

    public IEnumerable<ChatStreamEntry> DrainEntries()
    {
        while (_incoming.TryDequeue(out var entry))
            yield return entry;
    }

    public IReadOnlyList<ChatMessage> GetHistory() => _history;

    public ChatSessionSnapshot GetSnapshot() => new() { History = [.. _history] };

    public void LoadSnapshot(ChatSessionSnapshot snapshot)
    {
        _history.Clear();
        foreach (var msg in snapshot.History)
            _history.Add(msg);
    }

    public void ToggleIgnore(int index)
    {
        if (index < 0 || index >= _history.Count) return;
        var msg = _history[index];
        if (msg.IsPinned) return;
        msg.Ignore = !msg.Ignore;
    }

    public void ResetCache() { }

    public void Interrupt()
    {
        _cts.Cancel();
        _cts.Dispose();
        _cts = new CancellationTokenSource();
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
