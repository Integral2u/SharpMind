using NStack;
using SharpMind.CUI.App;
using SharpMind.Inference.Chat;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// The chat screen: scrolling transcript, live status sidebar, input line.
/// Sub-agent attribution logic (Executing announces a name, Researching
/// confirms it's a sub-agent and streams its text, flush on phase
/// transition rather than waiting for the turn's single Complete) mirrors
/// the earlier console-UI version — that logic was never about the
/// rendering layer, it's about correctly reading what ChatSession already
/// puts on the wire. Only the drawing/input mechanism changed here.
///
/// Terminal.Gui has no manual frame loop to drain the bridge's queues from,
/// so this polls via Application.MainLoop.AddTimeout — the standard
/// Terminal.Gui mechanism for periodic work that isn't triggered by a UI
/// event.
/// </summary>
public sealed class ChatView : View
{
    private readonly string _agentName;
    private readonly IChatBridge _bridge;
    private readonly CuiToolContext? _cuiContext;
    private readonly Action _onExit;

    private readonly TextView _transcriptView;
    private readonly TextField _inputField;
    private readonly Label _statusLabel;
    private readonly Label _agentLabel;
    private readonly Label _strategyLabel;
    private readonly Label _toolLabel;
    private readonly Label _speedLabel;

    private readonly System.Text.StringBuilder _liveResponse = new();
    private readonly System.Text.StringBuilder _subAgentBuffer = new();
    private string? _pendingSpeakerName;
    private string? _activeSubAgentName;
    private bool _generating;
    private object? _timeoutToken;
    private bool _disposed;

    public ChatView(string agentName, SessionOptions options, IChatBridge bridge, CuiToolContext? cuiContext, Action onExit)
    {
        _agentName = agentName;
        _bridge = bridge;
        _cuiContext = cuiContext;
        _onExit = onExit;

        int sidebarWidth = 24;

        _transcriptView = new TextView
        {
            X = 0, Y = 0,
            Width = Dim.Fill(sidebarWidth + 1),
            Height = Dim.Fill(2),
            ReadOnly = true,
            WordWrap = true
        };

        var sidebarFrame = new FrameView("Status")
        {
            X = Pos.Right(_transcriptView) + 1, Y = 0,
            Width = sidebarWidth, Height = Dim.Fill(2)
        };

        _statusLabel = new Label("Ready") { X = 0, Y = 0, Width = Dim.Fill() };
        _agentLabel = new Label(agentName) { X = 0, Y = 2, Width = Dim.Fill() };
        _strategyLabel = new Label($"{options.Generator}\n{options.Cache}\n{(options.UseGpu ? $"GPU ({options.HardwareTier})" : options.HardwareTier.ToString())}")
        { X = 0, Y = 4, Width = Dim.Fill(), Height = 3 };
        _toolLabel = new Label("") { X = 0, Y = 8, Width = Dim.Fill() };
        _speedLabel = new Label("--") { X = 0, Y = 10, Width = Dim.Fill() };

        sidebarFrame.Add(
            new Label("Status:") { X = 0, Y = 0 }, _statusLabel,
            new Label("Agent:") { X = 0, Y = 2 }, _agentLabel,
            new Label("Strategy:") { X = 0, Y = 3 }, _strategyLabel,
            _toolLabel,
            new Label("Speed:") { X = 0, Y = 9 }, _speedLabel);

        _inputField = new TextField("") { X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill() };
        _inputField.KeyPress += OnInputKeyPress;

        Add(_transcriptView, sidebarFrame, _inputField);

        KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == Key.Esc && !_generating) { _onExit(); args.Handled = true; }
        };

        // 60fps-equivalent poll for background-thread updates (stream entries, choice requests) —
        // the same cadence the old manual frame loop used, expressed as a Terminal.Gui timeout instead.
        _timeoutToken = Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(16), PollBackgroundState);

        _inputField.SetFocus();
    }

    private void OnInputKeyPress(KeyEventEventArgs args)
    {
        if (args.KeyEvent.Key != Key.Enter || _generating) return;

        string text = _inputField.Text.ToString() ?? "";
        if (text.Length == 0) { args.Handled = true; return; }

        _inputField.Text = "";
        AppendTranscript($"You: {text}");
        _generating = true;
        _statusLabel.Text = "Thinking...";
        _bridge.SubmitUserInput(text);
        args.Handled = true;
    }

    /// <summary>Returning true keeps the timeout recurring — see Application.MainLoop.AddTimeout's contract.</summary>
    private bool PollBackgroundState(MainLoop _)
    {
        if (_disposed) return false;

        foreach (var entry in _bridge.DrainEntries())
            OnStreamEntry(entry);

        if (_cuiContext?.TakePending() is { } request)
            ShowChoiceDialog(request);

        return true;
    }

    private void OnStreamEntry(ChatStreamEntry entry)
    {
        if (_activeSubAgentName is not null && entry.Status != ChatStatus.Researching)
        {
            if (_subAgentBuffer.Length > 0)
                AppendTranscript($"{_activeSubAgentName}: {_subAgentBuffer}");
            _subAgentBuffer.Clear();
            _activeSubAgentName = null;
        }

        _statusLabel.Text = StatusLabel(entry.Status);
        _speedLabel.Text = entry.TokensPerSecond is { } tps ? $"{tps:F1} tok/s" : "--";

        if (entry.Status == ChatStatus.Executing)
        {
            _toolLabel.Text = entry.Token ?? "";
            _pendingSpeakerName = entry.Token;
        }

        if (entry.Status == ChatStatus.Researching)
        {
            _activeSubAgentName ??= _pendingSpeakerName;
            if (entry.Token is not null) _subAgentBuffer.Append(entry.Token);
        }

        if (entry.Status == ChatStatus.Responding && entry.Token is not null)
            _liveResponse.Append(entry.Token);

        if (entry.IsComplete || entry.Status is ChatStatus.Complete or ChatStatus.Interrupted)
        {
            if (_liveResponse.Length > 0)
                AppendTranscript($"{_agentName}: {_liveResponse}");
            _liveResponse.Clear();
            _generating = false;
            _toolLabel.Text = "";
            _pendingSpeakerName = null;
            _statusLabel.Text = "Ready";
        }

        SetNeedsDisplay();
    }

    private void AppendTranscript(string line)
    {
        _transcriptView.Text = _transcriptView.Text.ToString() + line + "\n\n";
        _transcriptView.MoveEnd();
        _transcriptView.SetNeedsDisplay();
    }

    private static string StatusLabel(ChatStatus s) => s switch
    {
        ChatStatus.Thinking => "Thinking...",
        ChatStatus.Updating => "Updating...",
        ChatStatus.Executing => "Executing...",
        ChatStatus.Responding => "Responding...",
        ChatStatus.Waiting => "Ready",
        ChatStatus.Researching => "Researching...",
        ChatStatus.Interrupted => "Interrupted",
        ChatStatus.Complete => "Done",
        _ => s.ToString()
    };

    /// <summary>
    /// Modal choice dialog for UIShowOptionSelection — a real Terminal.Gui
    /// Dialog with RadioGroup + optional free-text field, replacing the
    /// hand-rolled overlay from the console-UI version. Resolves the
    /// blocked tool call the instant a button is pressed.
    /// </summary>
    private void ShowChoiceDialog(ChoiceRequest request)
    {
        var dialog = new Dialog("Choose an option", 60, Math.Min(20, request.Options.Count + (request.AllowFreeText ? 8 : 5)));

        var radio = new RadioGroup(request.Options.Select(p => (ustring)p).ToArray()) { X = 1, Y = 1 };
        dialog.Add(new Label(request.Prompt) { X = 1, Y = 0, Width = Dim.Fill(2) }, radio);

        TextField? freeTextField = null;
        if (request.AllowFreeText)
        {
            dialog.Add(new Label("Or type your own:") { X = 1, Y = Pos.Bottom(radio) + 1 });
            freeTextField = new TextField("") { X = 1, Y = Pos.Bottom(radio) + 2, Width = Dim.Fill(2) };
            dialog.Add(freeTextField);
        }

        var okButton = new Button("OK") { IsDefault = true };
        okButton.Clicked += () =>
        {
            string chosen = (freeTextField is not null && freeTextField.Text.ToString()!.Length > 0)
                ? freeTextField.Text.ToString()!
                : request.Options[radio.SelectedItem];
            request.Resolve(chosen);
            Application.RequestStop();
        };
        dialog.AddButton(okButton);

        Application.Run(dialog);
    }

    protected override void Dispose(bool disposing)
    {
        _disposed = true;
        if (disposing && _timeoutToken is not null)
            Application.MainLoop.RemoveTimeout(_timeoutToken);
        base.Dispose(disposing);
    }
}
