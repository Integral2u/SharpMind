using NStack;
using SharpMind;
using SharpMind.CUI.App;
using SharpMind.Inference.Chat;
using Terminal.Gui;
using System.Runtime.Intrinsics.X86;

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
    public string AgentName { get; private set; }
    public string SessionDisplayName { get; set; }

    private readonly IChatBridge _bridge;
    private readonly CuiToolContext? _cuiContext;
    private readonly Action _onExit;
    private readonly Action<bool>? _onGeneratingChanged;

    private readonly TextView _transcriptView;
    private readonly TextField _inputField;
    private readonly Button _interruptButton;
    private readonly Button _sendButton;
    private readonly Button _getButton;
    private readonly Label _statusLabel;
    private readonly Label _agentLabel;
    private readonly Label _strategyLabel;
    private readonly Label _toolLabel;
    private readonly Label _speedLabel;
    private readonly Label _ttftLabel;
    private readonly Label _memLabel;

    private readonly System.Text.StringBuilder _liveResponse = new();
    private readonly System.Text.StringBuilder _subAgentBuffer = new();
    private string? _pendingSpeakerName;
    private string? _activeSubAgentName;
    private bool _generating;
    private bool _interruptDialogOpen;
    private object? _timeoutToken;
    private bool _disposed;

    // Sidebar stats persist after a turn completes instead of resetting to
    // "--" — per request, the last known speed/TTFT should stay visible
    // until the next turn actually produces new numbers, not blank out the
    // moment a response finishes.
    private float? _lastTokensPerSecond;
    private float? _lastTimeToFirstToken;

    public ChatView(string agentName, SessionOptions options, IChatBridge bridge, CuiToolContext? cuiContext, Action onExit, Action<bool>? onGeneratingChanged = null)
    {
        AgentName = agentName;
        SessionDisplayName = agentName;
        _bridge = bridge;
        _cuiContext = cuiContext;
        _onExit = onExit;
        _onGeneratingChanged = onGeneratingChanged;

        int sidebarWidth = 26;

        _transcriptView = new TextView
        {
            X = 0,
            Y = 0,
            Width = Dim.Fill(sidebarWidth + 1),
            Height = Dim.Fill(2),
            ReadOnly = true,
            WordWrap = true,
            // CanFocus stays true so PageUp/PageDown/arrow scrolling still
            // works, but it's never the initial or default focus target —
            // see SetFocus() below, which always lands on the input field
            // instead. "Not selectable for input" really meant "don't let
            // typed characters go here", which ReadOnly already prevents;
            // this just makes sure focus doesn't land here by default either.
            TabStop = false
        };
        _transcriptView.ContextMenu.MenuItems.Children = [.. _transcriptView.ContextMenu.MenuItems.Children.Where(p =>
        {
            var s = p.Title.ToString();
            return (s == "Cu_t" || 
            s == "_Delete All" ||
            s == "Cu_t" ||
            s == "_Paste" ||
            s == "_Undo" ||
            s == "_Redo") == false;
        }
        )];
        var sidebarFrame = new FrameView("Status")
        {
            X = Pos.Right(_transcriptView) + 1,
            Y = 0,
            Width = sidebarWidth,
            Height = Dim.Fill(2)
        };

        _statusLabel = new Label("Ready") { X = 0, Y = 0, Width = Dim.Fill() };
        _agentLabel = new Label(agentName) { X = 0, Y = 2, Width = Dim.Fill() };

        var resolvedHw = options.HardwareTier switch
        {
            HardwareTier.Auto => Fma.IsSupported ? "FMA" : Avx2.IsSupported ? "AVX2" : Sse3.IsSupported ? "SSE" : "Scalar",
            _ => options.HardwareTier.ToString()
        };
        string hwDisplay = options.UseGpu ? $"GPU ({resolvedHw})" : resolvedHw;

        _strategyLabel = new Label($"Generator: {options.Generator}\nKV Cache: {options.Cache}\nHW Tier: {hwDisplay}")
        { X = 0, Y = 4, Width = Dim.Fill(), Height = 4 };
        _toolLabel = new Label("") { X = 0, Y = 8, Width = Dim.Fill() };
        _speedLabel = new Label("--") { X = 0, Y = 10, Width = Dim.Fill() };
        _ttftLabel = new Label("--") { X = 0, Y = 12, Width = Dim.Fill() };
        _memLabel = new Label(FormatMemory()) { X = 0, Y = 14, Width = Dim.Fill(), Height = 3 };

        sidebarFrame.Add(
            new Label("Status:") { X = 0, Y = 0 }, _statusLabel,
            new Label("Agent:") { X = 0, Y = 2 }, _agentLabel,
            new Label("Strategy:") { X = 0, Y = 3 }, _strategyLabel,
            _toolLabel,
            new Label("Speed:") { X = 0, Y = 9 }, _speedLabel,
            new Label("Time to 1st tok:") { X = 0, Y = 11 }, _ttftLabel,
            _memLabel);

        _getButton = new Button("Get")
        {
            X = 0,
            Y = Pos.AnchorEnd(1),
            Width = 6
        };
        _getButton.Clicked += OnGetArtifact;

        _inputField = new TextField("")
        {
            X = Pos.Right(_getButton) + 1,
            Y = Pos.AnchorEnd(1),
            Width = Dim.Fill(6 + 7 + 13 + 3)
        };
        _inputField.KeyPress += OnInputKeyPress;

        _sendButton = new Button("Send")
        {
            X = Pos.AnchorEnd(7 + 1 + 13),
            Y = Pos.AnchorEnd(1),
            Width = 7,
            Enabled = true
        };
        _sendButton.Clicked += OnSendArtifact;

        _interruptButton = new Button("Interrupt")
        {
            X = Pos.AnchorEnd(13),
            Y = Pos.AnchorEnd(1),
            Width = 13,
            Enabled = false
        };
        _interruptButton.Clicked += RequestInterrupt;

        Add(_transcriptView, sidebarFrame, _getButton, _inputField, _sendButton, _interruptButton);

        KeyPress += (args) =>
        {
            // Esc behaviour depends on what's actually happening: mid-generation
            // it interrupts the in-flight turn (matching the button); idle, it
            // exits the chat screen. Either way Esc always does *something*
            // useful here rather than only working when nothing is running.
            if (args.KeyEvent.Key == Key.Esc)
            {
                if (_generating) RequestInterrupt();
                else _onExit();
                args.Handled = true;
            }
        };

        // 60fps-equivalent poll for background-thread updates (stream entries, choice requests) —
        // the same cadence the old manual frame loop used, expressed as a Terminal.Gui timeout instead.
        _timeoutToken = Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(16), PollBackgroundState);

        _inputField.SetFocus();
    }

    private static string FormatMemory()
    {
        // GC.GetGCMemoryInfo mirrors what CuiTools.UIGetFreeMemory already
        // reports to the model — reusing the same source here means the
        // sidebar and a model's own UIGetFreeMemory answer can't disagree.
        var info = GC.GetGCMemoryInfo();
        long totalMb = info.TotalAvailableMemoryBytes / (1024 * 1024);
        long usedMb = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024);
        return $"Mem: {usedMb}MB used\nFree: {totalMb - usedMb}MB";
    }

    private void OnInputKeyPress(KeyEventEventArgs args)
    {
        if (args.KeyEvent.Key != Key.Enter) return;

        if (_generating)
        {
            var dialog = new Dialog("Interrupt generation?", 60, 5);
            var ok = new Button("Yes, interrupt") { X = 1, Y = 2, IsDefault = true };
            ok.Clicked += () => { RequestInterrupt(); Application.RequestStop(); };
            var cancel = new Button("No, keep going") { X = Pos.Right(ok) + 2, Y = 2 };
            cancel.Clicked += () => Application.RequestStop();
            dialog.Add(new Label("The model is still responding. Do you want to stop it now?") { X = 1, Y = 1, Width = Dim.Fill(2) }, ok, cancel);
            
            _interruptDialogOpen = true;
            Application.Run(dialog);
            _interruptDialogOpen = false;
            
            args.Handled = true;
            return;
        }

        string text = _inputField.Text.ToString() ?? "";
        if (text.Length == 0) { args.Handled = true; return; }

        _inputField.Text = "";
        AppendTranscript($"You: {text}");
        SetGenerating(true);
        _statusLabel.Text = "Thinking...";
        _bridge.SubmitUserInput(text);
        args.Handled = true;
    }

    private void SetGenerating(bool generating)
    {
        _generating = generating;
        _interruptButton.Enabled = generating;
        _onGeneratingChanged?.Invoke(generating);
    }

    /// <summary>
    /// No per-turn cancellation exists on IChatSession itself — only the
    /// whole-session CancellationToken ChatSessionBridge already owns. A
    /// true "stop just this response, keep the session" button would need
    /// that capability added to ChatSession upstream; what this can
    /// honestly do today is end the whole bridge (same as closing the
    /// session) and report that plainly, rather than implying a softer
    /// in-place stop that doesn't actually exist yet.
    /// </summary>
    private void OnSendArtifact()
    {
        if (_generating) return;
        var path = FilePickerDialog.Show("Send artifact", Directory.GetCurrentDirectory(), PickerMode.File);
        if (path is null) return;

        byte[] content;
        try
        {
            content = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Error", $"Could not read file:\n{ex.Message}", "OK");
            return;
        }

        var ext = Path.GetExtension(path)?.ToLowerInvariant();
        string type = ext switch
        {
            ".json" => "json",
            ".cs" or ".py" or ".js" or ".ts" or ".rs" or ".cpp" or ".c" or ".h" => "code",
            ".png" or ".jpg" or ".jpeg" or ".gif" or ".bmp" => "image",
            ".md" or ".txt" or ".xml" or ".yaml" or ".yml" => "text",
            _ => "text"
        };

        var artifact = new ChatArtifact
        {
            Type = type,
            Content = content,
            FileName = Path.GetFileName(path),
            Language = type == "code" ? ext?.TrimStart('.') : null
        };

        string text = _inputField.Text.ToString() ?? "";
        _inputField.Text = "";
        AppendTranscript($"You: {text}");
        SetGenerating(true);
        _statusLabel.Text = "Thinking...";
        _bridge.SubmitUserInput(text, [artifact]);
    }

    private void OnGetArtifact()
    {
        var artifacts = _bridge.LastArtifacts;
        if (artifacts is null || artifacts.Length == 0)
        {
            MessageBox.Query("No artifacts", "The last response has no artifacts to retrieve.", "OK");
            return;
        }

        // Copy to a mutable list so the dialog can remove items
        var remaining = artifacts.ToList();
        ShowArtifactListDialog(remaining);
    }

    private void ShowArtifactListDialog(List<ChatArtifact> artifacts)
    {
        var dialog = new Dialog("Artifacts", 60, 20);
        var listView = new ListView
        {
            X = 1, Y = 0,
            Width = Dim.Fill(2),
            Height = Dim.Fill(4)
        };
        dialog.Add(listView);

        void RefreshList()
        {
            listView.SetSource(artifacts.Select(a =>
            {
                string name = a.FileName;
                if (string.IsNullOrEmpty(name))
                    name = string.IsNullOrEmpty(a.Type) ? "artifact" : $"artifact.{a.Type}";
                string size = a.Content is null || a.Content.Length == 0 ? " [empty]" : $" ({a.Content.Length}B)";
                return (ustring)$"{name} ({a.Type}){size}";
            }).ToList());
        }

        var saveBtn = new Button("Save selected") { X = 1, Y = Pos.AnchorEnd(2) };
        saveBtn.Clicked += () =>
        {
            int idx = listView.SelectedItem;
            if (idx < 0 || idx >= artifacts.Count) return;
            var artifact = artifacts[idx];
            if (artifact.Content is null || artifact.Content.Length == 0)
            {
                MessageBox.Query("Empty content", "This artifact has no content to save.", "OK");
                return;
            }
            string fileName = artifact.FileName;
            if (string.IsNullOrEmpty(fileName))
                fileName = string.IsNullOrEmpty(artifact.Type) ? "artifact" : $"artifact.{artifact.Type}";
            string? savePath = FilePickerDialog.Show($"Save: {fileName}", Directory.GetCurrentDirectory(), PickerMode.SaveFile, "*.*");
            if (savePath is null) return;
            if (File.Exists(savePath))
            {
                int overwrite = MessageBox.Query("File exists", $"Overwrite existing file?\n{savePath}", "Yes", "No");
                if (overwrite != 0) return;
            }
            try
            {
                File.WriteAllBytes(savePath, artifact.Content);
            }
            catch (Exception ex)
            {
                MessageBox.ErrorQuery("Error", $"Could not save:\n{ex.Message}", "OK");
            }
        };

        var removeBtn = new Button("Remove") { X = Pos.Right(saveBtn) + 2, Y = Pos.AnchorEnd(2) };
        removeBtn.Clicked += () =>
        {
            int idx = listView.SelectedItem;
            if (idx < 0 || idx >= artifacts.Count) return;
            artifacts.RemoveAt(idx);
            if (artifacts.Count == 0)
                Application.RequestStop();
            else
                RefreshList();
        };

        var clearBtn = new Button("Clear all") { X = Pos.Right(removeBtn) + 2, Y = Pos.AnchorEnd(2) };
        clearBtn.Clicked += () =>
        {
            artifacts.Clear();
            Application.RequestStop();
        };

        var closeBtn = new Button("Close") { X = Pos.AnchorEnd(9), Y = Pos.AnchorEnd(2), Width = 9 };
        closeBtn.Clicked += () => Application.RequestStop();

        dialog.Add(saveBtn, removeBtn, clearBtn, closeBtn);
        RefreshList();
        listView.SetFocus();
        Application.Run(dialog);
    }

    private void RequestInterrupt()
    {
        if (!_generating) return;
        _bridge.Interrupt();
        SetGenerating(false);
    }

    /// <summary>Returning true keeps the timeout recurring — see Application.MainLoop.AddTimeout's contract.</summary>
    private bool PollBackgroundState(MainLoop _)
    {
        if (_disposed) return false;

        if (_interruptDialogOpen && !_generating)
        {
            Application.RequestStop();
        }

        _memLabel.Text = FormatMemory();

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

        if (entry.TokensPerSecond is { } tps) _lastTokensPerSecond = tps;
        if (entry.TimeToFirstToken is { } ttft) _lastTimeToFirstToken = ttft;
        // Sidebar keeps showing the last known values rather than resetting to
        // "--" between turns — per request, these shouldn't clear after a
        // response finishes streaming.
        _speedLabel.Text = _lastTokensPerSecond is { } s ? $"{s:F1} tok/s" : "--";
        _ttftLabel.Text = _lastTimeToFirstToken is { } t ? $"{t:F2}s" : "--";

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
        
        if ((entry.Status == ChatStatus.Responding || (entry.Status == ChatStatus.Thinking && _bridge.ShowThinking)) && entry.Token is not null)
        {
            _liveResponse.Append(entry.Token);
            // The actual streaming fix: render the in-progress response on
            // every token instead of only once the whole turn finishes.
            // Previously _liveResponse only got flushed to the transcript at
            // Complete/Interrupted, so nothing was visible until the entire
            // answer had already finished generating — correct text, just
            // displayed with no streaming at all.
            RenderLiveResponse();
        }

        if (entry.IsComplete || entry.Status is ChatStatus.Complete or ChatStatus.Interrupted)
        {
            if (_liveResponse.Length > 0)
                CommitLiveResponse();
            SetGenerating(false);
            _toolLabel.Text = "";
            _pendingSpeakerName = null;
            _statusLabel.Text = "Ready";
        }

        SetNeedsDisplay();
    }

    // Tracks where the in-progress line starts in the TextView's underlying
    // text, so each new token can replace just that trailing line instead of
    // re-appending the whole growing response (which would duplicate it).
    private int _liveResponseStartOffset = -1;

    private void RenderLiveResponse()
    {
        string full = _transcriptView.Text.ToString() ?? "";
        if (_liveResponseStartOffset < 0)
        {
            // First token of this turn: remember where the live line begins
            // so subsequent tokens overwrite it in place rather than append.
            _liveResponseStartOffset = full.Length;
            full += $"{AgentName}: ";
        }

        string prefix = full[.._liveResponseStartOffset];
        _transcriptView.Text = prefix + $"{AgentName}: " + _liveResponse;
        _transcriptView.MoveEnd();
        _transcriptView.SetNeedsDisplay();
    }

    private void CommitLiveResponse()
    {
        // The live line is already showing the full text by the time Complete
        // arrives (RenderLiveResponse kept it current token-by-token) — just
        // close it off with the trailing blank line the permanent transcript
        // uses between entries, and reset tracking for the next turn.
        _transcriptView.Text = _transcriptView.Text.ToString() + "\n\n";
        _transcriptView.MoveEnd();
        _transcriptView.SetNeedsDisplay();
        _liveResponse.Clear();
        _liveResponseStartOffset = -1;
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
        dialog.Add(new Label((ustring)request.Prompt) { X = 1, Y = 0, Width = Dim.Fill(2) }, radio);

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
