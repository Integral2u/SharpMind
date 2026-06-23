using SharpMind.CUI.Screen;
using SharpMind.Inference.Chat;

namespace SharpMind.CUI.App;

/// <summary>
/// The chat screen: scrolling transcript on the left, live status sidebar on
/// the right (this is what finally gives the existing ChatStatus enum
/// somewhere to be seen — Thinking/Executing/Responding/Waiting were already
/// being emitted by ChatSession, nothing in the engine needed to change),
/// input line and status bar pinned to the bottom.
/// </summary>
public sealed class ChatScreen(string agentName, SessionOptions options)
{
    private readonly List<(string speaker, string text)> _transcript = [];
    private readonly System.Text.StringBuilder _inputLine = new();
    private readonly System.Text.StringBuilder _liveResponse = new();
    private ChatStatus _status = ChatStatus.Waiting;
    private string? _activeToolName;
    private float? _tokensPerSecond;
    private bool _generating;
    private int _scrollOffset; // 0 = pinned to bottom

    public bool ExitRequested { get; private set; }

    public void HandleKey(ConsoleKeyInfo key)
    {
        if (_generating)
        {
            // IChatSession doesn't expose a per-turn cancellation hook — only
            // the whole-session CancellationToken passed into StartChatAsync.
            // Interrupting just the in-flight response without ending the
            // session isn't something this UI layer can add on its own; it
            // would need a cancel-current-turn method on ChatSession itself.
            // So: no input is accepted while generating, full stop, rather
            // than pretending Esc does something it doesn't.
            return;
        }

        switch (key.Key)
        {
            case ConsoleKey.Escape:
                ExitRequested = true;
                return;
            case ConsoleKey.Enter:
                if (_inputLine.Length > 0)
                {
                    PendingSubmission = _inputLine.ToString();
                    _inputLine.Clear();
                }
                return;
            case ConsoleKey.Backspace:
                if (_inputLine.Length > 0) _inputLine.Length--;
                return;
            case ConsoleKey.PageUp:
                _scrollOffset += 5;
                return;
            case ConsoleKey.PageDown:
                _scrollOffset = Math.Max(0, _scrollOffset - 5);
                return;
            default:
                if (!char.IsControl(key.KeyChar))
                    _inputLine.Append(key.KeyChar);
                return;
        }
    }

    /// <summary>Set by HandleKey on Enter, consumed by the app loop to drive the actual chat call.</summary>
    public string? PendingSubmission { get; private set; }
    public void ClearPendingSubmission() => PendingSubmission = null;

    public void BeginGenerating(string userText)
    {
        _transcript.Add(("You", userText));
        _liveResponse.Clear();
        _generating = true;
        _status = ChatStatus.Thinking;
        _scrollOffset = 0;
    }

    private string? _debugSpeakerOverride;

    /// <summary>
    /// Debug-only: lets <see cref="DebugChatBridge"/> attribute a response to
    /// a simulated sub-agent name instead of the session's top-level agent
    /// name. This is a CUI-side convention, not something the real engine
    /// supports — <see cref="SharpMind.Inference.Chat.ChatStreamEntry"/> has
    /// no speaker-identity field at all, so a real model's sub-agent
    /// delegation currently has no way to tag which agent actually produced
    /// a given piece of streamed output. TestAgent exercises what the
    /// transcript display *would* look like if that engine-side gap were
    /// closed; it does not close it. Call with null to clear the override
    /// once a turn finishes.
    /// </summary>
    public void SetDebugSpeakerOverride(string? speakerName) => _debugSpeakerOverride = speakerName;

    /// <summary>Feed each streamed entry in as it arrives.</summary>
    public void OnStreamEntry(ChatStreamEntry entry)
    {
        _status = entry.Status;
        _tokensPerSecond = entry.TokensPerSecond;
        _activeToolName = entry.Status == ChatStatus.Executing ? entry.Token : _activeToolName;

        if (entry.Status == ChatStatus.Responding && entry.Token is not null)
            _liveResponse.Append(entry.Token);

        if (entry.IsComplete || entry.Status is ChatStatus.Complete or ChatStatus.Interrupted)
        {
            if (_liveResponse.Length > 0)
                _transcript.Add((_debugSpeakerOverride ?? agentName, _liveResponse.ToString()));
            _liveResponse.Clear();
            _generating = false;
            _activeToolName = null;
            _status = ChatStatus.Waiting;
        }
    }

    public void Draw(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor bg = theme.Background;
        ConsoleColor fg = theme.Text;
        ConsoleColor accent = theme.Accent;
        ConsoleColor dim = theme.DimText;

        int sidebarW = 24;
        int mainW = w - sidebarW - 1;
        int inputRow = h - 3;
        int transcriptH = inputRow - 1;

        buf.FillRect(x, y, w, h, ' ', fg, bg);

        // --- main transcript pane -------------------------------------------------
        buf.DrawBox(x, y, mainW, transcriptH, accent, bg);
        DrawTranscript(buf, x + 1, y + 1, mainW - 2, transcriptH - 2, theme);

        // --- status sidebar ---------------------------------------------------
        int sbX = x + mainW + 1;
        buf.DrawBox(sbX, y, sidebarW, transcriptH, accent, bg, doubleLine: false);
        DrawSidebar(buf, sbX + 1, y + 1, sidebarW - 2, transcriptH - 2, theme);

        // --- input line ---------------------------------------------------
        buf.DrawBox(x, y + transcriptH, w, 3, accent, bg);
        string prompt = _generating ? "..." : ">";
        buf.Write(x + 2, y + transcriptH + 1, prompt, accent, bg);
        string visibleInput = _inputLine.ToString();
        int maxInputW = w - 6;
        if (visibleInput.Length > maxInputW) visibleInput = visibleInput[^maxInputW..];
        buf.Write(x + 4, y + transcriptH + 1, visibleInput, fg, bg);
        if (!_generating)
            buf.Set(x + 4 + visibleInput.Length, y + transcriptH + 1, '_', accent, bg);

        // --- bottom status bar ---------------------------------------------------
        DrawStatusBar(buf, x, y + h - 1, w, theme);
    }

    private void DrawTranscript(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor fg = theme.Text, bg = theme.Background;

        // Wrap every (speaker, text) pair into display lines, oldest first.
        var lines = new List<(string text, ConsoleColor color)>();
        foreach (var (speaker, text) in _transcript)
        {
            var speakerColor = speaker == "You" ? theme.Text : theme.DimText;
            lines.Add(($"{speaker}:", speakerColor));
            foreach (var wrapped in WrapText(text, w))
                lines.Add((wrapped, fg));
            lines.Add(("", fg)); // blank separator
        }
        if (_generating && _liveResponse.Length > 0)
        {
            lines.Add(($"{agentName}:", theme.DimText));
            foreach (var wrapped in WrapText(_liveResponse.ToString(), w))
                lines.Add((wrapped, fg));
        }

        int total = lines.Count;
        int firstVisible = Math.Max(0, total - h - _scrollOffset);
        for (int row = 0; row < h; row++)
        {
            int idx = firstVisible + row;
            buf.FillRect(x, y + row, w, 1, ' ', fg, bg);
            if (idx >= 0 && idx < total)
                buf.Write(x, y + row, lines[idx].text, lines[idx].color, bg);
        }
    }

    private static List<string> WrapText(string text, int width)
    {
        var result = new List<string>();
        foreach (var paragraph in text.Split('\n'))
        {
            if (paragraph.Length == 0) { result.Add(""); continue; }
            int i = 0;
            while (i < paragraph.Length)
            {
                int len = Math.Min(width, paragraph.Length - i);
                result.Add(paragraph.Substring(i, len));
                i += len;
            }
        }
        return result;
    }

    private void DrawSidebar(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor fg = theme.Text, bg = theme.Background, dim = theme.DimText;
        int row = 0;
        void Line(string text, ConsoleColor color)
        {
            if (row < h) buf.Write(x, y + row, text.Length > w ? text[..w] : text, color, bg);
            row++;
        }

        Line("STATUS", dim);
        Line(StatusLabel(_status), StatusColor(_status, theme));
        row++;
        Line("AGENT", dim);
        Line(agentName.Length > w ? agentName[..w] : agentName, fg);
        row++;
        Line("STRATEGY", dim);
        Line(options.Generator.ToString(), fg);
        Line(options.Cache.ToString(), fg);
        row++;
        if (_activeToolName is not null)
        {
            Line("TOOL", dim);
            Line(_activeToolName, theme.Accent);
            row++;
        }
        Line("SPEED", dim);
        Line(_tokensPerSecond is { } tps ? $"{tps:F1} tok/s" : "--", fg);
    }

    private void DrawStatusBar(ScreenBuffer buf, int x, int y, int w, Theme theme)
    {
        ConsoleColor barFg = theme.BarFg;
        ConsoleColor barBg = theme.BarBg;
        buf.FillRect(x, y, w, 1, ' ', barFg, barBg);
        string hint = _generating
            ? "Generating... (no per-turn cancel yet; wait for completion)"
            : "Esc=Exit  Enter=Send  PgUp/PgDn=Scroll";
        buf.Write(x + 1, y, hint, barFg, barBg);
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

    private static ConsoleColor StatusColor(ChatStatus s, Theme theme) => s switch
    {
        ChatStatus.Waiting => theme.Success,
        ChatStatus.Interrupted => theme.Error,
        ChatStatus.Executing => theme.Accent,
        _ => theme.Text
    };
}
