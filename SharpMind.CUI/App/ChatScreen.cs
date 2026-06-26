using SharpMind.CUI.Screen;
using SharpMind.Inference.Chat;

namespace SharpMind.CUI.App;

/// <summary>
/// The chat screen: scrolling transcript on the left, live status sidebar on
/// the right, input line and status bar pinned to the bottom.
///
/// Sub-agent attribution: ChatSession already streams everything needed to
/// know which sub-agent produced a given piece of output — it yields
/// <c>ChatStatus.Executing</c> with the agent's name in <c>Token</c> right
/// before delegating, then <c>ChatStatus.Researching</c> with the sub-agent's
/// own generated text as it streams. No engine change was needed for this;
/// the gap was entirely here, in not reading those two statuses as agent
/// identity. See the bookkeeping fields below for why a naive "attribute
/// everything to whoever's name was last seen, flush on Complete" approach
/// is wrong: ChatSession only emits one Complete per *turn*, and a turn that
/// delegates to a sub-agent loops back into Responding for the top-level
/// agent's own follow-up before that Complete ever fires.
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

    // Sub-agent attribution bookkeeping. ChatSession only yields one
    // ChatStatus.Complete entry per *turn*, at the very end — a turn that
    // delegates to a sub-agent along the way streams Researching(fragments)
    // for the sub-agent, then loops back into Thinking/Responding for the
    // top-level model's own follow-up, all before that single Complete
    // fires. So "wait for Complete to attribute the buffer" is wrong here:
    // it would credit the sub-agent's words and the top-level model's words
    // to whichever name happened to be active first. Instead, the
    // sub-agent's accumulated text is flushed as its own transcript entry
    // the moment the status stops being Researching, before any other
    // entry's text gets appended to what's now a fresh buffer.
    private string? _pendingSpeakerName;     // name seen on the most recent Executing entry
    private string? _activeSubAgentName;     // confirmed (via a Researching entry) sub-agent currently streaming
    private readonly System.Text.StringBuilder _subAgentBuffer = new();

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

    /// <summary>Feed each streamed entry in as it arrives.</summary>
    public void OnStreamEntry(ChatStreamEntry entry)
    {
        // Flush the sub-agent buffer the instant the stream leaves Researching —
        // whatever comes next (a new Executing, a Responding token from the
        // top-level model picking back up, or Complete) must not have its text
        // mixed into a buffer that belongs to the sub-agent's turn.
        if (_activeSubAgentName is not null && entry.Status != ChatStatus.Researching)
        {
            if (_subAgentBuffer.Length > 0)
                _transcript.Add((_activeSubAgentName!, _subAgentBuffer.ToString()));
            _subAgentBuffer.Clear();
            _activeSubAgentName = null;
        }

        _status = entry.Status;
        _tokensPerSecond = entry.TokensPerSecond;

        if (entry.Status == ChatStatus.Executing)
        {
            // ChatSession overloads Token to carry a name at this moment —
            // either a tool name (plain tool call) or a sub-agent name (about
            // to delegate via {{agent:name:query}}). Captured here but not
            // yet trusted as a sub-agent name: a following Researching entry
            // is what actually confirms that, since plain tool calls never
            // emit Researching.
            _activeToolName = entry.Token;
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
                _transcript.Add((agentName, _liveResponse.ToString()));
            _liveResponse.Clear();
            _generating = false;
            _activeToolName = null;
            _pendingSpeakerName = null;
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
        if (_generating && _subAgentBuffer.Length > 0)
        {
            string speaker = _activeSubAgentName ?? "agent";
            lines.Add(($"{speaker}:", theme.DimText));
            foreach (var wrapped in WrapText(_subAgentBuffer.ToString(), w))
                lines.Add((wrapped, fg));
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
        Line(options.UseGpu ? $"GPU ({options.HardwareTier})" : options.HardwareTier.ToString(), fg);
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
