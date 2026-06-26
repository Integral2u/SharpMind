using System.Text;

namespace SharpMind.CUI.Screen;

/// <summary>
/// Reads keys (and, where possible, mouse events) on a background thread.
///
/// Two genuinely different code paths live here, picked once at
/// <see cref="Start"/> based on the actual OS:
///
/// <b>Windows</b> uses <see cref="WindowsConsoleInput"/>, a direct
/// <c>ReadConsoleInput</c> P/Invoke. This exists because classic conhost —
/// the default console host on Windows, what a plain command prompt and
/// Visual Studio's "Console Host" debug option both launch into — reports
/// mouse activity exclusively as native <c>MOUSE_EVENT_RECORD</c>s through
/// the Win32 console API, never as ANSI/VT escape sequences in the input
/// stream. No amount of escape-sequence parsing on this end can see
/// something conhost never sends; the two console hosts use fundamentally
/// different transports for mouse data.
///
/// <b>Everywhere else</b> (Linux, macOS, and Windows Terminal specifically,
/// which — unlike conhost — does speak VT mouse sequences) keeps using
/// <see cref="Console.ReadKey"/> as the actual read primitive, since
/// <c>ReadKey(intercept: true)</c> is what puts the terminal into raw/no-echo
/// mode under the hood on those platforms, and <c>System.Console</c> exposes
/// no public way to do that ourselves. The problem this branch works around:
/// when a terminal reports a mouse click via an SGR escape sequence
/// (<c>ESC [ &lt; Cb ; Cx ; Cy M</c>), each byte still arrives as its own
/// individual <see cref="ConsoleKeyInfo"/> from <c>ReadKey</c> — there's no
/// single call that returns "a mouse event". So whenever a read comes back
/// as Escape, this pump keeps reading (with a short timeout) to see whether
/// more keys immediately follow that match the mouse-sequence shape.
/// </summary>
public sealed class InputQueue
{
    private readonly object _gate = new();
    private readonly Queue<ConsoleKeyInfo> _keyQueue = new();
    private readonly Queue<MouseEvent> _mouseQueue = new();
    private volatile bool _running;
    private Thread? _thread;
    private bool _useWindowsNativeInput;

    /// <summary>
    /// How long to wait, after seeing Escape, for the rest of a multi-key
    /// escape sequence to arrive. Real escape sequences arrive in a single
    /// burst (the terminal writes all the bytes at once), so this only needs
    /// to be long enough to not split that burst under load — not anywhere
    /// near long enough for a person to notice as input lag on a genuine
    /// standalone Escape press. Only relevant on the non-Windows path.
    /// </summary>
    private static readonly TimeSpan SequenceGapTimeout = TimeSpan.FromMilliseconds(15);

    /// <summary>True once a mouse sequence has actually been seen, i.e. the terminal does support it.</summary>
    public bool MouseSupportDetected { get; private set; }

    public void Start()
    {
        _useWindowsNativeInput = WindowsConsoleInput.IsSupported;

        if (_useWindowsNativeInput)
            WindowsConsoleInput.Enable();
        else
            EnableMouseReporting();

        _running = true;
        _thread = new Thread(Pump) { IsBackground = true, Name = "SharpMind.CUI.Input" };
        _thread.Start();
    }

    public void Stop()
    {
        _running = false;

        if (_useWindowsNativeInput)
            WindowsConsoleInput.Restore();
        else
            DisableMouseReporting();
    }

    /// <summary>
    /// SGR extended mouse mode (1006) reports coordinates beyond 223 cells
    /// correctly, unlike the older X10 mode. Button-event tracking (1000) is
    /// included so motion is only reported while a button is held, keeping
    /// the stream quiet when the mouse is just sitting still. Terminals that
    /// don't understand these sequences simply ignore them — there's no
    /// error, no visible effect, and the app falls back to keyboard-only
    /// navigation automatically since no mouse events will ever arrive. Only
    /// used on the non-Windows path; see <see cref="WindowsConsoleInput"/>
    /// for how mouse reporting is enabled on Windows instead.
    /// </summary>
    private static void EnableMouseReporting()
    {
        try { Console.Out.Write("\x1b[?1000h\x1b[?1006h"); } catch { /* not every host accepts writes here */ }
    }

    private static void DisableMouseReporting()
    {
        try { Console.Out.Write("\x1b[?1000l\x1b[?1006l"); } catch { }
    }

    private void Pump()
    {
        if (_useWindowsNativeInput)
        {
            PumpWindowsNative();
            return;
        }

        while (_running)
        {
            ConsoleKeyInfo key;
            try { key = Console.ReadKey(intercept: true); }
            catch { continue; }

            if (key.Key == ConsoleKey.Escape)
            {
                TryConsumeEscapeSequence();
                continue;
            }

            lock (_gate) _keyQueue.Enqueue(key);
        }
    }

    /// <summary>
    /// Windows-only read loop. Replaces <see cref="Console.ReadKey"/>
    /// entirely on this path — see the class doc comment for why both
    /// readers can't safely run concurrently against the same console input
    /// buffer. <see cref="WindowsConsoleInput.ReadBatch"/> decodes both keys
    /// and mouse events from the same underlying <c>ReadConsoleInput</c>
    /// call, so nothing is lost by not using <c>ReadKey</c> here.
    /// </summary>
    private void PumpWindowsNative()
    {
        // Mouse reporting is a real, always-on capability on this path —
        // unlike the VT path, which only learns a terminal supports mouse
        // sequences by actually observing one arrive.
        MouseSupportDetected = true;

        while (_running)
        {
            var (keys, mouseEvents) = WindowsConsoleInput.ReadBatch();

            if (keys.Count > 0 || mouseEvents.Count > 0)
            {
                lock (_gate)
                {
                    foreach (var k in keys) _keyQueue.Enqueue(k);
                    foreach (var m in mouseEvents) _mouseQueue.Enqueue(m);
                }
            }
        }
    }

    /// <summary>
    /// Already consumed the leading Escape. Tries to read the next couple of
    /// keys within <see cref="SequenceGapTimeout"/> to see whether this is a
    /// real lone Escape, an arrow/Home/End sequence, or a full SGR mouse
    /// report. Whatever it turns out to be gets queued as the right kind of
    /// event; nothing is ever silently dropped.
    /// </summary>
    private void TryConsumeEscapeSequence()
    {
        if (!TryReadKeyWithTimeout(out var k1))
        {
            lock (_gate) _keyQueue.Enqueue(new ConsoleKeyInfo((char)0x1b, ConsoleKey.Escape, false, false, false));
            return;
        }

        if (k1.KeyChar != '[')
        {
            // On Windows, Alt+letter already arrives as a single ReadKey call
            // with ConsoleModifiers.Alt set, so this path is never reached for
            // it there. On xterm-family Unix terminals, Alt+letter has no
            // modifier flag at all — it arrives as plain ESC followed by the
            // bare letter, indistinguishable at the byte level from someone
            // pressing Escape and then separately pressing that letter a
            // moment later. The SequenceGapTimeout is what makes the
            // distinction: if the letter arrives within the same burst as the
            // Escape, it's overwhelmingly likely to be Alt+letter, not two
            // separate keystrokes — a human can't physically type two keys
            // inside a 15ms window.
            if (char.IsLetter(k1.KeyChar))
            {
                lock (_gate) _keyQueue.Enqueue(new ConsoleKeyInfo(k1.KeyChar, k1.Key, false, true, false));
                return;
            }

            // Not a recognised sequence shape — surface plain Escape, then replay k1 untouched.
            lock (_gate)
            {
                _keyQueue.Enqueue(new ConsoleKeyInfo((char)0x1b, ConsoleKey.Escape, false, false, false));
                _keyQueue.Enqueue(k1);
            }
            return;
        }

        if (!TryReadKeyWithTimeout(out var k2))
        {
            // ESC [ with nothing else — not a sequence this app knows; surface both raw.
            lock (_gate)
            {
                _keyQueue.Enqueue(new ConsoleKeyInfo((char)0x1b, ConsoleKey.Escape, false, false, false));
                _keyQueue.Enqueue(k1);
            }
            return;
        }

        if (k2.KeyChar == '<')
        {
            TryConsumeMouseSequence();
            return;
        }

        // ESC [ <letter> is the classic arrow/Home/End shape.
        ConsoleKey? mapped = k2.KeyChar switch
        {
            'A' => ConsoleKey.UpArrow,
            'B' => ConsoleKey.DownArrow,
            'C' => ConsoleKey.RightArrow,
            'D' => ConsoleKey.LeftArrow,
            'H' => ConsoleKey.Home,
            'F' => ConsoleKey.End,
            _ => null
        };

        if (mapped is { } navKey)
        {
            lock (_gate) _keyQueue.Enqueue(new ConsoleKeyInfo('\0', navKey, false, false, false));
        }
        else
        {
            lock (_gate)
            {
                _keyQueue.Enqueue(new ConsoleKeyInfo((char)0x1b, ConsoleKey.Escape, false, false, false));
                _keyQueue.Enqueue(k1);
                _keyQueue.Enqueue(k2);
            }
        }
    }

    /// <summary>Already consumed "ESC [ &lt;". Reads the rest of the SGR body up to the terminating M/m.</summary>
    private void TryConsumeMouseSequence()
    {
        var sb = new StringBuilder();
        char terminator = '\0';

        while (TryReadKeyWithTimeout(out var k))
        {
            if (k.KeyChar == 'M' || k.KeyChar == 'm') { terminator = k.KeyChar; break; }
            sb.Append(k.KeyChar);
            if (sb.Length > 32) return; // malformed/unexpected — bail rather than loop forever
        }

        if (terminator == '\0') return; // timed out before a terminator arrived; drop the partial sequence

        var parts = sb.ToString().Split(';');
        if (parts.Length != 3 ||
            !int.TryParse(parts[0], out int cb) ||
            !int.TryParse(parts[1], out int cx) ||
            !int.TryParse(parts[2], out int cy))
            return;

        bool isRelease = terminator == 'm';

        MouseButton button;
        MouseAction action;
        if ((cb & 64) != 0)
        {
            // Bit 6 marks a scroll-wheel event; the button-id bits select direction instead of a button.
            button = MouseButton.None;
            action = (cb & 1) != 0 ? MouseAction.ScrollDown : MouseAction.ScrollUp;
        }
        else
        {
            button = (cb & 3) switch { 0 => MouseButton.Left, 1 => MouseButton.Middle, 2 => MouseButton.Right, _ => MouseButton.None };
            bool isDrag = (cb & 32) != 0; // bit 5 marks motion-while-pressed
            action = isDrag ? MouseAction.Move : (isRelease ? MouseAction.Up : MouseAction.Down);
        }

        // SGR coordinates are 1-based; ScreenBuffer coordinates are 0-based.
        var evt = new MouseEvent(cx - 1, cy - 1, button, action);
        lock (_gate) _mouseQueue.Enqueue(evt);
        MouseSupportDetected = true;
    }

    /// <summary>
    /// Blocking-equivalent ReadKey, bounded by <see cref="SequenceGapTimeout"/>
    /// so an isolated Escape never hangs the input thread. Implemented as a
    /// tight <see cref="Console.KeyAvailable"/> poll rather than
    /// <c>Task.Run(...).Wait(timeout)</c> deliberately — this method runs on
    /// every arrow key, Home, End, and Escape press in the whole app, and
    /// spinning up a thread-pool task for each one adds real scheduling
    /// overhead on top of the timeout itself. Polling on the thread we're
    /// already on has none of that cost; the 1ms sleep keeps the poll from
    /// busy-spinning the CPU while still resolving well within human
    /// perception once a key actually arrives.
    /// </summary>
    private static bool TryReadKeyWithTimeout(out ConsoleKeyInfo key)
    {
        var deadline = DateTime.UtcNow + SequenceGapTimeout;
        while (DateTime.UtcNow < deadline)
        {
            if (Console.KeyAvailable)
            {
                key = Console.ReadKey(intercept: true);
                return true;
            }
            Thread.Sleep(1);
        }
        key = default;
        return false;
    }

    /// <summary>Drains all keys received since the last call. Cheap, non-blocking.</summary>
    public List<ConsoleKeyInfo> DrainPending()
    {
        lock (_gate)
        {
            if (_keyQueue.Count == 0) return [];
            var result = new List<ConsoleKeyInfo>(_keyQueue.Count);
            while (_keyQueue.Count > 0) result.Add(_keyQueue.Dequeue());
            return result;
        }
    }

    /// <summary>Drains all mouse events received since the last call.</summary>
    public List<MouseEvent> DrainMousePending()
    {
        lock (_gate)
        {
            if (_mouseQueue.Count == 0) return [];
            var result = new List<MouseEvent>(_mouseQueue.Count);
            while (_mouseQueue.Count > 0) result.Add(_mouseQueue.Dequeue());
            return result;
        }
    }
}
