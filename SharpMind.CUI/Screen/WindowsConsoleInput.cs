using System.Runtime.InteropServices;

namespace SharpMind.CUI.Screen;

/// <summary>
/// Reads keyboard and mouse input directly from the Windows console input
/// buffer via <c>ReadConsoleInput</c>, instead of going through
/// <see cref="Console.ReadKey"/> at all.
///
/// Why this exists as a separate path: classic conhost (the default console
/// host on Windows — what you get from a plain command prompt, and what
/// Visual Studio's "Console Host" debug option launches into) reports mouse
/// activity exclusively as native <c>MOUSE_EVENT_RECORD</c> structures
/// through the Win32 console API. It does not emit ANSI/VT mouse-tracking
/// escape sequences into the input stream under any virtual-terminal mode —
/// that VT behaviour is specific to Windows Terminal and xterm-family
/// terminals, which is what <see cref="InputQueue"/>'s SGR-sequence parsing
/// is built for. There is no escape sequence for this reader to fail to
/// receive; conhost simply never sends one. The two console hosts use
/// fundamentally different transports for mouse data, not just different
/// formats of the same transport.
///
/// Both keyboard and mouse records have to be read from this single Windows
/// API call, not split across this and <see cref="Console.ReadKey"/> — they
/// both ultimately read from the same underlying console input buffer, and
/// running both concurrently on separate threads would race for the same
/// records, occasionally handing a real keystroke to whichever reader
/// happened to win the race and silently dropping it from the other. So on
/// Windows, this class owns *all* input, keys included, via a direct
/// key-record decode mirroring what <see cref="Console.ReadKey"/> would have
/// produced for the common cases this app actually binds to.
/// </summary>
internal static class WindowsConsoleInput
{
    public static bool IsSupported => OperatingSystem.IsWindows();

    private const int STD_INPUT_HANDLE = -10;
    private const uint ENABLE_MOUSE_INPUT = 0x0010;
    private const uint ENABLE_EXTENDED_FLAGS = 0x0080;
    private const uint ENABLE_QUICK_EDIT_MODE = 0x0040; // must be OFF, or conhost intercepts drag-clicks for text selection instead of passing them through
    private const uint ENABLE_PROCESSED_INPUT = 0x0001;

    private const ushort KEY_EVENT = 0x0001;
    private const ushort MOUSE_EVENT = 0x0002;

    private const uint MOUSE_MOVED = 0x0001;
    private const uint DOUBLE_CLICK = 0x0002;
    private const uint MOUSE_WHEELED = 0x0004;

    private const int FROM_LEFT_1ST_BUTTON_PRESSED = 0x0001;
    private const int RIGHTMOST_BUTTON_PRESSED = 0x0002;
    private const int FROM_LEFT_2ND_BUTTON_PRESSED = 0x0004; // middle button on a standard 3-button mouse

    private const int SHIFT_PRESSED = 0x0010;
    private const int LEFT_ALT_PRESSED = 0x0002;
    private const int RIGHT_ALT_PRESSED = 0x0001;
    private const int LEFT_CTRL_PRESSED = 0x0008;
    private const int RIGHT_CTRL_PRESSED = 0x0004;

    [StructLayout(LayoutKind.Sequential)]
    private struct COORD { public short X; public short Y; }

    [StructLayout(LayoutKind.Sequential)]
    private struct KEY_EVENT_RECORD
    {
        public bool bKeyDown;
        public ushort wRepeatCount;
        public ushort wVirtualKeyCode;
        public ushort wVirtualScanCode;
        public char UnicodeChar;
        public uint dwControlKeyState;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MOUSE_EVENT_RECORD
    {
        public COORD dwMousePosition;
        public uint dwButtonState;
        public uint dwControlKeyState;
        public uint dwEventFlags;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct INPUT_RECORD_UNION
    {
        [FieldOffset(0)] public KEY_EVENT_RECORD KeyEvent;
        [FieldOffset(0)] public MOUSE_EVENT_RECORD MouseEvent;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct INPUT_RECORD
    {
        public ushort EventType;
        public INPUT_RECORD_UNION Event;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetStdHandle(int nStdHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool ReadConsoleInput(IntPtr hConsoleInput, [Out] INPUT_RECORD[] lpBuffer, uint nLength, out uint lpNumberOfEventsRead);

    private static IntPtr _handle = IntPtr.Zero;
    private static uint _originalMode;

    /// <summary>
    /// Switches the console into mouse-reporting mode. Must be balanced with
    /// <see cref="Restore"/> on shutdown — this changes real console state
    /// that outlives the process if left in a modified mode (e.g. quick-edit
    /// staying disabled for whatever runs in that window next).
    /// </summary>
    public static void Enable()
    {
        _handle = GetStdHandle(STD_INPUT_HANDLE);
        if (_handle == IntPtr.Zero || !GetConsoleMode(_handle, out _originalMode)) return;

        uint newMode = (_originalMode | ENABLE_MOUSE_INPUT | ENABLE_EXTENDED_FLAGS)
            & ~ENABLE_QUICK_EDIT_MODE
            & ~ENABLE_PROCESSED_INPUT; // Ctrl+C should arrive as a normal key event, not terminate the process mid-render
        SetConsoleMode(_handle, newMode);
    }

    public static void Restore()
    {
        if (_handle != IntPtr.Zero) SetConsoleMode(_handle, _originalMode);
    }

    /// <summary>
    /// Blocks until at least one input record is available, then returns
    /// everything currently queued, decoded into this app's own
    /// <see cref="ConsoleKeyInfo"/>/<see cref="MouseEvent"/> shapes. Mirrors
    /// <see cref="Console.ReadKey"/>'s blocking behaviour so the calling loop
    /// doesn't need two different waiting strategies depending on platform.
    /// </summary>
    public static (List<ConsoleKeyInfo> Keys, List<MouseEvent> MouseEvents) ReadBatch()
    {
        var keys = new List<ConsoleKeyInfo>();
        var mouseEvents = new List<MouseEvent>();

        if (_handle == IntPtr.Zero) return (keys, mouseEvents);

        var buffer = new INPUT_RECORD[16];
        if (!ReadConsoleInput(_handle, buffer, (uint)buffer.Length, out uint count) || count == 0)
            return (keys, mouseEvents);

        for (int i = 0; i < count; i++)
        {
            ref readonly INPUT_RECORD record = ref buffer[i];
            if (record.EventType == KEY_EVENT)
            {
                var k = record.Event.KeyEvent;
                if (!k.bKeyDown) continue; // mirror ReadKey: one event per press, not per release

                var modifiers = (ConsoleModifiers)0;
                if ((k.dwControlKeyState & SHIFT_PRESSED) != 0) modifiers |= ConsoleModifiers.Shift;
                if ((k.dwControlKeyState & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED)) != 0) modifiers |= ConsoleModifiers.Alt;
                if ((k.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0) modifiers |= ConsoleModifiers.Control;

                // ConsoleKey's enum values are deliberately aligned with Win32
                // VK_ virtual-key codes for exactly this kind of interop, so a
                // direct cast is the documented-correct mapping here, not a
                // coincidence being relied on.
                var consoleKey = Enum.IsDefined(typeof(ConsoleKey), (int)k.wVirtualKeyCode)
                    ? (ConsoleKey)k.wVirtualKeyCode
                    : ConsoleKey.NoName;

                keys.Add(new ConsoleKeyInfo(k.UnicodeChar, consoleKey,
                    modifiers.HasFlag(ConsoleModifiers.Shift),
                    modifiers.HasFlag(ConsoleModifiers.Alt),
                    modifiers.HasFlag(ConsoleModifiers.Control)));
            }
            else if (record.EventType == MOUSE_EVENT)
            {
                var m = record.Event.MouseEvent;

                MouseAction action;
                if ((m.dwEventFlags & MOUSE_WHEELED) != 0)
                {
                    // High word of dwButtonState is a signed wheel-delta on this event; sign determines direction.
                    bool up = unchecked((int)m.dwButtonState) > 0;
                    action = up ? MouseAction.ScrollUp : MouseAction.ScrollDown;
                }
                else if ((m.dwEventFlags & MOUSE_MOVED) != 0)
                {
                    action = MouseAction.Move;
                }
                else
                {
                    // Plain button transition. conhost doesn't separately flag press vs
                    // release the way SGR does — a zero button state here means "buttons
                    // that were down are now up", anything nonzero means "now pressed".
                    action = m.dwButtonState != 0 ? MouseAction.Down : MouseAction.Up;
                }

                MouseButton button = MouseButton.None;
                if ((m.dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED) != 0) button = MouseButton.Left;
                else if ((m.dwButtonState & RIGHTMOST_BUTTON_PRESSED) != 0) button = MouseButton.Right;
                else if ((m.dwButtonState & FROM_LEFT_2ND_BUTTON_PRESSED) != 0) button = MouseButton.Middle;

                mouseEvents.Add(new MouseEvent(m.dwMousePosition.X, m.dwMousePosition.Y, button, action));
            }
            // Other record types (window resize, focus, menu) are intentionally ignored —
            // resize is already handled by App's own Console.WindowWidth/Height polling.
        }

        return (keys, mouseEvents);
    }
}
