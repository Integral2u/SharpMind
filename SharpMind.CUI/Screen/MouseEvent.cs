namespace SharpMind.CUI.Screen;

public enum MouseButton { Left, Middle, Right, None }
public enum MouseAction { Down, Up, Move, ScrollUp, ScrollDown }

/// <summary>
/// A single mouse event, already translated into screen-cell coordinates
/// (not pixels — there's no such thing in a terminal). Produced by parsing
/// SGR mouse-tracking escape sequences off the raw input stream; see
/// <see cref="InputQueue"/> for where that parsing happens.
/// </summary>
public readonly record struct MouseEvent(int X, int Y, MouseButton Button, MouseAction Action);
