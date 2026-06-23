namespace SharpMind.CUI.Screen;

/// <summary>
/// A single character cell: glyph plus foreground/background colour.
/// </summary>
public readonly struct Cell(char ch, ConsoleColor fg, ConsoleColor bg)
{
    public readonly char Ch = ch;
    public readonly ConsoleColor Fg = fg;
    public readonly ConsoleColor Bg = bg;

    public static readonly Cell Blank = new(' ', ConsoleColor.Gray, ConsoleColor.Blue);

    public bool Equals(Cell other) => Ch == other.Ch && Fg == other.Fg && Bg == other.Bg;
}

/// <summary>
/// A full-screen grid of <see cref="Cell"/>s. Drawing methods just write into
/// the grid; nothing touches the real console until <see cref="Present"/> is
/// called, which diffs against the previously presented frame and only emits
/// the cells that actually changed. This is the entire trick that makes a
/// flicker-free full-screen text UI possible with nothing but
/// <see cref="System.Console"/> — no terminal library required.
///
/// Performance notes, since this is the part of the app actually on the hot
/// path 60 times a second: the diff itself was never the bottleneck — it's
/// O(cells) array reads, which is fast. The real costs are <em>console API
/// calls</em>, not the in-memory bookkeeping around them:
/// <see cref="Console.ForegroundColor"/>/<see cref="Console.BackgroundColor"/>
/// setters and <see cref="Console.SetCursorPosition"/> each cost a real
/// terminal round-trip on most platforms — they are not free property
/// assignments. So the actual optimisation target is minimising how many
/// *times* those are called, not how many *cells* get compared.
/// </summary>
public sealed class ScreenBuffer
{
    private Cell[] _front;   // what's currently on screen
    private Cell[] _back;    // what we're about to draw
    private bool _firstFrame = true;
    private bool _cursorHidden;

    // Reused across frames instead of allocated fresh per run — Present()
    // runs 60 times a second, and the old code built a new StringBuilder for
    // every single changed run on every single frame.
    private readonly System.Text.StringBuilder _runBuilder = new(256);

    // Tracks the console's currently-set colours so Present() only issues a
    // ForegroundColor/BackgroundColor assignment when the colour actually
    // needs to change between runs, instead of once per run unconditionally.
    private ConsoleColor? _consoleFg;
    private ConsoleColor? _consoleBg;

    // Indices that changed this frame. Present() needs _back to start every
    // frame as an exact copy of _front — anything else risks a future screen
    // that doesn't FillRect its entire area first showing stale cells from
    // two frames ago. A full Array.Copy would be correct but pointless (most
    // of the screen is usually unchanged); copying back only the cells this
    // frame actually touched gets the same correctness for the cost of the
    // change-set instead of the whole buffer.
    private readonly List<int> _dirtyIndices = new(256);

    public int Width { get; private set; }
    public int Height { get; private set; }

    public ScreenBuffer(int width, int height)
    {
        Width = width;
        Height = height;
        _front = new Cell[width * height];
        _back = new Cell[width * height];
        Array.Fill(_front, Cell.Blank);
        Array.Fill(_back, Cell.Blank);
    }

    /// <summary>Resizes the buffers, discarding diff history (forces a full repaint next Present).</summary>
    public void Resize(int width, int height)
    {
        Width = width;
        Height = height;
        _front = new Cell[width * height];
        _back = new Cell[width * height];
        Array.Fill(_front, Cell.Blank);
        Array.Fill(_back, Cell.Blank);
        _firstFrame = true;
    }

    public void Clear(ConsoleColor bg = ConsoleColor.Blue) =>
        Array.Fill(_back, new Cell(' ', ConsoleColor.Gray, bg));

    public void Set(int x, int y, char ch, ConsoleColor fg, ConsoleColor bg)
    {
        if (x < 0 || y < 0 || x >= Width || y >= Height) return;
        _back[y * Width + x] = new Cell(ch, fg, bg);
    }

    /// <summary>Writes text left-to-right starting at (x, y), clipped to buffer bounds.</summary>
    public void Write(int x, int y, string text, ConsoleColor fg, ConsoleColor bg)
    {
        for (int i = 0; i < text.Length; i++)
            Set(x + i, y, text[i], fg, bg);
    }

    /// <summary>Writes text centred within a field of the given width starting at x.</summary>
    public void WriteCentered(int x, int y, int fieldWidth, string text, ConsoleColor fg, ConsoleColor bg)
    {
        int pad = Math.Max(0, (fieldWidth - text.Length) / 2);
        Write(x + pad, y, text, fg, bg);
    }

    public void FillRect(int x, int y, int w, int h, char ch, ConsoleColor fg, ConsoleColor bg)
    {
        for (int row = 0; row < h; row++)
            for (int col = 0; col < w; col++)
                Set(x + col, y + row, ch, fg, bg);
    }

    /// <summary>Draws a box using IBM box-drawing characters. Double-line for emphasis, single for nested panels.</summary>
    public void DrawBox(int x, int y, int w, int h, ConsoleColor fg, ConsoleColor bg, bool doubleLine = true)
    {
        char tl = doubleLine ? '╔' : '┌';
        char tr = doubleLine ? '╗' : '┐';
        char bl = doubleLine ? '╚' : '└';
        char br = doubleLine ? '╝' : '┘';
        char hz = doubleLine ? '═' : '─';
        char vt = doubleLine ? '║' : '│';

        Set(x, y, tl, fg, bg);
        Set(x + w - 1, y, tr, fg, bg);
        Set(x, y + h - 1, bl, fg, bg);
        Set(x + w - 1, y + h - 1, br, fg, bg);

        for (int i = 1; i < w - 1; i++)
        {
            Set(x + i, y, hz, fg, bg);
            Set(x + i, y + h - 1, hz, fg, bg);
        }
        for (int i = 1; i < h - 1; i++)
        {
            Set(x, y + i, vt, fg, bg);
            Set(x + w - 1, y + i, vt, fg, bg);
        }
    }

    /// <summary>
    /// Diffs against the last presented frame and writes only the cells that
    /// changed. See the class doc comment for why minimising console API
    /// calls — not minimising cell comparisons — is what actually matters
    /// here.
    /// </summary>
    public void Present()
    {
        // Only ever issued once, not every frame: the cursor-visibility state
        // doesn't change frame to frame, so there's nothing to re-assert here
        // after the first call.
        if (!_cursorHidden)
        {
            try { Console.CursorVisible = false; } catch { /* not every terminal host supports this */ }
            _cursorHidden = true;
        }

        _dirtyIndices.Clear();

        for (int y = 0; y < Height; y++)
        {
            int rowBase = y * Width;
            int x = 0;
            while (x < Width)
            {
                ref readonly Cell cell = ref _back[rowBase + x];
                int idx = rowBase + x;
                if (!_firstFrame && cell.Equals(_front[idx]))
                {
                    x++;
                    continue;
                }

                // Start of a changed run — extend while fg/bg stay constant.
                int runStart = x;
                ConsoleColor fg = cell.Fg, bg = cell.Bg;
                _runBuilder.Clear();
                while (x < Width)
                {
                    ref readonly Cell c = ref _back[rowBase + x];
                    bool changed = _firstFrame || !c.Equals(_front[rowBase + x]);
                    if (!changed || c.Fg != fg || c.Bg != bg) break;
                    _runBuilder.Append(c.Ch);
                    _dirtyIndices.Add(rowBase + x);
                    x++;
                }

                Console.SetCursorPosition(runStart, y);

                // Only touch the console's colour state when it's actually
                // different from what's already set — these setters are real
                // terminal writes, not free property assignments, and a typical
                // frame has many short same-colour runs back to back (label,
                // value, border, label, value...) that would otherwise reissue
                // the same colour over and over for no visible benefit.
                if (_consoleFg != fg) { Console.ForegroundColor = fg; _consoleFg = fg; }
                if (_consoleBg != bg) { Console.BackgroundColor = bg; _consoleBg = bg; }

                Console.Write(_runBuilder);
            }
        }

        // Bring _front up to date for exactly the cells that changed, then
        // mirror those same cells into _back so the next frame's drawing
        // calls start from an accurate baseline — this is the part that
        // keeps the "every screen happens to FillRect everything" assumption
        // from being load-bearing. Without it, a screen that only redraws
        // part of itself would reveal stale content from two frames back.
        foreach (var idx in _dirtyIndices)
        {
            _front[idx] = _back[idx];
        }

        _firstFrame = false;
    }
}
