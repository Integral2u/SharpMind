using SharpMind.CUI.Screen;
using SharpMind.Model.Format;

namespace SharpMind.CUI.App;

/// <summary>
/// Lets the user pick a project folder, then browse the GGUF files inside it
/// and see a quick metadata preview (architecture, layer count, quant type)
/// before committing to a full weight load.
/// </summary>
public sealed class ModelBrowserScreen
{
    private string _currentPath;
    private ListPicker _picker;
    private string[] _entries = [];
    private string? _previewError;
    private (string arch, long tensors, string quantHint)? _preview;

    // Cached from the last Draw call so mouse hit-testing matches exactly what's on screen.
    private (int X, int Y, int W, int H) _listBounds;
    private int _lastClickIndex = -1;
    private DateTime _lastClickTime = DateTime.MinValue;

    public bool Cancelled { get; private set; }
    public string? ChosenModelPath { get; private set; }

    public ModelBrowserScreen(string startPath)
    {
        _currentPath = Directory.Exists(startPath) ? startPath : Directory.GetCurrentDirectory();
        _picker = new ListPicker([]);
        Refresh();
    }

    private void Refresh()
    {
        var dirs = Directory.Exists(_currentPath)
            ? Directory.GetDirectories(_currentPath).Select(d => "[DIR] " + Path.GetFileName(d)).OrderBy(s => s).ToArray()
            : [];
        var ggufs = Directory.Exists(_currentPath)
            ? Directory.GetFiles(_currentPath, "*.gguf").Select(Path.GetFileName).OrderBy(s => s).ToArray()
            : [];

        var entries = new List<string> { ".. (up one level)" };
        entries.AddRange(dirs);
        entries.AddRange(ggufs.Select(f => f!));
        _entries = entries.ToArray();
        _picker = new ListPicker(_entries);
        UpdatePreview();
    }

    private void UpdatePreview()
    {
        _preview = null;
        _previewError = null;
        var selected = _picker.SelectedItem;
        if (selected is null || !selected.EndsWith(".gguf")) return;

        try
        {
            var meta = GgufLoader.LoadMeta(Path.Combine(_currentPath, selected));
            var arch = meta.GetString("general.architecture", "unknown");
            var quant = meta.Tensors.Count > 0 ? meta.Tensors[0].Dtype.ToString() : "unknown";
            _preview = (arch, meta.TensorCount, quant);
        }
        catch (Exception ex)
        {
            _previewError = ex.Message;
        }
    }

    public void HandleKey(ConsoleKeyInfo key)
    {
        switch (key.Key)
        {
            case ConsoleKey.UpArrow: _picker.MoveUp(); UpdatePreview(); break;
            case ConsoleKey.DownArrow: _picker.MoveDown(); UpdatePreview(); break;
            case ConsoleKey.Home: _picker.MoveHome(); UpdatePreview(); break;
            case ConsoleKey.End: _picker.MoveEnd(); UpdatePreview(); break;
            case ConsoleKey.Escape: Cancelled = true; break;
            case ConsoleKey.Enter: Activate(); break;
        }
    }

    /// <summary>Acts on whatever's currently selected — shared by Enter and by double-click.</summary>
    private void Activate()
    {
        var sel = _picker.SelectedItem;
        if (sel is null) return;
        if (sel.StartsWith(".. "))
        {
            var parent = Directory.GetParent(_currentPath);
            if (parent is not null) { _currentPath = parent.FullName; Refresh(); }
        }
        else if (sel.StartsWith("[DIR] "))
        {
            _currentPath = Path.Combine(_currentPath, sel[6..]);
            Refresh();
        }
        else if (sel.EndsWith(".gguf"))
        {
            ChosenModelPath = Path.Combine(_currentPath, sel);
        }
    }

    /// <summary>
    /// Single click on a row selects it (and refreshes the preview, same as
    /// arrow-key navigation). A second click on the same row within 400ms —
    /// the conventional double-click window — activates it, same as pressing
    /// Enter. There's no separate "is this a double click" plumbing in
    /// <see cref="MouseEvent"/> itself; tracking the previous click's index
    /// and timestamp here is simpler than adding click-counting to the input
    /// layer for what is, so far, the only screen that needs it.
    /// </summary>
    public void HandleMouse(MouseEvent evt)
    {
        if (evt.Action != MouseAction.Down) return;
        if (evt.X < _listBounds.X || evt.X >= _listBounds.X + _listBounds.W ||
            evt.Y < _listBounds.Y || evt.Y >= _listBounds.Y + _listBounds.H)
            return;

        int clickedRow = evt.Y - _listBounds.Y;
        int clickedIndex = _picker.IndexAtRow(clickedRow);
        if (clickedIndex < 0) return;

        bool isDoubleClick = clickedIndex == _lastClickIndex && (DateTime.UtcNow - _lastClickTime) < TimeSpan.FromMilliseconds(400);
        _lastClickIndex = clickedIndex;
        _lastClickTime = DateTime.UtcNow;

        _picker.SetSelectedIndex(clickedIndex);
        UpdatePreview();
        if (isDoubleClick) Activate();
    }

    public void Draw(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor bg = theme.Background;
        ConsoleColor fg = theme.Text;
        ConsoleColor accent = theme.Accent;

        buf.FillRect(x, y, w, h, ' ', fg, bg);
        buf.DrawBox(x, y, w, h, accent, bg);
        buf.WriteCentered(x, y, w, " Select Model ", theme.Text, bg);

        string pathLine = _currentPath.Length > w - 4 ? "..." + _currentPath[^(w - 7)..] : _currentPath;
        buf.Write(x + 2, y + 1, pathLine, theme.DimText, bg);

        int listW = (w - 4) * 2 / 3;
        int listH = h - 5;
        _picker.Draw(buf, x + 2, y + 3, listW, listH, fg, bg, theme.SelectionFg, theme.SelectionBg);
        _listBounds = (x + 2, y + 3, listW, listH);

        // preview pane
        int prevX = x + 2 + listW + 1;
        int prevW = w - 4 - listW - 1;
        buf.DrawBox(prevX, y + 3, prevW, listH, theme.DimText, bg, doubleLine: false);
        if (_previewError is not null)
        {
            buf.Write(prevX + 1, y + 4, "Read error:", theme.Error, bg);
            buf.Write(prevX + 1, y + 5, _previewError.Length > prevW - 2 ? _previewError[..(prevW - 2)] : _previewError, theme.Error, bg);
        }
        else if (_preview is { } p)
        {
            buf.Write(prevX + 1, y + 4, "Architecture:", theme.DimText, bg);
            buf.Write(prevX + 1, y + 5, p.arch, fg, bg);
            buf.Write(prevX + 1, y + 7, "Tensors:", theme.DimText, bg);
            buf.Write(prevX + 1, y + 8, p.tensors.ToString(), fg, bg);
            buf.Write(prevX + 1, y + 10, "Quant (first tensor):", theme.DimText, bg);
            buf.Write(prevX + 1, y + 11, p.quantHint, fg, bg);
        }
        else
        {
            buf.Write(prevX + 1, y + 4, "Select a .gguf file", theme.DimText, bg);
            buf.Write(prevX + 1, y + 5, "to preview it here.", theme.DimText, bg);
        }

        buf.Write(x + 2, y + h - 2, "Up/Down Navigate   Enter/Double-click Open/Select   Esc Cancel", theme.DimText, bg);
    }
}
