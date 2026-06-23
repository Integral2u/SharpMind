using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>
/// A scrollable, keyboard-navigable list rendered inside a box. Used for the
/// model browser, the path/skill pickers, and any "choose one of these enum
/// values" field on the options screen — one widget, several call sites.
/// </summary>
public sealed class ListPicker(IReadOnlyList<string> items)
{
    public int SelectedIndex { get; private set; }
    private int _scrollTop;

    public string? SelectedItem => items.Count == 0 ? null : items[SelectedIndex];

    public void MoveUp() => SelectedIndex = Math.Max(0, SelectedIndex - 1);
    public void MoveDown() => SelectedIndex = Math.Min(items.Count - 1, SelectedIndex + 1);
    public void MoveHome() => SelectedIndex = 0;
    public void MoveEnd() => SelectedIndex = Math.Max(0, items.Count - 1);

    /// <summary>Directly sets the selection, clamped to valid bounds. Used by mouse click handling.</summary>
    public void SetSelectedIndex(int index) => SelectedIndex = Math.Clamp(index, 0, Math.Max(0, items.Count - 1));

    /// <summary>
    /// Translates a row offset (0 = top visible row) into an item index,
    /// using the scroll position established by the most recent
    /// <see cref="Draw"/> call. Returns -1 if the row is outside the current
    /// item range (e.g. a click below the last item in a partially-filled
    /// list).
    /// </summary>
    public int IndexAtRow(int row)
    {
        int idx = _scrollTop + row;
        return idx >= 0 && idx < items.Count ? idx : -1;
    }

    /// <summary>Draws the list within the given inner content rect (already inside any border).</summary>
    public void Draw(ScreenBuffer buf, int x, int y, int w, int h, ConsoleColor fg, ConsoleColor bg, ConsoleColor selFg, ConsoleColor selBg)
    {
        // Keep the selection visible by scrolling the window as needed.
        if (SelectedIndex < _scrollTop) _scrollTop = SelectedIndex;
        if (SelectedIndex >= _scrollTop + h) _scrollTop = SelectedIndex - h + 1;
        _scrollTop = Math.Max(0, Math.Min(_scrollTop, Math.Max(0, items.Count - h)));

        for (int row = 0; row < h; row++)
        {
            int idx = _scrollTop + row;
            buf.FillRect(x, y + row, w, 1, ' ', fg, bg);
            if (idx >= items.Count) continue;

            bool selected = idx == SelectedIndex;
            string text = items[idx].Length > w ? items[idx][..w] : items[idx];
            buf.Write(x, y + row, text, selected ? selFg : fg, selected ? selBg : bg);
        }
    }
}
