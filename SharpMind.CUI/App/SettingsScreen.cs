using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>
/// App-level preferences, as distinct from <see cref="OptionsScreen"/> which
/// configures one session. Same field-list form pattern, deliberately —
/// consistency between the two screens matters more here than any layout
/// novelty would add.
/// </summary>
public sealed class SettingsScreen(AppSettings settings)
{
    private enum Field { DefaultModelFolder, ToolsFolder, Theme, Save }

    private static readonly Field[] FieldOrder = Enum.GetValues<Field>();
    private int _fieldIndex;
    private string? _textEditBuffer;

    public bool Cancelled { get; private set; }
    public bool SaveRequested { get; private set; }
    public string? LastSaveError { get; private set; }

    public void AcknowledgeSaveRequest() => SaveRequested = false;

    private Field Current => FieldOrder[_fieldIndex];

    public void HandleKey(ConsoleKeyInfo key)
    {
        if (_textEditBuffer is not null)
        {
            HandleTextEditKey(key);
            return;
        }

        switch (key.Key)
        {
            case ConsoleKey.Escape: Cancelled = true; return;
            case ConsoleKey.Tab when key.Modifiers.HasFlag(ConsoleModifiers.Shift):
                _fieldIndex = (_fieldIndex - 1 + FieldOrder.Length) % FieldOrder.Length; return;
            case ConsoleKey.Tab:
            case ConsoleKey.DownArrow:
                _fieldIndex = (_fieldIndex + 1) % FieldOrder.Length; return;
            case ConsoleKey.UpArrow:
                _fieldIndex = (_fieldIndex - 1 + FieldOrder.Length) % FieldOrder.Length; return;
            case ConsoleKey.LeftArrow: CycleTheme(-1); return;
            case ConsoleKey.RightArrow: CycleTheme(1); return;
            case ConsoleKey.Enter: ActivateField(); return;
        }
    }

    private void CycleTheme(int dir)
    {
        if (Current != Field.Theme) return;
        var values = Enum.GetValues<ThemeKind>();
        int idx = Array.IndexOf(values, settings.Theme);
        idx = (idx + dir + values.Length) % values.Length;
        settings.Theme = values[idx];
    }

    private void ActivateField()
    {
        switch (Current)
        {
            case Field.DefaultModelFolder: _textEditBuffer = settings.DefaultModelFolder ?? ""; break;
            case Field.ToolsFolder: _textEditBuffer = settings.ToolsFolder ?? ""; break;
            case Field.Save: SaveRequested = true; break;
        }
    }

    private void HandleTextEditKey(ConsoleKeyInfo key)
    {
        switch (key.Key)
        {
            case ConsoleKey.Enter: CommitTextEdit(); _textEditBuffer = null; return;
            case ConsoleKey.Escape: _textEditBuffer = null; return;
            case ConsoleKey.Backspace:
                if (_textEditBuffer!.Length > 0) _textEditBuffer = _textEditBuffer[..^1];
                return;
            default:
                if (!char.IsControl(key.KeyChar)) _textEditBuffer += key.KeyChar;
                return;
        }
    }

    private void CommitTextEdit()
    {
        var value = _textEditBuffer ?? "";
        switch (Current)
        {
            case Field.DefaultModelFolder: settings.DefaultModelFolder = string.IsNullOrWhiteSpace(value) ? null : value; break;
            case Field.ToolsFolder: settings.ToolsFolder = string.IsNullOrWhiteSpace(value) ? null : value; break;
        }
    }

    /// <summary>Called by the app right after attempting Save(), success or not, so the result is visible on screen.</summary>
    public void ReportSaveResult(bool success, string? error)
    {
        LastSaveError = success ? null : error;
    }

    public void Draw(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor bg = theme.Background;
        ConsoleColor fg = theme.Text;
        ConsoleColor accent = theme.Accent;
        ConsoleColor label = theme.DimText;

        buf.FillRect(x, y, w, h, ' ', fg, bg);
        buf.DrawBox(x, y, w, h, accent, bg);
        buf.WriteCentered(x, y, w, " Settings ", theme.Text, bg);

        int row = y + 2;
        int labelW = 22;

        void DrawField(Field f, string labelText, string value)
        {
            bool selected = f == Current;
            string text = f == Current && _textEditBuffer is not null ? _textEditBuffer + "_" : value;
            buf.Write(x + 2, row, labelText.PadRight(labelW), selected ? theme.SelectionFg : label, selected ? theme.SelectionBg : bg);
            buf.Write(x + 2 + labelW, row, text.Length > w - labelW - 4 ? text[..(w - labelW - 4)] : text,
                selected ? theme.SelectionFg : fg, selected ? theme.SelectionBg : bg);
            row++;
        }

        DrawField(Field.DefaultModelFolder, "Default model folder", settings.DefaultModelFolder ?? "(not set)");
        DrawField(Field.ToolsFolder, "Tools folder", settings.ToolsFolder ?? "(not set)");
        row++;
        DrawField(Field.Theme, "Color theme", $"< {settings.Theme} >");
        row++;

        bool saveSelected = Current == Field.Save;
        buf.Write(x + 2, row, "[ Save Settings ]", saveSelected ? theme.SelectionFg : theme.Success, saveSelected ? theme.SelectionBg : bg);
        row += 2;

        if (LastSaveError is { } err)
        {
            string msg = $"Save failed: {err}";
            buf.Write(x + 2, row, msg.Length > w - 4 ? msg[..(w - 4)] : msg, theme.Error, bg);
        }

        buf.Write(x + 2, y + h - 2, "Tab move   Left/Right cycle theme   Enter edit/activate   Esc back",
            theme.DimText, bg);
    }
}
