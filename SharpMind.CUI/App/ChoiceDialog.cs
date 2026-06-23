using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>
/// Renders one <see cref="ChoiceRequest"/> as a centred modal overlay: a
/// radio-button-style list of the offered options, plus — if the request
/// allows it — a free-text line as an explicit alternative to picking one of
/// them. Up/Down (or number keys 1-9) to pick, Enter to confirm, Tab to jump
/// into the free-text line when one is offered.
///
/// Like <see cref="MenuBar"/>'s dropdown, this is modal while showing: the
/// app routes all input here first and the screen underneath simply doesn't
/// see it until the request resolves.
/// </summary>
public sealed class ChoiceDialog(ChoiceRequest request)
{
    private int _selectedIndex;
    private bool _editingFreeText;
    private string _freeTextBuffer = "";

    public bool IsResolved { get; private set; }

    public void HandleKey(ConsoleKeyInfo key)
    {
        if (_editingFreeText)
        {
            HandleFreeTextKey(key);
            return;
        }

        switch (key.Key)
        {
            case ConsoleKey.UpArrow:
                _selectedIndex = (_selectedIndex - 1 + request.Options.Count) % request.Options.Count;
                return;
            case ConsoleKey.DownArrow:
                _selectedIndex = (_selectedIndex + 1) % request.Options.Count;
                return;
            case ConsoleKey.Tab when request.AllowFreeText:
                _editingFreeText = true;
                return;
            case ConsoleKey.Enter:
                Confirm(request.Options[_selectedIndex]);
                return;
            case ConsoleKey.Escape when request.AllowFreeText:
                // Esc only backs out of the dialog into free-text editing when free
                // text is offered — otherwise there is no "cancel" affordance at all,
                // deliberately: the calling tool is synchronously blocked waiting on
                // an answer, so dismissing the dialog with nothing chosen would just
                // leave the model's tool call hanging with no result to give back.
                _editingFreeText = true;
                return;
        }

        // Number-key shortcut: "1" picks the first option, etc. — faster than
        // arrowing down a long list, and mirrors how most CLI prompts already work.
        if (key.KeyChar is >= '1' and <= '9')
        {
            int idx = key.KeyChar - '1';
            if (idx < request.Options.Count) Confirm(request.Options[idx]);
        }
    }

    private void HandleFreeTextKey(ConsoleKeyInfo key)
    {
        switch (key.Key)
        {
            case ConsoleKey.Enter:
                if (_freeTextBuffer.Length > 0) Confirm(_freeTextBuffer);
                return;
            case ConsoleKey.Escape:
            case ConsoleKey.Tab:
                _editingFreeText = false;
                return;
            case ConsoleKey.Backspace:
                if (_freeTextBuffer.Length > 0) _freeTextBuffer = _freeTextBuffer[..^1];
                return;
            default:
                if (!char.IsControl(key.KeyChar)) _freeTextBuffer += key.KeyChar;
                return;
        }
    }

    private void Confirm(string chosenText)
    {
        request.Resolve(chosenText);
        IsResolved = true;
    }

    public void Draw(ScreenBuffer buf, int screenW, int screenH, Theme theme)
    {
        int w = Math.Min(screenW - 6, Math.Max(50, request.Options.Max(o => o.Length) + 10));
        int extraForFreeText = request.AllowFreeText ? 3 : 0;
        int h = Math.Min(screenH - 4, request.Options.Count + 6 + extraForFreeText + WrapCount(request.Prompt, w - 4));
        int x = (screenW - w) / 2;
        int y = (screenH - h) / 2;

        ConsoleColor bg = theme.Background;
        ConsoleColor fg = theme.Text;
        ConsoleColor accent = theme.Accent;

        // Dim the area behind the dialog slightly by overwriting it solid first —
        // there's no real "dim" effect available with 16-colour ConsoleColor, so
        // a plain fill is the honest version of a modal backdrop here rather than
        // attempting a translucency effect this renderer can't actually produce.
        buf.FillRect(x, y, w, h, ' ', fg, bg);
        buf.DrawBox(x, y, w, h, accent, bg);

        int row = y + 1;
        foreach (var line in Wrap(request.Prompt, w - 4))
        {
            buf.Write(x + 2, row, line, theme.Text, bg);
            row++;
        }
        row++;

        for (int i = 0; i < request.Options.Count; i++)
        {
            bool selected = !_editingFreeText && i == _selectedIndex;
            string marker = selected ? "(*)" : "( )";
            string line = $"{marker} {i + 1}. {request.Options[i]}";
            ConsoleColor lineFg = selected ? theme.SelectionFg : fg;
            ConsoleColor lineBg = selected ? theme.SelectionBg : bg;
            buf.FillRect(x + 2, row, w - 4, 1, ' ', lineFg, lineBg);
            buf.Write(x + 2, row, line.Length > w - 4 ? line[..(w - 4)] : line, lineFg, lineBg);
            row++;
        }

        if (request.AllowFreeText)
        {
            row++;
            buf.Write(x + 2, row, "Or type your own:", theme.DimText, bg);
            row++;
            ConsoleColor textFg = _editingFreeText ? theme.SelectionFg : fg;
            ConsoleColor textBg = _editingFreeText ? theme.SelectionBg : bg;
            buf.FillRect(x + 2, row, w - 4, 1, ' ', textFg, textBg);
            string display = _editingFreeText ? _freeTextBuffer + "_" : (_freeTextBuffer.Length > 0 ? _freeTextBuffer : "(Tab to type)");
            buf.Write(x + 3, row, display.Length > w - 6 ? display[..(w - 6)] : display, textFg, textBg);
        }

        string hint = _editingFreeText
            ? "Enter confirm   Tab/Esc back to list"
            : request.AllowFreeText
                ? "1-9/Up/Down select   Enter confirm   Tab type your own"
                : "1-9/Up/Down select   Enter confirm";
        buf.Write(x + 2, y + h - 2, hint.Length > w - 4 ? hint[..(w - 4)] : hint, theme.DimText, bg);
    }

    private static int WrapCount(string text, int width) => Wrap(text, width).Count;

    private static List<string> Wrap(string text, int width)
    {
        var result = new List<string>();
        if (width <= 0) { result.Add(text); return result; }
        int i = 0;
        while (i < text.Length)
        {
            int len = Math.Min(width, text.Length - i);
            result.Add(text.Substring(i, len));
            i += len;
        }
        if (result.Count == 0) result.Add("");
        return result;
    }
}
