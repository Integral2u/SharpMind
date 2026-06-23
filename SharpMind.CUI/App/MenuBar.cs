using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>A single action inside a dropdown menu.</summary>
public sealed record MenuItem(string Label, char Hotkey, AppScreenKind Target, bool IsExit = false, bool IsNewSession = false, bool IsDebugSession = false);

/// <summary>
/// The MS Edit/QBasic-style top menu bar: a row of top-level items, each
/// opened either by clicking it or by Alt+&lt;underlined letter&gt; (F10 also
/// opens the first menu, matching the old DOS convention). Once a menu is
/// open it's modal — every key and click goes to the menu until something
/// closes it — exactly like the originals, where you couldn't interact with
/// the document while a menu was dropped down.
/// </summary>
public sealed class MenuBar
{
    private readonly (string Label, char Hotkey, MenuItem[] Items)[] _menus =
    [
        ("File", 'F', [
            new MenuItem("New session...", 'N', AppScreenKind.ModelBrowser, IsNewSession: true),
            new MenuItem("Welcome screen", 'W', AppScreenKind.Welcome),
            new MenuItem("Settings...", 'S', AppScreenKind.Settings),
            new MenuItem("Exit", 'X', AppScreenKind.Welcome, IsExit: true)
        ]),
        ("Model", 'M', [
            new MenuItem("Browse for model...", 'B', AppScreenKind.ModelBrowser),
            new MenuItem("Debug session (no model)...", 'D', AppScreenKind.Options, IsDebugSession: true)
        ]),
        ("Options", 'O', [new MenuItem("Session options...", 'S', AppScreenKind.Options)]),
        ("Chat", 'C', [new MenuItem("Go to chat", 'G', AppScreenKind.Chat)]),
    ];

    private int? _openMenuIndex;
    private int _highlightedItem;

    /// <summary>True while a dropdown is open, i.e. while this component should be treated as modal by the caller.</summary>
    public bool IsMenuOpen => _openMenuIndex is not null;

    /// <summary>Set for exactly one frame when the user picks something; the caller is expected to act on it and it's then cleared.</summary>
    public AppScreenKind? SelectedTarget { get; private set; }
    public bool ExitSelected { get; private set; }
    public bool NewSessionSelected { get; private set; }
    public bool DebugSessionSelected { get; private set; }

    public void ClearSelection()
    {
        SelectedTarget = null;
        ExitSelected = false;
        NewSessionSelected = false;
        DebugSessionSelected = false;
    }

    /// <summary>Cached layout from the last Draw call, so click hit-testing matches exactly what's on screen.</summary>
    private readonly List<(int X, int Width)> _itemBounds = [];

    public void HandleKey(ConsoleKeyInfo key)
    {
        if (_openMenuIndex is null)
        {
            // Closed: Alt+letter or F10 opens a menu. Nothing else is this component's concern.
            if (key.Key == ConsoleKey.F10) { OpenMenu(0); return; }
            if (key.Modifiers.HasFlag(ConsoleModifiers.Alt))
            {
                for (int i = 0; i < _menus.Length; i++)
                {
                    if (char.ToUpperInvariant(key.KeyChar) == char.ToUpperInvariant(_menus[i].Hotkey))
                    {
                        OpenMenu(i);
                        return;
                    }
                }
            }
            return;
        }

        // Open: this is modal, so handle navigation/selection/close and nothing falls through.
        var menu = _menus[_openMenuIndex.Value];
        switch (key.Key)
        {
            case ConsoleKey.Escape:
                CloseMenu();
                return;
            case ConsoleKey.LeftArrow:
                OpenMenu((_openMenuIndex.Value - 1 + _menus.Length) % _menus.Length);
                return;
            case ConsoleKey.RightArrow:
                OpenMenu((_openMenuIndex.Value + 1) % _menus.Length);
                return;
            case ConsoleKey.UpArrow:
                _highlightedItem = (_highlightedItem - 1 + menu.Items.Length) % menu.Items.Length;
                return;
            case ConsoleKey.DownArrow:
                _highlightedItem = (_highlightedItem + 1) % menu.Items.Length;
                return;
            case ConsoleKey.Enter:
                Activate(menu.Items[_highlightedItem]);
                return;
        }

        // Mnemonic letter inside the open menu (e.g. press 'B' for "Browse for model...").
        foreach (var item in menu.Items)
        {
            if (char.ToUpperInvariant(key.KeyChar) == char.ToUpperInvariant(item.Hotkey))
            {
                Activate(item);
                return;
            }
        }
    }

    public void HandleMouse(MouseEvent evt)
    {
        if (evt.Action != MouseAction.Down) return;

        if (_openMenuIndex is null)
        {
            // Closed: a click on row 0 within an item's bounds opens that menu.
            if (evt.Y != 0) return;
            for (int i = 0; i < _itemBounds.Count; i++)
            {
                var (x, w) = _itemBounds[i];
                if (evt.X >= x && evt.X < x + w) { OpenMenu(i); return; }
            }
            return;
        }

        // Open: clicking the bar switches menus; clicking a dropdown row activates it;
        // clicking anywhere else closes the menu without acting (matches Edit's behaviour).
        if (evt.Y == 0)
        {
            for (int i = 0; i < _itemBounds.Count; i++)
            {
                var (x, w) = _itemBounds[i];
                if (evt.X >= x && evt.X < x + w) { OpenMenu(i); return; }
            }
            CloseMenu();
            return;
        }

        var menu = _menus[_openMenuIndex.Value];
        int dropdownTop = 1;
        int row = evt.Y - dropdownTop;
        if (row >= 0 && row < menu.Items.Length)
        {
            Activate(menu.Items[row]);
        }
        else
        {
            CloseMenu();
        }
    }

    private void OpenMenu(int index)
    {
        _openMenuIndex = index;
        _highlightedItem = 0;
    }

    private void CloseMenu() => _openMenuIndex = null;

    private void Activate(MenuItem item)
    {
        if (item.IsExit) ExitSelected = true;
        else if (item.IsNewSession) NewSessionSelected = true;
        else if (item.IsDebugSession) DebugSessionSelected = true;
        else SelectedTarget = item.Target;
        CloseMenu();
    }

    public void Draw(ScreenBuffer buf, int width, AppScreenKind current, Theme theme)
    {
        ConsoleColor barFg = theme.BarFg;
        ConsoleColor barBg = theme.BarBg;
        ConsoleColor activeFg = theme.Background;
        ConsoleColor activeBg = theme.Accent;
        ConsoleColor hotkeyFg = theme.Error;

        buf.FillRect(0, 0, width, 1, ' ', barFg, barBg);

        _itemBounds.Clear();
        int x = 1;
        for (int i = 0; i < _menus.Length; i++)
        {
            var (label, hotkey, _) = _menus[i];
            bool open = i == _openMenuIndex;
            string text = $" {label} ";
            ConsoleColor fg = open ? activeFg : barFg;
            ConsoleColor bg = open ? activeBg : barBg;
            buf.Write(x, 0, text, fg, bg);

            // Underline-equivalent: render the hotkey letter in a distinct colour since
            // terminal cells have no underline attribute in this renderer.
            int hotkeyOffset = label.IndexOf(hotkey);
            if (hotkeyOffset >= 0)
                buf.Set(x + 1 + hotkeyOffset, 0, label[hotkeyOffset], open ? theme.Background : hotkeyFg, bg);

            _itemBounds.Add((x, text.Length));
            x += text.Length + 1;
        }

        string title = "SharpMind CUI";
        buf.Write(width - title.Length - 1, 0, title, barFg, barBg);
    }

    /// <summary>
    /// Draws the open dropdown, if any. Deliberately separate from
    /// <see cref="Draw"/> and meant to be called by the app *after* the
    /// active screen has drawn its own content — every screen fills its
    /// entire area starting at row 1, which is exactly where a dropdown
    /// lives, so drawing the dropdown as part of the same pass as the bar
    /// would just get painted over a frame later.
    /// </summary>
    public void DrawDropdownOverlay(ScreenBuffer buf, Theme theme)
    {
        if (_openMenuIndex is { } openIdx)
            DrawDropdown(buf, openIdx, theme);
    }

    private void DrawDropdown(ScreenBuffer buf, int menuIndex, Theme theme)
    {
        var (_, _, items) = _menus[menuIndex];
        int x = _itemBounds[menuIndex].X;
        int w = Math.Max(24, items.Max(i => i.Label.Length) + 4);
        int h = items.Length + 2;

        ConsoleColor fg = theme.BarFg;
        ConsoleColor bg = theme.BarBg;
        ConsoleColor selFg = theme.SelectionFg;
        ConsoleColor selBg = theme.SelectionBg;

        buf.FillRect(x, 1, w, h, ' ', fg, bg);
        buf.DrawBox(x, 1, w, h, fg, bg, doubleLine: false);

        for (int i = 0; i < items.Length; i++)
        {
            bool highlighted = i == _highlightedItem;
            ConsoleColor itemFg = highlighted ? selFg : fg;
            ConsoleColor itemBg = highlighted ? selBg : bg;
            buf.FillRect(x + 1, 1 + 1 + i, w - 2, 1, ' ', itemFg, itemBg);
            buf.Write(x + 2, 1 + 1 + i, items[i].Label, itemFg, itemBg);

            int hotkeyOffset = items[i].Label.IndexOf(items[i].Hotkey);
            if (hotkeyOffset >= 0)
                buf.Set(x + 2 + hotkeyOffset, 1 + 1 + i, items[i].Label[hotkeyOffset],
                    highlighted ? theme.Accent : theme.Error, itemBg);
        }
    }
}

public enum AppScreenKind { Welcome, ModelBrowser, Options, Chat, Settings }
