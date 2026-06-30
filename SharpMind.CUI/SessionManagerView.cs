using NStack;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// Lists every currently-open chat session. Avoids fighting Terminal.Gui's
/// static-at-construction MenuBar model (rebuilding a submenu's children
/// dynamically as sessions open/close is possible but fiddly) by being one
/// ordinary view instead — the Chat menu just has a single static entry
/// that opens this.
/// </summary>
public sealed class SessionManagerView : View
{
    private readonly Func<IReadOnlyList<ChatSessionState>> _getSessions;
    private readonly Action<ChatSessionState> _onSwitch;
    private readonly Action<ChatSessionState> _onClose;
    private readonly Action _onBack;
    private readonly ListView _listView;

    public SessionManagerView(
        Func<IReadOnlyList<ChatSessionState>> getSessions,
        Action<ChatSessionState> onSwitch,
        Action<ChatSessionState> onClose,
        Action onBack)
    {
        _getSessions = getSessions;
        _onSwitch = onSwitch;
        _onClose = onClose;
        _onBack = onBack;

        Add(new Label("Open sessions:") { X = 1, Y = 0 });

        _listView = new ListView { X = 1, Y = 2, Width = Dim.Fill(2), Height = Dim.Fill(4) };
        _listView.OpenSelectedItem += (_) => SwitchToSelected();
        Add(_listView);

        var switchButton = new Button("Switch to") { X = 1, Y = Pos.AnchorEnd(2), IsDefault = true };
        switchButton.Clicked += SwitchToSelected;

        var renameButton = new Button("Rename") { X = Pos.Right(switchButton) + 2, Y = Pos.AnchorEnd(2) };
        renameButton.Clicked += RenameSelected;

        var closeButton = new Button("Close session") { X = Pos.Right(renameButton) + 2, Y = Pos.AnchorEnd(2) };
        closeButton.Clicked += CloseSelected;

        var backButton = new Button("Back") { X = Pos.Right(closeButton) + 2, Y = Pos.AnchorEnd(2) };
        backButton.Clicked += () => _onBack();

        Add(switchButton, renameButton, closeButton, backButton);

        KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == Key.Esc) { _onBack(); args.Handled = true; }
        };

        Refresh();
        _listView.SetFocus();
    }

    /// <summary>Call after any change that might affect the session list (open, close, rename) to keep this view's list current.</summary>
    public void Refresh()
    {
        var sessions = _getSessions();
        int previousSelection = _listView.SelectedItem;
        _listView.SetSource(sessions.Select(s => s.DisplayName).ToList());
        if (sessions.Count > 0)
            _listView.SelectedItem = Math.Clamp(previousSelection, 0, sessions.Count - 1);
        SetNeedsDisplay();
    }

    private ChatSessionState? Selected()
    {
        var sessions = _getSessions();
        int idx = _listView.SelectedItem;
        return idx >= 0 && idx < sessions.Count ? sessions[idx] : null;
    }

    private void SwitchToSelected()
    {
        if (Selected() is { } s) _onSwitch(s);
    }

    private void RenameSelected()
    {
        if (Selected() is not { } s) return;

        bool confirmed = false;
        string newName = s.DisplayName;

        var dialog = new Dialog((ustring)"Rename session", 50, 7);
        var field = new TextField((ustring)s.DisplayName) { X = 1, Y = 1, Width = Dim.Fill(2) };
        var ok = new Button("OK") { IsDefault = true };
        ok.Clicked += () => { newName = field.Text.ToString() ?? s.DisplayName; confirmed = true; Application.RequestStop(); };
        var cancel = new Button("Cancel");
        cancel.Clicked += () => Application.RequestStop();
        dialog.Add(field);
        dialog.AddButton(ok);
        dialog.AddButton(cancel);
        Application.Run(dialog);

        if (confirmed && !string.IsNullOrWhiteSpace(newName))
        {
            s.DisplayName = newName;
            s.View.SessionDisplayName = newName;
            Refresh();
        }
    }

    private void CloseSelected()
    {
        if (Selected() is { } s) { _onClose(s); Refresh(); }
    }
}
