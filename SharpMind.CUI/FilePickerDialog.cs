using NStack;
using Terminal.Gui;

namespace SharpMind.CUI;

public enum PickerMode { File, Folder }

/// <summary>
/// A small, predictable file/folder browser dialog. Built from scratch
/// rather than using Terminal.Gui's own OpenDialog — that control's
/// directory-chooser mode has a long-standing, still-open GitHub issue
/// describing it returning a different path than what's actually shown
/// selected in the UI, which is exactly the kind of silent-wrong-result bug
/// not worth risking for something as simple as "pick a folder".
/// </summary>
public static class FilePickerDialog
{
    /// <summary>Shows the picker modally and returns the chosen path, or null if cancelled.</summary>
    public static string? Show(string title, string startPath, PickerMode mode, string filePattern = "*.*")
    {
        string? result = null;
        string currentPath = Directory.Exists(startPath) ? startPath : Directory.GetCurrentDirectory();

        var dialog = new Dialog((ustring)title, 70, 20);
        var pathLabel = new Label((ustring)currentPath) { X = 1, Y = 0, Width = Dim.Fill(2) };
        var listView = new ListView { X = 1, Y = 2, Width = Dim.Fill(2), Height = Dim.Fill(4) };
        List<string> entries = [];

        void Refresh()
        {
            pathLabel.Text = currentPath;
            var dirs = Directory.GetDirectories(currentPath).Select(d => "[DIR] " + Path.GetFileName(d)).OrderBy(s => s);
            entries = new List<string> { ".. (up one level)" };
            entries.AddRange(dirs);
            if (mode == PickerMode.File)
                entries.AddRange(Directory.GetFiles(currentPath, filePattern).Select(f => Path.GetFileName(f)!).OrderBy(s => s));
            listView.SetSource(entries);
            dialog.SetNeedsDisplay();
        }

        void Activate()
        {
            int idx = listView.SelectedItem;
            if (idx < 0 || idx >= entries.Count) return;
            string sel = entries[idx];

            if (sel.StartsWith(".. "))
            {
                var parent = Directory.GetParent(currentPath);
                if (parent is not null) { currentPath = parent.FullName; Refresh(); }
            }
            else if (sel.StartsWith("[DIR] "))
            {
                currentPath = Path.Combine(currentPath, sel[6..]);
                Refresh();
            }
            else
            {
                result = Path.Combine(currentPath, sel);
                Application.RequestStop();
            }
        }

        listView.OpenSelectedItem += (_) => Activate();

        var selectButton = new Button(mode == PickerMode.Folder ? "Select this folder" : "Select")
        {
            X = 1, Y = Pos.AnchorEnd(2), IsDefault = mode == PickerMode.Folder
        };
        selectButton.Clicked += () =>
        {
            if (mode == PickerMode.Folder) { result = currentPath; Application.RequestStop(); }
            else Activate();
        };

        var cancelButton = new Button("Cancel") { X = Pos.Right(selectButton) + 2, Y = Pos.AnchorEnd(2) };
        cancelButton.Clicked += () => Application.RequestStop();

        dialog.Add(pathLabel, listView, selectButton, cancelButton);
        Refresh();
        listView.SetFocus();

        Application.Run(dialog);
        return result;
    }
}
