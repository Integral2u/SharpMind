using NStack;
using Terminal.Gui;

namespace SharpMind.CUI;

public enum PickerMode { File, Folder, SaveFile }

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
    public static string? Show(string title, string startPath, PickerMode mode, string filePattern = "*.*",
        string defaultName = "", params string[] patterns)
    {
        string? result = null;
        string currentPath = Directory.Exists(startPath) ? startPath : Directory.GetCurrentDirectory();

        var dialog = new Dialog((ustring)title, 70, 22);
        var pathLabel = new Label((ustring)currentPath) { X = 1, Y = 0, Width = Dim.Fill(2) };
        // In SaveFile mode the name field lives near the bottom (AnchorEnd 7) and
        // the buttons at AnchorEnd 2, so the list must end well above them.
        var listView = new ListView
        {
            X = 1,
            Y = 2,
            Width = Dim.Fill(2),
            Height = mode == PickerMode.SaveFile ? Dim.Fill(10) : Dim.Fill(6),
        };
        List<string> entries = [];

        TextField? saveNameField = null;
        if (mode == PickerMode.SaveFile)
        {
            dialog.Add(new Label("File name:") { X = 1, Y = Pos.AnchorEnd(7), Width = 10 });
            saveNameField = new TextField((ustring)defaultName) { X = 12, Y = Pos.AnchorEnd(7), Width = Dim.Fill(2) };
            dialog.Add(saveNameField);
        }

        // In SaveFile mode the file list defaults to the format implied by the
        // pre-filled name (standard save behavior: filter to the save format).
        // Explicit patterns (e.g. the source picker's *.gguf + *.smm) win.
        // A single pattern may list several via ';' (e.g. "*.txt;*.md").
        string[] activePatterns = patterns is { Length: > 0 }
            ? patterns
            : mode == PickerMode.SaveFile && Path.GetExtension(defaultName) is { Length: > 1 } ext
                ? [$"*{ext}"]
                : filePattern.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        void Refresh()
        {
            pathLabel.Text = currentPath;
            var dirs = Directory.GetDirectories(currentPath).Select(d => "[DIR] " + Path.GetFileName(d)).OrderBy(s => s);
            entries = new List<string> { ".. (up one level)" };
            entries.AddRange(dirs);
            if (mode == PickerMode.File || mode == PickerMode.SaveFile)
                entries.AddRange(activePatterns
                    .SelectMany(p => Directory.GetFiles(currentPath, p))
                    .Select(f => Path.GetFileName(f)!)
                    .Distinct()
                    .OrderBy(s => s));
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
            else if (mode == PickerMode.SaveFile && saveNameField is not null)
            {
                saveNameField.Text = sel;
            }
            else
            {
                result = Path.Combine(currentPath, sel);
                Application.RequestStop();
            }
        }

        listView.OpenSelectedItem += (_) => Activate();

        Button selectButton;
        if (mode == PickerMode.SaveFile)
        {
            selectButton = new Button("Save")
            {
                X = 1, Y = Pos.AnchorEnd(2), IsDefault = true
            };
            selectButton.Clicked += () =>
            {
                string name = saveNameField?.Text.ToString() ?? "";
                if (string.IsNullOrWhiteSpace(name))
                {
                    MessageBox.Query("No name", "Enter a file name before saving.", "OK");
                    return;
                }
                result = Path.Combine(currentPath, name);
                Application.RequestStop();
            };
        }
        else
        {
            selectButton = new Button(mode == PickerMode.Folder ? "Select this folder" : "Select")
            {
                X = 1, Y = Pos.AnchorEnd(2), IsDefault = mode == PickerMode.Folder
            };
            selectButton.Clicked += () =>
            {
                if (mode == PickerMode.Folder) { result = currentPath; Application.RequestStop(); }
                else Activate();
            };
        }

        var cancelButton = new Button("Cancel") { X = Pos.Right(selectButton) + 2, Y = Pos.AnchorEnd(2) };
        cancelButton.Clicked += () => Application.RequestStop();

        dialog.Add(pathLabel, listView, selectButton, cancelButton);
        Refresh();
        if (saveNameField is not null) saveNameField.SetFocus();
        else listView.SetFocus();

        Application.Run(dialog);
        return result;
    }
}
