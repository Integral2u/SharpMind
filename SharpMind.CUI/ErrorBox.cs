using NStack;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// Error dialog with a Copy button so message-box errors (especially long
/// load/launch failure traces) can be pasted out of the terminal app into an
/// issue report or a chat. Drop-in for <c>MessageBox.ErrorQuery(...)</c> — the
/// third argument keeps the button label the caller expects.
/// </summary>
public static class ErrorBox
{
    /// <summary>
    /// Shows a modal error dialog with a Copy button plus an OK button.
    /// <paramref name="width"/>/<paramref name="height"/> override the
    /// content-derived size (the viewer scrolls either way).
    /// </summary>
    public static void Show(string title, string message, string okButtonText = "OK", int? width = null, int? height = null)
    {
        string text = message ?? "";
        string[] lines = text.Split('\n');
        int maxLine = lines.DefaultIfEmpty("").Max(l => l.Length);
        int dialogWidth = width ?? Math.Clamp(maxLine + 5, 60, 100);
        int dialogHeight = height ?? Math.Clamp(lines.Length + 8, 10, 28);

        var dialog = new Dialog((ustring)title, dialogWidth, dialogHeight);

        var viewer = new TextView
        {
            X = 1,
            Y = 1,
            Width = Dim.Fill(2),
            Height = Dim.Fill(3),
            Text = (ustring)text,
            ReadOnly = true,
            WordWrap = true,
            TabStop = false
        };

        var status = new Label(" ") { Y = Pos.AnchorEnd(2) };

        var ok = new Button(okButtonText) { X = 1, Y = Pos.AnchorEnd(1), IsDefault = true };
        ok.Clicked += () => Application.RequestStop();

        var copy = new Button("_Copy") { X = Pos.Right(ok) + 2, Y = Pos.AnchorEnd(1) };
        void DoCopy()
        {
            bool copied = ClipboardUtil.CopyToClipboard(text);
            status.Text = copied ? "Copied to clipboard." : "Copy failed — use the terminal's own selection to copy.";
            status.SetNeedsDisplay();
        }
        copy.Clicked += DoCopy;

        dialog.KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == (Key.CtrlMask | Key.C))
            {
                DoCopy();
                args.Handled = true;
            }
        };

        dialog.Add(viewer, status, copy, ok);
        Application.Run(dialog);
    }
}