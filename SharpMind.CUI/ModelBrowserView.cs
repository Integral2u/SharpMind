using SharpMind.Model.Format;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>Lets the user navigate folders and pick a .gguf file, with a live metadata preview pane.</summary>
public sealed class ModelBrowserView : View
{
    private readonly Action<string> _onChosen;
    private readonly Action _onCancel;

    private string _currentPath;
    private readonly ListView _listView;
    private readonly Label _pathLabel;
    private readonly Label _previewLabel;
    private List<string> _entries = [];

    public ModelBrowserView(string startPath, Action<string> onChosen, Action onCancel)
    {
        _onChosen = onChosen;
        _onCancel = onCancel;
        _currentPath = Directory.Exists(startPath) ? startPath : Directory.GetCurrentDirectory();

        _pathLabel = new Label("") { X = 0, Y = 0, Width = Dim.Fill() };

        _listView = new ListView
        {
            X = 0,
            Y = 2,
            Width = Dim.Percent(60),
            Height = Dim.Fill(2)
        };
        _listView.OpenSelectedItem += (_) => Activate();
        _listView.SelectedItemChanged += (_) => UpdatePreview();

        var previewFrame = new FrameView("Preview")
        {
            X = Pos.Right(_listView) + 1,
            Y = 2,
            Width = Dim.Fill(),
            Height = Dim.Fill(2)
        };
        _previewLabel = new Label("") { X = 1, Y = 1, Width = Dim.Fill(2), Height = Dim.Fill(1) };
        previewFrame.Add(_previewLabel);

        var hint = new Label("Enter/double-click: open or select   Esc: cancel")
        {
            X = 0,
            Y = Pos.AnchorEnd(1),
            Width = Dim.Fill()
        };

        Add(_pathLabel, _listView, previewFrame, hint);

        KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == Key.Esc) { onCancel(); args.Handled = true; }
        };

        Refresh();
        _listView.SetFocus();
    }

    private void Refresh()
    {
        _pathLabel.Text = _currentPath;

        var dirs = Directory.Exists(_currentPath)
            ? Directory.GetDirectories(_currentPath).Select(d => "[DIR] " + Path.GetFileName(d)).OrderBy(s => s)
            : Enumerable.Empty<string>();
        var ggufs = Directory.Exists(_currentPath)
            ? Directory.GetFiles(_currentPath, "*.gguf").Select(f => Path.GetFileName(f)!).OrderBy(s => s)
            : Enumerable.Empty<string>();

        _entries = new List<string> { ".. (up one level)" };
        _entries.AddRange(dirs);
        _entries.AddRange(ggufs);

        // SetSource, not mutating the existing list in place — ListView doesn't
        // observe changes to the underlying IList, only to Source itself.
        _listView.SetSource(_entries);
        UpdatePreview();
        SetNeedsDisplay();
    }

    private void UpdatePreview()
    {
        int idx = _listView.SelectedItem;
        if (idx < 0 || idx >= _entries.Count) { _previewLabel.Text = ""; return; }

        string sel = _entries[idx];
        if (!sel.EndsWith(".gguf")) { _previewLabel.Text = sel.StartsWith("[DIR]") ? "Folder" : ""; return; }

        try
        {
            var meta = GgufLoaderFactory.Default.LoadMeta(Path.Combine(_currentPath, sel));
            string arch = meta.GetString("general.architecture", "unknown");
            string quant = meta.Tensors.Count > 0 ? meta.Tensors[0].Dtype.ToString() : "unknown";
            _previewLabel.Text = $"Architecture: {arch}\nTensors: {meta.TensorCount}\nQuant (first tensor): {quant}";
        }
        catch (Exception ex)
        {
            _previewLabel.Text = $"Read error: {ex.Message}";
        }
    }

    private void Activate()
    {
        int idx = _listView.SelectedItem;
        if (idx < 0 || idx >= _entries.Count) return;
        string sel = _entries[idx];

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
            _onChosen(Path.Combine(_currentPath, sel));
        }
    }
}
