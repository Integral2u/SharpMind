using NStack;
using SharpMind.Data.Metadata;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>Result of the component picker: the chosen descriptor + its filled parameter values.</summary>
public sealed class PickedComponent
{
    public required ComponentDescriptor Descriptor { get; init; }
    public required Dictionary<string, string> Values { get; init; }
}

/// <summary>
/// Lists every component (source or stage) available to the training wizard —
/// built-ins plus anything found in the plugins folder — and hands the selection
/// straight into <see cref="ComponentParamDialog"/> for its parameter form.
/// Returns null if cancelled at either step.
/// </summary>
public static class ComponentPickerDialog
{
    public static PickedComponent? Show(
        string? pluginsFolder,
        ComponentKind kind,
        IReadOnlyDictionary<string, string>? prefill = null)
    {
        var registry = ComponentRegistry.ScanFolder(pluginsFolder ?? "", out var warnings);
        var candidates = registry
            .Where(c => c.Kind == kind)
            .OrderBy(c => c.Name)
            .ToList();

        if (candidates.Count == 0)
        {
            MessageBox.ErrorQuery("No components", $"No {(kind == ComponentKind.Source ? "data sources" : "stages")} are available.", "OK");
            return null;
        }

        var dialog = new Dialog((ustring)(kind == ComponentKind.Source ? "Choose data source" : "Choose stage"), 70, Math.Min(24, candidates.Count + 6));
        var list = new ListView(candidates.Select(c => (ustring)$"{c.Name} — {c.Description}").ToArray())
        {
            X = 1, Y = 0, Width = Dim.Fill(2), Height = Dim.Fill(5)
        };
        var hint = new Label("Up/Down: navigate   PageUp/PageDown: jump   Enter: configure   Esc: cancel")
        {
            X = 1, Y = Pos.AnchorEnd(4), Width = Dim.Fill(2)
        };
        var ok = new Button("Choose") { X = 1, Y = Pos.AnchorEnd(1), IsDefault = true };
        var cancel = new Button("Cancel") { X = Pos.Right(ok) + 2, Y = Pos.AnchorEnd(1) };
        dialog.Add(list, hint, ok, cancel);

        PickedComponent? result = null;

        void Complete()
        {
            int idx = list.SelectedItem;
            if (idx < 0 || idx >= candidates.Count) return;
            var descriptor = candidates[idx];
            var values = ComponentParamDialog.Show(descriptor, prefill);
            if (values is null) return; // cancelled inside the param form
            result = new PickedComponent { Descriptor = descriptor, Values = values };
            Application.RequestStop();
        }

        list.OpenSelectedItem += (_) => Complete();
        ok.Clicked += () => { if (list.SelectedItem >= 0) Complete(); };
        cancel.Clicked += () => Application.RequestStop();

        if (warnings.Count > 0)
        {
            var w = new Label(("Plugins: " + string.Join(" | ", warnings))) { X = 1, Y = 0, Width = Dim.Fill(2) };
            list.Y = 1;
            list.Height = Dim.Fill(6);
            dialog.Add(w);
        }

        list.SetFocus();
        Application.Run(dialog);
        return result;
    }
}

/// <summary>
/// A modal form built from <see cref="ComponentDescriptor"/> metadata: one
/// control per constructor parameter. File/folder params get a text field plus
/// a "Browse" file-picker button; enums and <see cref="ChoicesAttribute"/>
/// become radio groups; booleans become check boxes; numerics are validated
/// number fields; plain strings are text fields.
/// </summary>
public static class ComponentParamDialog
{
    /// <summary>
    /// Shows the form modally for <paramref name="descriptor"/>. Returns the
    /// entered parameter-name→value map on OK, or null when cancelled.
    /// </summary>
    public static Dictionary<string, string>? Show(
        ComponentDescriptor descriptor,
        IReadOnlyDictionary<string, string>? existing = null)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        if (existing is not null)
            foreach (var (k, v) in existing) values[k] = v;

        int rows = ContentRows(descriptor) + 4;

        var dialog = new Dialog((ustring)descriptor.Name, 74, Math.Min(28, rows));
        int row = 0;

        var descLabel = new Label((ustring)descriptor.Description) { X = 1, Y = 0, Width = Dim.Fill(2) };
        dialog.Add(descLabel);
        row += 2;

        var errorLabel = new Label("") { X = 1, Y = Pos.AnchorEnd(2), Width = Dim.Fill(2) };
        dialog.Add(errorLabel);

        foreach (var p in descriptor.Parameters)
        {
            string current = values.TryGetValue(p.Name, out var v) ? v : InitialValue(p);
            string label = TruncateLabel(p.Tooltip?.Text is { } tip ? tip : p.Name);
            var labelView = new Label((ustring)(label + ":")) { X = 1, Y = row, Width = 22 };

            if (p.FileChooser is { } fc)
            {
                var field = new TextField((ustring)current) { X = 24, Y = row, Width = Dim.Fill(12) };
                var btn = new Button("Browse") { X = Pos.Right(field) + 1, Y = row };
                btn.Clicked += () =>
                {
                    var path = FilePickerDialog.Show(
                        fc.Help ?? $"Select {p.Name}",
                        StartPath(current), PickerMode.File, fc.Pattern);
                    if (path is not null) { field.Text = path; values[p.Name] = path; }
                };
                field.TextChanged += (_) => values[p.Name] = field.Text.ToString();
                dialog.Add(labelView, field, btn);
            }
            else if (p.FolderChooser is { } fo)
            {
                var field = new TextField((ustring)current) { X = 24, Y = row, Width = Dim.Fill(10) };
                var btn = new Button("Browse...") { X = Pos.Right(field) + 1, Y = row };
                btn.Clicked += () =>
                {
                    var dir = FilePickerDialog.Show(
                        fo.Help ?? $"Select {p.Name}",
                        StartPath(current), PickerMode.Folder);
                    if (dir is not null) { field.Text = dir; values[p.Name] = dir; }
                };
                field.TextChanged += (_) => values[p.Name] = field.Text.ToString();
                dialog.Add(labelView, field, btn);
            }
            else if (p.Type == typeof(bool))
            {
                var cb = new CheckBox((ustring)label, bool.TryParse(current, out var b) && b) { X = 1, Y = row };
                cb.Toggled += (_) => values[p.Name] = cb.Checked ? "true" : "false";
                dialog.Add(cb);
            }
            else if (p.Type.IsEnum)
            {
                var names = Enum.GetNames(p.Type);
                var radio = new RadioGroup(names.Select(n => (ustring)n).ToArray()) { X = 24, Y = row, SelectedItem = IndexOf(names, current) };
                radio.SelectedItemChanged += (a) => values[p.Name] = names[a.SelectedItem];
                dialog.Add(labelView, radio);
                row += names.Length + 1;
            }
            else if (p.Choices is { Length: > 0 } choices)
            {
                var items = choices.Select(c => (ustring)c).ToArray();
                var radio = new RadioGroup(items) { X = 24, Y = row, SelectedItem = Math.Max(0, Array.IndexOf(choices, current)) };
                radio.SelectedItemChanged += (a) => values[p.Name] = choices[a.SelectedItem];
                dialog.Add(labelView, radio);
                row += choices.Length + 1;
            }
            else
            {
                var field = new TextField((ustring)current) { X = 24, Y = row, Width = Dim.Fill(2) };
                field.TextChanged += (_) => values[p.Name] = field.Text.ToString();
                dialog.Add(labelView, field);
                row++;
            }

            if (!(p.Type.IsEnum || p.Choices is { Length: > 0 }))
                row++;
        }

        var ok = new Button("OK") { X = 1, Y = Pos.AnchorEnd(1), IsDefault = true };
        ok.Clicked += () =>
        {
            foreach (var p in descriptor.Parameters)
            {
                if (p.MinMax is null) continue;
                if (!values.TryGetValue(p.Name, out var raw) || string.IsNullOrWhiteSpace(raw)) continue;

                if (!double.TryParse(raw, System.Globalization.CultureInfo.InvariantCulture, out var dv)
                    || dv < p.MinMax.Min || dv > p.MinMax.Max)
                {
                    errorLabel.Text = $"{p.Name} must be within {p.MinMax.Min}..{p.MinMax.Max}.";
                    return;
                }
            }
            Application.RequestStop();
        };
        var cancel = new Button("Cancel") { X = Pos.Right(ok) + 2, Y = Pos.AnchorEnd(1) };
        cancel.Clicked += () => { values = null!; Application.RequestStop(); };
        dialog.Add(ok, cancel);

        Application.Run(dialog);
        return values;
    }

    private static string InitialValue(ComponentParameter p)
    {
        if (p.DefaultValue is { } dv) return dv.Value;
        if (!p.IsRequired && p.Parameter.DefaultValue?.ToString() is { } d) return d;
        return p.Type == typeof(bool) ? "false" : "";
    }

    private static string StartPath(string hint) =>
        Directory.Exists(hint) ? hint : Directory.GetCurrentDirectory();

    /// <summary>
    /// Mirrors the <c>row</c> counter used while building the parameter form so the
    /// dialog height covers every control (including enum/choices radios, which
    /// advance <c>row</c> by <c>names.Length + 1</c> rather than 1).
    /// </summary>
    private static int ContentRows(ComponentDescriptor descriptor)
    {
        int row = 2; // description label + one blank line
        foreach (var p in descriptor.Parameters)
        {
            if (p.Type == typeof(bool) || p.FileChooser is { } || p.FolderChooser is { })
            {
                row += 2;
            }
            else if (p.Type.IsEnum)
            {
                row += Enum.GetNames(p.Type).Length + 2;
            }
            else if (p.Choices is { Length: > 0 } choices)
            {
                row += choices.Length + 2;
            }
            else
            {
                row += 2;
            }
        }
        return row;
    }

    /// <summary>
    /// Caps a parameter label to the 22-column label gutter. <see cref="Label"/>
    /// in Terminal.Gui v1 does not clip at its <see cref="Label.Width"/>, so long
    /// tooltip-derived labels would spill into the value control's columns and be
    /// partially overpainted by the opaque TextField/RadioGroup. Truncate here so
    /// every row keeps "label : value" side by side.
    /// </summary>
    private static string TruncateLabel(string label, int maxWidth = 20)
    {
        if (string.IsNullOrEmpty(label)) return "";
        if (label.Length <= maxWidth) return label;
        return label[..(maxWidth - 1)] + "…";
    }

    private static int IndexOf(string[] names, string? current)
    {
        if (current is null) return 0;
        for (int i = 0; i < names.Length; i++)
            if (string.Equals(names[i], current, StringComparison.OrdinalIgnoreCase)) return i;
        return 0;
    }
}