using NStack;
using SharpMind.CUI.App;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>App-level preferences, as distinct from OptionsView which configures one session.</summary>
public sealed class SettingsView : View
{
    private readonly AppSettings _settings;
    public Action<ThemeKind>? OnThemeChanged;

    public SettingsView(AppSettings settings, Action onBack)
    {
        _settings = settings;
        int row = 0;

        Add(new Label("Default model folder:") { X = 1, Y = row });
        var modelFolderField = new TextField(settings.DefaultModelFolder ?? "") { X = 25, Y = row, Width = Dim.Fill(2) };
        modelFolderField.TextChanged += (_) => _settings.DefaultModelFolder =
            string.IsNullOrWhiteSpace(modelFolderField.Text.ToString()) ? null : modelFolderField.Text.ToString();
        Add(modelFolderField);
        row += 2;

        Add(new Label("Tools folder:") { X = 1, Y = row });
        var toolsFolderField = new TextField(settings.ToolsFolder ?? "") { X = 25, Y = row, Width = Dim.Fill(2) };
        toolsFolderField.TextChanged += (_) => _settings.ToolsFolder =
            string.IsNullOrWhiteSpace(toolsFolderField.Text.ToString()) ? null : toolsFolderField.Text.ToString();
        Add(toolsFolderField);
        row += 2;

        Add(new Label("Color theme:") { X = 1, Y = row });
        var themeRadio = new RadioGroup([.. Enum.GetNames<ThemeKind>().Select(p=>(ustring)p)]) { X = 25, Y = row, SelectedItem = (int)settings.Theme };
        themeRadio.SelectedItemChanged += (args) =>
        {
            _settings.Theme = (ThemeKind)args.SelectedItem;
            OnThemeChanged?.Invoke(_settings.Theme);
        };
        Add(themeRadio);
        row += Enum.GetValues<ThemeKind>().Length + 2;

        var saveButton = new Button("Save Settings") { X = 1, Y = row, IsDefault = true };
        var errorLabel = new Label("") { X = 1, Y = row + 2, Width = Dim.Fill(2) };
        saveButton.Clicked += () =>
        {
            bool ok = _settings.Save(out var error);
            errorLabel.Text = ok ? "Saved." : $"Save failed: {error}";
            SetNeedsDisplay();
        };
        var backButton = new Button("Back") { X = Pos.Right(saveButton) + 2, Y = row };
        backButton.Clicked += () => onBack();
        Add(saveButton, backButton, errorLabel);

        KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == Key.Esc) { onBack(); args.Handled = true; }
        };
    }
}
