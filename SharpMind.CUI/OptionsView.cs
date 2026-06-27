using NStack;
using SharpMind;
using SharpMind.CUI.App;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>Per-session configuration form. Replaces the old hand-rolled Tab/Left-Right field list with real focusable controls.</summary>
public sealed class OptionsView : View
{
    private readonly SessionOptions _options;

    public OptionsView(SessionOptions options, Action onLaunch, Action onCancel)
    {
        _options = options;

        int row = 0;
        Label AddLabel(string text)
        {
            var l = new Label(text) { X = 1, Y = row };
            Add(l);
            return l;
        }

        AddLabel("Model:");
        var modelLabel = new Label(options.ModelPath is not null ? Path.GetFileName(options.ModelPath) : "(none — UIDebug mode)")
        { X = 22, Y = row };
        Add(modelLabel);
        row += 2;

        AddLabel("Generator strategy:");
        var generatorRadio = new RadioGroup(Enum.GetNames<GeneratorStrategy>().Select(p => (ustring)p).ToArray()) { X = 22, Y = row, SelectedItem = (int)options.Generator };
        generatorRadio.SelectedItemChanged += (args) => _options.Generator = (GeneratorStrategy)args.SelectedItem;
        Add(generatorRadio);
        row += Enum.GetValues<GeneratorStrategy>().Length + 1;

        AddLabel("KV cache strategy:");
        var cacheRadio = new RadioGroup(Enum.GetNames<CacheStrategy>().Select(p => (ustring)p).ToArray()) { X = 22, Y = row, SelectedItem = (int)options.Cache };
        cacheRadio.SelectedItemChanged += (args) => _options.Cache = (CacheStrategy)args.SelectedItem;
        Add(cacheRadio);
        row += Enum.GetValues<CacheStrategy>().Length + 1;

        AddLabel("Hardware tier:");
        var hwRadio = new RadioGroup(Enum.GetNames<HardwareTier>().Select(p=>(ustring)p).ToArray()) { X = 22, Y = row, SelectedItem = (int)options.HardwareTier };
        hwRadio.SelectedItemChanged += (args) => _options.HardwareTier = (HardwareTier)args.SelectedItem;
        Add(hwRadio);
        row += Enum.GetValues<HardwareTier>().Length + 1;

        var gpuCheck = new CheckBox("Use GPU (requires SharpMind.GPU reference)", options.UseGpu) { X = 1, Y = row };
        gpuCheck.Toggled += (_) => _options.UseGpu = gpuCheck.Checked;
        Add(gpuCheck);
        row += 2;

        AddLabel("Temperature:");
        var tempField = new TextField(options.Sampling.Temperature.ToString("F2")) { X = 22, Y = row, Width = 10 };
        tempField.TextChanged += (_) =>
        {
            if (float.TryParse(tempField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { Temperature = v };
        };
        Add(tempField);
        row++;

        AddLabel("Top-K:");
        var topKField = new TextField(options.Sampling.TopK.ToString()) { X = 22, Y = row, Width = 10 };
        topKField.TextChanged += (_) =>
        {
            if (int.TryParse(topKField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { TopK = v };
        };
        Add(topKField);
        row++;

        AddLabel("Top-P:");
        var topPField = new TextField(options.Sampling.TopP.ToString("F2")) { X = 22, Y = row, Width = 10 };
        topPField.TextChanged += (_) =>
        {
            if (float.TryParse(topPField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { TopP = v };
        };
        Add(topPField);
        row++;

        AddLabel("Max new tokens:");
        var maxTokensField = new TextField(options.Generation.MaxNewTokens.ToString()) { X = 22, Y = row, Width = 10 };
        maxTokensField.TextChanged += (_) =>
        {
            if (int.TryParse(maxTokensField.Text.ToString(), out var v))
                _options.Generation = _options.Generation with { MaxNewTokens = v };
        };
        Add(maxTokensField);
        row += 2;

        AddLabel("Agent name:");
        var agentNameField = new TextField(options.AgentName) { X = 22, Y = row, Width = 30 };
        agentNameField.TextChanged += (_) => _options.AgentName = agentNameField.Text.ToString();
        Add(agentNameField);
        row++;

        var agentsCheck = new CheckBox("Enable sub-agents", options.AgentsEnabled) { X = 1, Y = row };
        agentsCheck.Toggled += (_) => _options.AgentsEnabled = agentsCheck.Checked;
        Add(agentsCheck);
        row += 2;

        AddLabel("Skill folders (;-sep):");
        var skillField = new TextField(string.Join(";", options.SkillFolders)) { X = 22, Y = row, Width = Dim.Fill(2) };
        skillField.TextChanged += (_) => _options.SkillFolders = skillField.Text.ToString()
            .Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
        Add(skillField);
        row++;

        AddLabel("Tool DLLs (;-sep):");
        var toolField = new TextField(string.Join(";", options.ToolAssemblyPaths)) { X = 22, Y = row, Width = Dim.Fill(2) };
        toolField.TextChanged += (_) => _options.ToolAssemblyPaths = toolField.Text.ToString()
            .Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
        Add(toolField);
        row++;

        AddLabel("Tools folder (scanned live):");
        var toolsFolderField = new TextField(options.ToolsFolder ?? "") { X = 30, Y = row, Width = Dim.Fill(2) };
        toolsFolderField.TextChanged += (_) => _options.ToolsFolder = string.IsNullOrWhiteSpace(toolsFolderField.Text.ToString()) ? null : toolsFolderField.Text.ToString();
        Add(toolsFolderField);
        row += 2;

        var launchButton = new Button("Launch Session") { X = 1, Y = row, IsDefault = true };
        launchButton.Clicked += () => onLaunch();
        var cancelButton = new Button("Cancel") { X = Pos.Right(launchButton) + 2, Y = row };
        cancelButton.Clicked += () => onCancel();
        Add(launchButton, cancelButton);
    }
}
