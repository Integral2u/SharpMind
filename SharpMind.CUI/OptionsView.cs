using System.Drawing;
using NStack;
using SharpMind;
using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
using SharpMind.Model.Format;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>Per-session configuration form, wrapped in a ScrollView since the field count exceeds most terminal window heights.</summary>
public sealed class OptionsView : View
{
    private readonly SessionOptions _options;
    private readonly View _formContent;

    public OptionsView(SessionOptions options, Action onLaunch, Action onCancel)
    {
        _options = options;

        // ScrollView's own Width/Height become the visible viewport; formContent
        // is sized to its full (taller-than-the-window) extent and set as
        // ScrollView.ContentSize. Several Terminal.Gui v1 GitHub issues report
        // ScrollView being unreliable about auto-detecting content size from
        // its children, so ContentSize is set explicitly from the actual final
        // row count below rather than left for the control to infer. Fixed
        // widths are used for the wide text fields below rather than
        // Dim.Fill(), for the same reason — Dim.Fill() resolving correctly
        // inside a ScrollView's content area is one of the specific things
        // those issues flagged as unpredictable.
        var scrollView = new ScrollView
        {
            X = 0, Y = 0,
            Width = Dim.Fill(),
            Height = Dim.Fill(),
            ShowVerticalScrollIndicator = true,
            ShowHorizontalScrollIndicator = false
        };

        _formContent = new View { X = 0, Y = 0, Width = Dim.Fill() };

        int row = 0;
        Label AddLabel(string text)
        {
            var l = new Label((ustring)text) { X = 1, Y = row };
            _formContent.Add(l);
            return l;
        }

        AddLabel("Model:");
        var modelLabel = new Label((ustring)(options.ModelPath is not null ? Path.GetFileName(options.ModelPath) : "(none — UIDebug mode)"))
        { X = 30, Y = row };
        _formContent.Add(modelLabel);
        row += 2;

        AddLabel("Project path:");
        var projectPathField = new TextField((ustring)(options.ProjectPath ?? "")) { X = 30, Y = row, Width = 45 };
        projectPathField.TextChanged += (_) => _options.ProjectPath = string.IsNullOrWhiteSpace(projectPathField.Text.ToString()) ? null : projectPathField.Text.ToString();
        var projectPathOpen = new Button("Open...") { X = Pos.Right(projectPathField) + 1, Y = row };
        projectPathOpen.Clicked += () =>
        {
            var picked = FilePickerDialog.Show("Select project folder", options.ProjectPath ?? Directory.GetCurrentDirectory(), PickerMode.Folder);
            if (picked is not null) { projectPathField.Text = picked; _options.ProjectPath = picked; }
        };
        _formContent.Add(projectPathField, projectPathOpen);
        row += 2;

        AddLabel("Generator strategy:");
        var generatorRadio = new RadioGroup(Enum.GetNames<GeneratorStrategy>().Select(p => (ustring)p).ToArray()) { X = 30, Y = row, SelectedItem = (int)options.Generator };
        generatorRadio.SelectedItemChanged += (args) => _options.Generator = (GeneratorStrategy)args.SelectedItem;
        _formContent.Add(generatorRadio);
        row += Enum.GetValues<GeneratorStrategy>().Length + 1;

        AddLabel("KV cache strategy:");
        var cacheRadio = new RadioGroup(Enum.GetNames<CacheStrategy>().Select(p => (ustring)p).ToArray()) { X = 30, Y = row, SelectedItem = (int)options.Cache };
        cacheRadio.SelectedItemChanged += (args) => _options.Cache = (CacheStrategy)args.SelectedItem;
        _formContent.Add(cacheRadio);
        row += Enum.GetValues<CacheStrategy>().Length + 1;



        AddLabel("Hardware tier:");
        var hwRadio = new RadioGroup(Enum.GetNames<HardwareTier>().Select(p => (ustring)p).ToArray()) { X = 30, Y = row, SelectedItem = (int)options.HardwareTier };
        hwRadio.SelectedItemChanged += (args) => _options.HardwareTier = (HardwareTier)args.SelectedItem;
        _formContent.Add(hwRadio);
        row += Enum.GetValues<HardwareTier>().Length + 1;

        var gpuCheck = new CheckBox("Use GPU (requires SharpMind.GPU reference)", options.UseGpu) { X = 1, Y = row };
        gpuCheck.Toggled += (_) => _options.UseGpu = gpuCheck.Checked;
        _formContent.Add(gpuCheck);
        row++;

        var gpuNonQuantCheck = new CheckBox("  GPU: non-quantized ops (pointwise, gate, softmax, rmsnorm)", options.GpuNonQuant) { X = 1, Y = row };
        gpuNonQuantCheck.Toggled += (_) => _options.GpuNonQuant = gpuNonQuantCheck.Checked;
        _formContent.Add(gpuNonQuantCheck);
        row++;

        var gpuVecDotCheck = new CheckBox("  GPU: quantized vector dot", options.GpuVecDot) { X = 1, Y = row };
        gpuVecDotCheck.Toggled += (_) => _options.GpuVecDot = gpuVecDotCheck.Checked;
        _formContent.Add(gpuVecDotCheck);
        row++;

        var gpuMatMulCheck = new CheckBox("  GPU: quantized matrix multiply", options.GpuMatMul) { X = 1, Y = row };
        gpuMatMulCheck.Toggled += (_) => _options.GpuMatMul = gpuMatMulCheck.Checked;
        _formContent.Add(gpuMatMulCheck);
        row++;

        var parallelCheck = new CheckBox("Use parallel kernels (faster on multi-core CPU)", options.UseParallelKernels) { X = 1, Y = row };
        parallelCheck.Toggled += (_) => _options.UseParallelKernels = parallelCheck.Checked;
        row += 2;

        AddLabel("Temperature:");
        var tempField = new TextField((ustring)options.Sampling.Temperature.ToString("F2")) { X = 30, Y = row, Width = 10 };
        tempField.TextChanged += (_) =>
        {
            if (float.TryParse(tempField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { Temperature = v };
        };
        _formContent.Add(tempField);
        row++;

        AddLabel("Top-K:");
        var topKField = new TextField((ustring)options.Sampling.TopK.ToString()) { X = 30, Y = row, Width = 10 };
        topKField.TextChanged += (_) =>
        {
            if (int.TryParse(topKField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { TopK = v };
        };
        _formContent.Add(topKField);
        row++;

        AddLabel("Top-P:");
        var topPField = new TextField((ustring)options.Sampling.TopP.ToString("F2")) { X = 30, Y = row, Width = 10 };
        topPField.TextChanged += (_) =>
        {
            if (float.TryParse(topPField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { TopP = v };
        };
        _formContent.Add(topPField);
        row++;

        AddLabel("Max new tokens:");
        var maxTokensField = new TextField((ustring)options.Generation.MaxNewTokens.ToString()) { X = 30, Y = row, Width = 10 };
        maxTokensField.TextChanged += (_) =>
        {
            if (int.TryParse(maxTokensField.Text.ToString(), out var v))
                _options.Generation = _options.Generation with { MaxNewTokens = v };
        };
        _formContent.Add(maxTokensField);
        row += 2;

        AddLabel("Agent name:");
        var agentNameField = new TextField((ustring)options.AgentName) { X = 30, Y = row, Width = 30 };
        agentNameField.TextChanged += (_) => _options.AgentName = agentNameField.Text.ToString();
        _formContent.Add(agentNameField);
        row++;

        var agentsCheck = new CheckBox("Enable sub-agents", options.AgentsEnabled) { X = 1, Y = row };
        agentsCheck.Toggled += (_) => _options.AgentsEnabled = agentsCheck.Checked;
        _formContent.Add(agentsCheck);
        row++;

        var enableThinkingCheck = new CheckBox("Enable model thinking (Qwen3 reasoning)", options.EnableThinking) { X = 1, Y = row };
        enableThinkingCheck.Toggled += (_) => _options.EnableThinking = enableThinkingCheck.Checked;
        _formContent.Add(enableThinkingCheck);
        row++;

        AddLabel("Context compaction:");
        var compactorRadio = new RadioGroup(Enum.GetNames<CompactorStrategy>().Select(p => (ustring)p).ToArray()) { X = 30, Y = row, SelectedItem = (int)options.Compactor };
        compactorRadio.SelectedItemChanged += (args) => _options.Compactor = (CompactorStrategy)args.SelectedItem;
        _formContent.Add(compactorRadio);
        row += Enum.GetValues<CompactorStrategy>().Length + 2;

        AddLabel("File access:");
        var fileAccessRadio = new RadioGroup(Enum.GetNames<ToolPermission>().Select(p => (ustring)p).ToArray()) { X = 30, Y = row, SelectedItem = (int)options.FileAccess };
        fileAccessRadio.SelectedItemChanged += (args) => _options.FileAccess = (ToolPermission)args.SelectedItem;
        _formContent.Add(fileAccessRadio);
        row += Enum.GetValues<ToolPermission>().Length + 1;

        AddLabel("Network access:");
        var networkAccessRadio = new RadioGroup(Enum.GetNames<ToolPermission>().Select(p => (ustring)p).ToArray()) { X = 30, Y = row, SelectedItem = (int)options.NetworkAccess };
        networkAccessRadio.SelectedItemChanged += (args) => _options.NetworkAccess = (ToolPermission)args.SelectedItem;
        _formContent.Add(networkAccessRadio);
        row += Enum.GetValues<ToolPermission>().Length + 1;

        AddLabel("Skill folders (;-sep):");
        var skillField = new TextField((ustring)string.Join(";", options.SkillFolders)) { X = 30, Y = row, Width = 40 };
        skillField.TextChanged += (_) => _options.SkillFolders = skillField.Text.ToString()
            .Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
        var skillOpen = new Button("Add...") { X = Pos.Right(skillField) + 1, Y = row };
        skillOpen.Clicked += () =>
        {
            var picked = FilePickerDialog.Show("Add skill folder", _options.ProjectPath ?? Directory.GetCurrentDirectory(), PickerMode.Folder);
            if (picked is null) return;
            var current = _options.SkillFolders.ToList();
            if (!current.Contains(picked)) current.Add(picked);
            _options.SkillFolders = current;
            skillField.Text = string.Join(";", current);
        };
        _formContent.Add(skillField, skillOpen);
        row++;

        AddLabel("Tool DLLs (;-sep):");
        var toolField = new TextField((ustring)string.Join(";", options.ToolAssemblyPaths)) { X = 30, Y = row, Width = 40 };
        toolField.TextChanged += (_) => _options.ToolAssemblyPaths = toolField.Text.ToString()
            .Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
        var toolOpen = new Button("Add...") { X = Pos.Right(toolField) + 1, Y = row };
        toolOpen.Clicked += () =>
        {
            var picked = FilePickerDialog.Show("Add tool DLL", _options.ToolsFolder ?? _options.ProjectPath ?? Directory.GetCurrentDirectory(), PickerMode.File, "*.dll");
            if (picked is null) return;
            var current = _options.ToolAssemblyPaths.ToList();
            if (!current.Contains(picked)) current.Add(picked);
            _options.ToolAssemblyPaths = current;
            toolField.Text = string.Join(";", current);
        };
        _formContent.Add(toolField, toolOpen);
        row++;

        AddLabel("Tools folder (scanned live):");
        var toolsFolderField = new TextField((ustring)(options.ToolsFolder ?? "")) { X = 30, Y = row, Width = 40 };
        toolsFolderField.TextChanged += (_) => _options.ToolsFolder = string.IsNullOrWhiteSpace(toolsFolderField.Text.ToString()) ? null : toolsFolderField.Text.ToString();
        var toolsFolderOpen = new Button("Open...") { X = Pos.Right(toolsFolderField) + 1, Y = row };
        toolsFolderOpen.Clicked += () =>
        {
            var picked = FilePickerDialog.Show("Select tools folder", _options.ToolsFolder ?? _options.ProjectPath ?? Directory.GetCurrentDirectory(), PickerMode.Folder);
            if (picked is not null) { toolsFolderField.Text = picked; _options.ToolsFolder = picked; }
        };
        _formContent.Add(toolsFolderField, toolsFolderOpen);
        row += 2;

        var manageToolsBtn = new Button("Manage Tools...") { X = 1, Y = row, Width = 15 };
        manageToolsBtn.Clicked += ManageTools;
        _formContent.Add(manageToolsBtn);
        row += 2;

        var launchButton = new Button("Launch Session") { X = 1, Y = row, IsDefault = true };
        launchButton.Clicked += () => onLaunch();
        var cancelButton = new Button("Cancel") { X = Pos.Right(launchButton) + 2, Y = row };
        cancelButton.Clicked += () => onCancel();
        var saveAsButton = new Button("Save Options As...") { X = Pos.Right(cancelButton) + 2, Y = row };
        saveAsButton.Clicked += SaveOptionsAs;
        var openPresetButton = new Button("Open Preset...") { X = Pos.Right(saveAsButton) + 2, Y = row };
        openPresetButton.Clicked += OpenPreset;
        _formContent.Add(launchButton, cancelButton, saveAsButton, openPresetButton);
        row += 2;

        _formContent.Height = row;
        scrollView.ContentSize = new Terminal.Gui.Size(100, row);
        scrollView.Add(_formContent);
        Add(scrollView);

        KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == Key.Esc) { onCancel(); args.Handled = true; }
        };
    }

    /// <summary>Saves the current SessionOptions as a named preset — no model needs to be loaded, no session needs to be running. Usable purely for code/scripted launches too, since it's just the plain JSON SessionLauncher.BuildSession already consumes.</summary>
    private void SaveOptionsAs()
    {
        bool confirmed = false;
        string name = "preset";

        var dialog = new Dialog("Save Options As", 50, 7);
        var field = new TextField((ustring)name) { X = 1, Y = 1, Width = Dim.Fill(2) };
        var ok = new Button("Save") { IsDefault = true };
        ok.Clicked += () => { name = field.Text.ToString() ?? name; confirmed = true; Application.RequestStop(); };
        var cancel = new Button("Cancel");
        cancel.Clicked += () => Application.RequestStop();
        dialog.Add(new Label("Preset name:") { X = 1, Y = 0 }, field);
        dialog.AddButton(ok);
        dialog.AddButton(cancel);
        Application.Run(dialog);

        if (!confirmed || string.IsNullOrWhiteSpace(name)) return;

        var saved = new SavedSession { Name = name, Options = _options };
        string safeFileName = string.Concat(name.Split(Path.GetInvalidFileNameChars()));
        string path = Path.Combine(SavedSession.DefaultFolder, $"{safeFileName}.json");

        if (SavedSession.Save(saved, path, out var error))
            MessageBox.Query("Saved", $"Saved preset to:\n{path}", "OK");
        else
            MessageBox.ErrorQuery("Save failed", error ?? "Unknown error", "OK");
    }

    private void OpenPreset()
    {
        var files = SavedSession.ListSaved(SavedSession.DefaultFolder);
        if (files.Count == 0)
        {
            MessageBox.Query("No presets", "No saved presets were found yet.\nUse \"Save Options As...\" first.", "OK");
            return;
        }

        var picked = FilePickerDialog.Show("Open preset", SavedSession.DefaultFolder, PickerMode.File, "*.json");
        if (picked is null) return;

        var loaded = SavedSession.Load(picked, out var error);
        if (loaded is null)
        {
            MessageBox.ErrorQuery("Load failed", error ?? "Unknown error", "OK");
            return;
        }

        // Copies every field from the loaded preset onto the existing
        // SessionOptions instance rather than replacing the reference — the
        // controls built above all closed over _options/the original
        // instance, so swapping the object out from under them would leave
        // every control showing stale values until the whole view is rebuilt.
        CopyOptionsInto(_options, loaded.Options);
        MessageBox.Query("Preset loaded", $"Loaded \"{loaded.Name}\".\nReopen Options to see the loaded values reflected in every field.", "OK");
    }

    private void ManageTools()
    {
        var available = SessionLauncher.GetAvailableTools(_options);
        if (available.Count == 0)
        {
            MessageBox.Query("No tools", "No available tools were found in the current configuration.", "OK");
            return;
        }

        var dialog = new Dialog("Manage Tools", 40, 20);
        var scroll = new ScrollView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        var content = new View { Width = Dim.Fill() };
        
        int row = 0;
        foreach (var tool in available)
        {
            var cb = new CheckBox($"Enable {tool}", !_options.DisabledTools.Contains(tool)) { X = 1, Y = row };
            cb.Toggled += (_) =>
            {
                if (cb.Checked) _options.DisabledTools.Remove(tool);
                else _options.DisabledTools.Add(tool);
            };
            content.Add(cb);
            row++;
        }
        
        content.Height = row;
        scroll.ContentSize = new Terminal.Gui.Size(40, row);
        scroll.Add(content);
        dialog.Add(scroll);
        var closeBtn = new Button("Close");
        closeBtn.Clicked += () => Application.RequestStop();
        dialog.AddButton(closeBtn);
        
        Application.Run(dialog);
    }

    private static void CopyOptionsInto(SessionOptions target, SessionOptions source)
    {
        target.ModelPath = source.ModelPath;
        target.ProjectPath = source.ProjectPath;
        target.SkillFolders = source.SkillFolders;
        target.ToolAssemblyPaths = source.ToolAssemblyPaths;
        target.ToolsFolder = source.ToolsFolder;
        target.Generator = source.Generator;
        target.Cache = source.Cache;
        target.HardwareTier = source.HardwareTier;
        target.UseGpu = source.UseGpu;
        target.GpuNonQuant = source.GpuNonQuant;
        target.GpuVecDot = source.GpuVecDot;
        target.GpuMatMul = source.GpuMatMul;
        target.FileAccess = source.FileAccess;
        target.NetworkAccess = source.NetworkAccess;
        target.Sampling = source.Sampling;
        target.Generation = source.Generation;
        target.AgentName = source.AgentName;
        target.AgentsEnabled = source.AgentsEnabled;
        target.EnableThinking = source.EnableThinking;
        target.MaxAgentDepth = source.MaxAgentDepth;
        target.MaxToolCallsPerTurn = source.MaxToolCallsPerTurn;
        target.DisabledTools = new HashSet<string>(source.DisabledTools);
    }
}
