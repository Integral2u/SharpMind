using NStack;
using SharpMind.Core;
using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model.Config;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>Per-session configuration form, wrapped in a ScrollView since the field count exceeds most terminal window heights.</summary>
public sealed class OptionsView : View
{
    private readonly SessionOptions _options;
    private readonly AppSettings _settings;
    private readonly View _formContent;
    private readonly TextField projectPathField;
    private readonly RadioGroup cacheRadio;
    private readonly RadioGroup loadModeRadio;
    private readonly RadioGroup formatterRadio;
    private readonly RadioGroup hwRadio;
    private readonly CheckBox gpuCheck;
    private readonly CheckBox gpuVecDotCheck;
    private readonly CheckBox gpuMatMulCheck;
    private readonly CheckBox parallelCheck;
    private readonly TextField topKField;
    private readonly TextField topPField;
    private readonly TextField repetitionPenaltyField;
    private readonly TextField maxTokensField;
    private readonly TextField maxContextTokensField;
    private readonly TextField userNameField;
    private readonly RadioGroup generatorRadio;
    private readonly CheckBox gpuNonQuantCheck;
    private readonly TextField tempField;
    private readonly TextField repetitionWindowField;
    private readonly TextField agentNameField;
    private readonly CheckBox agentsCheck;
    private readonly CheckBox enableThinkingCheck;
    private readonly CheckBox skipAgentPromptCheck;
    private readonly CheckBox disableToolsCheck;
    private readonly RadioGroup compactorRadio;
    private readonly RadioGroup fileAccessRadio;
    private readonly RadioGroup networkAccessRadio;
    private readonly TextField skillField;
    private readonly TextField toolsFolderField;
    private readonly RadioGroup? pluginCompactorRadio;
    private readonly TextField toolField;
    private readonly PluginLoadResult pluginResult = new();
    private readonly EmbeddedPluginInfo? _embedded;
    private readonly List<IContextCompactor> _pluginCompactors = [];
    private readonly ustring[] pluginCompactorNames = [];

    public OptionsView(SessionOptions options, AppSettings settings, Action onLaunch, Action onCancel)
    {
        _options = options;
        _settings = settings;
        _embedded = SessionLauncher.LoadEmbeddedPlugins(options.ModelPath);
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
            Height = Dim.Fill(1),
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
        projectPathField = new TextField((ustring)(options.ProjectPath ?? "")) { X = 30, Y = row, Width = 45 };
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
        generatorRadio = new RadioGroup([.. Enum.GetNames<GeneratorStrategy>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.Generator };
        generatorRadio.SelectedItemChanged += (args) => _options.Generator = (GeneratorStrategy)args.SelectedItem;
        _formContent.Add(generatorRadio);
        row += Enum.GetValues<GeneratorStrategy>().Length + 1;

        AddLabel("KV cache strategy:");
        cacheRadio = new RadioGroup([.. Enum.GetNames<CacheStrategy>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.Cache };
        cacheRadio.SelectedItemChanged += (args) => _options.Cache = (CacheStrategy)args.SelectedItem;
        _formContent.Add(cacheRadio);
        row += Enum.GetValues<CacheStrategy>().Length + 1;

        AddLabel("Weight load mode:");
        loadModeRadio = new RadioGroup([.. Enum.GetNames<LoadMode>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.LoadMode };
        loadModeRadio.SelectedItemChanged += (args) => _options.LoadMode = (LoadMode)args.SelectedItem;
        _formContent.Add(loadModeRadio);
        row += Enum.GetValues<LoadMode>().Length + 1;

        AddLabel("Formatter strategy:");
        formatterRadio = new RadioGroup([.. Enum.GetNames<FormatterStrategy>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.Formatter };
        formatterRadio.SelectedItemChanged += (args) => _options.Formatter = (FormatterStrategy)args.SelectedItem;
        _formContent.Add(formatterRadio);
        row += Enum.GetValues<FormatterStrategy>().Length + 1;

        AddLabel("Hardware tier:");
        hwRadio = new RadioGroup([.. Enum.GetNames<HardwareTier>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.HardwareTier };
        hwRadio.SelectedItemChanged += (args) => _options.HardwareTier = (HardwareTier)args.SelectedItem;
        _formContent.Add(hwRadio);
        row += Enum.GetValues<HardwareTier>().Length + 1;
        
        gpuCheck = new CheckBox("Use GPU (requires SharpMind.GPU reference)", options.UseGpu) { X = 1, Y = row };
        gpuCheck.Toggled += (_) => _options.UseGpu = gpuCheck.Checked;
        _formContent.Add(gpuCheck);
        row++;

        gpuNonQuantCheck = new CheckBox("  GPU: non-quantized ops (pointwise, gate, softmax, rmsnorm)", options.GpuNonQuant) { X = 1, Y = row };
        gpuNonQuantCheck.Toggled += (_) => _options.GpuNonQuant = gpuNonQuantCheck.Checked;
        _formContent.Add(gpuNonQuantCheck);
        row++;

        gpuVecDotCheck = new CheckBox("  GPU: quantized vector dot", options.GpuVecDot) { X = 1, Y = row };
        gpuVecDotCheck.Toggled += (_) => _options.GpuVecDot = gpuVecDotCheck.Checked;
        _formContent.Add(gpuVecDotCheck);
        row++;

        gpuMatMulCheck = new CheckBox("  GPU: quantized matrix multiply", options.GpuMatMul) { X = 1, Y = row };
        gpuMatMulCheck.Toggled += (_) => _options.GpuMatMul = gpuMatMulCheck.Checked;
        _formContent.Add(gpuMatMulCheck);
        row++;

        parallelCheck = new CheckBox("Use parallel kernels (faster on multi-core CPU)", options.UseParallelKernels) { X = 1, Y = row };
        parallelCheck.Toggled += (_) => _options.UseParallelKernels = parallelCheck.Checked;
        row += 2;

        AddLabel("Temperature:");
        tempField = new TextField((ustring)options.Sampling.Temperature.ToString("F2")) { X = 30, Y = row, Width = 10 };
        tempField.TextChanged += (_) =>
        {
            if (float.TryParse(tempField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { Temperature = v };
        };
        _formContent.Add(tempField);
        row++;

        AddLabel("Top-K:");
        topKField = new TextField((ustring)options.Sampling.TopK.ToString()) { X = 30, Y = row, Width = 10 };
        topKField.TextChanged += (_) =>
        {
            if (int.TryParse(topKField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { TopK = v };
        };
        _formContent.Add(topKField);
        row++;

        AddLabel("Top-P:");
        topPField = new TextField((ustring)options.Sampling.TopP.ToString("F2")) { X = 30, Y = row, Width = 10 };
        topPField.TextChanged += (_) =>
        {
            if (float.TryParse(topPField.Text.ToString(), out var v))
                _options.Sampling = _options.Sampling with { TopP = v };
        };
        _formContent.Add(topPField);
        row++;
        AddLabel("Repitition Penalty:");
        repetitionPenaltyField = new TextField((ustring)options.Generation.RepetitionPenalty.ToString("F2")) { X = 30, Y = row, Width = 10 };
        repetitionPenaltyField.TextChanged += (_) =>
        {
            if (float.TryParse(repetitionPenaltyField.Text.ToString(), out var v))
                _options.Generation = _options.Generation with { RepetitionPenalty = v };
        };
        _formContent.Add(repetitionPenaltyField);
        row++;
        AddLabel("Repitition Window:");
        repetitionWindowField = new TextField((ustring)options.Generation.RepetitionWindow.ToString()) { X = 30, Y = row, Width = 10 };
        repetitionWindowField.TextChanged += (_) =>
        {
            if (int.TryParse(repetitionWindowField.Text.ToString(), out var v))
                _options.Generation = _options.Generation with { RepetitionWindow = v };
        };
        _formContent.Add(repetitionWindowField);
        row++;

        AddLabel("Max new tokens:");
        maxTokensField = new TextField((ustring)options.Generation.MaxNewTokens.ToString()) { X = 30, Y = row, Width = 10 };
        maxTokensField.TextChanged += (_) =>
        {
            if (int.TryParse(maxTokensField.Text.ToString(), out var v))
                _options.Generation = _options.Generation with { MaxNewTokens = v };
        };
        _formContent.Add(maxTokensField);
        row++;

        AddLabel("Max context tokens (0 = full):");
        maxContextTokensField = new TextField((ustring)(options.MaxTokens?.ToString() ?? "0")) { X = 30, Y = row, Width = 10 };
        maxContextTokensField.TextChanged += (_) =>
        {
            if (int.TryParse(maxContextTokensField.Text.ToString(), out var v) && v > 0)
                _options.MaxTokens = v;
            else
                _options.MaxTokens = null;
        };
        _formContent.Add(maxContextTokensField);
        row += 2;

        AddLabel("Agent name:");
        agentNameField = new TextField((ustring)options.AgentName) { X = 30, Y = row, Width = 30 };
        agentNameField.TextChanged += (_) => _options.AgentName = agentNameField.Text.ToString() ?? string.Empty;
        _formContent.Add(agentNameField);
        row++;

        AddLabel("User name:");
        userNameField = new TextField((ustring)options.UserName) { X = 30, Y = row, Width = 30 };
        userNameField.TextChanged += (_) => _options.UserName = userNameField.Text.ToString() ?? string.Empty;
        _formContent.Add(userNameField);
        row++;

        agentsCheck = new CheckBox("Enable sub-agents", options.AgentsEnabled) { X = 1, Y = row };
        agentsCheck.Toggled += (_) => _options.AgentsEnabled = agentsCheck.Checked;
        _formContent.Add(agentsCheck);
        row++;

        enableThinkingCheck = new CheckBox("Enable model thinking (Qwen3 reasoning)", options.EnableThinking) { X = 1, Y = row };
        enableThinkingCheck.Toggled += (_) => _options.EnableThinking = enableThinkingCheck.Checked;
        _formContent.Add(enableThinkingCheck);
        row++;

        skipAgentPromptCheck = new CheckBox("Skip agent prompt (faster first turn)", options.SkipAgentPrompt) { X = 1, Y = row };
        skipAgentPromptCheck.Toggled += (_) => _options.SkipAgentPrompt = skipAgentPromptCheck.Checked;
        _formContent.Add(skipAgentPromptCheck);
        row++;

        disableToolsCheck = new CheckBox("Disable tools (no tool JSON / tool loop)", options.DisableTools) { X = 1, Y = row };
        disableToolsCheck.Toggled += (_) => _options.DisableTools = disableToolsCheck.Checked;
        _formContent.Add(disableToolsCheck);
        row++;

        AddLabel("Context compaction:");
        compactorRadio = new RadioGroup([.. Enum.GetNames<CompactorStrategy>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.Compactor };
        compactorRadio.SelectedItemChanged += (args) => _options.Compactor = (CompactorStrategy)args.SelectedItem;
        _formContent.Add(compactorRadio);
        row += Enum.GetValues<CompactorStrategy>().Length + 2;

        AddLabel("File access:");
        fileAccessRadio = new RadioGroup([.. Enum.GetNames<ToolPermission>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.FileAccess };
        fileAccessRadio.SelectedItemChanged += (args) => _options.FileAccess = (ToolPermission)args.SelectedItem;
        _formContent.Add(fileAccessRadio);
        row += Enum.GetValues<ToolPermission>().Length + 1;

        AddLabel("Network access:");
        networkAccessRadio = new RadioGroup([.. Enum.GetNames<ToolPermission>().Select(p => (ustring)p)]) { X = 30, Y = row, SelectedItem = (int)options.NetworkAccess };
        networkAccessRadio.SelectedItemChanged += (args) => _options.NetworkAccess = (ToolPermission)args.SelectedItem;
        _formContent.Add(networkAccessRadio);
        row += Enum.GetValues<ToolPermission>().Length + 1;

        AddLabel("Skill folders (;-sep):");
        skillField = new TextField((ustring)string.Join(";", options.SkillFolders)) { X = 30, Y = row, Width = 40 };
        skillField.TextChanged += (_) => _options.SkillFolders = [.. (skillField.Text.ToString() ?? string.Empty).Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)];
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
        toolField = new TextField((ustring)string.Join(";", options.ToolAssemblyPaths)) { X = 30, Y = row, Width = 40 };
        toolField.TextChanged += (_) => _options.ToolAssemblyPaths = [.. (toolField.Text.ToString() ?? string.Empty).Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)];
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
        toolsFolderField = new TextField((ustring)(options.ToolsFolder ?? "")) { X = 30, Y = row, Width = 40 };
        toolsFolderField.TextChanged += (_) => _options.ToolsFolder = string.IsNullOrWhiteSpace(toolsFolderField.Text.ToString()) ? null : toolsFolderField.Text.ToString();
        var toolsFolderOpen = new Button("Open...") { X = Pos.Right(toolsFolderField) + 1, Y = row };
        toolsFolderOpen.Clicked += () =>
        {
            var picked = FilePickerDialog.Show("Select tools folder", _options.ToolsFolder ?? _options.ProjectPath ?? Directory.GetCurrentDirectory(), PickerMode.Folder);
            if (picked is not null) { toolsFolderField.Text = picked; _options.ToolsFolder = picked; }
        };
        _formContent.Add(toolsFolderField, toolsFolderOpen);
        row += 2;

        var manageToolsBtn = new Button("Manage Tools...") { X = 1, Y = row, Width = 19 };
        manageToolsBtn.Clicked += ManageTools;
        var manageProcessorsBtn = new Button("Manage Pre/Post...") { X = Pos.Right(manageToolsBtn) + 2, Y = row, Width = 19 };
        manageProcessorsBtn.Clicked += ManageProcessors;
        _formContent.Add(manageToolsBtn, manageProcessorsBtn);
        row += 2;

        // --- Plugin status -------------------------------------------------
        pluginResult = PluginLoader.LoadFrom(Path.Combine(AppContext.BaseDirectory, "plugins"));
        _pluginCompactors.Clear();
        _pluginCompactors.AddRange(pluginResult.Compactors);
        if (_embedded is not null && _embedded.HasPlugins)
            _pluginCompactors.AddRange(_embedded.Plugins.Compactors);

        string embeddedNote = _embedded is not null && _embedded.HasPlugins
            ? $", {_embedded.AssemblyNames.Count} embedded ({string.Join(", ", _embedded.AssemblyNames)})"
            : "";
        var pluginLabel = new Label((ustring)$"Plugins: {pluginResult.Compactors.Count} compactors, {pluginResult.PreProcessors.Count} pre, {pluginResult.PostProcessors.Count} post, {pluginResult.Generators.Count} generators{embeddedNote}")
        { X = 1, Y = row };
        _formContent.Add(pluginLabel);
        if (pluginResult.Warnings.Count > 0)
        {
            var warnLabel = new Label((ustring)$"  ! {pluginResult.Warnings.Count} plugin warning(s)")
            { X = 1, Y = row + 1 };
            _formContent.Add(warnLabel);
            row++;
        }
        row++;

        // --- Plugin compactor selection ------------------------------------
        if (_pluginCompactors.Count > 0)
        {
            AddLabel("Plugin compactor:");
            pluginCompactorNames = [.. _pluginCompactors.Select(c => (ustring)(c.Name + (_embedded?.Plugins.Compactors.Contains(c) == true ? "  (Plugin Embedded)" : "")))];
            var selectedIdx = _options.PluginCompactorName is not null
                ? _pluginCompactors.FindIndex(c =>
                    string.Equals(c.Name, _options.PluginCompactorName, StringComparison.OrdinalIgnoreCase))
                : -1;
            pluginCompactorRadio = new RadioGroup(pluginCompactorNames) { X = 30, Y = row, SelectedItem = Math.Max(0, selectedIdx) };
            pluginCompactorRadio.SelectedItemChanged += (args) =>
                _options.PluginCompactorName = _pluginCompactors[args.SelectedItem].Name;
            _formContent.Add(pluginCompactorRadio);
            row += pluginCompactorNames.Length + 2;
        }

        row += 2; // reserved for fixed button bar below

        _formContent.Height = row;
        scrollView.ContentSize = new Terminal.Gui.Size(100, row);
        scrollView.Add(_formContent);
        Add(scrollView);

        var buttonBar = new View { X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill(), Height = 1 };
        var launchButton = new Button("Launch Session") { X = 1, Y = 0, IsDefault = true };
        launchButton.Clicked += () => onLaunch();
        var cancelButton = new Button("Cancel") { X = Pos.Right(launchButton) + 2, Y = 0 };
        cancelButton.Clicked += () => onCancel();
        var saveAsButton = new Button("Save Options As...") { X = Pos.Right(cancelButton) + 2, Y = 0 };
        saveAsButton.Clicked += SaveOptionsAs;
        var openPresetButton = new Button("Open Preset...") { X = Pos.Right(saveAsButton) + 2, Y = 0 };
        openPresetButton.Clicked += OpenPreset;
        var ResetPresetButton = new Button("Reset Defaults") { X = Pos.Right(openPresetButton) + 2, Y = 0 };
        ResetPresetButton.Clicked += ResetPreset;
        buttonBar.Add(launchButton, cancelButton, saveAsButton, openPresetButton, ResetPresetButton);
        Add(buttonBar);

        KeyPress += (args) =>
        {
            if (args.KeyEvent.Key == Key.Esc) { onCancel(); args.Handled = true; }
        };
    }

    private void SetOptionsFrom(SessionOptions options)
    {
        projectPathField.Text = (ustring)options.ProjectPath ?? "";
        generatorRadio.SelectedItem = (int)options.Generator;
        cacheRadio.SelectedItem = (int)options.Cache;
        loadModeRadio.SelectedItem = (int)options.LoadMode;
        formatterRadio.SelectedItem = (int)options.Formatter;
        hwRadio.SelectedItem = (int)options.HardwareTier;
        gpuCheck.Checked = options.UseGpu;
        gpuNonQuantCheck.Checked = options.GpuNonQuant;
        gpuVecDotCheck.Checked = options.GpuVecDot;
        gpuMatMulCheck.Checked = options.GpuMatMul;
        parallelCheck.Checked = options.UseParallelKernels;
        tempField.Text = (ustring)_options.Sampling.Temperature.ToString("F2");
        topKField.Text = (ustring)_options.Sampling.TopK.ToString("F2");
        topPField.Text = (ustring)_options.Sampling.TopP.ToString("F2");
        repetitionPenaltyField.Text = (ustring)_options.Generation.RepetitionPenalty.ToString("F2");
        repetitionWindowField.Text = (ustring)_options.Generation.RepetitionWindow.ToString("F2");
        maxTokensField.Text = (ustring)_options.Generation.MaxNewTokens.ToString();
        maxContextTokensField.Text = (ustring)(_options.MaxTokens?.ToString() ?? "0");
        agentNameField.Text = (ustring)_options.AgentName;
        userNameField.Text = (ustring)_options.UserName;
        agentsCheck.Checked = options.AgentsEnabled;
        enableThinkingCheck.Checked = options.EnableThinking;
        skipAgentPromptCheck.Checked = options.SkipAgentPrompt;
        disableToolsCheck.Checked = options.DisableTools;
        compactorRadio.SelectedItem = (int)options.Compactor;
        fileAccessRadio.SelectedItem = (int)options.FileAccess;
        networkAccessRadio.SelectedItem = (int)options.NetworkAccess;
        skillField.Text = (ustring)string.Join(";", options.SkillFolders);
        toolField.Text = (ustring)string.Join(";", options.ToolAssemblyPaths);
        toolsFolderField.Text = (ustring)options.ToolsFolder??string.Empty;
        var selectedIdx = _options.PluginCompactorName is not null ? 
            _pluginCompactors.FindIndex(c =>
            string.Equals(c.Name, _options.PluginCompactorName, StringComparison.OrdinalIgnoreCase))
            : -1;
        pluginCompactorRadio?.SelectedItem = Math.Max(0, selectedIdx);
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
        SetOptionsFrom(_options);
        MessageBox.Query("Preset loaded", $"Loaded \"{loaded.Name}\".", "OK");
    }

    private void ResetPreset()
    {
        var mPath = _options.ModelPath;
        var options = SessionOptions.Default;
        options.ModelPath = mPath;
        options.ProjectPath = _settings.ResolvedModelFolder;
        options.ToolsFolder = _settings.ToolsFolder;
        CopyOptionsInto(_options, options);
        SetOptionsFrom(_options);
        MessageBox.Query("Defaults Reset", $"Defaults have be reset.", "OK");
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

    private void ManageProcessors()
    {
        var pre = new List<(string Name, string Desc)>
        {
            ("Simple Artifact Injection", "Inlines text artifacts; adds path hints for binary files")
        };
        pre.AddRange(SessionLauncher.GetAvailablePreProcessors()
            .Select(p => (p.Name, p.Description)));
        if (_embedded is not null && _embedded.HasPlugins)
            pre.AddRange(_embedded.Plugins.PreProcessors.Select(p => (p.Name, p.Description + "  (Plugin Embedded)")));

        var post = SessionLauncher.GetAvailablePostProcessors()
            .Select(p => (p.Name, p.Description)).ToList();
        if (_embedded is not null && _embedded.HasPlugins)
            post.AddRange(_embedded.Plugins.PostProcessors.Select(p => (p.Name, p.Description + "  (Plugin Embedded)")));

        int totalItems = pre.Count + post.Count + (pre.Count > 0 && post.Count > 0 ? 1 : 0);
        if (totalItems == 0)
        {
            MessageBox.Query("No processors", "No pre/post processors are available.", "OK");
            return;
        }

        var dialog = new Dialog("Manage Pre/Post Processors", 60, 20);
        var scroll = new ScrollView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        var content = new View { Width = Dim.Fill() };

        int row = 0;
        var preLabel = new Label("Pre-processors:") { X = 1, Y = row };
        content.Add(preLabel);
        row++;
        foreach (var (name, desc) in pre)
        {
            var cb = new CheckBox($"  {name}  ({desc})", !_options.DisabledPreProcessors.Contains(name)) { X = 1, Y = row };
            cb.Toggled += (_) =>
            {
                if (cb.Checked) _options.DisabledPreProcessors.Remove(name);
                else _options.DisabledPreProcessors.Add(name);
            };
            content.Add(cb);
            row++;
        }

        if (post.Count > 0)
        {
            row++;
            var postLabel = new Label("Post-processors:") { X = 1, Y = row };
            content.Add(postLabel);
            row++;
            foreach (var (name, desc) in post)
            {
                var cb = new CheckBox($"  {name}  ({desc})", !_options.DisabledPostProcessors.Contains(name)) { X = 1, Y = row };
                cb.Toggled += (_) =>
                {
                    if (cb.Checked) _options.DisabledPostProcessors.Remove(name);
                    else _options.DisabledPostProcessors.Add(name);
                };
                content.Add(cb);
                row++;
            }
        }

        content.Height = row;
        scroll.ContentSize = new Terminal.Gui.Size(58, row);
        scroll.Add(content);
        dialog.Add(scroll);
        var closeBtn = new Button("Close");
        closeBtn.Clicked += () => Application.RequestStop();
        dialog.AddButton(closeBtn);

        Application.Run(dialog);
    }

    private static void CopyOptionsInto(SessionOptions target, SessionOptions source) => source.CopyTo(target);
}
