using NStack;
using SharpMind.CUI.App;
using SharpMind.Data.Metadata;
using SharpMind.Data.Sources;
using SharpMind.Model.Config;
using SharpMind.Training;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// The training job wizard: configure multiple data sources (each with its own
/// per-source stages), global stages, model size (manual or auto-sized), the
/// hyperparameters, and checkpoint/export settings, then save the job or start
/// training. The checkpoint directory is always derived from the export path.
/// Loading a saved job re-fills the form so it can be edited and resumed from a
/// checkpoint. Rendered as a scrollable form mirroring <see cref="OptionsView"/>.
/// </summary>
public sealed class TrainingWizardView : View
{
    private readonly AppSettings _settings;
    private TrainJobSettings _job;
    private readonly Action<TrainJobSettings> _onStart;
    private readonly Action? _onCancel;
    private string _savedPath = "";

    private readonly Label _jobPathLabel;
    private readonly Label _sourceLabel;
    private readonly Label _checkpointDirLabel;
    private readonly ListView _sourceList;
    private readonly ListView _stageList;
    private readonly ListView _pluginList;
    private readonly TextField _nameField;
    private readonly TextField _keepCountField;
    private readonly TextField _exportField;
    private readonly TextField _systemPromptField;
    private readonly TextField _skillsFolderField;
    private readonly RadioGroup _qatRadio;
    private readonly RadioGroup _keepModeRadio;
    private readonly Dictionary<string, TextField> _modelRows = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TextField> _hyperRows = new(StringComparer.Ordinal);

    private readonly string[] QatLabelsArr = ["F32 (off)", "F16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2"];
    private readonly string[] KeepLabelsArr = ["All", "Fixed", "None"];

    private IList<JobComponent>? SelectedStages
    {
        get
        {
            int i = _sourceList.SelectedItem;
            if (i < 0 || i >= _job.Sources.Count) return null;
            return _job.Sources[i].Stages;
        }
    }

    private JobSource? SelectedSource
        => _sourceList.SelectedItem >= 0 && _sourceList.SelectedItem < _job.Sources.Count
            ? _job.Sources[_sourceList.SelectedItem]
            : null;

    public TrainingWizardView(AppSettings settings, TrainJobSettings? job, Action<TrainJobSettings> onStart, Action? onCancel = null)
    {
        _settings = settings;
        _job = job ?? NewJob(settings);
        _onStart = onStart;
        _onCancel = onCancel;

        var scroll = new ScrollView
        {
            X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(1),
            ShowVerticalScrollIndicator = true, ShowHorizontalScrollIndicator = false
        };
        var form = new View { X = 0, Y = 0, Width = Dim.Fill() };
        int row = 0;

        Label AddLabel(string text)
        {
            var l = new Label((ustring)text) { X = 1, Y = row };
            form.Add(l);
            return l;
        }

        // --- Job header -------------------------------
        _jobPathLabel = new Label("New job") { X = 1, Y = row, Width = Dim.Fill(2) };
        form.Add(_jobPathLabel);
        row += 1;

        _nameField = new TextField((ustring)_job.Name) { X = 30, Y = row, Width = 30 };
        _nameField.TextChanged += (_) => _job.Name = string.IsNullOrWhiteSpace(_nameField.Text.ToString()) ? "Untitled" : _nameField.Text.ToString();
        var saveBtn = new Button("Save job") { X = 62, Y = row };
        saveBtn.Clicked += SaveJob;
        var loadBtn = new Button("Load job…") { X = Pos.Right(saveBtn) + 1, Y = row };
        loadBtn.Clicked += LoadJob;
        form.Add(AddLabel("Model name:"), _nameField, saveBtn, loadBtn);
        row += 2;

        // --- Sources --------------------------------
        var addSource = new Button("Add source…") { X = 30, Y = row };
        addSource.Clicked += ChooseSource;
        var editSource = new Button("Edit") { X = Pos.Right(addSource) + 1, Y = row };
        editSource.Clicked += EditSource;
        var removeSource = new Button("Remove") { X = Pos.Right(editSource) + 1, Y = row };
        removeSource.Clicked += RemoveSource;
        _sourceList = new ListView { X = 30, Y = row + 1, Width = 46, Height = 3 };
        _sourceList.OpenSelectedItem += (_) => EditSource();
        _sourceList.SelectedItemChanged += (_) => RefreshStages();
        form.Add(AddLabel("Data sources:"), addSource, editSource, removeSource, _sourceList);
        row += 5;

        // --- Stage chain (per current source) ----------
        _sourceLabel = new Label("Stages for: —") { X = 1, Y = row, Width = 28 };
        _stageList = new ListView { X = 30, Y = row + 1, Width = 46, Height = 3 };
        _stageList.OpenSelectedItem += (_) => EditStage();
        var addStage = new Button("Add stage…") { X = 30, Y = row };
        addStage.Clicked += AddStage;
        var editStage = new Button("Edit") { X = Pos.Right(addStage) + 1, Y = row };
        editStage.Clicked += EditStage;
        var upStage = new Button("Up") { X = Pos.Right(editStage) + 1, Y = row };
        upStage.Clicked += () => Move(SelectedStages, _stageList, +1);
        var downStage = new Button("Down") { X = Pos.Right(upStage) + 1, Y = row };
        downStage.Clicked += () => Move(SelectedStages, _stageList, -1);
        var delStage = new Button("Remove") { X = Pos.Right(downStage) + 1, Y = row };
        delStage.Clicked += RemoveStage;
        form.Add(_sourceLabel, addStage, editStage, upStage, downStage, delStage, _stageList);
        row += 5;

        // --- Model size -------------------------------
        row += 1;
        var autoBtn = new Button("Auto-size model…") { X = 30, Y = row };
        autoBtn.Clicked += AutoSize;
        form.Add(AddLabel("Model size:"), autoBtn);
        row += 2;

        row = NumRow(form, row, "Hidden dim:", _job.HiddenDim, v => _job.HiddenDim = v, _modelRows);
        row = NumRow(form, row, "Layers:", _job.NumLayers, v => _job.NumLayers = v, _modelRows);
        row = NumRow(form, row, "Heads:", _job.NumHeads, v => _job.NumHeads = v, _modelRows);
        row = NumRow(form, row, "KV heads:", _job.NumKvHeads, v => _job.NumKvHeads = v, _modelRows);
        row = NumRow(form, row, "FFN dim:", _job.FfnDim, v => _job.FfnDim = v, _modelRows);
        row = NumRow(form, row, "Context (max seq):", _job.MaxSeqLen, v => _job.MaxSeqLen = v, _modelRows);
        row = NumRow(form, row, "Vocab target:", _job.TokenizerVocabSize, v => _job.TokenizerVocabSize = v, _modelRows);
        row += 1;

        // --- Training hyperparameters -----------------
        row = NumRow(form, row, "Seq len:", _job.SeqLen, v => _job.SeqLen = v, _hyperRows);
        row = NumRow(form, row, "Batch size:", _job.BatchSize, v => _job.BatchSize = v, _hyperRows);
        row = NumRow(form, row, "Total steps:", _job.TotalSteps, v => _job.TotalSteps = v, _hyperRows);
        row = FloatRow(form, row, "Learning rate:", _job.LearningRate, v => _job.LearningRate = v, _hyperRows);
        row = NumRow(form, row, "Warmup steps:", _job.WarmupSteps, v => _job.WarmupSteps = v, _hyperRows);
        row = FloatRow(form, row, "Grad clip norm:", _job.GradClipNorm, v => _job.GradClipNorm = v, _hyperRows);
        row = FloatRow(form, row, "Label smoothing:", _job.LabelSmoothing, v => _job.LabelSmoothing = v, _hyperRows);
        row += 1;

        // --- QAT target -----------------------------------
        int qatIndex = QatIndexFor(_job.QuantAwareTraining);
        _qatRadio = new RadioGroup(QatLabelsArr.Select(q => (ustring)q).ToArray()) { X = 30, Y = row, SelectedItem = qatIndex };
        _qatRadio.SelectedItemChanged += (a) =>
            _job.QuantAwareTraining = QatStoredName(a.SelectedItem);
        form.Add(AddLabel("QAT target:"), _qatRadio);
        row += QatLabelsArr.Length + 1;

        // --- Checkpoints + export -------------------------
        row = NumRow(form, row, "Checkpoint interval:", _job.CheckpointInterval, v => _job.CheckpointInterval = v, _hyperRows);

        int keepIndex = KeepModeIndexFor(_job.KeepRecent);
        _keepModeRadio = new RadioGroup(KeepLabelsArr.Select(k => (ustring)k).ToArray())
        {
            X = 30, Y = row, SelectedItem = keepIndex
        };
        _keepCountField = new TextField((ustring)(_job.KeepRecent > 0 ? _job.KeepRecent : 3).ToString())
        {
            X = 42, Y = row, Width = 6, Visible = keepIndex == 1
        };
        _keepModeRadio.SelectedItemChanged += (a) =>
        {
            _job.KeepRecent = KeepRecentFromSelection(a.SelectedItem);
            _keepCountField.Visible = a.SelectedItem == 1;
            _checkpointDirLabel.Text = $"Checkpoints (derived): {_job.CheckpointDir}";
            SetNeedsDisplay();
        };
        _keepCountField.TextChanged += (_) =>
        {
            if (_job.KeepRecent > 0 && int.TryParse(_keepCountField.Text.ToString(), out var n) && n > 0)
                _job.KeepRecent = n;
        };
        form.Add(AddLabel("Keep recent ckpts:"), _keepModeRadio, _keepCountField);
        row += KeepLabelsArr.Length + 1;

        _checkpointDirLabel = new Label("") { X = 1, Y = row, Width = Dim.Fill(3) };
        form.Add(_checkpointDirLabel);
        row += 1;

        _exportField = new TextField((ustring)(_job.ExportPath ?? "")) { X = 30, Y = row, Width = 34 };
        _exportField.TextChanged += (_) =>
        {
            _job.ExportPath = string.IsNullOrWhiteSpace(_exportField.Text.ToString()) ? null : _exportField.Text.ToString();
            RefreshMap();
        };
        var expBrowse = new Button("Browse…") { X = Pos.Right(_exportField) + 1, Y = row };
        expBrowse.Clicked += () => BrowseFolder("Export .smm", _exportField, s => _job.ExportPath = s);
        form.Add(AddLabel("Export .smm:"), _exportField, expBrowse);
        row += 2;

        // --- Embed in the exported .smm --------------------------
        row += 1;
        _systemPromptField = new TextField((ustring)(_job.SystemPromptPath ?? "")) { X = 30, Y = row, Width = 34 };
        _systemPromptField.TextChanged += (_) =>
            _job.SystemPromptPath = string.IsNullOrWhiteSpace(_systemPromptField.Text.ToString()) ? null : _systemPromptField.Text.ToString();
        var promptBrowse = new Button("Browse…") { X = Pos.Right(_systemPromptField) + 1, Y = row };
        promptBrowse.Clicked += () =>
        {
            string start = Directory.Exists(_systemPromptField.Text.ToString())
                ? Path.GetDirectoryName(_systemPromptField.Text.ToString())!
                : Directory.GetCurrentDirectory();
            var picked = FilePickerDialog.Show("System prompt file (.txt/.md)", start, PickerMode.File,
                patterns: ["*.txt", "*.md"]);
            if (picked is not null) { _systemPromptField.Text = picked; _job.SystemPromptPath = picked; }
        };
        form.Add(AddLabel("System prompt:"), _systemPromptField, promptBrowse);
        row += 2;

        _skillsFolderField = new TextField((ustring)(_job.SkillsFolder ?? "")) { X = 30, Y = row, Width = 34 };
        _skillsFolderField.TextChanged += (_) =>
            _job.SkillsFolder = string.IsNullOrWhiteSpace(_skillsFolderField.Text.ToString()) ? null : _skillsFolderField.Text.ToString();
        var skillsBrowse = new Button("Browse…") { X = Pos.Right(_skillsFolderField) + 1, Y = row };
        skillsBrowse.Clicked += () => BrowseFolder("Skills folder (.md)", _skillsFolderField, s => _job.SkillsFolder = s);
        form.Add(AddLabel("Skills folder:"), _skillsFolderField, skillsBrowse);
        row += 2;

        var addPlugin = new Button("Add DLL…") { X = 30, Y = row };
        addPlugin.Clicked += AddPluginDll;
        var removePlugin = new Button("Remove") { X = Pos.Right(addPlugin) + 1, Y = row };
        removePlugin.Clicked += RemovePluginDll;
        _pluginList = new ListView { X = 30, Y = row + 1, Width = 46, Height = 3 };
        _pluginList.OpenSelectedItem += (_) => RemovePluginDll();
        form.Add(AddLabel("Plugin DLLs:"), addPlugin, removePlugin, _pluginList);
        row += 5;

        // --- Start ------------------------------------------
        var start = new Button("Start training…") { X = 1, Y = row, IsDefault = true };
        start.Clicked += StartTraining;
        form.Add(start);
        var cancel = new Button("Cancel") { X = Pos.Right(start) + 2, Y = row };
        cancel.Clicked += () => _onCancel?.Invoke();
        form.Add(cancel);
        row += 2;

        form.Height = row;
        scroll.ContentSize = new Terminal.Gui.Size(86, row);
        scroll.Add(form);
        Add(scroll);

        _nameField.SetFocus();
        RefreshAll();
    }

    // --- row helpers --------------------------------------------------------

    private static int NumRow(View form, int row, string label, int initial, Action<int> set, Dictionary<string, TextField> rows)
        => FieldRow(form, row, label, initial.ToString(), s => int.TryParse(s, out _), set, rows);

    private static int FloatRow(View form, int row, string label, float initial, Action<float> set, Dictionary<string, TextField> rows)
        => FieldRow(form, row, label, initial.ToString("0.#####"),
            s => float.TryParse(s, System.Globalization.CultureInfo.InvariantCulture, out _), set, rows);

    private static int FieldRow(
        View form, int row, string label, string initial,
        Func<string, bool> validate, Delegate set, Dictionary<string, TextField> rows)
    {
        var field = new TextField((ustring)initial) { X = 30, Y = row, Width = 12 };
        field.TextChanged += (_) =>
        {
            string text = field.Text.ToString();
            if (!validate(text)) return;
            if (set is Action<int> setInt && int.TryParse(text, out var i)) { setInt(i); return; }
            if (set is Action<float> setFloat && float.TryParse(text, System.Globalization.CultureInfo.InvariantCulture, out var f)) setFloat(f);
        };
        rows[label] = field;
        form.Add(new Label((ustring)label) { X = 1, Y = row }, field);
        return row + 1;
    }

    // --- actions -------------------------------------------------------------

    private void RefreshAll()
    {
        _jobPathLabel.Text = string.IsNullOrEmpty(_savedPath) ? "New job (not yet saved)" : $"Saved: {Path.GetFileName(_savedPath)}";
        _nameField.Text = (ustring)(_job.Name ?? "");
        RefreshMap();
        RefreshSources();
        RefreshStages();
        // keep count field reflects the current KeepRecent
        int keepIndex = KeepModeIndexFor(_job.KeepRecent);
        _keepModeRadio.SelectedItem = keepIndex;
        _keepCountField.Visible = keepIndex == 1;
        if (keepIndex == 1) _keepCountField.Text = (ustring)(_job.KeepRecent > 0 ? _job.KeepRecent : 3).ToString();
        _qatRadio.SelectedItem = QatIndexFor(_job.QuantAwareTraining);
        _systemPromptField.Text = (ustring)(_job.SystemPromptPath ?? "");
        _skillsFolderField.Text = (ustring)(_job.SkillsFolder ?? "");
        RefreshPlugins();
    }

    private void RefreshPlugins()
    {
        _pluginList.SetSource(_job.PluginDllPaths.Count == 0
            ? [(ustring)"(none — use Add DLL…)"]
            : _job.PluginDllPaths.Select(p => (ustring)Path.GetFileName(p)).ToArray());
    }

    private void AddPluginDll()
    {
        string start = _job.PluginDllPaths.FirstOrDefault(p => File.Exists(p)) is { } existing
            ? Path.GetDirectoryName(existing) ?? Directory.GetCurrentDirectory()
            : Directory.GetCurrentDirectory();
        var picked = FilePickerDialog.Show("Add tool DLL", start, PickerMode.File, "*.dll");
        if (picked is null) return;
        if (!_job.PluginDllPaths.Contains(picked, StringComparer.OrdinalIgnoreCase))
        {
            _job.PluginDllPaths.Add(picked);
            RefreshPlugins();
        }
    }

    private void RemovePluginDll()
    {
        int i = _pluginList.SelectedItem;
        if (i < 0 || i >= _job.PluginDllPaths.Count) return;
        _job.PluginDllPaths.RemoveAt(i);
        RefreshPlugins();
    }

    private void RefreshMap()
    {
        foreach (var (label, field) in _modelRows)
        {
            field.Text = label switch
            {
                "Hidden dim:" => _job.HiddenDim.ToString(),
                "Layers:" => _job.NumLayers.ToString(),
                "Heads:" => _job.NumHeads.ToString(),
                "KV heads:" => _job.NumKvHeads.ToString(),
                "FFN dim:" => _job.FfnDim.ToString(),
                "Context (max seq):" => _job.MaxSeqLen.ToString(),
                "Vocab target:" => _job.TokenizerVocabSize.ToString(),
                _ => field.Text.ToString(),
            };
        }
        _checkpointDirLabel.Text = $"Checkpoints (derived): {_job.CheckpointDir}";
    }

    private void RefreshSources()
    {
        _sourceList.SetSource(_job.Sources.Select((s, i) => (ustring)(s.DisplayName ?? $"(source {i + 1})")).ToArray());
        if (SelectedSource is null && _job.Sources.Count > 0) _sourceList.SelectedItem = 0;
        RefreshStages();
    }

    private void RefreshStages()
    {
        var stages = SelectedStages;
        if (stages is null)
        {
            _sourceLabel.Text = "Stages for: —";
            _stageList.SetSource(Array.Empty<ustring>());
            return;
        }
        var src = SelectedSource!;
        _sourceLabel.Text = $"Stages for: {src.Component.DisplayName ?? src.Component.TypeName.Split(',')[0]}";
        _stageList.SetSource(stages.Select((s, i) => (ustring)$"{i + 1}. {s.DisplayName ?? s.TypeName.Split(',')[0]}").ToArray());
        _stageList.SelectedItem = Math.Min(_stageList.SelectedItem, Math.Max(0, stages.Count - 1));
    }

    private PickedComponent? Pick(ComponentKind kind, IReadOnlyDictionary<string, string>? prefill = null)
        => ComponentPickerDialog.Show(_settings.PluginsFolder, kind, prefill);

    private JobComponent ToJobComponent(PickedComponent picked) => new()
    {
        DisplayName = picked.Descriptor.Name,
        TypeName = picked.Descriptor.Type.AssemblyQualifiedName!,
        Args = picked.Values,
    };

    private void ChooseSource()
    {
        var picked = Pick(ComponentKind.Source);
        if (picked is null) return;
        _job.Sources.Add(new JobSource { Component = ToJobComponent(picked) });
        _sourceList.SelectedItem = _job.Sources.Count - 1;
        RefreshSources();
    }

    private void EditSource()
    {
        var source = SelectedSource;
        if (source is null) return;
        var registry = ComponentRegistry.ScanFolder(_settings.PluginsFolder, out _);
        var descriptor = ComponentRegistry.Find(source.Component.TypeName, registry);
        if (descriptor is null)
        {
            MessageBox.ErrorQuery("Edit source", $"Unknown source: {source.Component.TypeName}", "OK");
            return;
        }
        var updated = ComponentParamDialog.Show(descriptor, source.Component.Args);
        if (updated is null) return;
        source.Component.Args = updated;
        RefreshSources();
    }

    private void RemoveSource()
    {
        int i = _sourceList.SelectedItem;
        if (i < 0 || i >= _job.Sources.Count) return;
        _job.Sources.RemoveAt(i);
        if (_job.Sources.Count > 0) _sourceList.SelectedItem = Math.Min(i, _job.Sources.Count - 1);
        RefreshSources();
    }

    private void AddStage()
    {
        var stages = SelectedStages;
        if (stages is null)
        {
            MessageBox.ErrorQuery("Add stage", "Select a data source first — each source has its own stage chain.", "OK");
            return;
        }
        var picked = Pick(ComponentKind.Stage);
        if (picked is null) return;
        stages.Add(ToJobComponent(picked));
        _stageList.SelectedItem = stages.Count - 1;
        RefreshStages();
    }

    private void EditStage()
    {
        var stages = SelectedStages;
        if (stages is null) return;
        int i = _stageList.SelectedItem;
        if (i < 0 || i >= stages.Count) return;
        var comp = stages[i];
        var registry = ComponentRegistry.ScanFolder(_settings.PluginsFolder, out _);
        var descriptor = ComponentRegistry.Find(comp.TypeName, registry);
        if (descriptor is null)
        {
            MessageBox.ErrorQuery("Edit stage", $"Unknown stage: {comp.TypeName}", "OK");
            return;
        }
        var updated = ComponentParamDialog.Show(descriptor, comp.Args);
        if (updated is null) return;
        comp.Args = updated;
        RefreshStages();
    }

    private void RemoveStage()
    {
        var stages = SelectedStages;
        if (stages is null) return;
        int i = _stageList.SelectedItem;
        if (i < 0 || i >= stages.Count) return;
        stages.RemoveAt(i);
        RefreshStages();
    }

    private void Move(List<JobComponent>? stages, ListView view, int delta)
    {
        if (stages is null) return;
        int i = view.SelectedItem;
        int j = i + delta;
        if (i < 0 || i >= stages.Count || j < 0 || j >= stages.Count) return;
        (stages[i], stages[j]) = (stages[j], stages[i]);
        view.SelectedItem = j;
        RefreshStages();
    }

    private void Move(IList<JobComponent>? stages, ListView view, int delta) => Move(stages as List<JobComponent>, view, delta);

    // --- Keep-recent / QAT mappings ----------------------------------------

    private int KeepModeIndexFor(int keepRecent)
    {
        if (keepRecent == 0) return 0;   // All
        if (keepRecent < 0) return 2;    // None
        return 1;                        // Fixed
    }

    private int KeepRecentFromSelection(int index) => index switch
    {
        0 => 0,          // All
        2 => -1,         // None
        _ => int.TryParse(_keepCountField.Text.ToString(), out var n) && n > 0 ? n : 3,
    };

    private int QatIndexFor(string? qat) => qat switch
    {
        null or "" or "F32" => 0,
        "F16" => 1,
        "Q8_K" or "Q8_0" => 2,   // legacy Q8_0 jobs display as Q8_K
        "Q6_K" => 3,
        "Q5_K" => 4,
        "Q4_K" or "Q4_0" => 5,   // legacy Q4_0 jobs display as Q4_K
        "Q3_K" => 6,
        "Q2_K" => 7,
        _ => 0,
    };

    private static string? QatStoredName(int index) => index switch
    {
        0 => null,
        1 => "F16",
        2 => "Q8_K",
        3 => "Q6_K",
        4 => "Q5_K",
        5 => "Q4_K",
        6 => "Q3_K",
        7 => "Q2_K",
        _ => null,
    };

    // --- Auto-size -----------------------------------------------------------

    private async void AutoSize()
    {
        if (_job.Sources.Count == 0)
        {
            MessageBox.ErrorQuery("Auto-size", "Add at least one data source first.", "OK");
            return;
        }

        IDataSource combined;
        try
        {
            var registry = ComponentRegistry.ScanFolder(_settings.PluginsFolder, out _);
            var built = new List<IDataSource>();
            foreach (var src in _job.Sources)
            {
                var descriptor = ComponentRegistry.Find(src.Component.TypeName, registry)
                    ?? throw new InvalidOperationException($"Unknown source: {src.Component.TypeName}");
                built.Add((IDataSource)ComponentRegistry.Build<IDataSource>(descriptor, src.Component.Args));
            }
            combined = built.Count == 1 ? built[0] : new CompositeSource(built);
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Auto-size", ex.Message, "OK");
            return;
        }

        var dialog = new Dialog("Auto-size", 60, 7);
        var statusLabel = new Label("Sampling corpus…") { X = 1, Y = 1, Width = Dim.Fill(2) };
        var cancelButton = new Button("Cancel") { X = 1, Y = Pos.AnchorEnd(1) };
        dialog.Add(statusLabel, cancelButton);

        ModelConfig? result = null;
        string? error = null;
        bool done = false;
        bool cancelled = false;

        var cts = new CancellationTokenSource();
        cancelButton.Clicked += () =>
        {
            cancelled = true;
            cts.Cancel();
        };

        var progress = new Progress<float>(p => Application.MainLoop.Invoke(() =>
            statusLabel.Text = $"Sizing… {p * 100:F0}%"));
        _ = Task.Run(async () =>
        {
            try
            {
                result = await ModelSizer.DetermineOptimalConfigAsync(combined, progress: progress, ct: cts.Token);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { error = ex.Message; }
            finally { done = true; }
        });

        object pollToken = null!;
        bool poll(MainLoop _)
        {
            if (done)
            {
                Application.MainLoop.RemoveTimeout(pollToken);
                Application.RequestStop();
                return false;
            }
            return true;
        }
        pollToken = Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(150), poll);
        Application.Run(dialog);

        if (cancelled) return;
        if (error is not null)
        {
            MessageBox.ErrorQuery("Auto-size", $"Sizing failed:\n{error}", "OK");
            return;
        }
        if (result is null) return;

        _job.AutoSized = true;
        _job.HiddenDim = result.HiddenDim;
        _job.NumLayers = result.NumLayers;
        _job.NumHeads = result.NumHeads;
        _job.NumKvHeads = result.NumKvHeads;
        _job.FfnDim = result.FfnDim;
        _job.MaxSeqLen = result.MaxSeqLen;
        RefreshMap();
        string summary = $"Hidden {result.HiddenDim}, {result.NumLayers} layers, FFN {result.FfnDim}, context {result.MaxSeqLen}.";
        MessageBox.Query("Auto-size", $"Model size set.\n{summary}\nVocab size stays {_job.TokenizerVocabSize}.", "OK");
    }

    // --- Save/Load --------------------------------------------------------

    private void SaveJob()
    {
        string path = Path.Combine(TrainJobSettings.DefaultFolder, Sanitize(_job.Name) + ".json");
        if (_job.Save(path, out var error))
        {
            _savedPath = path;
            _jobPathLabel.Text = $"Saved: {Path.GetFileName(path)}";
        }
        else MessageBox.ErrorQuery("Save job", error ?? "Unknown error", "OK");
    }

    private void LoadJob()
    {
        var files = TrainJobSettings.ListSaved();
        if (files.Count == 0)
        {
            MessageBox.ErrorQuery("Load job", "No saved training jobs found.", "OK");
            return;
        }
        var names = files.Select(f => (ustring)Path.GetFileNameWithoutExtension(f)).ToArray();
        int idx = 0;
        var dialog = new Dialog("Load training job", 50, Math.Min(12, files.Count + 5));
        var list = new ListView(names) { X = 1, Y = 0, Width = Dim.Fill(2), Height = Dim.Fill(3) };
        var ok = new Button("Load") { X = 1, Y = Pos.AnchorEnd(1), IsDefault = true };
        var cancel = new Button("Cancel") { X = Pos.Right(ok) + 2, Y = Pos.AnchorEnd(1) };
        ok.Clicked += () => { idx = list.SelectedItem; Application.RequestStop(); };
        cancel.Clicked += () => { idx = -1; Application.RequestStop(); };
        dialog.Add(list, ok, cancel);
        Application.Run(dialog);
        if (idx < 0 || idx >= files.Count) return;

        var loaded = TrainJobSettings.Load(files[idx], out var err);
        if (loaded is null) { MessageBox.ErrorQuery("Load job", err ?? "Not a valid job file.", "OK"); return; }
        _job = loaded;
        _savedPath = files[idx];
        RefreshAll();
        _nameField.SetFocus();
    }

    private void StartTraining()
    {
        if (_job.Sources.Count == 0)
        {
            MessageBox.ErrorQuery("Start training", "Add at least one data source first.", "OK");
            return;
        }
        if (!string.IsNullOrWhiteSpace(_job.SystemPromptPath) && !File.Exists(_job.SystemPromptPath))
        {
            MessageBox.ErrorQuery("Start training", $"System prompt file not found:\n{_job.SystemPromptPath}", "OK");
            return;
        }
        if (!string.IsNullOrWhiteSpace(_job.SkillsFolder))
        {
            if (!Directory.Exists(_job.SkillsFolder))
            {
                MessageBox.ErrorQuery("Start training", $"Skills folder not found:\n{_job.SkillsFolder}", "OK");
                return;
            }
            if (Directory.GetFiles(_job.SkillsFolder, "*.md", SearchOption.TopDirectoryOnly).Length == 0)
            {
                MessageBox.ErrorQuery("Start training", $"No *.md files found in:\n{_job.SkillsFolder}", "OK");
                return;
            }
        }
        foreach (var dll in _job.PluginDllPaths)
        {
            if (!File.Exists(dll))
            {
                MessageBox.ErrorQuery("Start training", $"Plugin DLL not found:\n{dll}", "OK");
                return;
            }
        }
        _onStart(_job);
    }

    private void BrowseFolder(string labelForPicker, TextField field, Action<string> set)
    {
        string start = Directory.Exists(field.Text.ToString()) ? field.Text.ToString() : Directory.GetCurrentDirectory();
        var picked = FilePickerDialog.Show(labelForPicker, start, PickerMode.Folder);
        if (picked is not null) { field.Text = picked; set(picked); }
    }

    private static TrainJobSettings NewJob(AppSettings settings) => new()
    {
        Name = $"{SharpMind.Inference.Agent.GreekTier.RandomDeity()}-{DateTime.Now:yyyyMMdd-HHmmss}",
        ExportPath = string.IsNullOrWhiteSpace(settings.LastExportPath) ? null : settings.LastExportPath,
    };

    private static string Sanitize(string name)
    {
        foreach (var c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return string.IsNullOrWhiteSpace(name) ? "job" : name.Trim();
    }
}