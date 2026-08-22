using NStack;
using SharpMind.CUI.App;
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// The application's single Window: a MenuBar pinned to the top, and a
/// content area that fills the rest, into which each "screen" gets swapped
/// via RemoveAll/Add. Owns the ModelCache (shared loaded models across
/// multiple named chat sessions) and the per-session PermissionGates
/// (Ask-mode file/network confirmation dialogs) — both need a place that
/// outlives any single screen, since sessions and their permission prompts
/// can outlast whatever screen happens to be showing at the time.
/// </summary>
public sealed class MainWindow : Window
{
    private readonly View _content;
    private readonly AppSettings _settings;
    private readonly ModelCache _modelCache = new();
    private readonly List<ChatSessionState> _sessions = [];
    private readonly Dictionary<Guid, PermissionGate> _activePermissionGates = [];

    private SessionOptions _options;
    private ChatSessionState? _currentSession;
    private TrainingProgressView? _activeTraining;
    private readonly object? _permissionPollToken;

    private static string LastUsedOptionsPath =>
        Path.Combine(SavedSession.DefaultFolder, "__last_used__.json");

    /// <summary>Returns the most recently saved session (excluding __last_used__.json), or null.</summary>
    private static (SavedSession session, string path)? FindLastSession()
    {
        try
        {
            var saved = SavedSession.ListSaved(SavedSession.DefaultFolder);
            // Prefer manually-saved sessions (named files), but fall back to
            // __last_used__.json when no manual saves exist — it now carries a
            // KV-cache snapshot on session launch.
            var manual = saved.FirstOrDefault(f =>
                !Path.GetFileName(f).Equals("__last_used__.json", StringComparison.OrdinalIgnoreCase));
            if (manual is not null)
                return (SavedSession.Load(manual, out _)!, manual);

            var autoSave = saved.FirstOrDefault(f =>
                Path.GetFileName(f).Equals("__last_used__.json", StringComparison.OrdinalIgnoreCase));
            if (autoSave is not null)
            {
                var session = SavedSession.Load(autoSave, out _);
                if (session?.Snapshot is not null)
                    return (session, autoSave);
            }
            return null;
        }
        catch { return null; }
    }

    private static string ModelNameFromPath(string? modelPath) =>
        Path.GetFileNameWithoutExtension(modelPath ?? "");

    public MainWindow()
    {
        Title = "SharpMind CUI";

        _settings = AppSettings.Load();
        SharpMindPaths.EnsureCreated();
        _options = LoadLastUsedOptions() ?? NewSessionOptionsFromSettings();

        ColorScheme = ThemeBuilder.Build(_settings.Theme);

        _content = new View
        {
            X = 0,
            Y = 1, // leave row 0 for the menu bar
            Width = Dim.Fill(),
            Height = Dim.Fill()
        };

        var menu = BuildMenuBar();

        Add(menu, _content);
        ShowWelcome();

        // Permission "Ask" dialogs can be posted by ANY open session's
        // background chat loop, not just whichever one is currently on
        // screen — so this polls at the MainWindow level, not inside
        // ChatView, the same way ModelCache's lifetime spans every screen.
        _permissionPollToken = Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(100), PollPermissionRequests);
    }

    private static void SaveLastUsedOptions(SessionOptions options, ChatSessionSnapshot? snapshot = null)
    {
        try
        {
            var dir = SavedSession.DefaultFolder;
            Directory.CreateDirectory(dir);
            var saved = new SavedSession { Name = "__last_used__", Options = options, Snapshot = snapshot };
            SavedSession.Save(saved, Path.Combine(dir, "__last_used__.json"), out _);
        }
        catch { /* best-effort save */ }
    }

    private static SessionOptions? LoadLastUsedOptions()
    {
        try
        {
            var path = LastUsedOptionsPath;
            if (!File.Exists(path)) return null;
            var loaded = SavedSession.Load(path, out _);
            return loaded?.Options;
        }
        catch { return null; }
    }

    private SessionOptions NewSessionOptionsFromSettings()
    {
        var options = SessionOptions.Default;
        options.ProjectPath = _settings.ResolvedModelFolder;
        options.ToolsFolder = _settings.ToolsFolder;
        return options;
    }

    private MenuBar BuildMenuBar()
    {
        return new MenuBar(
        [
            new("_File", new MenuItem[]
            {
                new("_New session...", "", StartNewSession, shortcut: Key.N | Key.CtrlMask),
                new("_Welcome screen", "", ShowWelcome),
                new("_Settings...", "", ShowSettings),
                new("_Resume last session...", "", ResumeLastSession),
                new("E_xit", "", () => Application.RequestStop())
            }),
            new("_Model", new MenuItem[]
            {
                new("_Browse for model...", "", ShowModelBrowser),
                new("_Debug session (no model)...", "", StartDebugSession),
                new("Modify _model metadata...", "", ShowModelMetaModifier),
                new("T_rain a model...", "", ShowTrainingWizard),
                new("_Quantize model...", "", ShowModelQuantizer),
                new("_Convert model...", "", ShowModelConverter)
            }),
            new("_Options", new MenuItem[]
            {
                new("_Session options...", "", ShowOptionsIfReachable)
            }),
            new("_Chat", new MenuItem[]
            {
                new("_Manage sessions...", "", ShowSessionManager),
                new("_Load session...", "", LoadSessionFromDisk),
                new("_Save current session...", "", SaveCurrentSession),
                new("Close _current session...", "", CloseCurrentSession),
                new("Toggle _show thinking", "", ToggleShowThinking),
                new("Toggle _enable_thinking (Qwen3)", "", ToggleTemplateThinking)
            }),
            new("_Help", new MenuItem[]
            {
                new("_About SharpMind…", "", ShowAbout)
            })
        ]);
    }

    // --- Screen navigation -------------------------------------------------

    private void SwapContent(View view)
    {
        foreach (var old in _content.Subviews.ToArray())
            old.Dispose();
        _content.RemoveAll();
        _content.Add(view);
        view.Width = Dim.Fill();
        view.Height = Dim.Fill();
        view.X = 0;
        view.Y = 0;
        _content.FocusFirst();
        _content.SetNeedsDisplay();
    }

    private void ShowWelcome()
    {
        var lastSession = FindLastSession();
        string lastModelName = _options.ModelPath is not null ? ModelNameFromPath(_options.ModelPath) : "";
        SwapContent(new WelcomeView(
            onBrowseModel: ShowModelBrowser,
            lastModelName: lastModelName,
            onContinueWithModel: _options.ModelPath is not null ? new Action(() =>
            {
                _options.ModelPath = _options.ModelPath!;
                ShowOptions();
            }) : null,
            lastSessionName: lastSession?.session.Name,
            onResumeLastSession: lastSession is not null ? new Action(() =>
            {
                _options = CloneOptions(lastSession.Value.session.Options);
                _options.PendingSnapshot = lastSession.Value.session.Snapshot;
                _options.SourceFilePath = lastSession.Value.path;
                ShowOptions();
            }) : null,
            onTrainModel: ShowTrainingWizard,
            onSponsor: () => OpenUrl("https://github.com/sponsors/Integral2u")));
    }

    private void ShowModelBrowser()
    {
        Action onCancel = _currentSession is not null ? ShowChat : ShowWelcome;
        SwapContent(new ModelBrowserView(
            startPath: _options.ProjectPath ?? Directory.GetCurrentDirectory(),
            onChosen: path =>
            {
                _options.ModelPath = path;
                _options.ProjectPath ??= Path.GetDirectoryName(path);

                if (string.IsNullOrWhiteSpace(_settings.DefaultModelFolder))
                {
                    _settings.DefaultModelFolder = Path.GetDirectoryName(path);
                    _settings.Save(out _);
                }

                ShowOptions();
            },
            onCancel: onCancel,
            lastModelPath: _options.ModelPath));
    }

    /// <summary>
    /// Model → Convert model...: pick a source (.gguf or .smm), a destination,
    /// then run a cancellable conversion with a live progress dialog. The
    /// converters write to a temp file and only move it into place on success,
    /// so cancelling never leaves a partial output behind.
    /// </summary>
    private void ShowModelConverter()
    {
        string start = _options.ProjectPath ?? _settings.ResolvedModelFolder ?? Directory.GetCurrentDirectory();
        string? source = FilePickerDialog.Show("Convert model", start, PickerMode.File,
            patterns: ["*.gguf", "*.smm"]);
        if (source is null) return;

        bool toSmm = source.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        bool toGguf = source.EndsWith(".smm", StringComparison.OrdinalIgnoreCase);
        if (!toSmm && !toGguf)
        {
            MessageBox.ErrorQuery("Convert model", "Pick a .gguf or .smm model file.", "OK");
            return;
        }

        string baseName = Path.GetFileNameWithoutExtension(source);
        string defaultName = toSmm ? baseName + ".smm" : baseName + ".gguf";
        string? target = FilePickerDialog.Show(
            toSmm ? "Save as .smm" : "Save as .gguf",
            Path.GetDirectoryName(source) ?? Directory.GetCurrentDirectory(),
            PickerMode.SaveFile, defaultName: defaultName);
        if (target is null) return;

        // Force the correct extension so the target is unambiguous.
        if (!target.EndsWith(toSmm ? ".smm" : ".gguf", StringComparison.OrdinalIgnoreCase))
            target += toSmm ? ".smm" : ".gguf";

        if (File.Exists(target))
        {
            int overwrite = MessageBox.Query("Convert model", $"Overwrite existing file?\n{target}", "Yes", "No");
            if (overwrite != 0) return;
        }

        var cancelSource = new CancellationTokenSource();

        string? error = null;
        bool done = false;
        int progressPercent = -1;

        var dialog = new Dialog("Convert model", 60, 8);
        var statusLabel = new Label("Converting…") { X = 1, Y = 1, Width = Dim.Fill(2) };
        var cancelButton = new Button("Cancel") { X = Pos.AnchorEnd(12), Y = Pos.AnchorEnd(1) };
        cancelButton.Clicked += () => cancelSource.Cancel();
        dialog.Add(statusLabel, cancelButton);

        var taskProgress = new Progress<float>(p => Application.MainLoop.Invoke(() =>
        {
            int percent = Math.Clamp((int)(p * 100), 0, 100);
            if (percent != progressPercent)
            {
                progressPercent = percent;
                statusLabel.Text = $"Converting… {percent}%";
            }
        }));

        _ = Task.Run(() =>
        {
            try
            {
                if (toSmm)
                    GgufToSmmConverter.Convert(source, target, new SmmWriteOptions { Source = "gguf" },
                        taskProgress, cancelSource.Token);
                else
                    SmmToGufConverter.Convert(source, target, taskProgress, cancelSource.Token);
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
        pollToken = Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(100), poll);
        Application.Run(dialog);

        if (cancelSource.IsCancellationRequested)
        {
            MessageBox.Query("Convert model", "Conversion cancelled — no output was written.", "OK");
        }
        else if (error is not null)
        {
            MessageBox.ErrorQuery("Convert model", $"Conversion failed:\n{error}", "OK");
        }
        else if (File.Exists(target))
        {
            MessageBox.Query("Convert model",
                $"Converted {Path.GetFileName(source)} → {target}\n({new FileInfo(target).Length:N0} bytes)", "OK");
        }
    }

    /// <summary>
    /// Model → Quantize model...: re-quantize an existing .SMM file's weights to
    /// a K-quant level (or a byte budget), saving to a chosen destination using
    /// the same tmp-file + atomic-replace safety as the converter. A small level
    /// dialog picks between a fixed K-quant and the automatic budget-floor mode.
    /// </summary>
    private void ShowModelQuantizer()
    {
        string start = _options.ProjectPath ?? _settings.ResolvedModelFolder ?? Directory.GetCurrentDirectory();
        string? sourcePath = FilePickerDialog.Show("Quantize model", start, PickerMode.File, patterns: ["*.smm"]);
        if (sourcePath is null) return;

        var levels = new[] { QuantDType.Q8_K, QuantDType.Q6_K, QuantDType.Q5_K, QuantDType.Q4_K, QuantDType.Q3_K, QuantDType.Q2_K };
        string[] options = ["Q8_K (best quality)", "Q6_K", "Q5_K", "Q4_K (balanced)", "Q3_K", "Q2_K (smallest)", "Fit byte budget…"];

        var picker = new Dialog("Quantize model", 46, 12);
        var prompt = new Label("Quantize weights to:") { X = 1, Y = 1 };
        var list = new ListView(options) { X = 1, Y = 2, Width = 44, Height = options.Length };
        var trigger = new Action(() =>
        {
            int i = list.SelectedItem;
            if (i < 0 || i >= options.Length) return;
            if (i < 6)
            {
                Application.RequestStop();
                RunQuantize(sourcePath, levels[i], $"Quantize model — {levels[i]}");
            }
            else
            {
                Application.RequestStop();
                AskBudgetFloorAndRun(sourcePath);
            }
        });
        var okButton = new Button("OK") { X = 1, Y = options.Length + 3 };
        okButton.Clicked += trigger;
        var cancelButton = new Button("Cancel") { X = Pos.Right(okButton) + 2, Y = options.Length + 3 };
        cancelButton.Clicked += () => Application.RequestStop();
        list.OpenSelectedItem += (_) => trigger();
        picker.Add(prompt, list, okButton, cancelButton);
        Application.Run(picker);
    }

    private void AskBudgetFloorAndRun(string sourcePath)
    {
        string[] floors = ["Q8_K", "Q6_K", "Q5_K", "Q4_K (default)", "Q3_K", "Q2_K"];
        var picker = new Dialog("Quantize by budget", 54, 16);
        var prompt = new Label("Quality floor (never go coarser than):") { X = 1, Y = 1 };
        var floorList = new ListView(floors) { X = 1, Y = 2, Width = 52, Height = floors.Length };
        var budgetPrompt = new Label("Target size (MB):") { X = 1, Y = floors.Length + 4 };
        var budgetInput = new TextView { X = 16, Y = floors.Length + 4, Width = 12, Height = 1, Text = "512" };
        var run = new Action(() =>
        {
            int fi = floorList.SelectedItem;
            if (fi < 0) fi = 2; // default Q4_K when nothing was highlighted
            var floor = new[] { QuantDType.Q8_K, QuantDType.Q6_K, QuantDType.Q5_K, QuantDType.Q4_K, QuantDType.Q3_K, QuantDType.Q2_K }[fi];
            if (!double.TryParse((budgetInput.Text.ToString() ?? string.Empty).Trim(), out double mb) || mb <= 0)
            {
                MessageBox.ErrorQuery("Quantize model", "Enter a positive target size in MB.", "OK");
                return;
            }
            Application.RequestStop();
            RunQuantizeBudget(sourcePath, (long)(mb * 1024 * 1024), floor);
        });
        var okButton = new Button("OK") { X = 1, Y = floors.Length + 6 };
        okButton.Clicked += run;
        var cancelButton = new Button("Cancel") { X = Pos.Right(okButton) + 2, Y = floors.Length + 6 };
        cancelButton.Clicked += () => Application.RequestStop();
        picker.Add(prompt, floorList, budgetPrompt, budgetInput, okButton, cancelButton);
        Application.Run(picker);
    }

    private void RunQuantize(string sourcePath, QuantDType target, string title)
    {
        var destPath = AskQuantizeDestination(sourcePath, $"-{target}", title);
        if (destPath is null) return;
        RunQuantizeCore(sourcePath, destPath, new SmmQuantOptions { DefaultLevel = target }, title);
    }

    private void RunQuantizeBudget(string sourcePath, long targetBytes, QuantDType floor)
    {
        var destPath = AskQuantizeDestination(sourcePath, "-quantized", "Quantize by budget");
        if (destPath is null) return;
        RunQuantizeCore(sourcePath, destPath, new SmmQuantOptions { TargetBytes = targetBytes, Floor = floor }, "Quantize by budget");
    }

    /// <summary>Save-as: lets the user choose where the quantized model goes. Defaults to the source folder.</summary>
    private static string? AskQuantizeDestination(string sourcePath, string suffix, string title)
    {
        string start = Path.GetDirectoryName(sourcePath) ?? Directory.GetCurrentDirectory();
        string defaultName = Path.GetFileNameWithoutExtension(sourcePath) + suffix + ".smm";
        return FilePickerDialog.Show(title + " — where to save?", start, PickerMode.SaveFile, defaultName: defaultName);
    }

    private void RunQuantizeCore(string sourcePath, string destPath, SmmQuantOptions options, string title)
    {
        long before = new FileInfo(sourcePath).Length;
        var cancelSource = new CancellationTokenSource();

        string? error = null;
        bool done = false;
        int progressPercent = -1;

        var dialog = new Dialog($"{title} — {Path.GetFileName(destPath)}", 60, 8);
        var statusLabel = new Label("Quantizing…") { X = 1, Y = 1, Width = Dim.Fill(2) };
        var cancelButton = new Button("Cancel") { X = Pos.AnchorEnd(12), Y = Pos.AnchorEnd(1) };
        cancelButton.Clicked += () => cancelSource.Cancel();
        dialog.Add(statusLabel, cancelButton);

        var taskProgress = new Progress<float>(p => Application.MainLoop.Invoke(() =>
        {
            int percent = Math.Clamp((int)(p * 100), 0, 100);
            if (percent != progressPercent)
            {
                progressPercent = percent;
                statusLabel.Text = $"Quantizing… {percent}%";
            }
        }));

        string src = sourcePath, dst = destPath;
        _ = Task.Run(() =>
        {
            try
            {
                SmmQuantizer.Quantize(src, dst, options, taskProgress, cancelSource.Token);
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
        pollToken = Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(100), poll);
        Application.Run(dialog);

        if (cancelSource.IsCancellationRequested)
        {
            MessageBox.Query(title, "Quantization cancelled — nothing was saved.", "OK");
        }
        else if (error is not null)
        {
            MessageBox.ErrorQuery(title, $"Quantization failed:\n{error}", "OK");
        }
        else if (File.Exists(destPath))
        {
            long after = new FileInfo(destPath).Length;
            MessageBox.Query(title,
                $"Quantized {Path.GetFileName(sourcePath)} → {Path.GetFileName(destPath)}\n" +
                $"({before:N0} → {after:N0} bytes, {(before > 0 ? 100 * (1 - (double)after / before) : 0):F1}% smaller)", "OK");
        }
    }

    /// <summary>
    /// Model → Modify model metadata…: edits the system prompt, skills and
    /// embedded plugins of an existing .SMM file in place (see
    /// <see cref="SmmModifier"/>), preserving the trained weights untouched.
    /// </summary>
    private void ShowModelMetaModifier()
    {
        string start = _options.ProjectPath ?? _settings.ResolvedModelFolder ?? Directory.GetCurrentDirectory();
        string? path = FilePickerDialog.Show("Modify .SMM metadata", start, PickerMode.File, patterns: ["*.smm"]);
        if (path is null) return;

        SmmModelDocument doc;
        try
        {
            doc = SmmModifier.Read(path);
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Modify metadata", $"Could not read {Path.GetFileName(path)}:\n{ex.Message}", "OK");
            return;
        }

        var dialog = new Dialog($"Modify .SMM metadata — {Path.GetFileName(path)}", 78, 26);

        var promptLabel = new Label("System prompt:") { X = 1, Y = 1 };
        var promptValue = new Label("") { X = 1, Y = 2, Width = Dim.Fill(2) };
        var editPrompt = new Button("Edit prompt…") { X = 1, Y = 3 };
        editPrompt.Clicked += () =>
        {
            var edited = EditTextModal("System prompt", doc.SystemPrompt);
            if (edited is not null) { doc.SystemPrompt = edited.Trim(); RefreshMetaDialog(promptValue, doc); }
        };
        var clearPrompt = new Button("Clear prompt") { X = Pos.Right(editPrompt) + 2, Y = 3 };
        clearPrompt.Clicked += () =>
        {
            doc.SystemPrompt = "";
            RefreshMetaDialog(promptValue, doc);
        };

        var skillsCaption = new Label("Skills: 0") { X = 1, Y = 5 };
        var skillList = new ListView { X = 1, Y = 6, Width = 42, Height = 5 };
        RefreshSkills(skillsCaption, skillList, doc);

        var addSkillFile = new Button("Add file…") { X = 46, Y = 6 };
        addSkillFile.Clicked += () =>
        {
            var picked = FilePickerDialog.Show("Add skill file (.md/.txt)", Directory.GetCurrentDirectory(), PickerMode.File,
                patterns: ["*.md", "*.txt"]);
            if (picked is null) return;
            doc.Skills.Add(File.ReadAllText(picked));
            RefreshSkills(skillsCaption, skillList, doc);
        };
        var addSkillManual = new Button("Add manual…") { X = 46, Y = 7 };
        addSkillManual.Clicked += () =>
        {
            var text = EditTextModal("Add skill", "");
            if (text is null) return;
            doc.Skills.Add(text);
            RefreshSkills(skillsCaption, skillList, doc);
        };
        var editSkill = new Button("Edit…") { X = 46, Y = 8 };
        editSkill.Clicked += () =>
        {
            int i = skillList.SelectedItem;
            if (i < 0 || i >= doc.Skills.Count) { MessageBox.ErrorQuery("Edit skill", "Select a skill first.", "OK"); return; }
            var edited = EditTextModal("Edit skill", doc.Skills[i]);
            if (edited is not null) { doc.Skills[i] = edited; RefreshSkills(skillsCaption, skillList, doc); }
        };
        var removeSkill = new Button("Remove") { X = 46, Y = 9 };
        removeSkill.Clicked += () =>
        {
            int i = skillList.SelectedItem;
            if (i < 0 || i >= doc.Skills.Count) return;
            doc.Skills.RemoveAt(i);
            RefreshSkills(skillsCaption, skillList, doc);
        };
        var clearSkills = new Button("Clear") { X = 46, Y = 10 };
        clearSkills.Clicked += () =>
        {
            doc.Skills.Clear();
            RefreshSkills(skillsCaption, skillList, doc);
        };

        var pluginsCaption = new Label("Plugins: 0") { X = 1, Y = 13 };
        var pluginList = new ListView { X = 1, Y = 14, Width = 42, Height = 4 };
        RefreshPluginsList(pluginsCaption, pluginList, doc);

        var addPlugin = new Button("Add DLL…") { X = 46, Y = 14 };
        addPlugin.Clicked += () =>
        {
            string dllStart = Directory.GetCurrentDirectory();
            var picked = FilePickerDialog.Show("Add tool DLL", dllStart, PickerMode.File, "*.dll");
            if (picked is null) return;
            doc.Plugins.Add(new SmmPluginEntry { Name = Path.GetFileName(picked), AssemblyBytes = File.ReadAllBytes(picked) });
            RefreshPluginsList(pluginsCaption, pluginList, doc);
        };
        var removePlugin = new Button("Remove") { X = 46, Y = 15 };
        removePlugin.Clicked += () =>
        {
            int i = pluginList.SelectedItem;
            if (i < 0 || i >= doc.Plugins.Count) return;
            doc.Plugins.RemoveAt(i);
            RefreshPluginsList(pluginsCaption, pluginList, doc);
        };
        var clearPlugins = new Button("Clear") { X = 46, Y = 16 };
        clearPlugins.Clicked += () =>
        {
            doc.Plugins.Clear();
            RefreshPluginsList(pluginsCaption, pluginList, doc);
        };

        var apply = new Button("Apply changes") { X = 1, Y = Pos.AnchorEnd(2), IsDefault = true };
        var cancel = new Button("Cancel") { X = Pos.Right(apply) + 2, Y = Pos.AnchorEnd(2) };
        bool applied = false;
        apply.Clicked += () =>
        {
            try
            {
                SmmModifier.Write(path, doc);
                applied = true;
                Application.RequestStop();
            }
            catch (Exception ex)
            {
                MessageBox.ErrorQuery("Modify metadata", $"Could not write {Path.GetFileName(path)}:\n{ex.Message}", "OK");
            }
        };
        cancel.Clicked += () => Application.RequestStop();

        dialog.Add(promptLabel, promptValue, editPrompt, clearPrompt,
            skillsCaption, skillList, addSkillFile, addSkillManual, editSkill, removeSkill, clearSkills,
            pluginsCaption, pluginList, addPlugin, removePlugin, clearPlugins,
            apply, cancel);
        RefreshMetaDialog(promptValue, doc);
        Application.Run(dialog);

        if (applied)
            MessageBox.Query("Modify metadata", $"Updated {Path.GetFileName(path)}.\nSystem prompt, skills and plugins are saved.", "OK");

        static void RefreshMetaDialog(Label label, SmmModelDocument d)
        {
            label.Text = string.IsNullOrEmpty(d.SystemPrompt)
                ? (ustring)"— none —"
                : (ustring)(d.SystemPrompt.Length <= 68 ? d.SystemPrompt : d.SystemPrompt[..68] + "…");
        }

        static void RefreshSkills(Label caption, ListView view, SmmModelDocument d)
        {
            caption.Text = $"Skills: {d.Skills.Count}";
            view.SetSource(d.Skills.Count == 0
                ? [(ustring)"(no skills embedded)"]
                : d.Skills.Select((s, i) => (ustring)$"{i + 1}. {FirstLine(s)}").ToArray());
        }

        static void RefreshPluginsList(Label caption, ListView view, SmmModelDocument d)
        {
            caption.Text = $"Plugins: {d.Plugins.Count}";
            view.SetSource(d.Plugins.Count == 0
                ? [(ustring)"(no plugins embedded)"]
                : d.Plugins.Select(p => (ustring)p.Name).ToArray());
        }

        static string FirstLine(string text)
        {
            int nl = text.IndexOf('\n');
            string line = nl >= 0 ? text[..nl] : text;
            return line.Length <= 50 ? line : line[..50] + "…";
        }
    }

    /// <summary>Modal multi-line text editor returning the text, or null when cancelled.</summary>
    private static string? EditTextModal(string title, string initial)
    {
        string? result = null;
        var editor = new Dialog(title, 74, 18);
        var textView = new TextView { X = 1, Y = 1, Width = Dim.Fill(2), Height = Dim.Fill(4), Text = (ustring)initial };
        var ok = new Button("OK") { X = 1, Y = Pos.AnchorEnd(1), IsDefault = true };
        ok.Clicked += () => { result = textView.Text.ToString(); Application.RequestStop(); };
        var cancel = new Button("Cancel") { X = Pos.Right(ok) + 2, Y = Pos.AnchorEnd(1) };
        cancel.Clicked += () => Application.RequestStop();
        editor.Add(textView, ok, cancel);
        Application.Run(editor);
        return result;
    }

    private void ShowTrainingWizard()
        => ShowTrainingWizard(job: null);

    /// <summary>
    /// Opens the training wizard for <paramref name="job"/>, or a fresh job when
    /// null. Passing the job back in lets the user return from a finished or
    /// interrupted training screen to the same editable settings — the wizard
    /// then shows whether a checkpoint exists to continue from.
    /// </summary>
    private void ShowTrainingWizard(TrainJobSettings? job)
    {
        // Only one Train run at a time: re-selecting training while a run is
        // active switches back to the running screen instead of starting anew.
        if (_activeTraining is not null)
        {
            SwapContent(_activeTraining);
            return;
        }

        Action onBack = _currentSession is not null ? ShowChat : ShowWelcome;
        SwapContent(new TrainingWizardView(
            _settings,
            job,
            onStart: ShowTrainingProgress,
            onCancel: onBack));
    }

    private void ShowTrainingProgress(TrainJobSettings job)
    {
        // Guard: never let a second Train run start while one is live.
        if (_activeTraining is not null)
        {
            SwapContent(_activeTraining);
            return;
        }

        if (!string.IsNullOrWhiteSpace(job.ExportPath) || !string.IsNullOrWhiteSpace(job.ExportFolder))
        {
            string folder = job.ExportFolder;
            if (!string.Equals(_settings.LastExportPath, folder, StringComparison.OrdinalIgnoreCase))
            {
                _settings.LastExportPath = folder;
                _settings.Save(out _);
            }
        }
        var view = new TrainingProgressView(
            _settings,
            job,
            browseModel: path =>
            {
                _options.ModelPath = path;
                _options.ProjectPath ??= Path.GetDirectoryName(path);
                ShowOptions();
            },
            onBack: () => ShowTrainingWizard(job),
            onDetach: () => _activeTraining = null);
        _activeTraining = view;
        SwapContent(view);
    }

    private void ShowOptions()
    {
        Action onCancel = _currentSession is not null ? ShowChat : ShowModelBrowser;
        SwapContent(new OptionsView(_options, _settings, onLaunch: LaunchSession, onCancel: onCancel));
    }

    /// <summary>Menu path into Options — declines quietly if there's nothing to configure yet.</summary>
    private void ShowOptionsIfReachable()
    {
        if (_options.ModelPath is not null || _options.Generator == GeneratorStrategy.UIDebug || _currentSession is not null)
            ShowOptions();
    }

    private void ShowSettings()
    {
        Action onBack = _currentSession is not null ? ShowChat : ShowWelcome;
        var view = new SettingsView(_settings, onBack: onBack)
        {
            OnThemeChanged = kind => ThemeBuilder.ApplyRecursively(this, ThemeBuilder.Build(kind))
        };
        SwapContent(view);
    }

    /// <summary>
    /// Help → About SharpMind…: project identity plus the GitHub repository and
    /// sponsor pages, with optional one-click open in the default browser.
    /// </summary>
    private static void ShowAbout()
    {
        const string repo = "https://github.com/Integral2u/SharpMind";
        const string sponsor = "https://github.com/sponsors/Integral2u";

        var dialog = new Dialog("About SharpMind", 62, 12);
        dialog.Add(new Label("SharpMind — an open source LLM toolkit written in pure C#.")
        {
            X = 1,
            Y = 1,
            Width = Dim.Fill(2),
        });
        dialog.Add(new Label((ustring)$"GitHub:  {repo}") { X = 1, Y = 3 });
        dialog.Add(new Label((ustring)$"Sponsor: {sponsor}") { X = 1, Y = 4 });

        var openRepo = new Button("Open GitHub") { X = 1, Y = 7 };
        openRepo.Clicked += () => OpenUrl(repo);
        var openSponsor = new Button("Open Sponsor") { X = Pos.Right(openRepo) + 2, Y = 7 };
        openSponsor.Clicked += () => OpenUrl(sponsor);
        var ok = new Button("OK") { X = Pos.Right(openSponsor) + 2, Y = 7, IsDefault = true };
        ok.Clicked += () => Application.RequestStop();
        dialog.Add(openRepo, openSponsor, ok);

        Application.Run(dialog);
    }

    /// <summary>Opens <paramref name="url"/> in the default browser (best effort).</summary>
    private static void OpenUrl(string url)
    {
        try
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(url)
            {
                UseShellExecute = true,
            });
        }
        catch
        {
            MessageBox.ErrorQuery("About SharpMind", $"Could not open the default browser.\n{url}", "OK");
        }
    }

    private void ShowChat()
    {
        if (_currentSession is null) return;
        SwapContent(_currentSession.View);
        // Defer transcript rebuild to the next main-loop tick so the view
        // is fully laid out and parented. A direct call here can race with
        // Terminal.Gui's first draw of the empty ChatView, leaving the
        // transcript blank even though history is present.
        Application.MainLoop.AddTimeout(TimeSpan.FromMilliseconds(50), _ =>
        {
            _currentSession?.View.RebuildTranscript();
            return false; // one-shot
        });
    }

    private void ShowSessionManager()
    {
        Action onBack = _currentSession is not null ? ShowChat : ShowWelcome;
        var view = new SessionManagerView(
            getSessions: () => _sessions,
            onSwitch: s => { _currentSession = s; ShowChat(); },
            onClose: CloseSession,
            onBack: onBack);
        SwapContent(view);
    }

    /// <summary>Chat → Close current session: closes the active session if any.</summary>
    private void CloseCurrentSession()
    {
        if (_currentSession is null)
        {
            MessageBox.ErrorQuery("Close current session", "No session is currently open.", "OK");
            return;
        }
        CloseSession(_currentSession);
    }

    private void ToggleShowThinking()
    {
        if (_currentSession?.Bridge is null) return;

        bool newValue = !_currentSession.Bridge.ShowThinking;
        _currentSession.Bridge.ShowThinking = newValue;
        _currentSession.Options.ShowThinking = newValue;
        _options.ShowThinking = newValue;
        _currentSession.View.RebuildTranscript();
        SaveLastUsedOptions(_options, _currentSession?.Bridge?.GetSnapshot());
        var thinking = _currentSession == null ? "off" : _currentSession.Bridge.ShowThinking ? "on" : "off";
        MessageBox.Query("Show Thinking",
            $"Show thinking is now {thinking}.",
            "OK");
    }

    private void ToggleTemplateThinking()
    {
        if (_currentSession?.Bridge is null) return;

        bool newValue = !_currentSession.Bridge.EnableThinking;
        _currentSession.Bridge.EnableThinking = newValue;
        _currentSession.Options.EnableThinking = newValue;
        _options.EnableThinking = newValue;
        SaveLastUsedOptions(_options, _currentSession?.Bridge?.GetSnapshot());
        var thinking = _currentSession == null ? "disabled" : _currentSession.Bridge.EnableThinking ? "enabled" : "disabled";
        MessageBox.Query("Enable Thinking (Template)",
            $"enable_thinking is now {thinking}.\nOnly takes effect on the next turn.",
            "OK");
    }

    // --- Session lifecycle -------------------------------------------------

    private void StartNewSession()
    {
        _currentSession = null; // switch context; doesn't close whatever else is open
        _options = LoadLastUsedOptions() ?? NewSessionOptionsFromSettings();
        ShowModelBrowser();
    }

    private void StartDebugSession()
    {
        _currentSession = null;
        _options = NewSessionOptionsFromSettings();
        _options.ModelPath = null;
        _options.Generator = GeneratorStrategy.UIDebug;
        LaunchSession();
    }

    private async void LaunchSession()
    {
        var progressView = new LoadingView("Preparing session...");
        SwapContent(progressView);

        var progress = new Progress<string>(s => Application.MainLoop.Invoke(() => progressView.SetMessage(s)));
        try
        {
            // Capture this launch's own options snapshot — _options is a shared,
            // mutable field that the next StartNewSession/StartDebugSession call
            // will reassign, so the session being built here needs its own copy
            // rather than chasing whatever _options points to by the time the
            // async load finishes.
            var launchOptions = CloneOptions(_options);
            // PendingSnapshot is transient — not carried by CopyTo — so transfer
            // it manually and clear the source so it doesn't leak into unrelated
            // sessions launched later.
            launchOptions.PendingSnapshot = _options.PendingSnapshot;
            _options.PendingSnapshot = null;

            if (launchOptions.Generator != GeneratorStrategy.UIDebug)
            {
                var cached = _modelCache.TryAcquire(launchOptions);
                LoadedModel loaded;
                if (cached is not null)
                {
                    loaded = cached;
                }
                else
                {
                    var loadResult = await SessionLauncher.LoadModelAsync(launchOptions, progress);
                    if (!loadResult.Success)
                    {
                        MessageBox.ErrorQuery("Load failed", loadResult.Error ?? "Unknown error", "OK");
                        ShowOptions();
                        return;
                    }
                    loaded = loadResult.Loaded!;
                    _modelCache.Register(launchOptions, loaded);
                }
                CreateAndShowSession(launchOptions, loaded, progressView);
            }
            else
            {
                CreateAndShowSession(launchOptions, null, progressView);
            }

        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Load failed", $"Unexpected error while loading the model:\n{ex.Message}", "OK");
            ShowOptions();
        }
    }

    private async void CreateAndShowSession(SessionOptions launchOptions, LoadedModel? loaded, LoadingView progressView)
    {
        try
        {
            var embedded = SessionLauncher.LoadEmbeddedPlugins(launchOptions.ModelPath);
            var gate = new PermissionGate();
            var result = SessionLauncher.BuildSession(launchOptions, loaded, gate.BuildCallback(launchOptions, embedded?.ToolNames), embedded);

            if (!result.Success)
            {
                MessageBox.ErrorQuery("Launch failed", result.Error ?? "Unknown error", "OK");
                ShowOptions();
                return;
            }

            var session = result.Session;
            if (session is not null)
            {
                // --- Combined build/rebuild progress (main-loop polling) ---
                // A single 0→100% progress across both InitializeChat and
                // WarmupPrefillAsync, updated by a main-loop timeout at ~10 fps.
                // LambdaProgress writes directly to the shared float from the
                // background thread (avoiding Progress<T>'s SynchronizationContext.Post).
                bool hasSnapshot = launchOptions.PendingSnapshot is not null;
                string progressLabel = hasSnapshot ? "Rebuilding KV cache..." : "Building KV cache...";
                float buildProgress = 0f;
                var buildTimer = Application.MainLoop.AddTimeout(
                    TimeSpan.FromMilliseconds(100), _ =>
                    {
                        progressView.SetMessage($"{progressLabel} {buildProgress * 100:F2}%");
                        return true;
                    });

                // Phase 1: InitializeChat — formatter, generator, agent, tokenizer (~0..0.5)
                await Task.Run(() => session.InitializeChat(new LambdaProgress<float>(p => buildProgress = p * 0.5f)));

                if (launchOptions.PendingSnapshot is { } snapshot)
                {
                    session.LoadSnapshot(snapshot);
                    launchOptions.PendingSnapshot = null;
                }

                // Phase 2: Warm-up prefill — encode prompt into KV cache (~0.5..1.0)
                if (session is ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder> concrete)
                {
                    await Task.Run(async () =>
                    {
                        concrete.PrefillProgress = fraction => buildProgress = 0.5f + (float)fraction * 0.5f;
                        await concrete.WarmupPrefillAsync();
                        concrete.PrefillProgress = null;
                    });
                }

                Application.MainLoop.RemoveTimeout(buildTimer);
                Application.MainLoop.Invoke(() => progressView.SetMessage("Starting session..."));
            }

            // disposeUnderlyingSession defaults to false here regardless of
            // generator/debug mode — the actual decision is made later, in
            // CloseSession, once ModelCache.Release knows whether this was the
            // last session sharing its model. Debug sessions have no
            // ChatSessionBridge at all (DebugChatBridge doesn't share this
            // concern, since there's no real Transformer underneath it).
            IChatBridge bridge = result.IsDebugMode
                ? new DebugChatBridge(result.CuiContext!, gate.BuildCallback(launchOptions, embedded?.ToolNames)) { UserName = launchOptions.UserName }
                : new ChatSessionBridge(result.Session!, disposeUnderlyingSession: false) { UserName = launchOptions.UserName };

            if (bridge is ChatSessionBridge realBridge) realBridge.Start();

            SaveLastUsedOptions(launchOptions, session?.GetSnapshot());

            string displayName = result.IsDebugMode ? $"{launchOptions.AgentName} [DEBUG]" : launchOptions.AgentName;
            var chatView = new ChatView(displayName, launchOptions, bridge, result.CuiContext, onExit: ShowSessionManager);

            var state = new ChatSessionState
            {
                Id = Guid.NewGuid(),
                DisplayName = MakeUniqueDisplayName(displayName),
                Options = launchOptions,
                Bridge = bridge,
                View = chatView,
                SourceFilePath = launchOptions.SourceFilePath
            };
            chatView.SessionDisplayName = state.DisplayName;

            _sessions.Add(state);
            _currentSession = state;
            _activePermissionGates[state.Id] = gate;

            if (result.Warnings.Count > 0)
                MessageBox.Query("Session started with warnings", string.Join("\n", result.Warnings), "OK");

            ShowChat();
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Launch failed", $"Unexpected error while starting the session:\n{ex.Message}", "OK");
            ShowOptions();
        }
    }

    private string MakeUniqueDisplayName(string baseName)
    {
        if (!_sessions.Any(s => s.DisplayName == baseName)) return baseName;
        int n = 2;
        while (_sessions.Any(s => s.DisplayName == $"{baseName} ({n})")) n++;
        return $"{baseName} ({n})";
    }

    /// <summary>
    /// Closes one session and removes it from the list. Whether this also
    /// disposes the underlying model depends entirely on ModelCache's ref
    /// count — see ChatSessionBridge's DisposeUnderlyingSession doc comment
    /// for why disposing a ChatSession that shares a Transformer with a
    /// still-open sibling would corrupt that sibling, and why this decision
    /// can only be made here, once ModelCache.Release has actually answered
    /// "was this the last one?".
    /// </summary>
    private async void CloseSession(ChatSessionState state)
    {
        bool wasCurrent = _currentSession == state;
        _sessions.Remove(state);
        _activePermissionGates.Remove(state.Id);
        try
        {
            bool isRealModelSession = state.Options.Generator != GeneratorStrategy.UIDebug;
            bool wasLastUser = isRealModelSession && _modelCache.Release(state.Options);

            if (state.Bridge is ChatSessionBridge realBridge)
                realBridge.DisposeUnderlyingSession = wasLastUser;

            await state.Bridge.DisposeAsync();
            state.View.Dispose();
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Close failed", $"Unexpected error while closing the session:\n{ex.Message}", "OK");
        }
        finally
        {
            if (wasCurrent)
            {
                _currentSession = _sessions.Count > 0 ? _sessions[0] : null;
                if (_currentSession is not null) ShowChat();
                else ShowWelcome();
            }
        }
    }

    // --- Resume last session -------------------------------------------------

    private void ResumeLastSession()
    {
        var lastSession = FindLastSession();
        if (lastSession is null)
        {
            MessageBox.Query("No saved session", "No saved sessions found.\nUse Chat > Save current session... to save one.", "OK");
            return;
        }

        _currentSession = null;
        _options = CloneOptions(lastSession.Value.session.Options);
        _options.PendingSnapshot = lastSession.Value.session.Snapshot;
        _options.SourceFilePath = lastSession.Value.path;
        ShowOptions();
    }

    // --- Save/Load session ---------------------------------------------------

    private void SaveCurrentSession()
    {
        if (_currentSession is null)
        {
            MessageBox.Query("No active session", "Switch to a chat session first.", "OK");
            return;
        }

        var saved = new SavedSession
        {
            Name = _currentSession.DisplayName,
            Options = _currentSession.Options,
            Snapshot = _currentSession.Bridge.GetSnapshot()
        };
        string safeFileName = string.Concat(_currentSession.DisplayName.Split(Path.GetInvalidFileNameChars()));
        //string defaultPath = Path.Combine(SavedSession.DefaultFolder, $"{safeFileName}.json");

        string? path;
        if (_currentSession.SourceFilePath is { } existing)
        {
            // Session was previously saved or loaded — ask before
            // overwriting the same file, and always offer Save As for a
            // different destination.
            int choice = MessageBox.Query("Save session",
                $"Save to:\n{existing}\n\nOverwrite or choose a different location?",
                "Overwrite", "Save As", "Cancel");
            if (choice == 0)
            {
                path = existing;
            }
            else if (choice == 1)
            {
                path = FilePickerDialog.Show("Save session as", SavedSession.DefaultFolder, PickerMode.SaveFile, $"{safeFileName}.json");
            }
            else
            {
                return; // Cancel
            }
        }
        else
        {
            // First save for this session — offer Save As with a sensible default.
            path = FilePickerDialog.Show("Save session", SavedSession.DefaultFolder, PickerMode.SaveFile, $"{safeFileName}.json");
        }

        if (path is null) return;

        if (SavedSession.Save(saved, path, out var error))
        {
            _currentSession.SourceFilePath = path;
            MessageBox.Query("Saved", $"Saved session to:\n{path}", "OK");
        }
        else
        {
            MessageBox.ErrorQuery("Save failed", error ?? "Unknown error", "OK");
        }
    }

    /// <summary>
    /// Loads a saved session's options and lands on the Options screen
    /// rather than launching immediately — the loaded model file might no
    /// longer exist, or the person might want to tweak something first, so
    /// committing to a load before they've seen what was actually restored
    /// would be presumptuous. ModelCache transparently reuses an
    /// already-loaded model if another open session happens to reference
    /// the same file, same as any other launch.
    /// </summary>
    private void LoadSessionFromDisk()
    {
        var picked = FilePickerDialog.Show("Load session", SavedSession.DefaultFolder, PickerMode.File, "*.json");
        if (picked is null) return;

        var loaded = SavedSession.Load(picked, out var error);
        if (loaded is null)
        {
            MessageBox.ErrorQuery("Load failed", error ?? "Unknown error", "OK");
            return;
        }

        _currentSession = null;
        _options = CloneOptions(loaded.Options);
        _options.PendingSnapshot = loaded.Snapshot;
        _options.SourceFilePath = picked;
        ShowOptions();
    }

    private static SessionOptions CloneOptions(SessionOptions source) => source.Clone();

    // --- Permission "Ask" dialogs --------------------------------------------

    private bool PollPermissionRequests(MainLoop _)
    {
        foreach (var gate in _activePermissionGates.Values.ToList())
        {
            if (gate.TakePending() is { } request)
                ShowPermissionDialog(request);
        }
        return true;
    }

    private static void ShowPermissionDialog(PermissionRequest request)
    {
        string kind = request.Context.Category == ToolCategory.Network ? "network" : "file";
        string message = $"Tool \"{request.Context.ToolName}\" wants {kind} access to:\n{request.Context.Resource}\n\nAllow this?";

        var dialog = new Dialog("Permission request", 60, 10);
        dialog.Add(new Label((NStack.ustring)message) { X = 1, Y = 1, Width = Dim.Fill(2), Height = 4 });

        // "Allow Once" and "Always Allow" both currently resolve to
        // ToolPermission.Always under the hood — ChatSession's permission
        // contract is per-call, not session-scoped, so there is no weaker
        // "just this one time" grant to give that's distinct from Always
        // without ChatSession itself tracking a one-shot exception. The two
        // buttons describe intent for the person answering THIS prompt; they
        // don't currently change future prompts differently from each other.
        var allowOnce = new Button("Allow Once") { X = 1, Y = 6, IsDefault = true };
        allowOnce.Clicked += () => { request.Resolve(ToolPermission.Always); Application.RequestStop(); };
        var allowAlways = new Button("Always Allow") { X = Pos.Right(allowOnce) + 2, Y = 6 };
        allowAlways.Clicked += () => { request.Resolve(ToolPermission.Always); Application.RequestStop(); };
        var deny = new Button("Deny") { X = Pos.Right(allowAlways) + 2, Y = 6 };
        deny.Clicked += () => { request.Resolve(ToolPermission.Never); Application.RequestStop(); };

        dialog.Add(allowOnce, allowAlways, deny);
        Application.Run(dialog);
    }

    /// <summary>
    /// Thin IProgress&lt;T&gt; adapter that invokes a delegate directly on the
    /// calling thread, avoiding Progress&lt;T&gt;'s SynchronizationContext.Post
    /// which in a console app routes to a thread-pool thread instead of the
    /// main loop — making main-loop timeout polling useless during awaited
    /// Task.Run blocks.
    /// </summary>
    private sealed class LambdaProgress<T>(Action<T> action) : IProgress<T>
    {
        public void Report(T value) => action(value);
    }
}
