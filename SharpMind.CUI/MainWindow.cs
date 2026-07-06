using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
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
    private readonly Dictionary<Guid, PermissionGate> _activePermissionGates = new();

    private SessionOptions _options;
    private ChatSessionState? _currentSession;
    private object? _permissionPollToken;

    public MainWindow()
    {
        Title = "SharpMind CUI";

        _settings = AppSettings.Load();
        _options = NewSessionOptionsFromSettings();

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

    private SessionOptions NewSessionOptionsFromSettings()
    {
        var options = SessionOptions.Default;
        options.ProjectPath = _settings.DefaultModelFolder;
        options.ToolsFolder = _settings.ToolsFolder;
        return options;
    }

    private MenuBar BuildMenuBar()
    {
        return new MenuBar(new MenuBarItem[]
        {
            new("_File", new MenuItem[]
            {
                new("_New session...", "", StartNewSession, shortcut: Key.N | Key.CtrlMask),
                new("_Welcome screen", "", ShowWelcome),
                new("_Settings...", "", ShowSettings),
                new("E_xit", "", () => Application.RequestStop())
            }),
            new("_Model", new MenuItem[]
            {
                new("_Browse for model...", "", ShowModelBrowser),
                new("_Debug session (no model)...", "", StartDebugSession)
            }),
            new("_Options", new MenuItem[]
            {
                new("_Session options...", "", ShowOptionsIfReachable)
            }),
            new("_Chat", new MenuItem[]
            {
                new("_Manage sessions...", "", ShowSessionManager),
                new("_Load session...", "", LoadSessionFromDisk),
                new("_Save current session...", "", SaveCurrentSession)
            })
        });
    }

    // --- Screen navigation -------------------------------------------------

    private void SwapContent(View view)
    {
        _content.RemoveAll();
        _content.Add(view);
        view.Width = Dim.Fill();
        view.Height = Dim.Fill();
        view.X = 0;
        view.Y = 0;
        _content.FocusFirst();
        _content.SetNeedsDisplay();
    }

    private void ShowWelcome() => SwapContent(new WelcomeView(onBrowseModel: ShowModelBrowser));

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
            onCancel: onCancel));
    }

    private void ShowOptions()
    {
        Action onCancel = _currentSession is not null ? ShowChat : ShowModelBrowser;
        SwapContent(new OptionsView(_options, onLaunch: LaunchSession, onCancel: onCancel));
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
        var view = new SettingsView(_settings, onBack: onBack);
        view.OnThemeChanged = kind => ThemeBuilder.ApplyRecursively(this, ThemeBuilder.Build(kind));
        SwapContent(view);
    }

    private void ShowChat()
    {
        if (_currentSession is null) return;
        SwapContent(_currentSession.View);
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

    // --- Session lifecycle -------------------------------------------------

    private void StartNewSession()
    {
        _currentSession = null; // switch context; doesn't close whatever else is open
        _options = NewSessionOptionsFromSettings();
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

        // Capture this launch's own options snapshot — _options is a shared,
        // mutable field that the next StartNewSession/StartDebugSession call
        // will reassign, so the session being built here needs its own copy
        // rather than chasing whatever _options points to by the time the
        // async load finishes.
        var launchOptions = CloneOptions(_options);

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
            progressView.SetMessage("Starting session...");
            CreateAndShowSession(launchOptions, loaded);
        }
        else
        {
            progressView.SetMessage("Starting session...");
            CreateAndShowSession(launchOptions, null);
        }
    }

    private void CreateAndShowSession(SessionOptions launchOptions, LoadedModel? loaded)
    {
        var gate = new PermissionGate();
        var result = SessionLauncher.BuildSession(launchOptions, loaded, gate.BuildCallback(launchOptions));

        if (!result.Success)
        {
            MessageBox.ErrorQuery("Launch failed", result.Error ?? "Unknown error", "OK");
            ShowOptions();
            return;
        }

        // disposeUnderlyingSession defaults to false here regardless of
        // generator/debug mode — the actual decision is made later, in
        // CloseSession, once ModelCache.Release knows whether this was the
        // last session sharing its model. Debug sessions have no
        // ChatSessionBridge at all (DebugChatBridge doesn't share this
        // concern, since there's no real Transformer underneath it).
        IChatBridge bridge = result.IsDebugMode
            ? new DebugChatBridge(result.CuiContext!, gate.BuildCallback(launchOptions))
            : new ChatSessionBridge(result.Session!, disposeUnderlyingSession: false);

        if (bridge is ChatSessionBridge realBridge) realBridge.Start();

        string displayName = result.IsDebugMode ? $"{launchOptions.AgentName} [DEBUG]" : launchOptions.AgentName;
        var chatView = new ChatView(displayName, launchOptions, bridge, result.CuiContext, onExit: ShowSessionManager);

        var state = new ChatSessionState
        {
            Id = Guid.NewGuid(),
            DisplayName = MakeUniqueDisplayName(displayName),
            Options = launchOptions,
            Bridge = bridge,
            View = chatView
        };
        chatView.SessionDisplayName = state.DisplayName;

        _sessions.Add(state);
        _currentSession = state;
        _activePermissionGates[state.Id] = gate;

        if (result.Warnings.Count > 0)
            MessageBox.Query("Session started with warnings", string.Join("\n", result.Warnings), "OK");

        ShowChat();
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

        bool isRealModelSession = state.Options.Generator != GeneratorStrategy.UIDebug;
        bool wasLastUser = isRealModelSession && _modelCache.Release(state.Options);

        if (state.Bridge is ChatSessionBridge realBridge)
            realBridge.DisposeUnderlyingSession = wasLastUser;

        await state.Bridge.DisposeAsync();
        state.View.Dispose();

        if (wasCurrent)
        {
            _currentSession = _sessions.Count > 0 ? _sessions[0] : null;
            if (_currentSession is not null) ShowChat();
            else ShowWelcome();
        }
    }

    // --- Save/Load session ---------------------------------------------------

    private void SaveCurrentSession()
    {
        if (_currentSession is null)
        {
            MessageBox.Query("No active session", "Switch to a chat session first.", "OK");
            return;
        }

        var saved = new SavedSession { Name = _currentSession.DisplayName, Options = _currentSession.Options };
        string safeFileName = string.Concat(_currentSession.DisplayName.Split(Path.GetInvalidFileNameChars()));
        string path = Path.Combine(SavedSession.DefaultFolder, $"{safeFileName}.json");

        if (SavedSession.Save(saved, path, out var error))
            MessageBox.Query("Saved", $"Saved session to:\n{path}", "OK");
        else
            MessageBox.ErrorQuery("Save failed", error ?? "Unknown error", "OK");
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
        ShowOptions();
    }

    private static SessionOptions CloneOptions(SessionOptions source) => new()
    {
        ModelPath = source.ModelPath,
        ProjectPath = source.ProjectPath,
        SkillFolders = source.SkillFolders.ToList(),
        ToolAssemblyPaths = source.ToolAssemblyPaths.ToList(),
        ToolsFolder = source.ToolsFolder,
        Generator = source.Generator,
        Cache = source.Cache,
        HardwareTier = source.HardwareTier,
        UseGpu = source.UseGpu,
        FileAccess = source.FileAccess,
        NetworkAccess = source.NetworkAccess,
        Sampling = source.Sampling,
        Generation = source.Generation,
        AgentName = source.AgentName,
        AgentsEnabled = source.AgentsEnabled,
        MaxAgentDepth = source.MaxAgentDepth,
        MaxToolCallsPerTurn = source.MaxToolCallsPerTurn,
        LoadMode = source.LoadMode,
        DisabledTools = new HashSet<string>(source.DisabledTools)
    };

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
}
