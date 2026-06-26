using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>
/// Top-level state machine: Welcome -&gt; ModelBrowser -&gt; Options -&gt; Chat,
/// with Esc generally stepping back one screen. Owns the render loop, which
/// is intentionally simple: drain input, update the current screen, redraw,
/// present, repeat, on a small fixed delay so it never spins the CPU just to
/// keep a blinking cursor blinking.
/// </summary>
public sealed class App : IAsyncDisposable
{
    private readonly ScreenBuffer _buf;
    private readonly InputQueue _input = new();
    private readonly MenuBar _menuBar = new();
    private AppScreenKind _screen = AppScreenKind.Welcome;

    private readonly AppSettings _settings;
    private Theme _theme;

    private SessionOptions _options;
    private ModelBrowserScreen? _browser;
    private OptionsScreen? _optionsScreen;
    private SettingsScreen? _settingsScreen;
    private ChatScreen? _chatScreen;
    private IChatBridge? _bridge;
    private CuiToolContext? _cuiContext;
    private ChoiceDialog? _choiceDialog;

    private string? _statusMessage;
    private bool _quit;

    public App()
    {
        _buf = new ScreenBuffer(Console.WindowWidth, Console.WindowHeight);
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        try { Console.CursorVisible = false; } catch { /* not every terminal supports this */ }

        _settings = AppSettings.Load();
        _theme = Theme.For(_settings.Theme);
        _options = NewSessionOptionsFromSettings();
    }

    /// <summary>
    /// Builds a fresh <see cref="SessionOptions"/> seeded from persisted
    /// <see cref="AppSettings"/> defaults (model folder, tools folder) —
    /// used both on first startup and every time File &gt; New Session resets
    /// the app back to picking a model.
    /// </summary>
    private SessionOptions NewSessionOptionsFromSettings()
    {
        var options = SessionOptions.Default;
        options.ProjectPath = _settings.DefaultModelFolder;
        options.ToolsFolder = _settings.ToolsFolder;
        return options;
    }

    public async Task RunAsync()
    {
        _input.Start();
        try
        {
            while (!_quit)
            {
                HandleResize();
                foreach (var key in _input.DrainPending())
                    await HandleKeyAsync(key);
                foreach (var mouse in _input.DrainMousePending())
                    HandleMouse(mouse);

                await PollPendingLaunchAsync();
                PollPendingChoiceRequest();

                Draw();
                _buf.Present();

                await Task.Delay(16); // ~60fps cap; this is a text UI, not a game, but smooth input feels right
            }
        }
        finally
        {
            _input.Stop();
            Console.ResetColor();
            Console.Clear();
            try { Console.CursorVisible = true; } catch { }
        }
    }

    private void HandleResize()
    {
        if (Console.WindowWidth != _buf.Width || Console.WindowHeight != _buf.Height)
            _buf.Resize(Console.WindowWidth, Console.WindowHeight);
    }

    private async Task HandleKeyAsync(ConsoleKeyInfo key)
    {
        // Explicit Ctrl+C handling, ahead of every other input path. On
        // Windows, WindowsConsoleInput deliberately clears
        // ENABLE_PROCESSED_INPUT so Ctrl+C arrives here as an ordinary key
        // event instead of the OS terminating the process mid-render — but
        // that means this app now has to be the one to honour the
        // conventional "Ctrl+C quits" reflex, or it would otherwise do
        // nothing at all on Windows while still working (via .NET's own
        // default SIGINT handling, unrelated to anything in this file) on
        // Linux/macOS. Binding it explicitly here makes the behaviour the
        // same deliberate choice on every platform instead of an accident of
        // whichever OS default happened to apply.
        if (key.Key == ConsoleKey.C && key.Modifiers.HasFlag(ConsoleModifiers.Control))
        {
            _quit = true;
            return;
        }

        if (_choiceDialog is not null)
        {
            _choiceDialog.HandleKey(key);
            if (_choiceDialog.IsResolved) _choiceDialog = null;
            // Highest-priority modal: a tool call is synchronously blocked
            // waiting on this answer, so nothing else — not even opening a
            // menu — should be reachable until it's resolved.
            return;
        }

        bool menuWasOpen = _menuBar.IsMenuOpen;
        bool couldOpenMenu = key.Key == ConsoleKey.F10 || key.Modifiers.HasFlag(ConsoleModifiers.Alt);

        if (menuWasOpen || couldOpenMenu)
        {
            _menuBar.HandleKey(key);
            ApplyMenuSelectionIfAny();
            // The menu bar is modal while open (or while a key was attempting to
            // open it) — it owns this keypress entirely, the active screen never
            // sees it. This matches how Edit/QBasic worked: you can't type into
            // the document while Alt or a menu is engaged.
            return;
        }

        switch (_screen)
        {
            case AppScreenKind.Welcome:
                if (key.Key == ConsoleKey.Escape) _quit = true;
                else if (key.Key == ConsoleKey.M || key.Key == ConsoleKey.Enter)
                {
                    _browser = new ModelBrowserScreen(_options.ProjectPath ?? Directory.GetCurrentDirectory());
                    _screen = AppScreenKind.ModelBrowser;
                }
                break;

            case AppScreenKind.ModelBrowser:
                _browser!.HandleKey(key);
                if (_browser.Cancelled) { _screen = AppScreenKind.Welcome; }
                else if (_browser.ChosenModelPath is { } path)
                {
                    _options.ModelPath = path;
                    _options.ProjectPath ??= Path.GetDirectoryName(path);

                    // First-run convenience: if no default model folder has ever been
                    // set, remember this one so next launch starts here instead of the
                    // process's current directory. Only fires once — after that, the
                    // folder is whatever Settings says, deliberately, even if the user
                    // browses somewhere else for one particular session.
                    if (string.IsNullOrWhiteSpace(_settings.DefaultModelFolder))
                    {
                        _settings.DefaultModelFolder = Path.GetDirectoryName(path);
                        _settings.Save(out _); // best-effort; a failed silent save here isn't worth surfacing
                    }

                    _optionsScreen = new OptionsScreen(_options, path);
                    _screen = AppScreenKind.Options;
                }
                break;

            case AppScreenKind.Options:
                _optionsScreen!.HandleKey(key);
                if (_optionsScreen.Cancelled) { _screen = AppScreenKind.ModelBrowser; _browser = new ModelBrowserScreen(_options.ProjectPath ?? "."); }
                else if (_optionsScreen.LaunchRequested && _pendingLaunch is null)
                {
                    _optionsScreen.AcknowledgeLaunchRequest();
                    LaunchSessionAsync();
                }
                break;

            case AppScreenKind.Chat:
                _chatScreen!.HandleKey(key);
                if (_chatScreen.PendingSubmission is { } text)
                {
                    _bridge!.SubmitUserInput(text);
                    _chatScreen.BeginGenerating(text);
                    _chatScreen.ClearPendingSubmission();
                }
                if (_chatScreen.ExitRequested)
                {
                    await TeardownSessionAsync();
                    _screen = AppScreenKind.Options;
                }
                break;

            case AppScreenKind.Settings:
                _settingsScreen!.HandleKey(key);
                _theme = Theme.For(_settings.Theme); // live preview - applies immediately, persisted only on Save
                if (_settingsScreen.Cancelled) { _screen = AppScreenKind.Welcome; }
                else if (_settingsScreen.SaveRequested)
                {
                    _settingsScreen.AcknowledgeSaveRequest();
                    bool ok = _settings.Save(out var err);
                    _settingsScreen.ReportSaveResult(ok, err);
                }
                break;
        }
    }

    private void HandleMouse(MouseEvent evt)
    {
        if (_choiceDialog is not null) return; // modal: no mouse interaction wired for the dialog yet, but it must still block clicks from leaking through to the screen underneath

        bool menuWasOpen = _menuBar.IsMenuOpen;
        bool clickedMenuRow = evt.Action == MouseAction.Down && evt.Y == 0;

        if (menuWasOpen || clickedMenuRow)
        {
            _menuBar.HandleMouse(evt);
            ApplyMenuSelectionIfAny();
            return;
        }

        // Mouse routing to individual screens (list clicks, field clicks) is not
        // wired up yet — this pass only covers the menu bar itself, which is the
        // gap that was actually raised. Per-screen click targets are a natural
        // follow-up once the menu plumbing above is confirmed to feel right.
        switch (_screen)
        {
            case AppScreenKind.ModelBrowser:
                _browser?.HandleMouse(evt);
                break;
        }
    }

    /// <summary>
    /// Reuses the exact same screen-construction logic the keyboard paths
    /// already use for each destination, so picking "Browse for model..."
    /// from the menu behaves identically to pressing M on the Welcome screen
    /// — same defaults, same starting folder, no second code path to drift
    /// out of sync with the first.
    /// </summary>
    private void ApplyMenuSelectionIfAny()
    {
        if (_menuBar.ExitSelected)
        {
            _menuBar.ClearSelection();
            _quit = true;
            return;
        }

        if (_menuBar.NewSessionSelected)
        {
            _menuBar.ClearSelection();
            StartNewSession();
            return;
        }

        if (_menuBar.DebugSessionSelected)
        {
            _menuBar.ClearSelection();
            StartDebugSession();
            return;
        }

        if (_menuBar.SelectedTarget is not { } target) return;
        _menuBar.ClearSelection();

        switch (target)
        {
            case AppScreenKind.Welcome:
                _screen = AppScreenKind.Welcome;
                break;
            case AppScreenKind.ModelBrowser:
                _browser = new ModelBrowserScreen(_options.ProjectPath ?? Directory.GetCurrentDirectory());
                _screen = AppScreenKind.ModelBrowser;
                break;
            case AppScreenKind.Options:
                if (_options.ModelPath is not null || _options.Generator == GeneratorStrategy.UIDebug)
                {
                    _optionsScreen = new OptionsScreen(_options, _options.ModelPath);
                    _screen = AppScreenKind.Options;
                }
                // No model chosen yet: silently decline rather than opening a
                // form with a blank model field — Options without a model
                // doesn't have anything meaningful to launch.
                break;
            case AppScreenKind.Chat:
                if (_chatScreen is not null) _screen = AppScreenKind.Chat;
                // No active session: silently decline — there is nothing to "go to" yet.
                break;
            case AppScreenKind.Settings:
                _settingsScreen = new SettingsScreen(_settings);
                _screen = AppScreenKind.Settings;
                break;
        }
    }

    /// <summary>
    /// File &gt; New Session: tears down whatever's currently running (if
    /// anything), resets session config back to fresh defaults seeded from
    /// AppSettings, and drops straight into the Model Browser — the same
    /// starting point as a cold launch, just without actually restarting the
    /// process.
    /// </summary>
    private void StartNewSession()
    {
        if (_bridge is not null)
        {
            // Fire-and-forget teardown: menu selection handling is synchronous
            // and this method isn't async, but disposing the bridge cleanly
            // doesn't need to block the UI from moving on immediately — the
            // background chat loop task will wind down on its own once
            // cancellation is requested.
            _ = TeardownSessionAsync();
        }

        _options = NewSessionOptionsFromSettings();
        _optionsScreen = null;
        _chatScreen = null;
        _statusMessage = null;
        _browser = new ModelBrowserScreen(_options.ProjectPath ?? Directory.GetCurrentDirectory());
        _screen = AppScreenKind.ModelBrowser;
    }

    /// <summary>
    /// Model &gt; Debug session: the whole point of UIDebug is that no model
    /// is required, so this skips the Model Browser entirely and drops
    /// straight onto the Options screen with GeneratorStrategy.UIDebug
    /// already selected — Launch is reachable in one screen, not three.
    /// </summary>
    private void StartDebugSession()
    {
        if (_bridge is not null) _ = TeardownSessionAsync();

        _options = NewSessionOptionsFromSettings();
        _options.ModelPath = null;
        _options.Generator = GeneratorStrategy.UIDebug;
        _chatScreen = null;
        _statusMessage = null;
        _optionsScreen = new OptionsScreen(_options, null);
        _screen = AppScreenKind.Options;
    }

    private Task<LaunchResult>? _pendingLaunch;

    private void LaunchSessionAsync()
    {
        _statusMessage = "Loading model...";
        var progress = new Progress<string>(s => _statusMessage = s);
        // Run on a background task rather than awaiting inline: LaunchAsync
        // does real synchronous-feeling work (mmap + copy gigabytes of
        // weights) that would otherwise block this thread and freeze the
        // render loop for the entire load, defeating the point of reporting
        // progress at all. The result is picked up next frame in Draw/Update.
        _pendingLaunch = SessionLauncher.LaunchAsync(_options, progress);
    }

    private async Task PollPendingLaunchAsync()
    {
        if (_pendingLaunch is null || !_pendingLaunch.IsCompleted) return;

        var result = await _pendingLaunch;
        _pendingLaunch = null;

        if (!result.Success)
        {
            _statusMessage = $"Launch failed: {result.Error}";
            return; // stay on Options screen so the user can adjust and retry
        }

        if (result.Warnings.Count > 0)
            _statusMessage = string.Join(" | ", result.Warnings);

        _cuiContext = result.CuiContext;
        _bridge = result.IsDebugMode
            ? new DebugChatBridge(result.CuiContext!)
            : new ChatSessionBridge(result.Session!);

        if (_bridge is ChatSessionBridge realBridge) realBridge.Start();

        string displayName = result.IsDebugMode ? $"{_options.AgentName} [DEBUG]" : _options.AgentName;
        _chatScreen = new ChatScreen(displayName, _options);
        _screen = AppScreenKind.Chat;
    }

    /// <summary>
    /// Checked once per frame: has a tool call (real or scripted) posted a
    /// choice request since the last frame? If so, wrap it in a
    /// <see cref="ChoiceDialog"/> so the next frame's Draw/HandleKey pick it
    /// up as the active modal.
    /// </summary>
    private void PollPendingChoiceRequest()
    {
        if (_choiceDialog is null && _cuiContext?.TakePending() is { } request)
            _choiceDialog = new ChoiceDialog(request);
    }

    private async Task TeardownSessionAsync()
    {
        if (_bridge is not null)
        {
            await _bridge.DisposeAsync();
            _bridge = null;
        }
        _chatScreen = null;
        _cuiContext = null;
        _choiceDialog = null;
    }

    private void Draw()
    {
        int w = _buf.Width, h = _buf.Height;
        _menuBar.Draw(_buf, w, _screen, _theme);

        int contentY = 1;
        int contentH = h - 1;

        switch (_screen)
        {
            case AppScreenKind.Welcome:
                WelcomeScreen.Draw(_buf, 0, contentY, w, contentH, _theme);
                break;
            case AppScreenKind.ModelBrowser:
                _browser?.Draw(_buf, 0, contentY, w, contentH, _theme);
                break;
            case AppScreenKind.Options:
                _optionsScreen?.Draw(_buf, 0, contentY, w, contentH, _theme);
                if (_pendingLaunch is not null)
                    _buf.Write(2, h - 1, "Loading, please wait...", _theme.SelectionFg, _theme.Accent);
                else if (_statusMessage is not null)
                    _buf.Write(2, h - 1, _statusMessage.Length > w - 4 ? _statusMessage[..(w - 4)] : _statusMessage, _theme.Accent, _theme.Background);
                break;
            case AppScreenKind.Chat:
                if (_chatScreen is not null)
                {
                    foreach (var entry in _bridge!.DrainEntries())
                        _chatScreen.OnStreamEntry(entry);
                    _chatScreen.Draw(_buf, 0, contentY, w, contentH, _theme);
                }
                break;
            case AppScreenKind.Settings:
                _settingsScreen?.Draw(_buf, 0, contentY, w, contentH, _theme);
                break;
        }

        _menuBar.DrawDropdownOverlay(_buf, _theme);

        _choiceDialog?.Draw(_buf, w, h, _theme);
    }

    public async ValueTask DisposeAsync()
    {
        await TeardownSessionAsync();
    }
}
