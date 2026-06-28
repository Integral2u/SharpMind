using SharpMind.CUI.App;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// The application's single Window: a MenuBar pinned to the top, and a
/// content area that fills the rest, into which each "screen" (Welcome,
/// model browser, options, settings, chat) gets swapped via RemoveAll/Add —
/// the standard Terminal.Gui v1 pattern for single-window navigation between
/// full-screen views. There is no longer a custom AppScreenKind enum or
/// frame-loop dispatch table; Terminal.Gui's own container/focus model
/// replaces all of that.
/// </summary>
public sealed class MainWindow : Window
{
    private readonly View _content;
    private readonly AppSettings _settings;
    private SessionOptions _options;

    private ChatSessionState? _activeSession; // null until a session is actually running

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
                new("_Session options...", "", ShowOptionsIfModelChosen)
            }),
            new("_Chat", new MenuItem[]
            {
                new("_Go to chat", "", ShowChatIfActive)
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

    private void ShowWelcome() => SwapContent(new WelcomeView(
        onBrowseModel: ShowModelBrowser));

    private void ShowModelBrowser()
    {
        Action onCancel = _activeSession is not null ? ShowChat : ShowWelcome;
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

    private void ShowOptions() => SwapContent(new OptionsView(_options,
        onLaunch: LaunchSession,
        onCancel: ShowModelBrowser));

    /// <summary>Menu path into Options — declines quietly if there's nothing to configure yet, same rule the old AppScreenKind dispatch enforced.</summary>
    private void ShowOptionsIfModelChosen()
    {
        if (_options.ModelPath is not null || _options.Generator == GeneratorStrategy.UIDebug)
            ShowOptions();
    }

    private void ShowSettings()
    {
        Action onBack = _activeSession is not null ? ShowChat : ShowWelcome;
        var view = new SettingsView(_settings, onBack: onBack);
        view.OnThemeChanged = kind => ThemeBuilder.ApplyRecursively(this, ThemeBuilder.Build(kind));
        SwapContent(view);
    }

    private void ShowChatIfActive()
    {
        if (_activeSession is not null) ShowChat();
    }

    private void ShowChat()
    {
        if (_activeSession is null) return;
        SwapContent(_activeSession.View);
    }

    // --- Session lifecycle -------------------------------------------------

    private void StartNewSession()
    {
        TeardownActiveSession();
        _options = NewSessionOptionsFromSettings();
        ShowModelBrowser();
    }

    private void StartDebugSession()
    {
        TeardownActiveSession();
        _options = NewSessionOptionsFromSettings();
        _options.ModelPath = null;
        _options.Generator = GeneratorStrategy.UIDebug;
        ShowOptions();
    }

    private async void LaunchSession()
    {
        var progressView = new LoadingView("Loading model...");
        SwapContent(progressView);

        var progress = new Progress<string>(s => Application.MainLoop.Invoke(() => progressView.SetMessage(s)));
        var result = await SessionLauncher.LaunchAsync(_options, progress);

        if (!result.Success)
        {
            MessageBox.ErrorQuery("Launch failed", result.Error ?? "Unknown error", "OK");
            ShowOptions();
            return;
        }

        IChatBridge bridge = result.IsDebugMode
            ? new DebugChatBridge(result.CuiContext!)
            : new ChatSessionBridge(result.Session!);

        if (bridge is ChatSessionBridge realBridge) realBridge.Start();

        string displayName = result.IsDebugMode ? $"{_options.AgentName} [DEBUG]" : _options.AgentName;
        var chatView = new ChatView(displayName, _options, bridge, result.CuiContext, onExit: () =>
        {
            TeardownActiveSession();
            ShowOptions();
        });

        _activeSession = new ChatSessionState(bridge, chatView);

        if (result.Warnings.Count > 0)
            MessageBox.Query("Session started with warnings", string.Join("\n", result.Warnings), "OK");

        ShowChat();
    }

    private void TeardownActiveSession()
    {
        if (_activeSession is null) return;
        _ = _activeSession.Bridge.DisposeAsync(); // fire-and-forget: see App.cs's earlier equivalent for why this is an accepted trade-off
        _activeSession.View.Dispose();
        _activeSession = null;
    }

    private sealed record ChatSessionState(IChatBridge Bridge, ChatView View);
}
