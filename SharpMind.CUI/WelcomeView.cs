using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>The landing view: banner, brief instructions, a button into the model browser.</summary>
public sealed class WelcomeView : View
{
    public WelcomeView(
        Action onBrowseModel,
        string? lastModelName = null,
        Action? onContinueWithModel = null,
        string? lastSessionName = null,
        Action? onResumeLastSession = null,
        Action? onTrainModel = null)
    {
        var bannerText = "  ___ _                  __  __ _         _ \n / __| |_  __ _ _ _ _ __|  \\/  (_)_ _  __| |\n \\__ \\ ' \\/ _` | '_| '_ \\ |\\/| | | ' \\/ _` |\n |___/_||_\\__,_|_| | .__/_|  |_|_|_||_\\__,_|\n                   |_|                      ";
        var banner = new Label(bannerText)
        {
            X = Pos.Center(),
            Y = Pos.Center() - 6,
            TextAlignment = TextAlignment.Centered
        };

        var subtitle = new Label("open source llm · pure c#")
        {
            X = Pos.Center(),
            Y = Pos.Center() - 1,
            TextAlignment = TextAlignment.Centered
        };

        var browseButton = new Button("Browse for a model")
        {
            X = Pos.Center(),
            Y = Pos.Center() + 1,
            IsDefault = true
        };
        browseButton.Clicked += () => onBrowseModel();

        var hint = new Label("Use the File / Model menus above for more options, or Esc to step back.")
        {
            X = Pos.Center(),
            Y = Pos.AnchorEnd() - 1,
            TextAlignment = TextAlignment.Centered
        };

        var items = new List<View> { banner, subtitle, browseButton, hint };

        if (lastModelName is not null && onContinueWithModel is not null)
        {
            var modelBtn = new Button($"Continue with {lastModelName}")
            {
                X = Pos.Center(),
                Y = Pos.Bottom(browseButton) + 1
            };
            modelBtn.Clicked += () => onContinueWithModel();
            items.Insert(items.Count - 1, modelBtn);
        }

        if (lastSessionName is not null && onResumeLastSession is not null)
        {
            var sessionBtn = new Button($"Resume session \"{lastSessionName}\"")
            {
                X = Pos.Center(),
                Y = Pos.Bottom((View)items[^2]) + 1
            };
            sessionBtn.Clicked += () => onResumeLastSession();
            items.Insert(items.Count - 1, sessionBtn);
        }

        if (onTrainModel is not null)
        {
            var trainBtn = new Button("Train a model…")
            {
                X = Pos.Center(),
                Y = Pos.Bottom((View)items[^2]) + 1
            };
            trainBtn.Clicked += () => onTrainModel();
            items.Insert(items.Count - 1, trainBtn);
        }

        Add(items.ToArray());
    }
}
