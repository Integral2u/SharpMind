using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>The landing view: banner, brief instructions, a button into the model browser.</summary>
public sealed class WelcomeView : View
{
    public WelcomeView(Action onBrowseModel)
    {
        var bannerText = "  ___ _                  __  __ _           _ \n / __| |_  __ _ _ _ _ __ |  \\/  (_)_ _  __| |\n \\__ \\ ' \\/ _` | '_| '_ \\| |\\/| | | ' \\/ _` |\n |___/_||_\\__,_|_| | .__/|_|  |_|_|_||_\\__,_|\n                   |_|                        ";
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
            Y = Pos.Center() + 3,
            TextAlignment = TextAlignment.Centered
        };

        Add(banner, subtitle, browseButton, hint);
    }
}
