using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>The landing view: banner, brief instructions, a button into the model browser.</summary>
public sealed class WelcomeView : View
{
    public WelcomeView(Action onBrowseModel)
    {
        var banner = new Label("SharpMind CUI")
        {
            X = Pos.Center(),
            Y = Pos.Center() - 4,
            TextAlignment = TextAlignment.Centered
        };

        var subtitle = new Label("An experimentation console for local language models")
        {
            X = Pos.Center(),
            Y = Pos.Center() - 2,
            TextAlignment = TextAlignment.Centered
        };

        var browseButton = new Button("Browse for a model")
        {
            X = Pos.Center(),
            Y = Pos.Center(),
            IsDefault = true
        };
        browseButton.Clicked += () => onBrowseModel();

        var hint = new Label("Use the File / Model menus above for more options, or Esc to step back.")
        {
            X = Pos.Center(),
            Y = Pos.Center() + 2,
            TextAlignment = TextAlignment.Centered
        };

        Add(banner, subtitle, browseButton, hint);
    }
}
