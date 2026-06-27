using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>Shown while SessionLauncher.LaunchAsync is running. SetMessage updates the text from the progress callback.</summary>
public sealed class LoadingView : View
{
    private readonly Label _label;

    public LoadingView(string initialMessage)
    {
        _label = new Label(initialMessage)
        {
            X = Pos.Center(),
            Y = Pos.Center(),
            TextAlignment = TextAlignment.Centered
        };
        Add(_label);
    }

    /// <summary>Must be called from the UI thread — wrap in Application.MainLoop.Invoke if called from a background continuation.</summary>
    public void SetMessage(string message)
    {
        _label.Text = message;
        SetNeedsDisplay();
    }
}
