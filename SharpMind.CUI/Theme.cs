using Terminal.Gui;

namespace SharpMind.CUI;

public enum ThemeKind { ClassicBlue, HighContrastDark, Monochrome }

/// <summary>
/// Builds a Terminal.Gui ColorScheme for each named theme. Assigned directly
/// to each top-level view's ColorScheme property — Terminal.Gui's global
/// Colors.ColorSchemes registry exists too, but per several maintainer
/// discussions its pickup is inconsistent across view types unless every
/// view explicitly references the named scheme, so setting it directly per
/// view is the more predictable path for a small app like this one.
/// </summary>
public static class ThemeBuilder
{
    public static ColorScheme Build(ThemeKind kind) => kind switch
    {
        ThemeKind.ClassicBlue => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.Gray, Color.Blue),
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.Cyan),
            HotNormal = new Terminal.Gui.Attribute(Color.BrightYellow, Color.Blue),
            HotFocus = new Terminal.Gui.Attribute(Color.BrightYellow, Color.Cyan),
            Disabled = new Terminal.Gui.Attribute(Color.DarkGray, Color.Blue)
        },
        ThemeKind.Monochrome => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.Gray, Color.Black),
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.Gray),
            HotNormal = new Terminal.Gui.Attribute(Color.White, Color.Black),
            HotFocus = new Terminal.Gui.Attribute(Color.White, Color.Gray),
            Disabled = new Terminal.Gui.Attribute(Color.DarkGray, Color.Black)
        },
        // HighContrastDark is the default: black background reads far easier
        // over long sessions than the classic bright-blue, which was the
        // actual readability complaint that prompted having more than one
        // option in the first place.
        _ => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.White, Color.Black),
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.White),
            HotNormal = new Terminal.Gui.Attribute(Color.BrightYellow, Color.Black),
            HotFocus = new Terminal.Gui.Attribute(Color.BrightYellow, Color.White),
            Disabled = new Terminal.Gui.Attribute(Color.DarkGray, Color.Black)
        }
    };

    /// <summary>
    /// Recursively applies the scheme to a view and every one of its
    /// subviews — Terminal.Gui views don't automatically inherit a parent's
    /// ColorScheme unless it's left null, and several controls (Button,
    /// TextField) set their own default on construction, so an explicit
    /// walk is the reliable way to actually theme an entire screen rather
    /// than just its outermost container.
    /// </summary>
    public static void ApplyRecursively(View view, ColorScheme scheme)
    {
        view.ColorScheme = scheme;
        foreach (var sub in view.Subviews)
            ApplyRecursively(sub, scheme);
    }
}
