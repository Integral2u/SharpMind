using Terminal.Gui;

namespace SharpMind.CUI;

public enum ThemeKind { ClassicBlue,Green, HighContrastDark, HighContrastLight, Monochrome }

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
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.BrightBlue),
            HotNormal = new Terminal.Gui.Attribute(Color.BrightYellow, Color.Blue),
            HotFocus = new Terminal.Gui.Attribute(Color.BrightYellow, Color.BrightBlue),
            Disabled = new Terminal.Gui.Attribute(Color.Black, Color.Blue)
        },
        ThemeKind.Green => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.Gray, Color.Green),
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.BrightGreen),
            HotNormal = new Terminal.Gui.Attribute(Color.BrightYellow, Color.Green),
            HotFocus = new Terminal.Gui.Attribute(Color.BrightYellow, Color.BrightGreen),
            Disabled = new Terminal.Gui.Attribute(Color.Black, Color.Green)
        },
        ThemeKind.Monochrome => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.Gray, Color.Black),
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.Gray),
            HotNormal = new Terminal.Gui.Attribute(Color.White, Color.Black),
            HotFocus = new Terminal.Gui.Attribute(Color.White, Color.Gray),
            Disabled = new Terminal.Gui.Attribute(Color.DarkGray, Color.Black)
        },
        ThemeKind.HighContrastLight => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.Black, Color.White),
            Focus = new Terminal.Gui.Attribute(Color.White, Color.Black),
            HotNormal = new Terminal.Gui.Attribute(Color.Red, Color.White),
            HotFocus = new Terminal.Gui.Attribute(Color.Red, Color.Black),
            Disabled = new Terminal.Gui.Attribute(Color.DarkGray, Color.White)
        },
        // HighContrastDark is the default: black background reads far easier
        // over long sessions than the classic bright-blue, which was the
        // actual readability complaint that prompted having more than one
        // option in the first place.
        _ => new ColorScheme
        {
            Normal = new Terminal.Gui.Attribute(Color.White, Color.Black),
            Focus = new Terminal.Gui.Attribute(Color.Black, Color.White),
            HotNormal = new Terminal.Gui.Attribute(Color.Red, Color.Black),
            HotFocus = new Terminal.Gui.Attribute(Color.Red, Color.White),
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
