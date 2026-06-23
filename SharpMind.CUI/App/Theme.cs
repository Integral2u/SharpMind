namespace SharpMind.CUI.App;

public enum ThemeKind { ClassicBlue, HighContrastDark, Monochrome }

/// <summary>
/// A named set of colour roles. Screens ask for "background" or "accent",
/// never a raw <see cref="ConsoleColor"/> — that's what makes switching
/// themes a one-line change at the call site instead of a find-and-replace
/// across seven files.
///
/// Note the real limitation here: <see cref="ConsoleColor"/> is a fixed
/// 16-colour enum with no true RGB. Switching themes means picking a
/// different combination from that same small palette, not unlocking new
/// colours outright — "bright" vs "easier to read" is mostly a question of
/// which 2 or 3 of those 16 you put text on top of each other with, not
/// something a wider colour gamut would meaningfully change here.
/// </summary>
public sealed record Theme(
    ConsoleColor Background,
    ConsoleColor Text,
    ConsoleColor DimText,
    ConsoleColor Accent,
    ConsoleColor SelectionFg,
    ConsoleColor SelectionBg,
    ConsoleColor Success,
    ConsoleColor Warning,
    ConsoleColor Error,
    ConsoleColor BarFg,
    ConsoleColor BarBg)
{
    /// <summary>
    /// The original palette: classic DOS blue background, yellow accents,
    /// white text. Kept available since it's the nostalgic reference point,
    /// but yellow-on-blue and white-on-blue are both genuinely low-contrast
    /// combinations to stare at for a while, which is exactly the complaint
    /// that prompted adding alternatives.
    /// </summary>
    public static readonly Theme ClassicBlue = new(
        Background: ConsoleColor.Blue,
        Text: ConsoleColor.Gray,
        DimText: ConsoleColor.Cyan,
        Accent: ConsoleColor.Yellow,
        SelectionFg: ConsoleColor.Black,
        SelectionBg: ConsoleColor.Yellow,
        Success: ConsoleColor.Green,
        Warning: ConsoleColor.Yellow,
        Error: ConsoleColor.Red,
        BarFg: ConsoleColor.Black,
        BarBg: ConsoleColor.Gray);

    /// <summary>
    /// Default theme. Black background instead of blue — white-on-black and
    /// cyan-on-black both read considerably easier over long sessions than
    /// anything-on-bright-blue, while keeping the same role assignments so it
    /// still feels like the same app, just calmer on the eyes.
    /// </summary>
    public static readonly Theme HighContrastDark = new(
        Background: ConsoleColor.Black,
        Text: ConsoleColor.White,
        DimText: ConsoleColor.Cyan,
        Accent: ConsoleColor.Yellow,
        SelectionFg: ConsoleColor.Black,
        SelectionBg: ConsoleColor.White,
        Success: ConsoleColor.Green,
        Warning: ConsoleColor.Yellow,
        Error: ConsoleColor.Red,
        BarFg: ConsoleColor.Black,
        BarBg: ConsoleColor.Gray);

    /// <summary>
    /// For terminals that remap their own 16-colour palette, or for anyone
    /// who just wants the lowest-strain option: greyscale only, no hue at
    /// all, contrast carried purely by light-vs-dark.
    /// </summary>
    public static readonly Theme Monochrome = new(
        Background: ConsoleColor.Black,
        Text: ConsoleColor.Gray,
        DimText: ConsoleColor.DarkGray,
        Accent: ConsoleColor.White,
        SelectionFg: ConsoleColor.Black,
        SelectionBg: ConsoleColor.Gray,
        Success: ConsoleColor.White,
        Warning: ConsoleColor.White,
        Error: ConsoleColor.White,
        BarFg: ConsoleColor.Black,
        BarBg: ConsoleColor.DarkGray);

    public static Theme For(ThemeKind kind) => kind switch
    {
        ThemeKind.ClassicBlue => ClassicBlue,
        ThemeKind.HighContrastDark => HighContrastDark,
        ThemeKind.Monochrome => Monochrome,
        _ => HighContrastDark
    };
}
