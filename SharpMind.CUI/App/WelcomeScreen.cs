using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>The landing screen: banner, brief instructions, nothing fancier than that.</summary>
public static class WelcomeScreen
{
    public static void Draw(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor bg = theme.Background;
        ConsoleColor fg = theme.Text;
        ConsoleColor accent = theme.Accent;

        buf.FillRect(x, y, w, h, ' ', fg, bg);

        string[] banner =
        [
            "  ___ _                  __  __ _           _ ",
            " / __| |_  __ _ _ _ _ __ |  \\/  (_)_ _  __| |",
            " \\__ \\ ' \\/ _` | '_| '_ \\| |\\/| | | ' \\/ _` |",
            " |___/_||_\\__,_|_| | .__/|_|  |_|_|_||_\\__,_|",
            "                   |_|                        "
        ];

        int bannerY = y + h / 2 - 6;
        foreach (var (line, i) in banner.Select((l, i) => (l, i)))
            buf.WriteCentered(x, bannerY + i, w, line, accent, bg);

        buf.WriteCentered(x, bannerY + banner.Length + 2, w, "An experimentation console for local language models", fg, bg);

        buf.WriteCentered(x, bannerY + banner.Length + 5, w, "[ M ]  Browse for a model", theme.Text, bg);
        buf.WriteCentered(x, bannerY + banner.Length + 6, w, "[ Alt+F ]  File menu (Settings, New Session)", theme.Text, bg);
        buf.WriteCentered(x, bannerY + banner.Length + 7, w, "[ Esc ]  Quit", theme.Text, bg);
    }
}
