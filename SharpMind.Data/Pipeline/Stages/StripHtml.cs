using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>
/// Removes HTML tags. Decodes common HTML entities (&amp;, &lt;, &gt;, &quot;, &apos;, &#NNN;).
/// </summary>
public sealed class StripHtml : ICleaningStage
{
    public string Name => "StripHtml";

    private static readonly Regex TagPattern =
        new(@"<[^>]+>", RegexOptions.Compiled, TimeSpan.FromSeconds(1));

    private static readonly Regex EntityPattern =
        new(@"&(?:#(\d+)|#x([0-9a-fA-F]+)|([a-zA-Z]+));",
            RegexOptions.Compiled, TimeSpan.FromSeconds(1));

    private static readonly Dictionary<string, string> NamedEntities = new()
    {
        ["amp"] = "&",
        ["lt"] = "<",
        ["gt"] = ">",
        ["quot"] = "\"",
        ["apos"] = "'",
        ["nbsp"] = " ",
    };

    public string? Process(string document)
    {
        string stripped = TagPattern.Replace(document, " ");
        string decoded = EntityPattern.Replace(stripped, DecodeEntity);
        string trimmed = decoded.Trim();
        return trimmed.Length == 0 ? null : trimmed;
    }

    private static string DecodeEntity(Match m)
    {
        if (m.Groups[1].Success && int.TryParse(m.Groups[1].Value, out int dec))
            return ((char)dec).ToString();
        if (m.Groups[2].Success && int.TryParse(m.Groups[2].Value,
                System.Globalization.NumberStyles.HexNumber, null, out int hex))
            return ((char)hex).ToString();
        if (m.Groups[3].Success &&
            NamedEntities.TryGetValue(m.Groups[3].Value.ToLowerInvariant(), out string? named))
            return named;
        return m.Value;
    }
}