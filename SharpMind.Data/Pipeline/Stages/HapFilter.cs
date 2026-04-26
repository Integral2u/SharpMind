using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;

/// <summary>
/// Used to Filters or redacts hate speech, abuse, and profanity.
/// List not provided in this source
/// </summary>
public sealed class HapFilter : ICleaningStage
{
    public enum HapMode { Discard, Redact }

    private readonly Regex _pattern;
    private readonly HapMode _mode;

    private static readonly string[] BuiltInTerms = Decrypt(
    [
        // This list is intentionally minimal and non-exhaustive.
        // Replace or augment with a comprehensive HAP lexicon for production. (encrypted)
        "ojhhfs","ojhhb","gbhhpu","ljlf","tqjd","dijol","hppl","xfucbdl","usbooz","sfubse",
        "dvou","gvdl","tiju","cbtubse"
    ]);

    /// <param name="mode">
    /// <see cref="HapMode.Discard"/> drops documents containing HAP content.
    /// <see cref="HapMode.Redact"/> replaces matches with <c>[HAP]</c>.
    /// </param>
    /// <param name="additionalTerms">Extra terms to add to the built-in list.</param>
    public HapFilter(HapMode mode, IEnumerable<string> filterList, bool useBuiltIn = true)
    {
        _mode = mode;        
        var terms = useBuiltIn? BuiltInTerms.Concat(filterList ?? []).Select(Regex.Escape) : filterList.Select(Regex.Escape);
        _pattern = new Regex(
            $@"\b(?:{string.Join('|', terms)})\b",
            RegexOptions.Compiled | RegexOptions.IgnoreCase,
            TimeSpan.FromSeconds(2));
    }

    public string Name => $"HapFilter({_mode})";

    public string? Process(string document)
    {
        if (!_pattern.IsMatch(document)) return document;
        return _mode == HapMode.Redact
            ? _pattern.Replace(document, "[HAP]")
            : null;
    }

    public static string[] Encrypt(string[] terms)
    {
        var ret = new string[terms.Length];
        for (int i = 0; i < terms.Length; i++)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(terms[i]);
            for (int j = 0; j < bytes.Length; j++) bytes[j] = (byte)(bytes[j] + 1);
            ret[i] = System.Text.Encoding.UTF8.GetString(bytes);
        }
        return ret;
    }

    public static string[] Decrypt(string[] terms)
    {
        var ret = new string[terms.Length];
        for (int i = 0; i < terms.Length; i++)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(terms[i]);
            for (int j = 0; j < bytes.Length; j++) bytes[j] = (byte)(bytes[j] - 1);
            ret[i] = System.Text.Encoding.UTF8.GetString(bytes);
        }
        return ret;
    }
}
