using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;

[Flags]
public enum PiiType
{
    None = 0,
    Email = 1 << 0,
    Phone = 1 << 1,
    Ssn = 1 << 2,
    CreditCard = 1 << 3,
    IpAddress = 1 << 4,
    Url = 1 << 5,
    All = Email | Phone | Ssn | CreditCard | IpAddress | Url
}

public sealed class PiiMasker : ICleaningStage
{
    private readonly (Regex Regex, string Replacement)[] _rules;

    private static readonly (string Pattern, string Replacement)[] DefaultPatterns =
    [
        (@"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
        (@"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
        (@"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "[SSN]"),
        (@"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b", "[CREDIT_CARD]"),
        (@"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_ADDRESS]"),   //IPv4
        (@"\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){2,7}\b", "[IP_ADDRESS]"), // IPv6 (simplified)
        (@"https?://[^\s]+", "[URL]"),
    ];

    public PiiMasker(PiiType types = PiiType.All)
    {
        var rules = new List<(Regex, string)>();
        if ((types & PiiType.Email) != 0)
            rules.Add(MakeRule(DefaultPatterns[0]));
        if ((types & PiiType.Phone) != 0)
            rules.Add(MakeRule(DefaultPatterns[1]));
        if ((types & PiiType.Ssn) != 0)
            rules.Add(MakeRule(DefaultPatterns[2]));
        if ((types & PiiType.CreditCard) != 0)
            rules.Add(MakeRule(DefaultPatterns[3]));
        if ((types & PiiType.IpAddress) != 0)
        {
            rules.Add(MakeRule(DefaultPatterns[4]));
            rules.Add(MakeRule(DefaultPatterns[5]));
        }
        if ((types & PiiType.Url) != 0)
            rules.Add(MakeRule(DefaultPatterns[6]));
        _rules = [.. rules];
    }

    public string Name => $"PiiMasker({string.Join(",", _rules.Select(r => r.Replacement))})";

    public string? Process(string document)
    {
        if (document is null) return null;
        string result = document;
        foreach (var (regex, replacement) in _rules)
        {
            result = regex.Replace(result, replacement);
        }
        return result;
    }

    private static (Regex, string) MakeRule((string Pattern, string Replacement) rule)
        => (new Regex(rule.Pattern, RegexOptions.Compiled, TimeSpan.FromSeconds(2)), rule.Replacement);
}