using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace SharpMind.Tokenization.PreTokeniser;

/// <summary>
/// llama.cpp's multi-regex pre-tokeniser (<c>unicode_regex_split</c>, b8683): each regex in turn splits every
/// piece produced so far into its matches and the gaps between them; the final pieces are the pre-tokens.
/// This is how the DeepSeek pre types split, so their token ids match llama.cpp for the same vocab.
///
/// Two llama.cpp mechanics are reproduced because its regexes run on code points and .NET's on UTF-16 units.
/// A stage that uses <c>\p{..}</c> classes runs on a <i>collapsed</i> view where every astral code point is one
/// BMP stand-in of its Unicode category (llama.cpp replaces every non-ASCII code point by a category marker):
/// <c>\p{S}</c> matches an emoji and no class can cut a surrogate pair in half. A stage without them runs on
/// the text itself with every non-ASCII whitespace shown as <c>\v</c> (llama.cpp's std::wregex <c>\s</c>
/// fallback), so a literal class such as <c>[\u3000-\u3002]</c> cannot claim an ideographic space.
/// </summary>
public sealed class SequentialRegexPreTokeniser : IPreTokeniser
{
    private readonly (Regex Regex, bool Collapse)[] _stages;

    /// <param name="name">llama.cpp's <c>tokenizer.ggml.pre</c> name; also the name in a saved tokenizer file.</param>
    /// <param name="stages">The regexes in llama.cpp's order.</param>
    public SequentialRegexPreTokeniser(string name, params Regex[] stages)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (stages.Length == 0) throw new ArgumentException("At least one stage is required.", nameof(stages));
        Name = name;
        _stages = [.. stages.Select(r => (r, r.ToString().Contains(@"\p{", StringComparison.Ordinal)))];
    }

    public string Name { get; }

    /// <summary>LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM (<c>deepseek-llm</c>).</summary>
    public static SequentialRegexPreTokeniser DeepSeekLlm { get; } = new("deepseek-llm",
        RegexGenerated.NewlineStage, RegexGenerated.DeepSeekLlmLetters, RegexGenerated.DeepSeekLlmPunctuation,
        RegexGenerated.TrailingWhitespace, RegexGenerated.DeepSeekCjk, RegexGenerated.DigitRun);

    /// <summary>LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER (<c>deepseek-coder</c>).</summary>
    public static SequentialRegexPreTokeniser DeepSeekCoder { get; } = new("deepseek-coder",
        RegexGenerated.NewlineStage, RegexGenerated.DeepSeekCoderLetters, RegexGenerated.DeepSeekCoderPunctuation,
        RegexGenerated.DeepSeekCjk, RegexGenerated.SingleDigit);

    /// <summary>LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM (<c>deepseek-v3</c>, also <c>hunyuan-dense</c>, <c>joyai-llm</c>).</summary>
    public static SequentialRegexPreTokeniser DeepSeek3 { get; } = new("deepseek-v3",
        RegexGenerated.DigitGroup, RegexGenerated.DeepSeek3Cjk, RegexGenerated.DeepSeek3Main);

    public IEnumerable<string> PreTokenise(string text)
    {
        if (text.Length == 0) yield break;
        List<(int Start, int Length)> pieces = [(0, text.Length)];
        foreach (var (regex, collapse) in _stages)
        {
            var next = new List<(int Start, int Length)>(pieces.Count);
            foreach (var piece in pieces)
                Split(regex, collapse, text, piece, next);
            pieces = next;
        }
        foreach (var (start, length) in pieces)
            yield return text.Substring(start, length);
    }

    /// <summary>
    /// llama.cpp <c>unicode_regex_split_stl</c>: the piece becomes its matches and the gaps between them, in order.
    /// Empty matches are skipped, as a regex iterator skips them.
    /// </summary>
    private static void Split(Regex regex, bool collapse, string text, (int Start, int Length) piece, List<(int Start, int Length)> output)
    {
        ReadOnlySpan<char> span = text.AsSpan(piece.Start, piece.Length);
        int[]? map = null; // view index -> offset in the piece, when the view is not the piece itself
        ReadOnlySpan<char> view = span;
        if (!Ascii.IsValid(span))
        {
            var sb = new StringBuilder(span.Length);
            var offsets = new List<int>(span.Length + 1);
            for (int i = 0; i < span.Length; i++)
            {
                char c = span[i];
                offsets.Add(i);
                if (collapse && char.IsHighSurrogate(c) && i + 1 < span.Length && char.IsLowSurrogate(span[i + 1]))
                {
                    sb.Append(CategoryStandIn(new Rune(c, span[i + 1])));
                    i++;
                }
                else if (!collapse && c > 0x7F && char.IsWhiteSpace(c))
                    sb.Append('\v');
                else
                    sb.Append(c);
            }
            offsets.Add(span.Length);
            map = [.. offsets];
            view = sb.ToString();
        }

        int start = 0;
        foreach (var match in regex.EnumerateMatches(view))
        {
            if (match.Length == 0) continue;
            if (match.Index > start) Emit(start, match.Index);
            Emit(match.Index, match.Index + match.Length);
            start = match.Index + match.Length;
        }
        if (start < view.Length) Emit(start, view.Length);

        void Emit(int from, int to)
        {
            int a = map is null ? from : map[from];
            int b = map is null ? to : map[to];
            output.Add((piece.Start + a, b - a));
        }
    }

    /// <summary>
    /// One BMP character per Unicode category, standing in for an astral code point the way llama.cpp's
    /// collapsed text uses one marker byte per category. Each is outside every literal class of the
    /// <c>\p{}</c> stages and is not whitespace.
    /// </summary>
    private static char CategoryStandIn(Rune rune) => Rune.GetUnicodeCategory(rune) switch
    {
        UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter => '\u00AA',           // U+00AA  Lo
        UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber => '\u00B2', // U+00B2 No
        UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation
            or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation
            or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation => '\u00A1', // U+00A1 Po
        UnicodeCategory.MathSymbol or UnicodeCategory.CurrencySymbol or UnicodeCategory.ModifierSymbol
            or UnicodeCategory.OtherSymbol => '\u00A2',                                             // U+00A2 Sc
        UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark => '\u0300', // Mn
        _ => '\u00AD',                                                                              // soft hyphen, Cf: matches no class
    };
}
