namespace SharpMind.Model.Tokenizer.PreTokeniser;

/// <summary>
/// Splits on whitespace. Fast, correct for many languages, but merges
/// punctuation with adjacent words (e.g. "hello," is one pre-token).
/// Good baseline for training from scratch on clean text.
/// </summary>
public sealed class WhitespacePreTokeniser : IPreTokeniser
{
    public IEnumerable<string> PreTokenise(string text)
    {
        static IEnumerable<Range> SplitToRanges(string s)
        {
            int start = -1;
            for (int i = 0; i <= s.Length; i++)
            {
                bool ws = i == s.Length || char.IsWhiteSpace(s[i]);
                if (!ws && start < 0) start = i;
                else if (ws && start >= 0) { yield return start..i; start = -1; }
            }
        }
        foreach (Range range in SplitToRanges(text))
        {
            ReadOnlySpan<char> word = text.AsSpan(range);
            if (!word.IsWhiteSpace())
                yield return word.ToString();
        }
    }
}
