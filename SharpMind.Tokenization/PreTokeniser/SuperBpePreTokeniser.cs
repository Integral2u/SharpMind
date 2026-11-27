namespace SharpMind.Tokenization.PreTokeniser;

/// <summary>
/// ZONOS3's SuperBPE pre-tokeniser. Unlike GPT-2 it does not split at word
/// boundaries: BPE merges may cross spaces. The Split regex (behavior Isolated) is
/// <c>\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)</c>, implemented exactly as
/// Zyphra's <c>zonos3-tokenizer.cpp</c> <c>pretokenize()</c>.
/// </summary>
public sealed class SuperBpePreTokeniser : IPreTokeniser
{
    public IEnumerable<string> PreTokenise(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        int length = text.Length;
        int gap = 0, i = 0;
        while (i < length)
        {
            int end = MatchAt(text, i);
            if (end == i)
            {
                i++;
                continue;
            }

            if (gap < i) yield return text[gap..i];
            yield return text[i..end];
            i = gap = end;
        }

        if (gap < length) yield return text[gap..];
    }

    private static int MatchAt(string text, int start)
    {
        int length = text.Length;

        // \p{N}{1,3}
        if (IsNumber(text[start]))
        {
            int j = start;
            while (j < length && j - start < 3 && IsNumber(text[j])) j++;
            return j;
        }

        // ' ?[^\s\p{L}\p{N}]{2,}[\r\n/]*
        {
            int j = start + (text[start] == ' ' ? 1 : 0);
            int k = j;
            while (k < length && IsOther(text[k])) k++;
            if (k - j >= 2)
            {
                while (k < length && text[k] is '\r' or '\n' or '/') k++;
                return k;
            }
        }

        // ' +(?!\S)
        if (text[start] == ' ')
        {
            int end = start;
            while (end < length && text[end] == ' ') end++;
            if (end == length || IsSpace(text[end])) return end;
            if (end - start >= 2) return end - 1;
        }

        return start;
    }

    private static bool IsNumber(char c) => char.IsNumber(c);

    private static bool IsLetter(char c) => char.IsLetter(c);

    private static bool IsOther(char c) => !IsSpace(c) && !IsLetter(c) && !IsNumber(c);

    // Unicode White_Space (Oniguruma \s), matching Zyphra's table.
    private static bool IsSpace(char c) =>
        (c >= '\u0009' && c <= '\u000D') || c == '\u0020' || c == '\u0085' || c == '\u00A0' ||
        c == '\u1680' || (c >= '\u2000' && c <= '\u200A') || c == '\u2028' || c == '\u2029' ||
        c == '\u202F' || c == '\u205F' || c == '\u3000';
}
