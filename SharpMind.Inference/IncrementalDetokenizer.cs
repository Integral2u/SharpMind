using SharpMind.Tokenization;

namespace SharpMind.Inference;

/// <summary>
/// Turns a growing token sequence into text deltas without splitting UTF-8.
/// Decoding one token at a time loses any character whose bytes span tokens:
/// each partial byte decodes to U+FFFD, and stripping those drops the character.
/// This keeps a window of tokens and emits only text that is complete, the way
/// Hugging Face's streamer does.
/// </summary>
internal static class IncrementalDetokenizer
{
    private const char Replacement = '\uFFFD';

    /// <summary>
    /// Returns the newly complete text for <paramref name="ids"/>. State lives in
    /// <paramref name="prefixStart"/> and <paramref name="readEnd"/>: start both at 0
    /// and pass the same refs on every call. Text ending in an incomplete sequence
    /// is held back until a later token completes it, unless <paramref name="final"/>
    /// is set. U+FFFD is removed from the output, as it was before.
    /// </summary>
    public static string Next(Tokenizer tokenizer, ReadOnlySpan<int> ids,
        ref int prefixStart, ref int readEnd, bool final = false)
    {
        if (ids.Length <= readEnd && !final) return "";

        string prefixText = readEnd > prefixStart
            ? tokenizer.Decode(ids[prefixStart..readEnd], skipSpecials: true)
            : "";
        string newText = ids.Length > prefixStart
            ? tokenizer.Decode(ids[prefixStart..], skipSpecials: true)
            : "";

        bool complete = final || !newText.EndsWith(Replacement);
        if (newText.Length <= prefixText.Length || !complete)
            return "";

        string delta = newText[prefixText.Length..].Replace(Replacement.ToString(), "");
        prefixStart = readEnd;
        readEnd = ids.Length;
        return delta;
    }
}
