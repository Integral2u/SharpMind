namespace SharpMind.Tokenizer.PreTokeniser;

/// <summary>
/// Splits raw text into "words" before BPE is applied.
/// Pre-tokenisation determines which character sequences can ever be merged —
/// BPE never merges across a pre-token boundary.
/// </summary>
public interface IPreTokeniser
{
    /// <summary>
    /// Splits <paramref name="text"/> into a sequence of pre-tokens.
    /// Each pre-token is later independently encoded by BPE.
    /// </summary>
    IEnumerable<string> PreTokenise(string text);
}