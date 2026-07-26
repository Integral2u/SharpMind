using System.Text.RegularExpressions;

namespace SharpMind.Tokenization.PreTokeniser;

/// <summary>
/// GPT-2 / RoBERTa-style pre-tokeniser.
/// Uses the regex from the original GPT-2 paper to split on:
///   contractions ('s, 're, 've, etc.)
///   punctuation
///   digits
///   whitespace-prefixed words (Ġ prefix convention)
///
/// This is the correct pre-tokeniser for any model that uses the GPT-2
/// vocabulary or merges (GPT-2, GPT-Neo, CodeGen, Bloom, Falcon).
/// </summary>
public sealed class Gpt2PreTokeniser : IPreTokeniser
{
    public IEnumerable<string> PreTokenise(string text)
    {
        foreach (Match m in RegexGenerated.Gpt2Pattern.Matches(text))
            if (m.Length > 0)
                yield return m.Value;
    }
}
