using System.Text.RegularExpressions;

namespace SharpMind.Model.Tokenizer.PreTokeniser;

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
    // The original GPT-2 regex from openai/gpt-2 encoder.py
    private static readonly Regex Pattern = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture,
        TimeSpan.FromSeconds(2));

    public IEnumerable<string> PreTokenise(string text)
    {
        foreach (Match m in Pattern.Matches(text))
            if (m.Length > 0)
                yield return m.Value;
    }
}
