using System.Text.RegularExpressions;

namespace SharpMind.Tokenization.PreTokeniser;

/// <summary>
/// cl100k_base-style pre-tokeniser used by tiktoken-based models
/// (Llama 3+, Qwen 2+/3, DeepSeek, Phi-3/4, Gemma 2, GPT-4o/o200k).
///
/// Identical to <see cref="Gpt2PreTokeniser"/> except contractions are
/// matched case-insensitively via (?i:...), so tokens like I'M, YOU'RE,
/// and SHE'D are split the same way as their lowercase counterparts.
/// This matches the behaviour of tiktoken's cl100k_base regex.
/// </summary>
public sealed class Cl100kPreTokeniser : IPreTokeniser
{
    public IEnumerable<string> PreTokenise(string text)
    {
        foreach (Match m in RegexGenerated.Cl100kPattern.Matches(text))
            if (m.Length > 0)
                yield return m.Value;
    }
}
