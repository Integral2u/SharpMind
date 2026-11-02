using System.Text.RegularExpressions;

namespace SharpMind.Tokenization.PreTokeniser;

/// <summary>Which tiktoken-family pre-tokenizer a <see cref="Cl100kPreTokeniser"/> applies (llama.cpp pre types).</summary>
public enum Cl100kVariant
{
    /// <summary>cl100k regex (digits in groups of up to three): DBRX, Smaug, ChatGLM4, and the fallback.</summary>
    Cl100k,
    /// <summary>Cl100k regex plus llama.cpp's <c>ignore_merges</c>: a pre-token that is itself a vocab entry is one token.</summary>
    Llama3,
    /// <summary>Single digits: Qwen2/Qwen3, StableLM2, Hunyuan, Solar Open.</summary>
    Qwen2,
    /// <summary>Qwen2 plus combining marks inside letter runs: Qwen3.5.</summary>
    Qwen35,
}

/// <summary>
/// tiktoken-family pre-tokeniser (Llama 3+, Qwen 2+/3/3.5, DeepSeek, Phi-4, ...). The regex per
/// <see cref="Cl100kVariant"/> is llama.cpp's, so token ids match llama.cpp for the same vocab.
/// </summary>
public sealed class Cl100kPreTokeniser(Cl100kVariant variant) : IPreTokeniser
{
    public Cl100kVariant Variant { get; } = variant;

    /// <summary>llama.cpp's <c>ignore_merges</c>, set for the LLAMA3 pre type only.</summary>
    public bool IgnoreMerges => Variant == Cl100kVariant.Llama3;

    private readonly Regex _pattern = variant switch
    {
        Cl100kVariant.Qwen2 => RegexGenerated.Qwen2Pattern,
        Cl100kVariant.Qwen35 => RegexGenerated.Qwen35Pattern,
        _ => RegexGenerated.Llama3Pattern,
    };

    public IEnumerable<string> PreTokenise(string text)
    {
        foreach (Match m in _pattern.Matches(text))
            if (m.Length > 0)
                yield return m.Value;
    }
}
