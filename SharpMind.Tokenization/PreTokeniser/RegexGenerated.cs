using System.Text.RegularExpressions;

namespace SharpMind.Tokenization.PreTokeniser;

/// <summary>
/// Pre-tokenizer regexes, taken from llama.cpp's <c>src/llama-vocab.cpp</c> (b8683) so token ids
/// match llama.cpp. Contractions are written <c>(?i:'s|'t|...)</c>, the same set as llama.cpp's
/// <c>'[sS]|'[tT]|...</c>.
/// </summary>
public partial class RegexGenerated
{
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex Gpt2Pattern { get; }

    /// <summary>LLAMA3 / DBRX / SMAUG / CHATGLM4 (tiktoken cl100k): digits in groups of up to three.</summary>
    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex Llama3Pattern { get; }

    /// <summary>QWEN2 / STABLELM2 / HUNYUAN / SOLAR_OPEN: single digits.</summary>
    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex Qwen2Pattern { get; }

    /// <summary>QWEN35: as Qwen2, with combining marks (<c>\p{M}</c>) kept inside letter runs.</summary>
    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex Qwen35Pattern { get; }
}
