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

    // -- DeepSeek stage lists (llama-vocab.cpp b8683, lines 308-341), run in sequence by SequentialRegexPreTokeniser.
    // Non-ASCII is written as \uXXXX so the file stays ASCII; the seven astral ranges of the DeepSeek-LLM letter
    // class are surrogate-pair alternatives because a .NET character class works on UTF-16 units.

    /// <summary>DEEPSEEK_LLM / DEEPSEEK_CODER stage 1: every CR or LF on its own.</summary>
    [GeneratedRegex(@"[\r\n]")]
    public static partial Regex NewlineStage { get; }

    /// <summary>DEEPSEEK_LLM stage 2: an optional whitespace and a run of (mostly cased) letters, listed explicitly.</summary>
    [GeneratedRegex(@"\s?(?:[A-Za-z\u00B5\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u01BA\u01BC-\u01BF\u01C4-\u0293\u0295-\u02AF\u0370-\u0373\u0376\u0377\u037B-\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u10A0-\u10C5\u13A0-\u13F5\u13F8-\u13FD\u1C90-\u1CBA\u1CBD-\u1CBF\u1D00-\u1D2B\u1D6B-\u1D77\u1D79-\u1D9A\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2134\u2139\u213C-\u213F\u2145-\u2149\u214E\u2183\u2184\u2C00-\u2C7B\u2C7E-\u2CE4\u2CEB-\u2CEE\u2CF2\u2CF3\uA640-\uA66D\uA680-\uA69B\uA722-\uA76F\uA771-\uA787\uA78B-\uA78E\uAB70-\uABBF\uFB00-\uFB06\uFB13-\uFB17\uFF21-\uFF3A\uFF41-\uFF5A]|\uD801[\uDC00-\uDC4F]|\uD801[\uDCB0-\uDCD3]|\uD801[\uDCD8-\uDCFB]|\uD803[\uDC80-\uDCB2]|\uD803[\uDCC0-\uDCF2]|\uD806[\uDCA0-\uDCDF]|\uD83A[\uDD00-\uDD43])+", RegexOptions.ExplicitCapture)]
    public static partial Regex DeepSeekLlmLetters { get; }

    /// <summary>DEEPSEEK_LLM stage 3: an optional whitespace and a run of ASCII, fullwidth or CJK punctuation.</summary>
    [GeneratedRegex(@"\s?[!-/:-~\uFF01-\uFF0F\uFF1A-\uFF5E\u2018-\u201F\u3000-\u3002]+")]
    public static partial Regex DeepSeekLlmPunctuation { get; }

    /// <summary>DEEPSEEK_LLM stage 4: whitespace that ends the piece.</summary>
    [GeneratedRegex(@"\s+$")]
    public static partial Regex TrailingWhitespace { get; }

    /// <summary>DEEPSEEK_LLM stage 5 / DEEPSEEK_CODER stage 4: CJK, Hangul and the block between.</summary>
    [GeneratedRegex(@"[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+")]
    public static partial Regex DeepSeekCjk { get; }

    /// <summary>DEEPSEEK_LLM stage 6: a run of digits.</summary>
    [GeneratedRegex(@"\p{N}+")]
    public static partial Regex DigitRun { get; }

    /// <summary>DEEPSEEK_CODER stage 2: an optional whitespace and a run of letters.</summary>
    [GeneratedRegex(@"\s?\p{L}+")]
    public static partial Regex DeepSeekCoderLetters { get; }

    /// <summary>DEEPSEEK_CODER stage 3: an optional whitespace and a run of punctuation.</summary>
    [GeneratedRegex(@"\s?\p{P}+")]
    public static partial Regex DeepSeekCoderPunctuation { get; }

    /// <summary>DEEPSEEK_CODER stage 5: single digits.</summary>
    [GeneratedRegex(@"\p{N}")]
    public static partial Regex SingleDigit { get; }

    /// <summary>DEEPSEEK3_LLM stage 1: digits in groups of up to three.</summary>
    [GeneratedRegex(@"\p{N}{1,3}")]
    public static partial Regex DigitGroup { get; }

    /// <summary>DEEPSEEK3_LLM stage 2: CJK, hiragana, katakana.</summary>
    [GeneratedRegex(@"[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+")]
    public static partial Regex DeepSeek3Cjk { get; }

    /// <summary>DEEPSEEK3_LLM stage 3: an ASCII punctuation mark glued to ASCII letters, letter runs, punctuation/symbol runs, whitespace.</summary>
    [GeneratedRegex(@"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex DeepSeek3Main { get; }
}
