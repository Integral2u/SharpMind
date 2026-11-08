using SharpMind.Tokenization.PreTokeniser;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// The multi-regex split llama.cpp uses for the DeepSeek pre types (<c>unicode_regex_split</c>): gaps are kept,
/// later stages split earlier pieces, and the two code-point mechanics (category stand-ins for astral
/// characters in <c>\p{}</c> stages, <c>\v</c> for non-ASCII whitespace in plain stages) hold. The llama.cpp
/// fixtures in <see cref="LlamaCppVocabParityTests"/> prove the end result for deepseek-coder and
/// deepseek-llm; deepseek-v3 has no fixture, so its rows here are what pins it.
/// </summary>
public sealed class SequentialRegexPreTokeniserTests
{
    [Fact]
    public void Stages_KeepGapsAndSplitEarlierPieces()
    {
        var pt = new SequentialRegexPreTokeniser("t", RegexGenerated.NewlineStage, RegexGenerated.SingleDigit);

        // Stage 1 leaves "a1" and "b22" as gaps around "\n"; stage 2 splits the digits out of both gaps.
        Assert.Equal(["a", "1", "\n", "b", "2", "2"], pt.PreTokenise("a1\nb22").ToArray());
    }

    [Fact]
    public void Stages_RunInOrder()
    {
        // Letters first: "ab" is one piece and the digits stay single. Digits first on the same text gives the
        // same pieces, so the order is pinned on a text where it matters: the digit-group stage can only group
        // digits it sees together, and a stage that first isolates single digits leaves nothing to group.
        var groupFirst = new SequentialRegexPreTokeniser("t", RegexGenerated.DigitGroup, RegexGenerated.SingleDigit);
        var singleFirst = new SequentialRegexPreTokeniser("t", RegexGenerated.SingleDigit, RegexGenerated.DigitGroup);

        Assert.Equal(["1", "2", "3", "4"], groupFirst.PreTokenise("1234").ToArray());
        Assert.Equal(["1", "2", "3", "4"], singleFirst.PreTokenise("1234").ToArray());
        Assert.Equal(["123", "4"], new SequentialRegexPreTokeniser("t", RegexGenerated.DigitGroup).PreTokenise("1234").ToArray());

        // On the real deepseek-v3 list the digit stage runs first, so the spaces before "1" end their own piece
        // and `\s+(?!\S)` keeps them together. With the main stage first they still precede a non-space and split.
        Assert.Equal(["  ", "1"], SequentialRegexPreTokeniser.DeepSeek3.PreTokenise("  1").ToArray());
        Assert.Equal([" ", " ", "1"],
            new SequentialRegexPreTokeniser("t", RegexGenerated.DeepSeek3Main, RegexGenerated.DigitGroup).PreTokenise("  1").ToArray());
    }

    [Fact]
    public void EmptyText_YieldsNothing()
        => Assert.Empty(SequentialRegexPreTokeniser.DeepSeekCoder.PreTokenise(""));

    // The first fixture case, by hand: "ied" and " months" are letter runs, "4" and "½" (No) are numbers, the
    // spaces between them are gaps. llama.cpp ids: 1050 207 19 207 19192 4217.
    [Theory]
    [InlineData("deepseek-coder")]
    [InlineData("deepseek-llm")]
    public void DeepSeek_SplitsTheFixtureOpener(string pre)
        => Assert.Equal(["ied", " ", "4", " ", "½", " months"], ByName(pre).PreTokenise("ied 4 ½ months").ToArray());

    [Theory]
    // \p{} stages see an astral code point as one character of its category, as llama.cpp's collapsed text does.
    [InlineData("deepseek-v3", " \U0001F680", new[] { " \U0001F680" })]          // space + rocket: one symbol run
    [InlineData("deepseek-v3", "\U0001F680a", new[] { "\U0001F680", "a" })]       // the pair is never cut before the letter
    [InlineData("deepseek-coder", " \U00020000a", new[] { " \U00020000a" })]      // CJK Extension B is \p{L}: one letter run
    [InlineData("deepseek-coder", "\U0001F680 (normal)", new[] { "\U0001F680", " (", "normal", ")" })]
    // A plain stage sees non-ASCII whitespace as \v (llama.cpp's std::wregex fallback), so U+3000 is whitespace,
    // not the first member of the literal [　-。] class.
    [InlineData("deepseek-llm", "!　", new[] { "!", "　" })]
    [InlineData("deepseek-llm", "　 ", new[] { "　 " })]
    // DeepSeek-LLM's explicit letter class keeps its astral ranges: a Deseret letter joins the run of Latin-1
    // letters. Next to ASCII letters it does not: stage 3's `:-~` range re-splits every earlier piece and
    // takes the ASCII letters out again, in llama.cpp as here.
    [InlineData("deepseek-llm", "é\U00010400é", new[] { "é\U00010400é" })]
    [InlineData("deepseek-llm", "a\U00010400b", new[] { "a", "\U00010400", "b" })]
    // deepseek-v3 stage set, no fixture upstream.
    [InlineData("deepseek-v3", "12345", new[] { "123", "45" })]
    [InlineData("deepseek-v3", "!abc", new[] { "!abc" })]
    [InlineData("deepseek-v3", "x ひらがな.", new[] { "x", " ", "ひらがな", "." })]
    [InlineData("deepseek-v3", "ns.Core;\nx", new[] { "ns", ".Core", ";\n", "x" })]
    public void DeepSeek_SplitsLikeLlamaCpp(string pre, string text, string[] expected)
        => Assert.Equal(expected, ByName(pre).PreTokenise(text).ToArray());

    private static SequentialRegexPreTokeniser ByName(string pre) => pre switch
    {
        "deepseek-llm" => SequentialRegexPreTokeniser.DeepSeekLlm,
        "deepseek-coder" => SequentialRegexPreTokeniser.DeepSeekCoder,
        _ => SequentialRegexPreTokeniser.DeepSeek3,
    };
}
