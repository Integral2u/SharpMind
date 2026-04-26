using SharpMind.Data.Pipeline.Stages;

namespace SharpMind.Tests.Data;

public sealed class BuiltInStageTests
{
    // ── NormaliseWhitespace ───────────────────────────────────────────────

    [Fact]
    public void NormaliseWhitespace_CollapsesRuns()
    {
        var stage  = new NormaliseWhitespace();
        string result = stage.Process("  hello   world  ")!;
        Assert.Equal("hello world", result);
    }

    [Fact]
    public void NormaliseWhitespace_NormalisesNewlinesAndTabs()
    {
        var stage = new NormaliseWhitespace();
        Assert.Equal("a b c", stage.Process("a\tb\nc")!);
    }

    [Fact]
    public void NormaliseWhitespace_AllWhitespace_ReturnsNull()
    {
        var stage = new NormaliseWhitespace();
        Assert.Null(stage.Process("   "));
    }

    // ── LowerCase ─────────────────────────────────────────────────────────

    [Fact]
    public void LowerCase_LowercasesAll()
    {
        Assert.Equal("hello world", new LowerCase().Process("HELLO WORLD"));
    }

    // ── StripHtml ─────────────────────────────────────────────────────────

    [Fact]
    public void StripHtml_RemovesTags()
    {
        var result = new StripHtml().Process("<p>Hello <b>world</b></p>")!;
        Assert.DoesNotContain("<", result);
        Assert.Contains("Hello", result);
        Assert.Contains("world", result);
    }

    [Theory]
    [InlineData("&amp;",  "&")]
    [InlineData("&lt;",   "<")]
    [InlineData("&gt;",   ">")]
    [InlineData("&quot;", "\"")]
    [InlineData("&#65;",  "A")]
    public void StripHtml_DecodesEntities(string entity, string expected)
    {
        var result = new StripHtml().Process(entity)!;
        Assert.Equal(expected, result.Trim());
    }

    // ── MinLengthFilter ───────────────────────────────────────────────────

    [Fact]
    public void MinLengthFilter_PassesLongEnough()
    {
        var stage = new MinLengthFilter(5);
        Assert.Equal("hello", stage.Process("hello"));
    }

    [Fact]
    public void MinLengthFilter_DiscardsShort()
    {
        var stage = new MinLengthFilter(10);
        Assert.Null(stage.Process("hi"));
    }

    // ── MaxLengthFilter ───────────────────────────────────────────────────

    [Fact]
    public void MaxLengthFilter_PassesShortEnough()
    {
        Assert.Equal("hi", new MaxLengthFilter(10).Process("hi"));
    }

    [Fact]
    public void MaxLengthFilter_DiscardsLong()
    {
        Assert.Null(new MaxLengthFilter(3).Process("toolong"));
    }

    // ── RegexFilter ───────────────────────────────────────────────────────

    [Fact]
    public void RegexFilter_DiscardsMatching()
    {
        var stage = new RegexFilter(@"\bspam\b");
        Assert.Null(stage.Process("this is spam content"));
        Assert.NotNull(stage.Process("this is clean content"));
    }

    [Fact]
    public void RegexKeepFilter_KeepsOnlyMatching()
    {
        var stage = new RegexKeepFilter(@"^\d+$");
        Assert.Equal("12345", stage.Process("12345"));
        Assert.Null(stage.Process("abc123"));
    }

    // ── DeduplicateFilter ─────────────────────────────────────────────────

    [Fact]
    public void DeduplicateFilter_DropsDuplicates()
    {
        var stage = new DeduplicateFilter(100);
        Assert.Equal("hello", stage.Process("hello"));
        Assert.Null(stage.Process("hello"));   // duplicate
        Assert.Equal("world", stage.Process("world"));
    }

    [Fact]
    public void DeduplicateFilter_WindowEvicts_AllowsRepeatAfterWindow()
    {
        var stage = new DeduplicateFilter(windowSize: 2);
        stage.Process("a");
        stage.Process("b");
        stage.Process("c"); // evicts "a"
        // "a" should now be accepted again
        Assert.Equal("a", stage.Process("a"));
    }
}
