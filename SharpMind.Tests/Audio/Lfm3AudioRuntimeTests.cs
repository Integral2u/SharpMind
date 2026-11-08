using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class Lfm3AudioRuntimeTests
{
    [Fact]
    public void Merge_DropsTheOverlappingBoundaryWords()
    {
        string text = Lfm3AudioRuntime.MergeTranscripts(
        [
            "And so, my fellow Americans,",
            "Americans, ask not what your",
            "what your country can do for you."
        ]);

        Assert.Equal("And so, my fellow Americans, ask not what your country can do for you.", text);
    }

    [Fact]
    public void Merge_AppendsPartsWithoutOverlap()
    {
        string text = Lfm3AudioRuntime.MergeTranscripts(["one two", "three four"]);

        Assert.Equal("one two three four", text);
    }

    [Fact]
    public void Merge_IgnoresCaseAndTrailingPunctuation()
    {
        string text = Lfm3AudioRuntime.MergeTranscripts(["Ask not what", "What your country can do."]);

        Assert.Equal("Ask not what your country can do.", text);
    }

    [Fact]
    public void Merge_SkipsEmptyParts()
    {
        string text = Lfm3AudioRuntime.MergeTranscripts(["hello", "", "   ", "there"]);

        Assert.Equal("hello there", text);
    }

    [Fact]
    public void Merge_KeepsARepeatedWordWhenOnlyTheLastWordMatches()
    {
        string text = Lfm3AudioRuntime.MergeTranscripts(["a a a", "a b"]);

        Assert.Equal("a a a b", text);
    }

    [Fact]
    public void Merge_EmptyInput_IsEmpty()
    {
        Assert.Equal("", Lfm3AudioRuntime.MergeTranscripts([]));
        Assert.Equal("", Lfm3AudioRuntime.MergeTranscripts(["", " "]));
    }
}
