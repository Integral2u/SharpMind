using SharpMind.Tokenization.PreTokeniser;
using Xunit;

namespace SharpMind.Tests.Tokenization;

public sealed class SuperBpePreTokeniserTests
{
    [Fact]
    public void KeepsAPhraseAsOneSegment()
    {
        var pieces = new SuperBpePreTokeniser().PreTokenise("The old lighthouse keeper").ToArray();
        Assert.Equal(["The old lighthouse keeper"], pieces);
    }

    [Fact]
    public void SplitsNumbersIntoGroupsOfAtMostThree()
    {
        var pieces = new SuperBpePreTokeniser().PreTokenise("12345").ToArray();
        Assert.Equal(["123", "45"], pieces);
    }

    [Fact]
    public void IsolatesARunOfPunctuation()
    {
        var pieces = new SuperBpePreTokeniser().PreTokenise("really!!!yes").ToArray();
        Assert.Equal(["really", "!!!", "yes"], pieces);
    }

    [Fact]
    public void KeepsASinglePeriodInsideTheSegment()
    {
        var pieces = new SuperBpePreTokeniser().PreTokenise("Hello world.").ToArray();
        Assert.Equal(["Hello world."], pieces);
    }
}
