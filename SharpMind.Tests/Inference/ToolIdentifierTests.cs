using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>Name normalization, typo distance and the Loose path guard.</summary>
public sealed class ToolIdentifierTests
{
    [Theory]
    [InlineData("action_type", "actiontype")]
    [InlineData("action-type", "actiontype")]
    [InlineData("Action Type", "actiontype")]
    [InlineData("ACTIONTYPE", "actiontype")]
    public void Normalize_RemovesSeparatorsAndCase(string input, string expected)
        => Assert.Equal(expected, ToolIdentifier.Normalize(input));

    [Fact]
    public void Distance_CountsAnAdjacentTranspositionAsOneEdit()
    {
        Assert.Equal(1, ToolIdentifier.Distance("promt", "prompt"));
        Assert.Equal(1, ToolIdentifier.Distance("pnig", "ping"));
        Assert.Equal(1, ToolIdentifier.Distance("presentchoice", "presentchoices"));
    }

    [Fact]
    public void Distance_CountsSubstitutionAndDeletion()
    {
        Assert.Equal(1, ToolIdentifier.Distance("paris", "paros"));
        Assert.Equal(1, ToolIdentifier.Distance("count", "counts"));
        Assert.Equal(2, ToolIdentifier.Distance("count", "cnt"));
    }

    [Theory]
    [InlineData("C:\\temp\\file.txt")]
    [InlineData("C:/temp/file.txt")]
    [InlineData("/etc/passwd")]
    [InlineData("../notes.md")]
    [InlineData("./run.sh")]
    [InlineData("~/notes.txt")]
    [InlineData("notes.txt")]
    public void LooksLikePath_RecognizesLocations(string value)
        => Assert.True(ToolIdentifier.LooksLikePath(value));

    [Theory]
    [InlineData("Paris")]
    [InlineData("celsius")]
    [InlineData("https://example.com")]
    [InlineData("1.5")]
    [InlineData("Go north")]
    public void LooksLikePath_DoesNotFlagOrdinaryValues(string value)
        => Assert.False(ToolIdentifier.LooksLikePath(value));

    [Fact]
    public void ContainsPathLikeArgument_FindsNestedValues()
    {
        var args = new JsonObject
        {
            ["city"] = "Paris",
            ["target"] = "C:\\secret.txt"
        };

        Assert.True(ToolIdentifier.ContainsPathLikeArgument(args));
    }

    [Fact]
    public void ContainsPathLikeArgument_IgnoresBareFilenameInFreeText()
    {
        // A choice detail may mention a filename; only a path-named argument
        // treats that as a filesystem target.
        Assert.False(ToolIdentifier.ContainsPathLikeArgument(
            new JsonObject { ["choiceDetail"] = "Read notes.txt" }));

        Assert.True(ToolIdentifier.ContainsPathLikeArgument(
            new JsonObject { ["path"] = "notes.txt" }));
    }

    [Fact]
    public void HasPathStructure_RequiresSeparatorsOrAPrefix()
    {
        Assert.True(ToolIdentifier.HasPathStructure("C:\\temp\\x"));
        Assert.True(ToolIdentifier.HasPathStructure("src/file.cs"));
        Assert.False(ToolIdentifier.HasPathStructure("notes.txt"));
        Assert.False(ToolIdentifier.HasPathStructure("https://example.com"));
    }

    [Fact]
    public void FuzzyThreshold_RejectsShortAndDistantNames()
    {
        Assert.True(ToolIdentifier.IsWithinFuzzyThreshold("pnig", "ping", 1));
        Assert.False(ToolIdentifier.IsWithinFuzzyThreshold("cat", "dog", 3));
        Assert.False(ToolIdentifier.IsWithinFuzzyThreshold("ping", "post", 2));
    }
}
