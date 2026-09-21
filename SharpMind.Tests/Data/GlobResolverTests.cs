using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

[Collection("Non-Parallel")]
public sealed class GlobResolverTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private string Root => Path.GetFullPath(_dir.Path);

    [Fact]
    public void RecursiveDoubleStar_Backslash_ReturnsAllFiles()
    {
        Directory.CreateDirectory(Path.Combine(_dir.Path, "sub"));
        _dir.Write("a.txt", "x");
        _dir.Write("sub\\b.txt", "x");

        var files = GlobResolver.Resolve(Path.Combine(_dir.Path, "**", "*.txt"));

        Assert.Equal(Expected(["a.txt", "sub\\b.txt"]), files);
    }

    [Fact]
    public void RecursiveDoubleStar_ForwardSlash_ReturnsAllFiles()
    {
        Directory.CreateDirectory(Path.Combine(_dir.Path, "sub"));
        _dir.Write("a.txt", "x");
        _dir.Write("sub\\b.txt", "x");

        string pattern = Root.Replace('\\', '/') + "/**/*.txt";
        var files = GlobResolver.Resolve(pattern);

        Assert.Equal(Expected(["a.txt", "sub\\b.txt"]), files);
    }

    [Fact]
    public void SingleStar_DoesNotDescendIntoSubdirectories()
    {
        Directory.CreateDirectory(Path.Combine(_dir.Path, "sub"));
        _dir.Write("a.txt", "x");
        _dir.Write("sub\\b.txt", "x");

        var files = GlobResolver.Resolve(Path.Combine(_dir.Path, "*.txt"));

        Assert.Equal(Expected(["a.txt"]), files);
    }

    [Fact]
    public void RelativePattern_ResolvesAgainstCurrentDirectory()
    {
        Directory.CreateDirectory(Path.Combine(_dir.Path, "sub"));
        _dir.Write("a.txt", "x");
        _dir.Write("sub\\b.txt", "x");

        string previous = Directory.GetCurrentDirectory();
        try
        {
            Directory.SetCurrentDirectory(Root);
            var files = GlobResolver.Resolve("**/*.txt");
            Assert.Equal(Expected(["a.txt", "sub\\b.txt"]), files);
        }
        finally
        {
            Directory.SetCurrentDirectory(previous);
        }
    }

    [Fact]
    public void QuestionMark_MatchesSingleCharacter()
    {
        _dir.Write("a1.txt", "x");
        _dir.Write("aa.txt", "x");
        _dir.Write("b1.txt", "x");

        var files = GlobResolver.Resolve(Path.Combine(_dir.Path, "a?.txt"));

        Assert.Equal(Expected(["a1.txt", "aa.txt"]), files);
    }

    [Fact]
    public void BareDoubleStar_TreatsAsAllFiles()
    {
        Directory.CreateDirectory(Path.Combine(_dir.Path, "sub"));
        _dir.Write("a.txt", "x");
        _dir.Write("sub\\b.txt", "x");

        var files = GlobResolver.Resolve(Path.Combine(_dir.Path, "**"));

        Assert.Equal(Expected(["a.txt", "sub\\b.txt"]), files);
    }

    [Fact]
    public void NoMatch_ReturnsEmpty()
    {
        _dir.Write("a.txt", "x");
        var files = GlobResolver.Resolve(Path.Combine(_dir.Path, "*.zzz"));
        Assert.Empty(files);
    }

    [Fact]
    public void LiteralExistingFile_ReturnsFullPath()
    {
        string path = _dir.Write("a.txt", "x");
        var files = GlobResolver.Resolve(path);
        Assert.Equal([Path.GetFullPath(path)], files);
    }

    [Fact]
    public void LiteralMissingFile_ReturnsEmpty()
    {
        var files = GlobResolver.Resolve(Path.Combine(_dir.Path, "nope.txt"));
        Assert.Empty(files);
    }

    [Fact]
    public void ResolveMany_DeduplicatesAndSorts()
    {
        _dir.Write("a.txt", "x");
        Directory.CreateDirectory(Path.Combine(_dir.Path, "sub"));
        _dir.Write("sub\\b.txt", "x");

        string pattern1 = Path.Combine(_dir.Path, "**", "*.txt");
        var files = GlobResolver.ResolveMany([pattern1, pattern1, Path.Combine(_dir.Path, "a.txt")]);

        Assert.Equal(Expected(["a.txt", "sub\\b.txt"]), files);
    }

    private string[] Expected(IEnumerable<string> relative)
        => [.. relative.Select(r => Path.GetFullPath(Path.Combine(_dir.Path, r))).Order()];
}