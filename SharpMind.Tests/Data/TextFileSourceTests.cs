using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

public sealed class TextFileSourceTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    [Fact]
    public async Task LineMode_YieldsEachNonEmptyLine()
    {
        string path = _dir.Write("a.txt", "hello\nworld\n\nfoo");
        var source  = new TextFileSource(path, TextFileSource.DocumentMode.LinePerDoc);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Equal(["hello", "world", "foo"], docs);
    }

    [Fact]
    public async Task FileMode_YieldsWholeFileAsOneDocument()
    {
        string path = _dir.Write("b.txt", "line one\nline two");
        var source  = new TextFileSource(path, TextFileSource.DocumentMode.FilePerDoc);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Single(docs);
        Assert.Contains("line one", docs[0]);
        Assert.Contains("line two", docs[0]);
    }

    [Fact]
    public async Task SkipsEmptyLines()
    {
        string path = _dir.Write("c.txt", "\n   \nhello\n\n");
        var source  = new TextFileSource(path);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Single(docs);
        Assert.Equal("hello", docs[0]);
    }

    [Fact]
    public async Task MultipleFiles_YieldsAllInOrder()
    {
        string p1 = _dir.Write("1.txt", "alpha");
        string p2 = _dir.Write("2.txt", "beta");
        var source = new TextFileSource([p1, p2]);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Equal(["alpha", "beta"], docs);
    }

    [Fact]
    public void MissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(
            () => new TextFileSource("/does/not/exist.txt"));
    }

    [Fact]
    public async Task EstimatedCount_FileMode_ReturnsFileCount()
    {
        string p1 = _dir.Write("x.txt", "a");
        string p2 = _dir.Write("y.txt", "b");
        var source = new TextFileSource([p1, p2], TextFileSource.DocumentMode.FilePerDoc);

        Assert.Equal(2L, source.EstimatedCount);
        await source.DisposeAsync();
    }

    [Fact]
    public async Task EstimatedCount_LineMode_ReturnsNull()
    {
        string path = _dir.Write("z.txt", "a\nb");
        var source  = new TextFileSource(path, TextFileSource.DocumentMode.LinePerDoc);

        Assert.Null(source.EstimatedCount);
        await source.DisposeAsync();
    }

    [Fact]
    public async Task Cancellation_StopsStream()
    {
        string path = _dir.Write("big.txt", string.Join('\n', Enumerable.Range(0, 1000)));
        var source  = new TextFileSource(path);
        var cts     = new CancellationTokenSource();

        var docs = new List<string>();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var doc in source.ReadAsync(cts.Token))
            {
                docs.Add(doc);
                if (docs.Count == 5) cts.Cancel();
            }
        });

        Assert.True(docs.Count <= 6); // cancelled shortly after 5
    }
}
