using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

public sealed class CompositeSourceTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private TextFileSource Source(string name, string content)
        => new(_dir.Write(name, content));

    [Fact]
    public async Task Concatenate_YieldsAllSourcesInOrder()
    {
        var composite = new CompositeSource(
            [Source("a.txt", "alpha"), Source("b.txt", "beta")],
            CompositeSource.CompositionMode.Concatenate);

        var docs = await composite.ReadAsync().ToListAsync();

        Assert.Equal(["alpha", "beta"], docs);
    }

    [Fact]
    public async Task RoundRobin_InterleavesSources()
    {
        var composite = new CompositeSource(
            [Source("a.txt", "a1\na2"), Source("b.txt", "b1\nb2")],
            CompositeSource.CompositionMode.RoundRobin);

        var docs = await composite.ReadAsync().ToListAsync();

        // Round-robin: a1, b1, a2, b2
        Assert.Equal(["a1", "b1", "a2", "b2"], docs);
    }

    [Fact]
    public async Task EstimatedCount_SumsWhenAllKnown()
    {
        var a = new TextFileSource(_dir.Write("a.txt", "x"), TextFileSource.DocumentMode.FilePerDoc);
        var b = new TextFileSource(_dir.Write("b.txt", "y"), TextFileSource.DocumentMode.FilePerDoc);
        var composite = new CompositeSource([a, b]);

        Assert.Equal(2L, composite.EstimatedCount);
        await composite.DisposeAsync();
    }

    [Fact]
    public async Task EstimatedCount_NullWhenAnyUnknown()
    {
        var a = new TextFileSource(_dir.Write("a.txt", "x"), TextFileSource.DocumentMode.FilePerDoc);
        var b = new TextFileSource(_dir.Write("b.txt", "x\ny"), TextFileSource.DocumentMode.LinePerDoc);
        var composite = new CompositeSource([a, b]);

        Assert.Null(composite.EstimatedCount);
        await composite.DisposeAsync();
    }

    [Fact]
    public void RequiresAtLeastOneSource()
    {
        Assert.Throws<ArgumentException>(
            () => new CompositeSource([]));
    }
}
