using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

public sealed class JsonlSourceTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    [Fact]
    public async Task ExtractsTextField()
    {
        string path = _dir.Write("a.jsonl",
            """{"text":"hello"}""" + "\n" +
            """{"text":"world"}""");
        var source = new JsonlSource(path);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Equal(["hello", "world"], docs);
    }

    [Fact]
    public async Task NestedFieldPath_DotNotation()
    {
        string path = _dir.Write("b.jsonl",
            """{"meta":{"content":"deep value"}}""");
        var source = new JsonlSource(path, textField: "meta.content");

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Equal("deep value", docs[0]);
    }

    [Fact]
    public async Task MalformedLine_Skipped_CountTracked()
    {
        string path = _dir.Write("c.jsonl",
            """{"text":"ok"}""" + "\n" +
            "NOT JSON\n" +
            """{"text":"also ok"}""");
        var source = new JsonlSource(path);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Equal(["ok", "also ok"], docs);
        Assert.Equal(1, source.SkippedLines);
    }

    [Fact]
    public async Task MissingField_Skipped()
    {
        string path = _dir.Write("d.jsonl",
            """{"other":"value"}""" + "\n" +
            """{"text":"present"}""");
        var source = new JsonlSource(path);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Single(docs);
        Assert.Equal("present", docs[0]);
        Assert.Equal(1, source.SkippedLines);
    }

    [Fact]
    public async Task EmptyTextValue_Skipped()
    {
        string path = _dir.Write("e.jsonl",
            """{"text":"   "}""" + "\n" +
            """{"text":"real"}""");
        var source = new JsonlSource(path);

        var docs = await source.ReadAsync().ToListAsync();

        Assert.Single(docs);
        Assert.Equal("real", docs[0]);
    }
}
