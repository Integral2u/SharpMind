using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

public sealed class CleaningPipelineTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private PipelineNode PipelineFrom(string content)
    {
        var source = new TextFileSource(_dir.Write("in.txt", content));
        return CleaningPipeline.From(source);
    }

    [Fact]
    public async Task SingleStage_TransformsDocuments()
    {
        var node = PipelineFrom("Hello\nWorld")
            .Pipe("ToUpper", s => s.ToUpperInvariant());

        var docs = await node.ReadAsync().ToListAsync();

        Assert.Equal(["HELLO", "WORLD"], docs);
    }

    [Fact]
    public async Task StageReturningNull_DiscardsDocument()
    {
        var node = PipelineFrom("keep\ndrop\nkeep")
            .Pipe("DropDrop", s => s == "drop" ? null : s);

        var docs = await node.ReadAsync().ToListAsync();

        Assert.Equal(["keep", "keep"], docs);
    }

    [Fact]
    public async Task ChainedStages_ApplyInOrder()
    {
        var node = PipelineFrom("  Hello  ")
            .Pipe(new NormaliseWhitespace())
            .Pipe(new LowerCase());

        var docs = await node.ReadAsync().ToListAsync();

        Assert.Single(docs);
        Assert.Equal("hello", docs[0]);
    }

    [Fact]
    public async Task LambdaStage_WorksWithoutClass()
    {
        var node = PipelineFrom("abc\nxyz")
            .Pipe("Reverse", s => new string([.. s.Reverse()]));

        var docs = await node.ReadAsync().ToListAsync();

        Assert.Equal(["cba", "zyx"], docs);
    }

    [Fact]
    public async Task Branch_SplitsStream()
    {
        var root              = PipelineFrom("short\na much longer document\nhi");
        var (match, other)    = root.Branch("LongDoc", s => s.Length > 10);

        var longDocs  = await match.ReadAsync().ToListAsync();
        var shortDocs = await other.ReadAsync().ToListAsync();

        Assert.Equal(["a much longer document"], longDocs);
        Assert.Equal(["short", "hi"], shortDocs);
    }

    [Fact]
    public async Task Merge_RejoinsAfterBranch()
    {
        string path = _dir.Write("merge.txt", "code\nprose\nmore prose");
        var root     = CleaningPipeline.From(new TextFileSource(path));

        var (code, prose) = root.Branch("IsCode", s => s == "code");

        var merged = CleaningPipeline.Merge(
            code.Pipe("UpperCode",  s => "[CODE] " + s),
            prose.Pipe("LowerProse", s => s.ToLowerInvariant()));

        var docs = await merged.ReadAsync().ToListAsync();

        Assert.Contains("[CODE] code", docs);
        Assert.Contains("prose", docs);
        Assert.Contains("more prose", docs);
        Assert.Equal(3, docs.Count);
    }

    [Fact]
    public void Merge_RequiresAtLeastTwoNodes()
    {
        var node = PipelineFrom("x");
        Assert.Throws<ArgumentException>(() => CleaningPipeline.Merge(node));
    }

    [Fact]
    public void Describe_IncludesAllStageNames()
    {
        var node = PipelineFrom("x")
            .Pipe(new NormaliseWhitespace())
            .Pipe(new LowerCase());

        string desc = node.Describe();

        Assert.Contains("NormaliseWhitespace", desc);
        Assert.Contains("LowerCase", desc);
    }
}
