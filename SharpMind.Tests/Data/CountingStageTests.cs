using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

/// <summary>
/// B1: <see cref="CountingStage"/> decorates any string stage and reports a
/// per-stage funnel (documents and UTF-8 bytes in/out, and how many were
/// dropped) without changing the stream. A three-stage pipeline's counters must
/// chain — each stage's input equals the previous stage's output — so the funnel
/// actually adds up.
/// </summary>
public sealed class CountingStageTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    // 6 documents; "hi" (2) and "golf" (4) fall below a 5-char minimum.
    private string Corpus() => _dir.Write("corpus.txt", string.Join('\n',
    [
        "alpha bravo",
        "charlie delta",
        "hi",
        "echo foxtrot",
        "golf",
        "hotel india",
    ]));

    [Fact]
    public async Task ThreeStagePipeline_ReportsChainedCounts()
    {
        var sink = new StageCountCollector();
        var pipeline = PipelineNode.From(new TextFileSource(Corpus()))
            .Pipe(new LowerCase(), sink)
            .Pipe(new MinLengthFilter(5), sink)
            .Pipe("bang", s => s + "!", sink);

        var output = new List<string>();
        await foreach (string doc in pipeline.ReadAsync())
            output.Add(doc);

        var lower = sink["LowerCase"];
        var minLen = sink["MinLength(5)"];
        var bang = sink["bang"];

        // Three stages reported; the stream kept 4 of the 6 documents.
        Assert.Equal(3, sink.Counters.Count);
        Assert.Equal(4, output.Count);

        // Counts chain: each stage is offered exactly what the previous kept.
        Assert.Equal(6, lower.In);
        Assert.Equal(6, lower.Out);
        Assert.Equal(0, lower.Dropped);

        Assert.Equal(lower.Out, minLen.In);
        Assert.Equal(4, minLen.Out);
        Assert.Equal(2, minLen.Dropped);

        Assert.Equal(minLen.Out, bang.In);
        Assert.Equal(4, bang.Out);
        Assert.Equal(0, bang.Dropped);

        // Bytes chain too, and reflect the drops.
        Assert.Equal(lower.BytesOut, minLen.BytesIn);
        Assert.Equal(minLen.BytesOut, bang.BytesIn);
        Assert.True(minLen.BytesOut < minLen.BytesIn, "dropping documents must reduce the byte total");
    }

    [Fact]
    public async Task CountingStage_DoesNotAlterTheStream()
    {
        var counted = PipelineNode.From(new TextFileSource(Corpus()))
            .Pipe(new LowerCase(), new StageCountCollector())
            .Pipe(new MinLengthFilter(5), new StageCountCollector())
            .Pipe("bang", s => s + "!", new StageCountCollector());

        var plain = PipelineNode.From(new TextFileSource(Corpus()))
            .Pipe(new LowerCase())
            .Pipe(new MinLengthFilter(5))
            .Pipe("bang", s => s + "!");

        Assert.Equal(await Collect(plain), await Collect(counted));

        static async Task<List<string>> Collect(PipelineNode node)
        {
            var list = new List<string>();
            await foreach (string doc in node.ReadAsync())
                list.Add(doc);
            return list;
        }
    }

    [Fact]
    public void StageName_DefaultsToInner_ButCanBeOverridden()
    {
        var inner = new MinLengthFilter(5);

        Assert.Equal(inner.Name, new CountingStage(inner, new StageCountCollector()).Name);
        Assert.Equal("mine", new CountingStage(inner, new StageCountCollector(), "mine").Name);
    }

    [Fact]
    public void DelegateSink_ReceivesASnapshotPerDocument()
    {
        var seen = new List<StageCounters>();
        var stage = new CountingStage(
            new LambdaStageUnderTest("noop"),
            new DelegateStageCountSink(seen.Add));

        Assert.Equal("one", stage.Process("one"));
        Assert.Equal("two", stage.Process("two"));

        Assert.Equal(2, seen.Count);
        Assert.Equal(1, seen[0].In);
        Assert.Equal(2, seen[1].In);
        Assert.Equal(2, seen[1].Out);
        Assert.Equal(0, seen[1].Dropped);
    }

    [Fact]
    public void DroppedDocuments_CountTowardInButNotOut()
    {
        var sink = new StageCountCollector();
        var stage = new CountingStage(new LambdaStageUnderTest("reject-all", _ => null), sink);

        Assert.Null(stage.Process("anything"));
        Assert.Null(stage.Process("else"));

        var c = sink["reject-all"];
        Assert.Equal(2, c.In);
        Assert.Equal(0, c.Out);
        Assert.Equal(2, c.Dropped);
    }

    /// <summary>An <see cref="ICleaningStage"/> for tests (the internal LambdaStage is not visible here).</summary>
    private sealed class LambdaStageUnderTest(string name, Func<string, string?>? fn = null) : ICleaningStage
    {
        public string Name { get; } = name;
        public string? Process(string document) => fn is null ? document : fn(document);
    }
}
