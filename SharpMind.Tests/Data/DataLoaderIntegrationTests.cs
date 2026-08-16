using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

public sealed class DataLoaderIntegrationTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static int[] Tokenise(string s) =>
        TestTokens.Encode(s, 500);

    [Fact]
    public async Task EndToEnd_TextFile_CleanPipeline_PackedBatches()
    {
        string path = _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, 20).Select(i => $"word{i} foo bar")));

        var pipeline = CleaningPipeline
            .From(new TextFileSource(path))
            .Pipe(new NormaliseWhitespace())
            .Pipe(new MinLengthFilter(3));

        var loader  = new DataLoader(pipeline, Tokenise,
                          new PackingBatcher(batchSize: 2, maxSeqLen: 16));

        var batches = new List<TrainingBatch>();
        await foreach (var batch in loader.LoadAsync())
            batches.Add(batch);

        Assert.True(batches.Count > 0);
        Assert.All(batches, b =>
        {
            Assert.Equal(2,  b.TokenIds.Shape.Rows);
            Assert.Equal(16, b.TokenIds.Shape.Cols);
            Assert.True(b.RealTokenCount > 0);
        });

        foreach (var b in batches) b.Dispose();
    }

    [Fact]
    public async Task LoadAsync_MaxBatches_LoopsPastSinglePass()
    {
        string path = _dir.Write("tiny.txt",
            string.Join('\n', Enumerable.Range(0, 20).Select(i => $"word{i} foo bar")));

        var pipeline = CleaningPipeline.From(new TextFileSource(path));
        var loader   = new DataLoader(pipeline, Tokenise,
                          new PackingBatcher(batchSize: 2, maxSeqLen: 16));

        // The single-pass batch count is well under 10 — with a budget the
        // loader must re-enumerate the pipeline until 10 batches are emitted.
        const int budget = 10;
        var batches = new List<TrainingBatch>();
        await foreach (var batch in loader.LoadAsync(maxBatches: budget))
            batches.Add(batch);

        Assert.Equal(budget, batches.Count);
        Assert.All(batches, b =>
        {
            Assert.Equal(2,  b.TokenIds.Shape.Rows);
            Assert.Equal(16, b.TokenIds.Shape.Cols);
            Assert.True(b.RealTokenCount > 0);
        });

        foreach (var b in batches) b.Dispose();
    }

    [Fact]
    public async Task LoadAsync_ConstructorBudget_AppliesWhenNoArgGiven()
    {
        string path = _dir.Write("tiny2.txt",
            string.Join('\n', Enumerable.Range(0, 20).Select(i => $"word{i} foo bar")));

        var pipeline = CleaningPipeline.From(new TextFileSource(path));
        var loader   = new DataLoader(pipeline, Tokenise,
                          new PackingBatcher(batchSize: 2, maxSeqLen: 16),
                          maxBatches: 7);

        var batches = new List<TrainingBatch>();
        await foreach (var batch in loader.LoadAsync())
            batches.Add(batch);

        Assert.Equal(7, batches.Count);
        foreach (var b in batches) b.Dispose();
    }

    [Fact]
    public async Task LoadAsync_MaxBatches_LessThanOnePass_StopsEarly()
    {
        string path = _dir.Write("tiny3.txt",
            string.Join('\n', Enumerable.Range(0, 20).Select(i => $"word{i} foo bar")));

        var pipeline = CleaningPipeline.From(new TextFileSource(path));
        var loader   = new DataLoader(pipeline, Tokenise,
                          new PackingBatcher(batchSize: 2, maxSeqLen: 16));

        var batches = new List<TrainingBatch>();
        await foreach (var batch in loader.LoadAsync(maxBatches: 2))
            batches.Add(batch);

        Assert.Equal(2, batches.Count);
        foreach (var b in batches) b.Dispose();
    }

    [Fact]
    public async Task EndToEnd_Jsonl_SafetyStages()
    {
        string path = _dir.Write("data.jsonl",
            """{"text":"hello world"}""" + "\n" +
            """{"text":"this is spam content"}""" + "\n" +
            """{"text":"email user@example.com here"}""" + "\n" +
            """{"text":"clean document text here"}""");

        var pipeline = CleaningPipeline
            .From(new JsonlSource(path))
            .Pipe(new BlocklistFilter(["spam"]))
            .Pipe(new PiiMasker(PiiType.Email));

        var loader  = new DataLoader(pipeline, Tokenise,
                          new PackingBatcher(batchSize: 1, maxSeqLen: 32));

        var allDocs = new List<string>();
        // Collect by running through pipeline without batching for assertion clarity
        await foreach (var doc in pipeline.ReadAsync())
            allDocs.Add(doc);

        // "spam content" doc discarded, email masked
        Assert.DoesNotContain(allDocs, d => d.Contains("spam"));
        Assert.DoesNotContain(allDocs, d => d.Contains("user@example.com"));
        Assert.Contains(allDocs, d => d.Contains("[EMAIL]"));
        Assert.Contains(allDocs, d => d.Contains("hello world"));
        Assert.Contains(allDocs, d => d.Contains("clean document"));
    }

    [Fact]
    public void Describe_ReturnsPipelineSummary()
    {
        string path = _dir.Write("x.txt", "hello");
        var pipeline = CleaningPipeline
            .From(new TextFileSource(path))
            .Pipe(new NormaliseWhitespace());

        var loader = new DataLoader(pipeline, Tokenise,
                         new PackingBatcher(batchSize: 1, maxSeqLen: 8));

        string desc = loader.Describe();
        Assert.Contains("TextFile", desc);
        Assert.Contains("NormaliseWhitespace", desc);
    }
}
