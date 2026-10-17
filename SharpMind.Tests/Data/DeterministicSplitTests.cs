using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

/// <summary>
/// B2: <see cref="DeterministicSplit.TrainVal"/> assigns documents to train/val
/// by a stable content hash, so the same corpus splits identically every run and
/// appending new files never moves an existing document to the other side.
/// </summary>
public sealed class DeterministicSplitTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static string Line(int i) => $"document number {i} with a little shared boilerplate";

    private string Corpus(string name, int count, int offset = 0)
        => _dir.Write(name, string.Join('\n', Enumerable.Range(offset, count).Select(Line)));

    private static PipelineNode Source(string path) => PipelineNode.From(new TextFileSource(path));

    private static async Task<List<string>> Collect(PipelineNode node)
    {
        var list = new List<string>();
        await foreach (string doc in node.ReadAsync())
            list.Add(doc);
        return list;
    }

    [Fact]
    public async Task SameCorpus_AlwaysSplitsIdentically()
    {
        string path = Corpus("corpus.txt", 500);

        var (train1, val1) = Source(path).TrainVal(0.2, seed: 7);
        var (train2, val2) = Source(path).TrainVal(0.2, seed: 7);

        var v1 = await Collect(val1);
        var v2 = await Collect(val2);

        Assert.Equal(v1, v2);
        Assert.Equal(await Collect(train1), await Collect(train2));
    }

    [Fact]
    public async Task AppendingFiles_KeepsExistingDocumentsOnTheSameSide()
    {
        string path = Corpus("corpus.txt", 200);
        var (_, before) = Source(path).TrainVal(0.25, seed: 1);
        var beforeVal = (await Collect(before)).ToHashSet();

        // Append 100 more documents, then re-read the same file.
        File.AppendAllText(path, "\n" + string.Join('\n', Enumerable.Range(200, 100).Select(Line)));

        var (_, after) = Source(path).TrainVal(0.25, seed: 1);
        var afterVal = (await Collect(after)).ToHashSet();

        // Every document that was validation before is still validation; the
        // originals are a subset of the new val set (new docs may join it).
        var originals = Enumerable.Range(0, 200).Select(Line).ToHashSet();
        Assert.All(beforeVal, doc => Assert.Contains(doc, afterVal));
        Assert.All(beforeVal, doc => Assert.Contains(doc, originals));
    }

    [Fact]
    public async Task SplitIsDisjoint_AndCoversEveryDocument()
    {
        string path = Corpus("corpus.txt", 400);
        var (train, val) = Source(path).TrainVal(0.3, seed: 0);

        var trainSet = (await Collect(train)).ToHashSet();
        var valSet = (await Collect(val)).ToHashSet();
        var all = Enumerable.Range(0, 400).Select(Line).ToHashSet();

        Assert.Empty(trainSet.Intersect(valSet));
        Assert.Equal(all, trainSet.Union(valSet).ToHashSet());
    }

    [Fact]
    public async Task ValFraction_IsApproximated()
    {
        string path = Corpus("corpus.txt", 4000);

        var (_, val) = Source(path).TrainVal(0.2, seed: 0);
        int valCount = (await Collect(val)).Count;

        // Hashing lands near the requested fraction; allow a small tolerance.
        Assert.InRange(valCount, 700, 900);
    }

    [Fact]
    public void IsValidation_MatchesTheBranchDecision()
    {
        string doc = Line(42);
        Assert.Equal(DeterministicSplit.IsValidation(doc, 0.2, 3),
                     DeterministicSplit.Bucket(doc, seed: 3) < (int)Math.Round(0.2 * DeterministicSplit.Buckets));
    }

    [Fact]
    public void Seed_ReshufflesTheSplit()
    {
        // Different seeds should send at least some documents the other way.
        int moved = Enumerable.Range(0, 500)
            .Count(i => DeterministicSplit.IsValidation(Line(i), 0.2, seed: 1)
                     != DeterministicSplit.IsValidation(Line(i), 0.2, seed: 2));

        Assert.True(moved > 0);
    }

    [Fact]
    public void ValFraction_OutOfRange_Throws()
    {
        string path = Corpus("corpus.txt", 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => Source(path).TrainVal(1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Source(path).TrainVal(-0.1));
    }
}
