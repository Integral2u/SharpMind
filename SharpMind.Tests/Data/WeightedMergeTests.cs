using SharpMind.Data.Pipeline;
using Xunit;

namespace SharpMind.Tests.Data;

/// <summary>
/// B7: <see cref="CleaningPipeline.MergeWeighted"/> mixes several streams in
/// proportion to their weights (replay mixing) and does so deterministically.
/// </summary>
public sealed class WeightedMergeTests
{
    private static PipelineNode Node(string prefix, int count)
        => CleaningPipeline.From(new InMemorySource(
            Enumerable.Range(0, count).Select(i => $"{prefix}{i}")));

    private static async Task<List<string>> Drain(PipelineNode node)
    {
        var docs = new List<string>();
        await foreach (var doc in node.ReadAsync())
            docs.Add(doc);
        return docs;
    }

    private static (int A, int B) Counts(IEnumerable<string> docs)
    {
        int a = 0, b = 0;
        foreach (var doc in docs)
        {
            if (doc.StartsWith('a')) a++;
            else if (doc.StartsWith('b')) b++;
        }
        return (a, b);
    }

    [Fact]
    public async Task ThreeToOne_ProducesTheWeightedProportions()
    {
        // Supply enough of each so neither runs out before the ratio is realised.
        var mixed = CleaningPipeline.MergeWeighted((Node("a", 300), 3.0), (Node("b", 100), 1.0));

        var docs = await Drain(mixed);

        Assert.Equal(400, docs.Count);
        Assert.Equal((300, 100), Counts(docs));

        // The ratio holds locally too, not just in aggregate.
        Assert.Equal((30, 10), Counts(docs.Take(40)));
    }

    [Fact]
    public async Task Mixing_InterleavesRatherThanFrontLoading()
    {
        var mixed = CleaningPipeline.MergeWeighted((Node("a", 300), 3.0), (Node("b", 100), 1.0));

        var docs = await Drain(mixed);

        // Both sources appear within the first few documents, so the mix is
        // interleaved across the run rather than all-a-then-all-b.
        Assert.Contains(docs.Take(8), d => d.StartsWith('a'));
        Assert.Contains(docs.Take(8), d => d.StartsWith('b'));
    }

    [Fact]
    public async Task SameInputs_ProduceTheSameOrder()
    {
        PipelineNode Build() => CleaningPipeline.MergeWeighted((Node("a", 40), 2.0), (Node("b", 40), 1.0));

        var first = await Drain(Build());
        var second = await Drain(Build());

        Assert.Equal(first, second);
    }

    [Fact]
    public async Task ZeroWeight_ExcludesThatSource()
    {
        var mixed = CleaningPipeline.MergeWeighted((Node("a", 10), 1.0), (Node("b", 10), 0.0));

        var docs = await Drain(mixed);

        Assert.Equal(10, docs.Count);
        Assert.All(docs, d => Assert.StartsWith("a", d));
    }

    [Fact]
    public async Task ExhaustedSource_DoesNotStallTheOthers()
    {
        var mixed = CleaningPipeline.MergeWeighted((Node("a", 10), 1.0), (Node("b", 100), 1.0));

        var docs = await Drain(mixed);

        Assert.Equal(110, docs.Count);
        Assert.Equal((10, 100), Counts(docs));
    }

    [Fact]
    public void AllZeroWeights_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            CleaningPipeline.MergeWeighted((Node("a", 1), 0.0), (Node("b", 1), 0.0)));
    }

    [Fact]
    public void FewerThanTwoNodes_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            CleaningPipeline.MergeWeighted((Node("a", 1), 1.0)));
    }

    [Fact]
    public void NegativeWeight_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CleaningPipeline.MergeWeighted((Node("a", 1), 1.0), (Node("b", 1), -1.0)));
    }

    [Fact]
    public void Describe_ReportsTheWeightedMerge()
    {
        var mixed = CleaningPipeline.MergeWeighted((Node("a", 1), 3.0), (Node("b", 1), 1.0));

        Assert.Contains("WeightedMerge", mixed.Describe());
    }
}
