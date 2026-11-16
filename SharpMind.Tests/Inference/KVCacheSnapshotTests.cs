using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The snapshot's hash can only prove an exact prompt match, which is why a
/// multi-turn chat never reused its cache: every turn grows the prompt. Prefix
/// reuse needs the tokens themselves so it can ask "does this prompt start with
/// what the cache holds", and get a truthful answer for old snapshots that
/// never recorded them.
/// </summary>
public sealed class KVCacheSnapshotTests
{
    private static KVCacheSnapshot Snapshot(int[]? tokens) => new()
    {
        PromptHash = KVCacheSnapshot.HashPromptTokens(tokens ?? []),
        PromptTokenCount = tokens?.Length ?? 0,
        PromptTokens = tokens,
        Layers = [],
    };

    [Fact]
    public void IsPrefixOf_TrueForTheSamePrompt() =>
        Assert.True(Snapshot([1, 2, 3]).IsPrefixOf([1, 2, 3]));

    [Fact]
    public void IsPrefixOf_TrueWhenThePromptContinuesTheCachedTokens() =>
        Assert.True(Snapshot([1, 2, 3]).IsPrefixOf([1, 2, 3, 4, 5]));

    [Fact]
    public void IsPrefixOf_FalseWhenThePromptDivergesInsideTheCachedRange()
    {
        Assert.False(Snapshot([1, 2, 3]).IsPrefixOf([1, 2, 9, 4]));
        Assert.False(Snapshot([1, 2, 3]).IsPrefixOf([1, 2, 9]));
    }

    [Fact]
    public void IsPrefixOf_FalseWhenThePromptIsShorterThanTheCache()
    {
        // The cache holds tokens this prompt does not; importing it would put the
        // cache ahead of the prompt and corrupt positions.
        Assert.False(Snapshot([1, 2, 3]).IsPrefixOf([1, 2]));
        Assert.False(Snapshot([1, 2, 3]).IsPrefixOf([]));
    }

    [Fact]
    public void IsPrefixOf_FalseForASnapshotThatNeverRecordedItsTokens()
    {
        // Snapshots written before PromptTokens existed still load; they simply
        // cannot be used as a prefix, only at the exact-match fast path.
        Assert.False(Snapshot(null).IsPrefixOf([1, 2, 3]));
        Assert.False(Snapshot([]).IsPrefixOf([1, 2, 3]));
    }
}
