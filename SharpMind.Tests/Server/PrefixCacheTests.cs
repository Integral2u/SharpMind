using System.Collections.Generic;
using System.Linq;
using SharpMind.Inference.Chat;
using SharpMind.Server;
using Xunit;

namespace SharpMind.Tests.Server;

/// <summary>
/// The prefix cache decides whether a request costs a prefill or not, and it is a
/// byte-bounded LRU keyed by conversation shape. These pin the two things that go
/// wrong in that kind of structure: picking the wrong entry (a shorter prefix
/// when a longer one exists) and eviction accounting (an entry that silently
/// never leaves, or a budget that grows past its cap).
/// </summary>
public sealed class PrefixCacheTests
{
    private static ChatSessionSnapshot Snap(int messages, int layerBytes = 100)
    {
        var history = new List<ChatMessage>();
        for (int i = 0; i < messages; i++)
        {
            history.Add(new ChatMessage
            {
                Role = i % 2 == 0 ? ChatRole.User : ChatRole.User,
                Content = $"m{i}",
            });
        }

        return new ChatSessionSnapshot
        {
            History = history,
            KVCache = new KVCacheSnapshot
            {
                PromptHash = "hash",
                PromptTokenCount = 10,
                PromptTokens = [1, 2, 3],
                Layers = [new byte[layerBytes]],
            },
        };
    }

    private static ChatMessage[] History(params string[] contents) =>
        [.. contents.Select((c, i) => new ChatMessage
        {
            Role = i % 2 == 0 ? ChatRole.User : ChatRole.User,
            Content = c,
        })];

    [Fact]
    public void DisabledCache_NeverMatchesAndStoresNothing()
    {
        var cache = new PrefixCache(0);

        cache.Put("model", Snap(2));

        Assert.False(cache.Enabled);
        Assert.Equal(0, cache.Count);
        Assert.Null(cache.Take("model", History("m0", "m1", "m2")));
    }

    [Fact]
    public void Take_PicksTheLongestMatchingPrefix()
    {
        var cache = new PrefixCache(1_000_000);

        cache.Put("model", Snap(1));   // [m0]
        cache.Put("model", Snap(3));   // [m0, m1, m2]

        var hit = cache.Take("model", History("m0", "m1", "m2", "m3"));

        Assert.NotNull(hit);
        Assert.Equal(3, hit!.History.Length);
        Assert.Equal(1, cache.Hits);
        Assert.Equal(3, cache.ReusedTokens);
    }

    [Fact]
    public void Take_DoesNotCrossModels()
    {
        var cache = new PrefixCache(1_000_000);
        cache.Put("model-a", Snap(2));

        Assert.Null(cache.Take("model-b", History("m0", "m1", "m2")));
        Assert.Equal(1, cache.Misses);
    }

    [Fact]
    public void Take_RequiresTheStoredHistoryToBeAPrefix()
    {
        var cache = new PrefixCache(1_000_000);
        cache.Put("model", Snap(2));   // [m0, m1]

        Assert.Null(cache.Take("model", History("m0", "different", "m2")));
    }

    [Fact]
    public void Put_EvictsLeastRecentlyUsedEntriesToStayInsideTheBudget()
    {
        // Each entry is 100 bytes of KV; a 250-byte budget holds two.
        var cache = new PrefixCache(250);

        cache.Put("model", Snap(2));
        cache.Put("model", Snap(3));
        cache.Put("model", Snap(4));

        Assert.Equal(2, cache.Count);
        Assert.True(cache.Bytes <= cache.MaxBytes);
        // The first entry is the one that went.
        Assert.Null(cache.Take("model", History("m0", "x")));
        Assert.NotNull(cache.Take("model", History("m0", "m1", "m2", "m3")));
    }

    [Fact]
    public void Put_RefusesAnEntryBiggerThanTheWholeBudget()
    {
        var cache = new PrefixCache(50);

        cache.Put("model", Snap(2, layerBytes: 100));

        Assert.Equal(0, cache.Count);
        Assert.Equal(0, cache.Bytes);
    }

    [Fact]
    public void Put_SkipsASessionThatNeverPrefilled()
    {
        var cache = new PrefixCache(1_000_000);
        var noKv = new ChatSessionSnapshot { History = Snap(2).History, KVCache = null };

        cache.Put("model", noKv);

        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void Put_ReplacesAnIdenticalHistoryInsteadOfDuplicatingIt()
    {
        var cache = new PrefixCache(1_000_000);

        cache.Put("model", Snap(2));
        cache.Put("model", Snap(2));

        Assert.Equal(1, cache.Count);
        Assert.Equal(100, cache.Bytes);
    }
}
