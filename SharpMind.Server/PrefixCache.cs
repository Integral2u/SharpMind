using SharpMind.Inference.Chat;

namespace SharpMind.Server;

/// <summary>
/// Cross-request KV prefix cache. A chat request re-states its whole history, so
/// the same tokens are prefilled on every turn; this keeps the KV that a finished
/// request produced and hands it to the next request whose history extends it.
///
/// Keyed by model plus the non-system message list, longest match wins. System
/// prompts are deliberately not part of the key: the session re-adds them, and the
/// session's own token-level check rejects a snapshot whose prompt does not really
/// start the same way (different tool schema, formatter, etc.).
///
/// Bounded by bytes of KV, not entry count, because one long conversation can dwarf
/// a hundred short ones.
/// </summary>
public sealed class PrefixCache(long maxBytes)
{
    private readonly LinkedList<Entry> _entries = new();

    public long MaxBytes { get; } = maxBytes;
    public long Bytes { get; private set; }
    public int Count => _entries.Count;
    public int Hits { get; private set; }
    public int Misses { get; private set; }
    public long ReusedTokens { get; private set; }

    public bool Enabled => MaxBytes > 0;

    public sealed class Entry
    {
        public required string ModelId { get; init; }
        public required ChatMessage[] History { get; init; }
        public required ChatSessionSnapshot Snapshot { get; init; }
        public required long Bytes { get; init; }
    }

    /// <summary>
    /// Longest stored history that is a prefix of <paramref name="history"/>, moved
    /// to the front. Null when nothing matches (or the cache is off).
    /// </summary>
    public Entry? Take(string modelId, IReadOnlyList<ChatMessage> history)
    {
        if (!Enabled) return null;

        Entry? best = null;
        foreach (var entry in _entries)
        {
            if (!string.Equals(entry.ModelId, modelId, StringComparison.Ordinal)) continue;
            if (entry.History.Length > history.Count) continue;
            if (!IsPrefix(entry.History, history)) continue;
            if (best is null || entry.History.Length > best.History.Length) best = entry;
        }

        if (best is null)
        {
            Misses++;
            return null;
        }

        Hits++;
        ReusedTokens += best.History.Length;
        _entries.Remove(best);
        _entries.AddFirst(best);
        return best;
    }

    /// <summary>
    /// Stores a finished session's cache. Entries with no KV (a session that never
    /// prefilled) are skipped, and a snapshot larger than the whole budget is not
    /// stored at all - caching it would evict everything for one entry that still
    /// cannot fit.
    /// </summary>
    public void Put(string modelId, ChatSessionSnapshot snapshot)
    {
        if (!Enabled || snapshot.KVCache is null) return;

        long bytes = 0;
        foreach (byte[] layer in snapshot.KVCache.Layers) bytes += layer.Length;
        if (bytes == 0 || bytes > MaxBytes) return;

        ChatMessage[] history = [.. snapshot.History.Where(m => m.Role != ChatRole.System)];
        if (history.Length == 0) return;

        // Replace an identical history rather than keeping two copies of it.
        foreach (var existing in _entries)
        {
            if (string.Equals(existing.ModelId, modelId, StringComparison.Ordinal)
                && existing.History.Length == history.Length
                && IsPrefix(existing.History, history))
            {
                _entries.Remove(existing);
                Bytes -= existing.Bytes;
                break;
            }
        }

        while (Bytes + bytes > MaxBytes && _entries.Last is { } last)
        {
            _entries.RemoveLast();
            Bytes -= last.Value.Bytes;
        }

        _entries.AddFirst(new Entry
        {
            ModelId = modelId,
            History = history,
            Snapshot = snapshot,
            Bytes = bytes,
        });
        Bytes += bytes;
    }

    private static bool IsPrefix(ChatMessage[] prefix, IReadOnlyList<ChatMessage> history)
    {
        for (int i = 0; i < prefix.Length; i++)
        {
            if (prefix[i].Role != history[i].Role) return false;
            if (!string.Equals(prefix[i].Content, history[i].Content, StringComparison.Ordinal)) return false;
        }
        return true;
    }
}
