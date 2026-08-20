using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json.Serialization;

namespace SharpMind.Inference.Chat;

/// <summary>
/// Persisted KV-cache state plus a SHA-256 hash of the prompt tokens that
/// were prefilled. On session resume, if the hash matches the freshly-built
/// prompt the warm-up prefill is skipped entirely — the cache is restored
/// from the saved bytes and the first user turn extends it incrementally.
/// </summary>
public sealed class KVCacheSnapshot
{
    /// <summary>SHA-256 hex of the prompt token IDs that were prefilled into the cache.</summary>
    public required string PromptHash { get; init; }

    /// <summary>Number of prompt tokens that were prefilled.</summary>
    public required int PromptTokenCount { get; init; }

    /// <summary>
    /// One binary blob per cache layer (KVCache / PagedKVCache /
    /// QuantizedKVCache), produced by <c>IKVCache.SnapshotBytes()</c>.
    /// </summary>
    public required List<byte[]> Layers { get; init; }

    /// <summary>
    /// Computes a deterministic SHA-256 hash from the prompt token IDs.
    /// </summary>
    public static string HashPromptTokens(int[] tokens)
    {
        var bytes = MemoryMarshal.AsBytes(tokens.AsSpan());
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(bytes, hash);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
