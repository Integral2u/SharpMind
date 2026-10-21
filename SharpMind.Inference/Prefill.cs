using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Shared chunked-prefill helper for the token generators.
///
/// Each generator sizes its <see cref="Core.Memory.Workspace"/> with
/// <see cref="Core.Memory.Workspace.CalculateRequiredSize"/>, which budgets for at
/// most <c>min(MaxSeqLen, 128)</c> prefill tokens. The generators used to send the
/// entire prompt to <see cref="Transformer.ForwardLastLogits"/> in one call, so a
/// prompt longer than that overflowed the workspace's bump allocator. The
/// <c>OutOfMemoryException</c> then travelled up to ChatSession's catch-all and
/// surfaced as an empty reply with no stream at all.
/// </summary>
public static class Prefill
{
    /// <summary>
    /// Maximum prompt tokens fed to a single <see cref="Transformer.ForwardLastLogits"/>
    /// call. Kept below the prefill sizing cap (128) used by
    /// <see cref="Core.Memory.Workspace.CalculateRequiredSize"/> so each chunk fits
    /// with headroom for the chunk input and logits tensors, even on models where
    /// the workspace is sized by the estimate itself rather than the 100 MB floor.
    /// </summary>
    public const int MaxChunkLength = 64;

    /// <summary>
    /// Runs <paramref name="promptIds"/> through the model in chunks of at most
    /// <see cref="MaxChunkLength"/>, accumulating the KV cache.
    /// Returns the logits for the last prompt token, valid until the caller resets
    /// <paramref name="workspace"/>.
    ///
    /// Chunking is numerically identical to a single-shot prefill: each chunk starts
    /// at the current cache length, so positions (learned embeddings, RoPE, KV slots)
    /// advance exactly as they would for the whole prompt at once. Every chunk but the
    /// last goes through <see cref="Transformer.ForwardFill"/>, which skips the
    /// vocabulary projection; only the final chunk goes through
    /// <see cref="Transformer.ForwardLastLogits"/>.
    ///
    /// When <paramref name="progress"/> is supplied it is invoked once per finished
    /// chunk with the overall fraction of the prompt prefilled so far (in
    /// <c>0..1</c>), on the calling thread. Useful for surfacing "Prefilling NN%"
    /// during the (potentially slow) first turn.
    ///
    /// <paramref name="startPosition"/> is the absolute position of the first prompt
    /// token. Omit it when the cache holds the full conversation prefix; pass the
    /// tracked position when sliding-window trimming has made cache length diverge
    /// from the absolute position.
    /// </summary>
    public static Tensor<float> ForwardLastLogitsChunked(
        Transformer model,
        IKVCache[] caches,
        int[] promptIds,
        Core.Memory.IWorkspace workspace,
        Action<double>? progress = null,
        int? startPosition = null)
    {
        if (promptIds.Length == 0)
            throw new ArgumentException("Prompt produced no token IDs; cannot prefill.", nameof(promptIds));

        // RoPE (and learned positional embeddings) index the absolute conversation
        // position, which no longer equals cache length once a sliding-window trim
        // has dropped early rows. Callers that track that position pass it in;
        // otherwise the historical "position == cache length" rule applies.
        int position = startPosition ?? caches[0].Length;
        int positionLimit = model.Config.MaxPosition;
        if (position < 0 || (long)position + promptIds.Length > positionLimit)
        {
            throw new InvalidOperationException(
                $"Prompt positions {position}..{position + promptIds.Length - 1} exceed the model's " +
                $"context window of {positionLimit} positions. Reduce the prompt length.");
        }

        // A single chunk that fits goes straight through. One that would overflow the
        // cache takes the loop below, which trims first, like any longer prompt.
        if (promptIds.Length <= MaxChunkLength && caches[0].Length + promptIds.Length <= caches[0].MaxSeqLen)
        {
            workspace.Reset();
            return RunChunk(model, caches, promptIds, 0, promptIds.Length, project: true, workspace, position)!;
        }

        Tensor<float>? logits = null;
        try
        {
            int processed = 0;
            int len = 0;
            for (int start = 0; start < promptIds.Length; start += len)
            {
                len = Math.Min(MaxChunkLength, promptIds.Length - start);

                // Sliding-window safety: when the next chunk would overflow
                // the KV cache, trim to half capacity so there is room to
                // continue without throwing. This discards early prompt tokens
                // the window can no longer hold — correct for SWA models.
                if (caches[0].Length + len > caches[0].MaxSeqLen)
                {
                    int keep = Math.Max(1, caches[0].MaxSeqLen / 2);
                    for (int i = 0; i < caches.Length; i++)
                        caches[i].TrimToLast(keep);
                }

                // Cap chunk length to remaining capacity after any trim.
                int available = caches[0].MaxSeqLen - caches[0].Length;
                if (len > available) len = available;
                if (len <= 0) break;

                workspace.Reset();
                bool last = start + len >= promptIds.Length;
                logits = RunChunk(model, caches, promptIds, start, len, project: last, workspace, position + start);
                processed += len;
                progress?.Invoke((double)processed / promptIds.Length);
            }
            // Only the last chunk projects, so a cache that fills up before it leaves no logits.
            return logits ?? throw new InvalidOperationException("The KV cache has no room left for the rest of the prompt.");
        }
        catch
        {
            logits?.Dispose();
            throw;
        }
    }

    private static Tensor<float>? RunChunk(
        Transformer model,
        IKVCache[] caches,
        int[] promptIds,
        int start,
        int len,
        bool project,
        Core.Memory.IWorkspace workspace,
        int position)
    {
        using var chunkInput = workspace.Rent<int>([1, len]);
        promptIds.AsSpan(start, len).CopyTo(chunkInput.Data);
        if (!project)
        {
            model.ForwardFill(chunkInput, caches, position, workspace);
            return null;
        }
        return model.ForwardLastLogits(chunkInput, caches, position, workspace);
    }
}
