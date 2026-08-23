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
internal static class Prefill
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
    /// Runs <paramref name="promptIds"/> through <see cref="Transformer.ForwardLastLogits"/>
    /// in chunks of at most <see cref="MaxChunkLength"/>, accumulating the KV cache.
    /// Returns the logits for the last prompt token, valid until the caller resets
    /// <paramref name="workspace"/>.
    ///
    /// Chunking is numerically identical to a single-shot prefill: each chunk starts
    /// at the current cache length, so positions (learned embeddings, RoPE, KV slots)
    /// advance exactly as they would for the whole prompt at once. Intermediate
    /// chunks' logits are discarded; only the final chunk's logits are returned.
    ///
    /// When <paramref name="progress"/> is supplied it is invoked once per finished
    /// chunk with the overall fraction of the prompt prefilled so far (in
    /// <c>0..1</c>), on the calling thread. Useful for surfacing "Prefilling NN%"
    /// during the (potentially slow) first turn.
    /// </summary>
    public static Tensor<float> ForwardLastLogitsChunked(
        Transformer model,
        IKVCache[] caches,
        int[] promptIds,
        Core.Memory.IWorkspace workspace,
        Action<double>? progress = null)
    {
        if (promptIds.Length == 0)
            throw new ArgumentException("Prompt produced no token IDs; cannot prefill.", nameof(promptIds));

        if (promptIds.Length <= MaxChunkLength)
        {
            workspace.Reset();
            return RunChunk(model, caches, promptIds, 0, promptIds.Length, workspace);
        }

        Tensor<float>? logits = null;
        try
        {
            int processed = 0;
            for (int start = 0; start < promptIds.Length; start += MaxChunkLength)
            {
                int len = Math.Min(MaxChunkLength, promptIds.Length - start);
                logits?.Dispose();
                workspace.Reset();
                logits = RunChunk(model, caches, promptIds, start, len, workspace);
                processed += len;
                progress?.Invoke((double)processed / promptIds.Length);
            }
            return logits!;
        }
        catch
        {
            logits?.Dispose();
            throw;
        }
    }

    private static Tensor<float> RunChunk(
        Transformer model,
        IKVCache[] caches,
        int[] promptIds,
        int start,
        int len,
        Core.Memory.IWorkspace workspace)
    {
        using var chunkInput = workspace.Rent<int>([1, len]);
        promptIds.AsSpan(start, len).CopyTo(chunkInput.Data);
        var result = model.ForwardLastLogits(chunkInput, caches, caches[0].Length, workspace);
        return result;
    }
}
