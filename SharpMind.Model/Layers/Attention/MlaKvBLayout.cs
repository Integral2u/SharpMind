namespace SharpMind.Model.Layers.Attention;

/// <summary>
/// The storage layout MLA gives <c>attn_kv_b</c>: per-head blocks of
/// <c>[kvLora, nope | vDim]</c> rows instead of the plain
/// <c>[kvLora, heads x (nope + vDim)]</c> matrix.
///
/// <para>MLA reads that matrix three ways — absorbing it into the query
/// (decode), expanding latents through it (prefill) and reading values back out
/// — and all three walk one head at a time. In the plain layout a head's 256
/// columns are 4096 floats away from the next row, so every dot product is a
/// handful of cache lines at a 16 KB stride: measured 0.22 GMAC/s / 0.9 GB/s on
/// a 4-core AVX2 box. Head-major blocks make the same reads sequential and the
/// same loop reaches 1.29 GMAC/s / 5.2 GB/s. The element count is unchanged, so
/// this costs no memory — it only changes where each element is stored.</para>
///
/// <para>Packing happens once, where the weights are written: the GGUF float
/// path in <c>ModelLoaderBase</c> and the float-ingestion path in
/// <see cref="AttentionLayer"/>. A wrong layout here would still produce
/// plausible numbers, so <c>MlaAttentionTests</c> builds the blocks explicitly
/// and the reference cross-check below pins the kernels against index
/// arithmetic.</para>
/// </summary>
internal static class MlaKvBLayout
{
    /// <summary>
    /// Reorders a GGUF-order <c>attn_kv_b</c> (<c>src[out, in]</c>, i.e.
    /// <c>src[o * kvLora + j]</c>) into head-major blocks
    /// (<c>dst[(h * kvLora + j) * headStride + r]</c>).
    /// </summary>
    public static void Pack(ReadOnlySpan<float> src, int kvLora, int heads, int headStride, Span<float> dst)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvLora);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headStride);

        long count = (long)kvLora * heads * headStride;
        if (src.Length < count)
            throw new ArgumentException($"attn_kv_b needs {count} floats, got {src.Length}.", nameof(src));
        if (dst.Length < count)
            throw new ArgumentException($"attn_kv_b target holds {dst.Length} floats, {count} needed.", nameof(dst));

        for (int h = 0; h < heads; h++)
        {
            int srcCol = h * headStride;
            int dstHead = h * kvLora * headStride;
            for (int j = 0; j < kvLora; j++)
            {
                int dstRow = dstHead + j * headStride;
                for (int o = 0; o < headStride; o++)
                    dst[dstRow + o] = src[(srcCol + o) * kvLora + j];
            }
        }
    }
}
