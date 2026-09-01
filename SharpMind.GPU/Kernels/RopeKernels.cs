using ILGPU;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// One thread per (row, head, pair). Two pairings, as RoPE.Apply: adjacent (2i, 2i+1)
/// or NeoX halves (i, i + ropeDim/2). sign = +1 forward, −1 backward (inverse rotation).
/// </summary>
internal static class RopeKernels
{
    public static void Rope(Index1D idx, ArrayView<float> x, ArrayView<float> cos, ArrayView<float> sin,
        int seqLen, int numHeads, int headDim, int ropeDim, int neox, float sign)
    {
        int pairs = ropeDim / 2;
        int pair = idx % pairs;
        int head = (idx / pairs) % numHeads;
        int row = idx / (pairs * numHeads);
        int pos = row % seqLen;
        long baseOff = (long)row * numHeads * headDim + (long)head * headDim;
        int i0 = neox != 0 ? pair : 2 * pair;
        int i1 = neox != 0 ? pair + pairs : 2 * pair + 1;
        float c = cos[pos * pairs + pair];
        float s = sin[pos * pairs + pair] * sign;
        float x0 = x[baseOff + i0], x1 = x[baseOff + i1];
        x[baseOff + i0] = x0 * c - x1 * s;
        x[baseOff + i1] = x1 * c + x0 * s;
    }

    /// <summary>
    /// Position-offset variant of <see cref="Rope"/>: rows are treated as sitting at absolute
    /// positions <c>[pos0, pos0 + seqLen)</c> rather than <c>[0, seqLen)</c>, so the cos/sin
    /// tables are indexed at <c>pos0 + row</c>. Needed by inference decode and continued prefill,
    /// which rotate a fresh token/K row at absolute position <c>pos0</c> (the current cache length)
    /// instead of at 0. <paramref name="pos0"/> is the absolute position of row 0.
    /// </summary>
    public static void RopePos(Index1D idx, ArrayView<float> x, ArrayView<float> cos, ArrayView<float> sin,
        int seqLen, int pos0, int numHeads, int headDim, int ropeDim, int neox, float sign)
    {
        int pairs = ropeDim / 2;
        int pair = idx % pairs;
        int head = (idx / pairs) % numHeads;
        int row = idx / (pairs * numHeads);
        int pos = pos0 + (row % seqLen);
        long baseOff = (long)row * numHeads * headDim + (long)head * headDim;
        int i0 = neox != 0 ? pair : 2 * pair;
        int i1 = neox != 0 ? pair + pairs : 2 * pair + 1;
        float c = cos[pos * pairs + pair];
        float s = sin[pos * pairs + pair] * sign;
        float x0 = x[baseOff + i0], x1 = x[baseOff + i1];
        x[baseOff + i0] = x0 * c - x1 * s;
        x[baseOff + i1] = x1 * c + x0 * s;
    }
}
