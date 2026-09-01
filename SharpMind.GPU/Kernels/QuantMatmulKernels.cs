using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// On-device fused dequant+matmul for GGUF quantized weights. Inference only (no backward):
/// the CPU engine's source of truth is <c>VecDotQ8_0_Scalar</c> for the block linears and the
/// weight-tied LM head, and the raw <c>token_embd.weight</c> gather. All three read the same
/// <c>[N, K]</c> Q8_0 layout (<c>VecDotQ8_0_Scalar(input, raw, col, K)</c>): column <c>col</c>'s
/// blocks live at <c>raw + col*nBlocks*34 + b*34</c>, each block a little-endian f16 scale then
/// 32 sbyteds. The LM head is just the same matmul with <c>K = hidden, N = vocab</c> and raw =
/// the embedding (weight-tied), which is why one kernel serves every call site.
///
/// The per-thread reduction replicates <c>_Scalar</c> accumulation order (F32 dot, not FMA tree)
/// so the result agrees with the CPU oracle on identical raw bytes. Scale conversion is pure
/// arithmetic — normal/subnormal half -> float — matching <c>QuantizationKernels.HalfToFloat_Scalar</c>
/// for every value a real weight scale can hold (block scales are never inf/nan).
/// </summary>
internal static class QuantMatmulKernels
{
    public const int QK = 32;
    public const int Q8BlockBytes = 34;

    /// <summary>
    /// y[i,o] = Σ_k x[i,k] · wQ8[o,k] for a [N, K] Q8_0 matrix in <see cref="VecDot"/> layout.
    /// One thread per output element (idx = i·N + o), reduction over K blocks in the same order
    /// as the CPU scalar. Used for every block linear (M=m, K=In, N=Out) and the weight-tied
    /// LM head (M=m, K=Hidden, N=Vocab) with the embedding's raw bytes.
    /// </summary>
    public static void Q8_0Matmul(Index1D idx, ArrayView<float> y, ArrayView<float> x,
        ArrayView<byte> w, int K, int N, int nBlocks)
    {
        int i = idx / N, o = idx - i * N;
        int inRow = i * K;
        double acc = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            long block = (long)o * nBlocks * Q8BlockBytes + (long)b * Q8BlockBytes;
            float d = BlockScale(w, block);
            int inBase = b * QK;
            int end = inBase + QK < K ? QK : K - inBase;
            long vBase = block + 2;
            for (int t = 0; t < end; t++)
                acc += (double)x[inRow + inBase + t] * (ToSByte(w[vBase + t]) * d);
        }
        y[idx] = (float)acc;
    }

    /// <summary>
    /// x[i,d] = dequant(embedding[ids[i]][d]) for a [V, K] Q8_0 embedding table. The embedding
    /// row for token <c>ids[i]</c> is output column <c>ids[i]</c> of the same layout the LM head
    /// matmul consumes, so the block addressing is identical. One thread per (row, feature).
    /// </summary>
    public static void Q8_0Gather(Index1D idx, ArrayView<float> x, ArrayView<byte> table,
        ArrayView<int> ids, int cols, int nBlocks)
    {
        int t = idx / cols, d = idx - t * cols;
        int token = ids[t];
        int b = d / QK;
        long block = (long)token * nBlocks * Q8BlockBytes + (long)b * Q8BlockBytes;
        float scale = BlockScale(table, block);
        x[idx] = ToSByte(table[block + 2 + (d - b * QK)]) * scale;
    }

    /// <summary>Q8_0 stores quantized values as two's-complement sbyte; reinterpret the raw byte.</summary>
    private static int ToSByte(byte b) => b < 128 ? (int)b : b - 256;

    /// <summary>
    /// Little-endian f16 at <paramref name="block"/> -> float. Pure arithmetic, matching
    /// <c>HalfToFloat_Scalar</c>: subnormal (exp5 == 0) is <c>mant · 2^-24</c>, everything in the
    /// normal range is <c>±(1 + mant/1024) · 2^(exp5-15)</c>. Scale values in real Q8_0 weights
    /// are finite normals (or zero), so the exp5==31 inf/nan path is unreachable.
    /// </summary>
    private static float BlockScale(ArrayView<byte> w, long block)
    {
        ushort v = (ushort)(w[block] | (w[block + 1] << 8));
        int exp5 = (v >> 10) & 0x1F;
        float sign = (v & 0x8000) != 0 ? -1f : 1f;
        float res;
        if (exp5 == 0)
            res = (v & 0x3FF) == 0 ? 0f : (float)(v & 0x3FF) * 5.960464477539063e-8f;
        else
            res = (1.0f + (float)(v & 0x3FF) / 1024.0f) * XMath.Exp2(exp5 - 15);
        return sign * res;
    }
}
