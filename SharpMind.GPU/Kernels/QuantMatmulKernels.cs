using ILGPU;
using ILGPU.Algorithms;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// On-device fused dequant+matmul for GGUF quantized weights. Inference only (no backward):
/// the CPU engine's source of truth is the <c>VecDot*_Scalar</c> family (Q8_0, Q4_0, Q4_1,
/// Q5_0, Q5_1) for the block linears and the weight-tied LM head, and the raw
/// <c>token_embd.weight</c> gather. All three read the same <c>[N, K]</c> block layout
/// (<c>VecDotQ8_0_Scalar(input, raw, col, K)</c>): output column <c>col</c>'s blocks live at
/// <c>raw + col·nBlocks·blockBytes + b·blockBytes</c>, each block a little-endian f16 scale then
/// the quant payload (32 elements per block). The LM head is just the same matmul with
/// <c>K = hidden, N = vocab</c> and raw = the embedding (weight-tied), which is why one kernel
/// serves every call site.
///
/// The per-thread reduction replicates the <c>_Scalar</c> accumulation order (double accumulator,
/// F32 products, same block/column iteration) so the result agrees with the CPU oracle on
/// identical raw bytes. Scale/min conversion is pure arithmetic — normal/subnormal half -> float —
/// matching <c>QuantizationKernels.HalfToFloat_Scalar</c> for every value a real weight can hold.
/// </summary>
internal static class QuantMatmulKernels
{
    public const int QK = 32;

    /// <summary>Little-endian f16 at <paramref name="block"/> -> float. Pure arithmetic, matching
    /// <c>HalfToFloat_Scalar</c>: subnormal (exp5 == 0) is <c>mant · 2^-24</c>, everything in the
    /// normal range is <c>±(1 + mant/1024) · 2^(exp5-15)</c>. Scale values in real weights are
    /// finite normals (or zero), so the exp5==31 inf/nan path is unreachable.</summary>
    private static float ReadHalf(ArrayView<byte> w, long offset)
    {
        ushort v = (ushort)(w[offset] | (w[offset + 1] << 8));
        int exp5 = (v >> 10) & 0x1F;
        float sign = (v & 0x8000) != 0 ? -1f : 1f;
        float res;
        if (exp5 == 0)
            res = (v & 0x3FF) == 0 ? 0f : (float)(v & 0x3FF) * 5.960464477539063e-8f;
        else
            res = (1.0f + (float)(v & 0x3FF) / 1024.0f) * XMath.Exp2(exp5 - 15);
        return sign * res;
    }

    /// <summary>Block size in raw bytes for the block quants this engine runs on the device.</summary>
    internal static int BlockBytes(QuantDType q) => q switch
    {
        QuantDType.Q8_0 => 34,
        QuantDType.Q4_0 => 18,
        QuantDType.Q4_1 => 20,
        QuantDType.Q5_0 => 22,
        QuantDType.Q5_1 => 24,
        _ => throw new NotSupportedException($"GPU inference supports only Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 quantized weights, got {q}."),
    };

    /// <summary>Throw-free <see cref="BlockBytes"/> for use inside device kernels (ILGPU rejects
    /// <c>throw</c> inlined into OpenCL code). The <c>DequantMatmul</c>/<c>DequantGather</c> entry
    /// points are only ever called with a supported quant from <see cref="GpuInferenceEngine"/>.</summary>
    private static int BlockSize(QuantDType q) => q == QuantDType.Q8_0 ? 34
        : q == QuantDType.Q4_0 ? 18
        : q == QuantDType.Q4_1 ? 20
        : q == QuantDType.Q5_0 ? 22 : 24;

    /// <summary>Q8_0 stores quantized values as two's-complement sbyte; reinterpret the raw byte.</summary>
    private static int ToSByte(byte b) => b < 128 ? (int)b : b - 256;

    private static int ReadU32(ArrayView<byte> w, long offset)
        => w[offset] | (w[offset + 1] << 8) | (w[offset + 2] << 16) | (w[offset + 3] << 24);

    /// <summary>Dequantises element <paramref name="t"/> (0..31) of the block at <paramref name="block"/>.
    /// Whitespace-free: all four nibble formats (Q4_0/Q4_1/Q5_0/Q5_1) share the low-nibble/high-nibble
    /// packing and differ only in scale offset, min ("m"), the high-quant bit table and the centering
    /// offset; Q8_0 is a plain sbyte.</summary>
    private static float Decode(QuantDType q, ArrayView<byte> w, long block, int t, float d, float m,
        int qh, int nibOff)
    {
        if (q == QuantDType.Q8_0)
            return ToSByte(w[block + 2 + t]) * d;
        int nib = t < 16 ? (w[block + nibOff + t] & 0x0F) : (w[block + nibOff + t - 16] >> 4);
        if (qh != 0 && ((qh >> t) & 1) != 0) nib |= 16;
        int off = q == QuantDType.Q4_0 ? 8 : q == QuantDType.Q5_0 ? 16 : 0;
        return (nib - off) * d + m;
    }

    /// <summary>
    /// y[i,o] = Σ_k x[i,k] · w[o,k] for a [N, K] block-quantized matrix in VecDot layout.
    /// One thread per output element (idx = i·N + o), reduction over K blocks in the same order
    /// as the CPU scalar. Used for every block linear (M=m, K=In, N=Out) and the weight-tied
    /// LM head (M=m, K=Hidden, N=Vocab) with the embedding's raw bytes.
    /// </summary>
    public static void DequantMatmul(Index1D idx, ArrayView<float> y, ArrayView<float> x,
        ArrayView<byte> w, int K, int N, int nBlocks, QuantDType q)
    {
        int blockBytes = BlockSize(q);
        bool hasMin = q == QuantDType.Q4_1 || q == QuantDType.Q5_1;
        bool hasQh = q == QuantDType.Q5_0 || q == QuantDType.Q5_1;
        int nibOff = q == QuantDType.Q8_0 ? 0 : (q == QuantDType.Q4_0 ? 2 : q == QuantDType.Q4_1 ? 4 : q == QuantDType.Q5_0 ? 6 : 8);
        int minOff = q == QuantDType.Q4_1 || q == QuantDType.Q5_1 ? 2 : -1;
        int qhOff = q == QuantDType.Q5_0 ? 2 : q == QuantDType.Q5_1 ? 4 : -1;
        int i = idx / N, o = idx - i * N;
        int inRow = i * K;
        double acc = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            long block = (long)o * nBlocks * blockBytes + (long)b * blockBytes;
            float d = ReadHalf(w, block);
            float m = hasMin ? ReadHalf(w, block + minOff) : 0f;
            int qh = hasQh ? ReadU32(w, block + qhOff) : 0;
            int inBase = b * QK;
            int end = inBase + QK < K ? QK : K - inBase;
            for (int t = 0; t < end; t++)
                acc += (double)x[inRow + inBase + t] * Decode(q, w, block, t, d, m, qh, nibOff);
        }
        y[idx] = (float)acc;
    }

    /// <summary>
    /// x[i,d] = dequant(embedding[ids[i]][d]) for a [V, K] block-quantized embedding table. The embedding
    /// row for token <c>ids[i]</c> is output column <c>ids[i]</c> of the same layout the LM head
    /// matmul consumes, so the block addressing is identical. One thread per (row, feature).
    /// </summary>
    public static void DequantGather(Index1D idx, ArrayView<float> x, ArrayView<byte> table,
        ArrayView<int> ids, int cols, int nBlocks, QuantDType q)
    {
        int blockBytes = BlockSize(q);
        bool hasMin = q == QuantDType.Q4_1 || q == QuantDType.Q5_1;
        bool hasQh = q == QuantDType.Q5_0 || q == QuantDType.Q5_1;
        int nibOff = q == QuantDType.Q8_0 ? 0 : (q == QuantDType.Q4_0 ? 2 : q == QuantDType.Q4_1 ? 4 : q == QuantDType.Q5_0 ? 6 : 8);
        int minOff = q == QuantDType.Q4_1 || q == QuantDType.Q5_1 ? 2 : -1;
        int qhOff = q == QuantDType.Q5_0 ? 2 : q == QuantDType.Q5_1 ? 4 : -1;
        int t = idx / cols, d = idx - t * cols;
        int token = ids[t];
        int b = d / QK;
        long block = (long)token * nBlocks * blockBytes + (long)b * blockBytes;
        float scale = ReadHalf(table, block);
        float m = hasMin ? ReadHalf(table, block + minOff) : 0f;
        int qh = hasQh ? ReadU32(table, block + qhOff) : 0;
        int tq = d - b * QK;
        x[idx] = Decode(q, table, block, tq, scale, m, qh, nibOff);
    }

    /// <summary>y[i,o] = Σ_k x[i,k]·wQ8[o,k]; see <see cref="DequantMatmul"/>.</summary>
    public static void Q8_0Matmul(Index1D idx, ArrayView<float> y, ArrayView<float> x,
        ArrayView<byte> w, int K, int N, int nBlocks)
        => DequantMatmul(idx, y, x, w, K, N, nBlocks, QuantDType.Q8_0);

    /// <summary>Gather over a Q8_0 embedding table; see <see cref="DequantGather"/>.</summary>
    public static void Q8_0Gather(Index1D idx, ArrayView<float> x, ArrayView<byte> table,
        ArrayView<int> ids, int cols, int nBlocks)
        => DequantGather(idx, x, table, ids, cols, nBlocks, QuantDType.Q8_0);
}