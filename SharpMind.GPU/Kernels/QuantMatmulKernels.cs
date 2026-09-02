using ILGPU;
using ILGPU.Algorithms;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// On-device fused dequant+matmul for GGUF quantized weights. Inference only (no backward):
/// the CPU engine's source of truth is the <c>VecDot*_Scalar</c> family — the block quants
/// Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 (32-element blocks, <see cref="DequantMatmul"/>) and the K-quants
/// Q2_K/Q3_K/Q4_K/Q5_K/Q6_K (256-element super-blocks, <see cref="DequantMatmulK"/>) — for the
/// block linears and the weight-tied LM head, and the raw <c>token_embd.weight</c> gather. The
/// block quants read the same <c>[N, K]</c> block layout (<c>VecDotQ8_0_Scalar(input, raw, col, K)</c>):
/// output column <c>col</c>'s blocks live at <c>raw + col·nBlocks·blockBytes + b·blockBytes</c>,
/// each block a little-endian f16 scale then the quant payload (32 elements per block). The K-quant
/// stream is one run of 256-element super-blocks over the flattened [N, K] tensor, so column
/// <c>col</c> spans super-blocks from <c>col·K / 256</c> with its first element at <c>col·K % 256</c>.
/// The LM head is just the same matmul with <c>K = hidden, N = vocab</c> and raw = the embedding
/// (weight-tied), which is why one kernel shape serves every call site.
///
/// The per-thread reduction replicates the <c>_Scalar</c> accumulation order (double accumulator,
/// F32 products, same block/column iteration) so the result agrees with the CPU oracle on
/// identical raw bytes. Scale/min conversion is pure arithmetic — normal/subnormal half -> float —
/// matching <c>QuantizationKernels.HalfToFloat_Scalar</c> for every value a real weight can hold.
/// </summary>
internal static class QuantMatmulKernels
{
    public const int QK = 32;
    public const int QK_K = 256;

    /// <summary>K-quant variant alias → the base GGML block it shares its byte layout with.</summary>
    private static QuantDType BaseK(QuantDType q) => q switch
    {
        QuantDType.Q2_K_S => QuantDType.Q2_K,
        QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L => QuantDType.Q3_K,
        QuantDType.Q4_K_S or QuantDType.Q4_K_M => QuantDType.Q4_K,
        QuantDType.Q5_K_S or QuantDType.Q5_K_M => QuantDType.Q5_K,
        QuantDType.Q6_K_S => QuantDType.Q6_K,
        _ => q,
    };

    /// <summary>True for the K-quant family (super-blocks of 256) this engine runs on the device,
    /// including the legacy <c>_S/_M/_L</c> aliases. Q8_K is deliberately absent: its block holds an
    /// F32 scale, which ILGPU device code cannot reinterpret without an FFI, so it has no kernel
    /// and the host's <see cref="GpuInferenceEngine.SupportedDtypes"/> gate refuses it up front.</summary>
    internal static bool IsKQuant(QuantDType q) => BaseK(q) is QuantDType.Q2_K or QuantDType.Q3_K
        or QuantDType.Q4_K or QuantDType.Q5_K or QuantDType.Q6_K;

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

    /// <summary>Block size for host-side guards of the K-quant super-block formats (throwing).</summary>
    internal static int BlockBytesK(QuantDType q)
    {
        switch (BaseK(q))
        {
            case QuantDType.Q2_K: return 84;
            case QuantDType.Q3_K: return 110;
            case QuantDType.Q4_K: return 144;
            case QuantDType.Q5_K: return 176;
            case QuantDType.Q6_K: return 210;
        }
        throw new NotSupportedException($"GPU inference supports only Q2_K/Q3_K/Q4_K/Q5_K/Q6_K K-quantized weights, got {q}.");
    }

    /// <summary>Throw-free <see cref="BlockBytesK"/> for device kernels (ILGPU rejects throw).</summary>
    private static int BlockSizeK(QuantDType q)
    {
        q = BaseK(q);
        if (q == QuantDType.Q2_K) return 84;
        if (q == QuantDType.Q3_K) return 110;
        if (q == QuantDType.Q4_K) return 144;
        if (q == QuantDType.Q5_K) return 176;
        return 210;   // Q6_K
    }

    /// <summary>The Q4_K/Q5_K six-bit scale and min of sub-block <paramref name="j"/> (0..7), packed
    /// across the 12 byte fields at scales + 0. Mirrors <c>GetScaleMinK4_*_Scalar</c>.</summary>
    private static int KScale(ArrayView<byte> w, long block, int j)
    {
        if (j < 4) return w[block + 4 + j] & 0x3F;
        return (w[block + 4 + j + 4] & 0x0F) | ((w[block + 4 + j - 4] >> 6) << 4);
    }

    private static int KMin(ArrayView<byte> w, long block, int j)
    {
        if (j < 4) return w[block + 4 + j + 4] & 0x3F;
        return (w[block + 4 + j + 4] >> 4) | ((w[block + 4 + j] >> 6) << 4);
    }

    /// <summary>The 16 Q3_K signed scale bytes and the 32-byte sign table of a super-block. The
    /// 12 scale bytes at +96 hold the 16 six-bit scales interleaved (llama's <c>get_scale_min_k8</c>);
    /// the byte-level unwinding below is algebraically identical to the CPU scalar's uint32 shuffle
    /// (<see cref="QuantizationKernels.VecDotQ3K_Scalar"/>) but reads only what one group needs.</summary>
    private static int K3Scale(ArrayView<byte> w, long block, int g)
    {
        int high, low;
        if (g < 4) { low = w[block + 96 + g] & 0x0F; high = w[block + 104 + g] & 0x03; }
        else if (g < 8) { low = w[block + 100 + g - 4] & 0x0F; high = (w[block + 104 + g - 4] >> 2) & 0x03; }
        else if (g < 12) { low = (w[block + 96 + g - 8] >> 4) & 0x0F; high = (w[block + 104 + g - 8] >> 4) & 0x03; }
        else { low = (w[block + 100 + g - 12] >> 4) & 0x0F; high = (w[block + 104 + g - 12] >> 6) & 0x03; }
        int sc8 = low | (high << 4);
        return (sc8 & 0x80) != 0 ? sc8 - 256 : sc8;     // signed byte
    }

    /// <summary>Dequantises K-quant element <paramref name="pos"/> (0..255) of the super-block at
    /// <paramref name="block"/>. One scalar reads exactly what the matching <c>VecDot*K_Scalar</c>
    /// computes for that element — scale batches read once per block by the CPU, per element here,
    /// which is byte-for-byte the same float arithmetic (products are F32, values int→float).</summary>
    private static float KValue(QuantDType q, ArrayView<byte> w, long block, int pos)
    {
        q = BaseK(q);
        if (q == QuantDType.Q2_K)
        {
            float dS = ReadHalf(w, block + 80);
            float mS = ReadHalf(w, block + 82);
            int g = pos >> 4;                                          // scales[16] pairs at block start
            int s0 = w[block + g] & 0x0F;
            int m0 = w[block + g] >> 4;
            int qsByte = (pos >> 7) * 32 + (pos & 31);
            int v = (w[block + 16 + qsByte] >> (((pos & 127) >> 5) << 1)) & 3;
            return (float)s0 * v * dS - (float)m0 * mS;
        }
        if (q == QuantDType.Q3_K)
        {
            float dAll = ReadHalf(w, block + 108);
            int g = pos >> 4;
            int sc = K3Scale(w, block, g) - 32;
            int qsByte = (pos >> 7) * 32 + (pos & 31);
            int s2 = (w[block + 32 + qsByte] >> (((pos & 127) >> 5) << 1)) & 3;
            int hBit = (w[block + (pos & 31)] >> (pos >> 5)) & 1;      // hmask[32] at block start
            int actual = s2 - (hBit == 0 ? 4 : 0);
            return dAll * sc * actual;
        }
        if (q == QuantDType.Q4_K)
        {
            float dS = ReadHalf(w, block);
            float mS = ReadHalf(w, block + 2);
            int j = pos >> 5;
            int s = KScale(w, block, j);
            int m = KMin(w, block, j);
            int qsByte = (pos >> 6) * 32 + (pos & 31);
            int v = (w[block + 16 + qsByte] >> (((pos & 63) >> 5) << 2)) & 0x0F;
            return (float)s * v * dS - (float)m * mS;
        }
        if (q == QuantDType.Q5_K)
        {
            float d = ReadHalf(w, block);
            float min = ReadHalf(w, block + 2);
            int j = pos >> 5;
            int sc = KScale(w, block, j);
            int mn = KMin(w, block, j);
            int idx32 = pos & 31, group64 = pos >> 6, half = (pos & 63) >> 5;
            int bitPos = group64 * 2 + half;
            int hAdd = (w[block + 16 + idx32] & (1 << bitPos)) != 0 ? 16 : 0;
            int q5 = half == 0
                ? (w[block + 48 + group64 * 32 + idx32] & 0x0F)
                : (w[block + 48 + group64 * 32 + idx32] >> 4);
            q5 |= hAdd;
            return (float)sc * q5 * d - (float)mn * min;
        }
        // Q6_K
        float dAll6 = ReadHalf(w, block + 208);
        int nOff = (pos >> 7) << 7;                                    // 0 or 128
        int l = pos & 31;
        int col = (pos - nOff) >> 5;                                   // 0..3
        int qlOff = nOff == 0 ? 0 : 64;
        int qhOff = nOff == 0 ? 0 : 32;
        int qlIdx = (col & 1) == 0 ? l : l + 32;
        int nib = (w[block + qlOff + qlIdx] >> ((col & 2) != 0 ? 4 : 0)) & 0x0F;
        int hi = (w[block + 128 + qhOff + l] >> (col << 1)) & 0x03;
        int qq = nib | (hi << 4);
        int scaleOff = 192 + (nOff == 0 ? 0 : 8) + (l >> 4) + (col << 1);
        int sc8 = ToSByte(w[block + scaleOff]);
        return dAll6 * sc8 * (qq - 32);
    }

    /// <summary>
    /// y[i,o] = Σ_k x[i,k]·wQK[o,k] for a [N, K] K-quant matrix. The K-quant raw buffer is one
    /// stream of 256-element super-blocks over the flattened [N, K] tensor (QK_K ≠ the small-quant
    /// 32), so column <paramref name="o"/> spans super-blocks starting at <c>o·K / QK_K</c>, and its
    /// elements are offset by <c>o·K % QK_K</c> within the first block. The per-block iteration below
    /// reproduces exactly the <c>startBlock</c>/<c>colBlockStart</c> addressing and accumulation order
    /// of the matching CPU <c>VecDot*K_Scalar</c> — including Q6_K's four-at-a-time add order and its
    /// partial-tail break — so the device result is bit-for-bit the oracle on identical raw bytes.
    /// Used for every K-quant block linear and the weight-tied LM head (raw = embedding, K = Hidden).
    /// </summary>
    public static void DequantMatmulK(Index1D idx, ArrayView<float> y, ArrayView<float> x,
        ArrayView<byte> w, int K, int N, int nBlocks, QuantDType q)
    {
        int i = idx / N, o = idx - i * N;
        int startBlock = (o * K) >> 8;
        int colBlockStart = o * K & 255;
        int inBase = i * K;
        double acc = 0;
        if (q == QuantDType.Q6_K || BaseK(q) == QuantDType.Q6_K)
        {
            for (int b = 0; b < nBlocks; b++)
            {
                long block = (startBlock + (long)b) * BlockSizeK(q);
                int curStart = b == 0 ? colBlockStart : 0;
                int blockEnd = K + colBlockStart - (b << 8);
                if (blockEnd > QK_K) blockEnd = QK_K;
                if (blockEnd <= curStart) continue;
                int bound = (b << 8) + blockEnd - colBlockStart;
                for (int nOff = curStart; nOff < blockEnd; nOff += 128)
                {
                    int halfRem = blockEnd - nOff;
                    if (halfRem > 128) halfRem = 128;
                    for (int l = 0; l < 32 && l < halfRem; l++)
                    {
                        int e1 = (b << 8) + nOff + l - colBlockStart;
                        if (e1 + 32 >= bound)
                        {
                            acc += (double)x[inBase + e1] * KValue(q, w, block, nOff + l);
                            break;
                        }
                        acc += (double)x[inBase + e1] * KValue(q, w, block, nOff + l);
                        acc += (double)x[inBase + e1 + 32] * KValue(q, w, block, nOff + l + 32);
                        acc += (double)x[inBase + e1 + 64] * KValue(q, w, block, nOff + l + 64);
                        acc += (double)x[inBase + e1 + 96] * KValue(q, w, block, nOff + l + 96);
                    }
                }
            }
        }
        else
        {
            for (int b = 0; b < nBlocks; b++)
            {
                long block = (startBlock + (long)b) * BlockSizeK(q);
                int curStart = b == 0 ? colBlockStart : 0;
                int blockEnd = K + colBlockStart - (b << 8);
                if (blockEnd > QK_K) blockEnd = QK_K;
                if (blockEnd <= curStart) continue;
                int inOfs = b << 8;
                for (int pos = curStart; pos < blockEnd; pos++)
                    acc += (double)x[inBase + inOfs + pos - colBlockStart] * KValue(q, w, block, pos);
            }
        }
        y[idx] = (float)acc;
    }

    /// <summary>
    /// x[i,d] = dequant(embedding[ids[i]][d]) from a [V, K] K-quant embedding table: the row for token
    /// <c>ids[i]</c> occupies the flattened stream positions <c>[ids[i]·K, (ids[i]+1)·K)</c>, the same
    /// column the LM head matmul reads. One thread per (row, feature).
    /// </summary>
    public static void DequantGatherK(Index1D idx, ArrayView<float> x, ArrayView<byte> table,
        ArrayView<int> ids, int cols, QuantDType q)
    {
        int t = idx / cols, d = idx - t * cols;
        long g = (long)ids[t] * cols + d;
        long block = (g >> 8) * BlockSizeK(q);
        int pos = (int)(g & 255);
        x[idx] = KValue(q, table, block, pos);
    }
}