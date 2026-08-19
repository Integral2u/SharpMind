using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{

    // ── SIMD unpack helpers ─────────────────────────────────────────────────
    //
    // These four were introduced alongside the blocked prefill path in the same
    // series of work. Upstream reimplemented that sibling change under different
    // names, so they are carried here to keep this commit self-contained: the
    // blocked dequant callbacks below are their only consumer.

    private static unsafe Vector256<float> Q5_0Codes8(byte* qs, int g, Vector256<uint> qhv, Vector256<uint> bitOfLane)
    {
        var nib = Avx2.ConvertToVector256Int32(qs + (g & 1) * 8);
        nib = g < 2 ? Avx2.And(nib, Vector256.Create(0x0F)) : Avx2.ShiftRightLogical(nib, 4);
        var hi = Avx2.And(Avx2.ShiftRightLogicalVariable(qhv, bitOfLane), Vector256.Create(1u));
        return Avx.ConvertToVector256Single(Avx2.Or(nib, Avx2.ShiftLeftLogical(hi, 4).AsInt32()));
    }


    private static readonly Vector256<uint> Q5Bits0 = Vector256.Create(0u, 1, 2, 3, 4, 5, 6, 7);
    private static readonly Vector256<uint> Q5Bits1 = Vector256.Create(8u, 9, 10, 11, 12, 13, 14, 15);
    private static readonly Vector256<uint> Q5Bits2 = Vector256.Create(16u, 17, 18, 19, 20, 21, 22, 23);
    private static readonly Vector256<uint> Q5Bits3 = Vector256.Create(24u, 25, 26, 27, 28, 29, 30, 31);

    private static unsafe Vector256<float> Q4KNibbles8(byte* qs, int basePos, int g)
    {
        var v = Avx2.ConvertToVector256Int32(qs + (basePos >> 6) * 32 + g * 8);
        v = (basePos & 32) == 0
            ? Avx2.And(v, Vector256.Create(0x0F))
            : Avx2.ShiftRightLogical(v, 4);   // bytes are < 256, so this is the high nibble
        return Avx.ConvertToVector256Single(v);
    }

    private static unsafe void Q6KCodes8(byte* pql, byte* pqh, int l,
        out Vector256<float> q1, out Vector256<float> q2, out Vector256<float> q3, out Vector256<float> q4)
    {
        var lo = Avx2.ConvertToVector256Int32(pql + l);
        var lo2 = Avx2.ConvertToVector256Int32(pql + l + 32);
        var h = Avx2.ConvertToVector256Int32(pqh + l);
        var m0F = Vector256.Create(0x0F);
        var m03 = Vector256.Create(0x03);

        // Bytes are < 256, so a plain right shift by 4 or 6 already isolates the
        // high field; only the middle fields need masking.
        q1 = Avx.ConvertToVector256Single(Avx2.Or(Avx2.And(lo, m0F), Avx2.ShiftLeftLogical(Avx2.And(h, m03), 4)));
        q2 = Avx.ConvertToVector256Single(Avx2.Or(Avx2.And(lo2, m0F), Avx2.ShiftLeftLogical(Avx2.And(Avx2.ShiftRightLogical(h, 2), m03), 4)));
        q3 = Avx.ConvertToVector256Single(Avx2.Or(Avx2.ShiftRightLogical(lo, 4), Avx2.ShiftLeftLogical(Avx2.And(Avx2.ShiftRightLogical(h, 4), m03), 4)));
        q4 = Avx.ConvertToVector256Single(Avx2.Or(Avx2.ShiftRightLogical(lo2, 4), Avx2.ShiftLeftLogical(Avx2.ShiftRightLogical(h, 6), 4)));
    }

    /// <summary>Dequantizes one weight column's K values into <c>dst[0..K)</c>.</summary>
    private unsafe delegate void DequantColumnFn(byte* rawWeights, int col, int K, float* dst);

    // ── Blocked M>1 driver for quantized formats ─────────────────────────────
    //
    // The per-row M>1 path ran one VecDot per (row, column), so every weight
    // column was UNPACKED ONCE PER ROW: a 128-token prefill chunk decoded every
    // Q4_K/Q6_K/Q5_0/Q8_0 weight 128 times to use it 128 times. Here each column
    // is dequantized exactly once into an F32 scratch tile and then consumed by
    // the same four-rows-per-weight-vector microkernel the F16/F32 paths use.
    //
    // Loop order is weight-tile-first: the outer loop walks tiles of columns
    // sized to ~256 KB of dequantized scratch (L2-resident), and ALL input rows
    // stream through each resident tile. The alternative (input-tile-first, as
    // F16BlockedColumns does) would re-dequantize the whole weight matrix once
    // per 16-row tile — for quantized weights the unpack is the expensive part,
    // so the tile that must stay resident is the weights, not the input.
    // (dotLLM measured the same ordering fastest for its int8 GEMM; its L2
    // budget heuristic — half a typical 512 KB L2 — is borrowed here.)
    //
    // Decode (M <= 1) is untouched: it keeps the column-parallel DecodeParallel
    // path, where each weight is used exactly once and a scratch would only add
    // traffic.

    private const int QuantScratchBudgetBytes = 256 * 1024;

    private static int QuantTileCols(int K, int maxCols)
    {
        int cols = QuantScratchBudgetBytes / (K * sizeof(float));
        cols &= ~3;                                   // multiples of 4 keep tails rare
        return Math.Clamp(cols, 4, Math.Max(4, maxCols));
    }

    private static unsafe void QuantBlockedColumns(
        DequantColumnFn dequant,
        float* input, byte* rawWeights, float* output,
        int M, int K, int N, int colStart, int colEnd)
    {
        int tileCols = QuantTileCols(K, colEnd - colStart);
        float* scratch = (float*)NativeMemory.AlignedAlloc((nuint)((long)tileCols * K * sizeof(float)), 64);
        try
        {
            for (int ct = colStart; ct < colEnd; ct += tileCols)
            {
                int tc = Math.Min(tileCols, colEnd - ct);
                for (int c = 0; c < tc; c++)
                    dequant(rawWeights, ct + c, K, scratch + (long)c * K);

                // Identical microkernel to F32BlockedColumns, reading the tile.
                for (int rowTile = 0; rowTile < M; rowTile += F16RowTile)
                {
                    int tileEnd = Math.Min(rowTile + F16RowTile, M);
                    for (int c = 0; c < tc; c++)
                    {
                        float* pW = scratch + (long)c * K;
                        int col = ct + c;
                        int r = rowTile;
                        for (; r + F16RowBlock <= tileEnd; r += F16RowBlock)
                        {
                            var a0 = Vector256<float>.Zero;
                            var a1 = Vector256<float>.Zero;
                            var a2 = Vector256<float>.Zero;
                            var a3 = Vector256<float>.Zero;
                            float* i0 = input + (long)(r + 0) * K;
                            float* i1 = input + (long)(r + 1) * K;
                            float* i2 = input + (long)(r + 2) * K;
                            float* i3 = input + (long)(r + 3) * K;

                            int k = 0;
                            for (; k <= K - 8; k += 8)
                            {
                                var vw = Vector256.LoadUnsafe(ref pW[k]);
                                a0 = Fma.MultiplyAdd(vw, Vector256.LoadUnsafe(ref i0[k]), a0);
                                a1 = Fma.MultiplyAdd(vw, Vector256.LoadUnsafe(ref i1[k]), a1);
                                a2 = Fma.MultiplyAdd(vw, Vector256.LoadUnsafe(ref i2[k]), a2);
                                a3 = Fma.MultiplyAdd(vw, Vector256.LoadUnsafe(ref i3[k]), a3);
                            }

                            float s0 = MathHelpers.HSum256_Avx(a0);
                            float s1 = MathHelpers.HSum256_Avx(a1);
                            float s2 = MathHelpers.HSum256_Avx(a2);
                            float s3 = MathHelpers.HSum256_Avx(a3);
                            for (; k < K; k++)
                            {
                                float wf = pW[k];
                                s0 += i0[k] * wf;
                                s1 += i1[k] * wf;
                                s2 += i2[k] * wf;
                                s3 += i3[k] * wf;
                            }

                            output[(long)(r + 0) * N + col] = s0;
                            output[(long)(r + 1) * N + col] = s1;
                            output[(long)(r + 2) * N + col] = s2;
                            output[(long)(r + 3) * N + col] = s3;
                        }

                        for (; r < tileEnd; r++)
                            output[(long)r * N + col] = VecDotF32_FMA(input + (long)r * K, (byte*)pW, 0, K);
                    }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(scratch);
        }
    }

    private static unsafe void QuantBlockedMatMul(
        DequantColumnFn dequant,
        float* input, byte* rawWeights, float* output,
        int M, int K, int N, bool parallel)
    {
        if (!parallel)
        {
            QuantBlockedColumns(dequant, input, rawWeights, output, M, K, N, 0, N);
            return;
        }

        // Same column split as the F16/F32 parallel paths: contiguous spans in
        // 16-column quanta so two threads never share an output cache line.
        int target = Math.Max(1, N / Environment.ProcessorCount);
        int chunkSize = (target + 15) & ~15;
        int numChunks = (N + chunkSize - 1) / chunkSize;

        long inputAddr = (long)input, weightsAddr = (long)rawWeights, outputAddr = (long)output;
        Parallel.For(0, numChunks, chunkIdx =>
        {
            int colStart = chunkIdx * chunkSize;
            int colEnd = Math.Min(colStart + chunkSize, N);
            QuantBlockedColumns(dequant, (float*)inputAddr, (byte*)weightsAddr, (float*)outputAddr,
                M, K, N, colStart, colEnd);
        });
    }

    // ── Per-format column dequantizers ───────────────────────────────────────
    // Each mirrors its VecDot*_FMA unpack exactly (same helpers, same aligned
    // fast path), storing the dequantized weight instead of multiplying by the
    // input. Unaligned starts and partial blocks use the per-element scalar
    // form from the corresponding VecDot*_Scalar.

    private static unsafe void DequantColumnQ8_0(byte* rawWeights, int col, int K, float* dst)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (K + QK - 1) / QK;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, K - b * QK);
            float* pOut = dst + b * QK;

            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
                Vector256.StoreUnsafe(
                    Avx.Multiply(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i)), vd),
                    ref pOut[i]);
            for (; i < blockEnd; i++)
                pOut[i] = values[i] * d;
        }
    }

    private static unsafe void DequantColumnQ5_0(byte* rawWeights, int col, int K, float* dst)
    {
        const int BLOCK_BYTES = 22;
        const int QK5 = 32;
        int nBlocks = (K + QK5 - 1) / QK5;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            uint qh = *(uint*)(block + 2);
            byte* qs = block + 6;
            int blockEnd = Math.Min(QK5, K - b * QK5);
            float* pOut = dst + b * QK5;
            int half = QK5 / 2;

            int i = 0;
            if (blockEnd == QK5)
            {
                var vd = Vector256.Create(d);
                var v16d = Vector256.Create(16 * d);
                var qhv = Vector256.Create(qh);
                Vector256.StoreUnsafe(Fma.MultiplySubtract(Q5_0Codes8(qs, 0, qhv, Q5Bits0), vd, v16d), ref pOut[0]);
                Vector256.StoreUnsafe(Fma.MultiplySubtract(Q5_0Codes8(qs, 1, qhv, Q5Bits1), vd, v16d), ref pOut[8]);
                Vector256.StoreUnsafe(Fma.MultiplySubtract(Q5_0Codes8(qs, 2, qhv, Q5Bits2), vd, v16d), ref pOut[16]);
                Vector256.StoreUnsafe(Fma.MultiplySubtract(Q5_0Codes8(qs, 3, qhv, Q5Bits3), vd, v16d), ref pOut[24]);
                i = QK5;
            }
            for (; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                pOut[i] = ((nib | h4) - 16) * d;
            }
        }
    }

    private static unsafe void DequantColumnQ4K(byte* rawWeights, int col, int K, float* dst)
    {
        const int BLOCK_BYTES = 144;
        const int QK_K = 256;
        int startBlock = (col * K) / QK_K;
        int colBlockStart = col * K % QK_K;
        // A column starting mid-super-block can span one more block than
        // ceil(K/QK_K). (The VecDot kernels use the shorter count and silently
        // skip the tail for such shapes — unreachable in real GGUFs, where
        // K-quant rows are multiples of 256, so columns start at offset 0 or 128
        // and the counts coincide.)
        int nBlocks = (K + colBlockStart + QK_K - 1) / QK_K;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 0));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, K + colBlockStart - b * QK_K);
            float* pOut = dst + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScaleMinK4_Scale_Scalar(isc, scales);
                    float m = GetScaleMinK4_Min_Scalar(isc, scales);

                    int subRem = Math.Min(32, blockEnd - basePos);
                    int l = 0;
                    if ((basePos & 31) == 0)
                    {
                        var vs = Vector256.Create(s * dSuper);
                        var vm = Vector256.Create(m * minSuper);
                        for (; l <= subRem - 8; l += 8)
                            Vector256.StoreUnsafe(
                                Fma.MultiplySubtract(Q4KNibbles8(qs, basePos, l >> 3), vs, vm),
                                ref pOut[basePos + l]);
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        pOut[idx] = s * v * dSuper - m * minSuper;
                    }
                }
            }
        }
    }

    private static unsafe void DequantColumnQ6K(byte* rawWeights, int col, int K, float* dst)
    {
        const int BLOCK_BYTES = 210;
        const int QK_K = 256;
        int startBlock = (col * K) / QK_K;
        int colBlockStart = col * K % QK_K;
        // A column starting mid-super-block can span one more block than
        // ceil(K/QK_K). (The VecDot kernels use the shorter count and silently
        // skip the tail for such shapes — unreachable in real GGUFs, where
        // K-quant rows are multiples of 256, so columns start at offset 0 or 128
        // and the counts coincide.)
        int nBlocks = (K + colBlockStart + QK_K - 1) / QK_K;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d = HalfToFloat_F16C(*(ushort*)(block + 208));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, K + colBlockStart - b * QK_K);
            float* pOut = dst + b * QK_K - colBlockStart;

            // Vector path only for a fully aligned, complete 128-half; everything
            // else per element (mid-block column starts, partial final blocks).
            for (int nOff = curBlockStart & ~127; nOff < blockEnd; nOff += 128)
            {
                byte* pql = ql + (nOff == 0 ? 0 : 64);
                byte* pqh = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                if (nOff >= curBlockStart && nOff + 128 <= blockEnd)
                {
                    for (int l = 0; l < 32; l += 8)
                    {
                        int is_ = l / 16;
                        Q6KCodes8(pql, pqh, l, out var q1, out var q2, out var q3, out var q4);
                        float s1 = d * psc[is_ + 0], s2 = d * psc[is_ + 2], s3 = d * psc[is_ + 4], s4 = d * psc[is_ + 6];
                        Vector256.StoreUnsafe(Fma.MultiplySubtract(q1, Vector256.Create(s1), Vector256.Create(32 * s1)), ref pOut[nOff + l]);
                        Vector256.StoreUnsafe(Fma.MultiplySubtract(q2, Vector256.Create(s2), Vector256.Create(32 * s2)), ref pOut[nOff + l + 32]);
                        Vector256.StoreUnsafe(Fma.MultiplySubtract(q3, Vector256.Create(s3), Vector256.Create(32 * s3)), ref pOut[nOff + l + 64]);
                        Vector256.StoreUnsafe(Fma.MultiplySubtract(q4, Vector256.Create(s4), Vector256.Create(32 * s4)), ref pOut[nOff + l + 96]);
                    }
                    continue;
                }

                int from = Math.Max(nOff, curBlockStart);
                int to = Math.Min(nOff + 128, blockEnd);
                for (int idx = from; idx < to; idx++)
                {
                    int r = idx - nOff;      // 0..127 within the half
                    int g = r / 32;
                    int l = r % 32;
                    int q = g switch
                    {
                        0 => (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4),
                        1 => (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4),
                        2 => ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4),
                        _ => ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4),
                    };
                    pOut[idx] = d * scales[(nOff == 0 ? 0 : 8) + g * 2 + l / 16] * (q - 32);
                }
            }
        }
    }
}
