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
                    int cPair = 0;

                    // Three columns at a time. The 4x1 shape this replaced did four FMAs per five
                    // loads, leaving the FMA units waiting on the load ports; a 4x3 tile does twelve
                    // per seven. Twelve accumulators plus three weight vectors is 15 of the 16 YMM
                    // registers, with the input side reachable as FMA memory operands.
                    //
                    // 4x3 is the measured optimum here, not a guess: 4x4 (sixteen accumulators) was
                    // built and measured level with 4x3, i.e. the spill cancels the extra work.
                    // llama.cpp's tinyBLAS independently picks 4x3 for 16-register targets and only
                    // widens to 4x6 when 32 vector registers are available (sgemm.cpp, the
                    // VECTOR_REGISTERS == 32 branch) - which .NET does not give us here.
                    for (; cPair + 3 <= tc; cPair += 3)
                    {
                        float* pW0 = scratch + (long)cPair * K;
                        float* pW1 = scratch + (long)(cPair + 1) * K;
                        float* pW2 = scratch + (long)(cPair + 2) * K;
                        int col0 = ct + cPair, col1 = col0 + 1, col2 = col0 + 2;
                        int r3 = rowTile;
                        for (; r3 + F16RowBlock <= tileEnd; r3 += F16RowBlock)
                        {
                            var x00 = Vector256<float>.Zero; var x01 = Vector256<float>.Zero; var x02 = Vector256<float>.Zero;
                            var x10 = Vector256<float>.Zero; var x11 = Vector256<float>.Zero; var x12 = Vector256<float>.Zero;
                            var x20 = Vector256<float>.Zero; var x21 = Vector256<float>.Zero; var x22 = Vector256<float>.Zero;
                            var x30 = Vector256<float>.Zero; var x31 = Vector256<float>.Zero; var x32 = Vector256<float>.Zero;
                            float* q0 = input + (long)(r3 + 0) * K;
                            float* q1 = input + (long)(r3 + 1) * K;
                            float* q2 = input + (long)(r3 + 2) * K;
                            float* q3 = input + (long)(r3 + 3) * K;

                            int k3 = 0;
                            for (; k3 <= K - 8; k3 += 8)
                            {
                                var w0 = Vector256.LoadUnsafe(ref pW0[k3]);
                                var w1 = Vector256.LoadUnsafe(ref pW1[k3]);
                                var w2 = Vector256.LoadUnsafe(ref pW2[k3]);
                                var u0 = Vector256.LoadUnsafe(ref q0[k3]);
                                x00 = Fma.MultiplyAdd(w0, u0, x00); x01 = Fma.MultiplyAdd(w1, u0, x01); x02 = Fma.MultiplyAdd(w2, u0, x02);
                                var u1 = Vector256.LoadUnsafe(ref q1[k3]);
                                x10 = Fma.MultiplyAdd(w0, u1, x10); x11 = Fma.MultiplyAdd(w1, u1, x11); x12 = Fma.MultiplyAdd(w2, u1, x12);
                                var u2 = Vector256.LoadUnsafe(ref q2[k3]);
                                x20 = Fma.MultiplyAdd(w0, u2, x20); x21 = Fma.MultiplyAdd(w1, u2, x21); x22 = Fma.MultiplyAdd(w2, u2, x22);
                                var u3 = Vector256.LoadUnsafe(ref q3[k3]);
                                x30 = Fma.MultiplyAdd(w0, u3, x30); x31 = Fma.MultiplyAdd(w1, u3, x31); x32 = Fma.MultiplyAdd(w2, u3, x32);
                            }

                            float* op0 = output + (long)(r3 + 0) * N; float* op1 = output + (long)(r3 + 1) * N;
                            float* op2 = output + (long)(r3 + 2) * N; float* op3 = output + (long)(r3 + 3) * N;
                            float s00 = MathHelpers.HSum256_Avx(x00), s01 = MathHelpers.HSum256_Avx(x01), s02 = MathHelpers.HSum256_Avx(x02);
                            float s10 = MathHelpers.HSum256_Avx(x10), s11 = MathHelpers.HSum256_Avx(x11), s12 = MathHelpers.HSum256_Avx(x12);
                            float s20 = MathHelpers.HSum256_Avx(x20), s21 = MathHelpers.HSum256_Avx(x21), s22 = MathHelpers.HSum256_Avx(x22);
                            float s30 = MathHelpers.HSum256_Avx(x30), s31 = MathHelpers.HSum256_Avx(x31), s32 = MathHelpers.HSum256_Avx(x32);
                            for (; k3 < K; k3++)
                            {
                                float f0 = pW0[k3], f1 = pW1[k3], f2 = pW2[k3];
                                s00 += q0[k3] * f0; s01 += q0[k3] * f1; s02 += q0[k3] * f2;
                                s10 += q1[k3] * f0; s11 += q1[k3] * f1; s12 += q1[k3] * f2;
                                s20 += q2[k3] * f0; s21 += q2[k3] * f1; s22 += q2[k3] * f2;
                                s30 += q3[k3] * f0; s31 += q3[k3] * f1; s32 += q3[k3] * f2;
                            }
                            op0[col0] = s00; op0[col1] = s01; op0[col2] = s02;
                            op1[col0] = s10; op1[col1] = s11; op1[col2] = s12;
                            op2[col0] = s20; op2[col1] = s21; op2[col2] = s22;
                            op3[col0] = s30; op3[col1] = s31; op3[col2] = s32;
                        }
                        for (; r3 < tileEnd; r3++)
                        {
                            output[(long)r3 * N + col0] = VecDotF32_FMA(input + (long)r3 * K, (byte*)pW0, 0, K);
                            output[(long)r3 * N + col1] = VecDotF32_FMA(input + (long)r3 * K, (byte*)pW1, 0, K);
                            output[(long)r3 * N + col2] = VecDotF32_FMA(input + (long)r3 * K, (byte*)pW2, 0, K);
                        }
                    }

                    // Two- and one-column remainders for a tile whose width is not a multiple
                    // of three.
                    for (; cPair + 2 <= tc; cPair += 2)
                    {
                        float* pWa = scratch + (long)cPair * K;
                        float* pWb = scratch + (long)(cPair + 1) * K;
                        int colA = ct + cPair, colB = ct + cPair + 1;
                        int rr = rowTile;
                        for (; rr + F16RowBlock <= tileEnd; rr += F16RowBlock)
                        {
                            var a0 = Vector256<float>.Zero; var b0 = Vector256<float>.Zero;
                            var a1 = Vector256<float>.Zero; var b1 = Vector256<float>.Zero;
                            var a2 = Vector256<float>.Zero; var b2 = Vector256<float>.Zero;
                            var a3 = Vector256<float>.Zero; var b3 = Vector256<float>.Zero;
                            float* j0 = input + (long)(rr + 0) * K;
                            float* j1 = input + (long)(rr + 1) * K;
                            float* j2 = input + (long)(rr + 2) * K;
                            float* j3 = input + (long)(rr + 3) * K;

                            int kk = 0;
                            for (; kk <= K - 8; kk += 8)
                            {
                                var wa = Vector256.LoadUnsafe(ref pWa[kk]);
                                var wb = Vector256.LoadUnsafe(ref pWb[kk]);
                                var v0 = Vector256.LoadUnsafe(ref j0[kk]);
                                var v1 = Vector256.LoadUnsafe(ref j1[kk]);
                                var v2 = Vector256.LoadUnsafe(ref j2[kk]);
                                var v3 = Vector256.LoadUnsafe(ref j3[kk]);
                                a0 = Fma.MultiplyAdd(wa, v0, a0); b0 = Fma.MultiplyAdd(wb, v0, b0);
                                a1 = Fma.MultiplyAdd(wa, v1, a1); b1 = Fma.MultiplyAdd(wb, v1, b1);
                                a2 = Fma.MultiplyAdd(wa, v2, a2); b2 = Fma.MultiplyAdd(wb, v2, b2);
                                a3 = Fma.MultiplyAdd(wa, v3, a3); b3 = Fma.MultiplyAdd(wb, v3, b3);
                            }

                            float ta0 = MathHelpers.HSum256_Avx(a0), tb0 = MathHelpers.HSum256_Avx(b0);
                            float ta1 = MathHelpers.HSum256_Avx(a1), tb1 = MathHelpers.HSum256_Avx(b1);
                            float ta2 = MathHelpers.HSum256_Avx(a2), tb2 = MathHelpers.HSum256_Avx(b2);
                            float ta3 = MathHelpers.HSum256_Avx(a3), tb3 = MathHelpers.HSum256_Avx(b3);
                            for (; kk < K; kk++)
                            {
                                float wfa = pWa[kk], wfb = pWb[kk];
                                ta0 += j0[kk] * wfa; tb0 += j0[kk] * wfb;
                                ta1 += j1[kk] * wfa; tb1 += j1[kk] * wfb;
                                ta2 += j2[kk] * wfa; tb2 += j2[kk] * wfb;
                                ta3 += j3[kk] * wfa; tb3 += j3[kk] * wfb;
                            }

                            output[(long)(rr + 0) * N + colA] = ta0; output[(long)(rr + 0) * N + colB] = tb0;
                            output[(long)(rr + 1) * N + colA] = ta1; output[(long)(rr + 1) * N + colB] = tb1;
                            output[(long)(rr + 2) * N + colA] = ta2; output[(long)(rr + 2) * N + colB] = tb2;
                            output[(long)(rr + 3) * N + colA] = ta3; output[(long)(rr + 3) * N + colB] = tb3;
                        }
                        for (; rr < tileEnd; rr++)
                        {
                            output[(long)rr * N + colA] = VecDotF32_FMA(input + (long)rr * K, (byte*)pWa, 0, K);
                            output[(long)rr * N + colB] = VecDotF32_FMA(input + (long)rr * K, (byte*)pWb, 0, K);
                        }
                    }

                    for (int c = cPair; c < tc; c++)
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
