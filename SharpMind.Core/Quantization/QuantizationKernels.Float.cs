using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{ 

    public static unsafe void QuantizedMatMulF32_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        float* w = (float*)rawWeights;
        for (int row = 0; row < M; row++)
        {
            float* pIn = input + (long)row * K;
            float* pOut = output + (long)row * N;
            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                float* pW = w + (long)col * K;
                for (int i = 0; i < K; i++)
                    sum += pIn[i] * pW[i];
                pOut[col] = sum;
            }
        }
    }

    public static unsafe void QuantizedMatMulF32_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotF32_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            float* w = (float*)rawWeights;
            Parallel.For(0, M, row =>
            {
                float* pIn = input + (long)row * K;
                float* pOut = output + (long)row * N;
                for (int col = 0; col < N; col++)
                {
                    float sum = 0;
                    float* pW = w + (long)col * K;
                    for (int i = 0; i < K; i++)
                        sum += pIn[i] * pW[i];
                    pOut[col] = sum;
                }
            });
        }
    }

    public static unsafe void QuantizedMatMulF16_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        ushort* w = (ushort*)rawWeights;
        for (int row = 0; row < M; row++)
        {
            float* pIn = input + (long)row * K;
            float* pOut = output + (long)row * N;
            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                ushort* pW = w + (long)col * K;
                for (int i = 0; i < K; i++)
                    sum += pIn[i] * HalfToFloat_Scalar(pW[i]);
                pOut[col] = sum;
            }
        }
    }

    public static unsafe void QuantizedMatMulF16_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotF16_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            ushort* w = (ushort*)rawWeights;
            Parallel.For(0, M, row =>
            {
                float* pIn = input + (long)row * K;
                float* pOut = output + (long)row * N;
                for (int col = 0; col < N; col++)
                {
                    float sum = 0;
                    ushort* pW = w + (long)col * K;
                    for (int i = 0; i < K; i++)
                        sum += pIn[i] * HalfToFloat_Scalar(pW[i]);
                    pOut[col] = sum;
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF32_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        float* w = (float*)rawWeights;
        double sum = 0;
        float* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF16_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        ushort* w = (ushort*)rawWeights;
        double sum = 0;
        ushort* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * HalfToFloat_Scalar(pW[i]);
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF32_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        float* w = (float*)rawWeights;
        float* pW = w + (long)col * inFeatures;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 16; i += 16)
        {
            var vi0 = Vector256.LoadUnsafe(ref pW[i]);
            var vi1 = Vector256.LoadUnsafe(ref pW[i + 8]);
            var vw0 = Vector256.LoadUnsafe(ref input[i]);
            var vw1 = Vector256.LoadUnsafe(ref input[i + 8]);
            vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
            vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
        }
        for (; i <= inFeatures - 8; i += 8)
        {
            var vi = Vector256.LoadUnsafe(ref pW[i]);
            var vw = Vector256.LoadUnsafe(ref input[i]);
            vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
        }
        float sum = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        for (; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return sum;
    }

    /// <summary>
    /// Widens 8 consecutive IEEE-754 half bit patterns to a Vector256&lt;float&gt;.
    ///
    /// .NET exposes no F16C intrinsic (vcvtph2ps) and Half is not a supported
    /// Vector128&lt;T&gt; element type, so this does it arithmetically. Multiplying
    /// the shifted magnitude by 2^112 rescales normals and denormals in one step;
    /// the blend then forces the float exponent to all-ones for half exponent 31,
    /// which keeps Inf as Inf and NaN as NaN.
    ///
    /// Bit-exact against <c>(float)Half</c> for all 65536 patterns — see
    /// <c>HalfWideningTests</c>. The scalar-per-element alternative is not just
    /// slower arithmetic: writing 8 floats to a stack buffer and then vector-loading
    /// that same buffer cannot store-forward, so it stalls once per 8 weights.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Vector256<float> WidenHalf8(ushort* p)
    {
        Vector256<int> h = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p));
        Vector256<int> sign = Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x8000)), 16);
        Vector256<int> mag = Avx2.And(h, Vector256.Create(0x7FFF));
        Vector256<int> shifted = Avx2.ShiftLeftLogical(mag, 13);
        Vector256<float> scaled = Avx.Multiply(
            shifted.AsSingle(),
            Vector256.Create(BitConverter.Int32BitsToSingle(0x7780_0000))); // 2^112
        Vector256<int> isSpecial = Avx2.CompareGreaterThan(mag, Vector256.Create(0x7BFF));
        Vector256<float> special = Avx2.Or(shifted, Vector256.Create(0x7F80_0000)).AsSingle();
        return Avx.Or(Avx.BlendVariable(scaled, special, isSpecial.AsSingle()), sign.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF16_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        ushort* w = (ushort*)rawWeights;
        ushort* pW = w + (long)col * inFeatures;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        var vacc2 = Vector256<float>.Zero;
        var vacc3 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 32; i += 32)
        {
            vacc0 = Fma.MultiplyAdd(WidenHalf8(pW + i), Vector256.LoadUnsafe(ref input[i]), vacc0);
            vacc1 = Fma.MultiplyAdd(WidenHalf8(pW + i + 8), Vector256.LoadUnsafe(ref input[i + 8]), vacc1);
            vacc2 = Fma.MultiplyAdd(WidenHalf8(pW + i + 16), Vector256.LoadUnsafe(ref input[i + 16]), vacc2);
            vacc3 = Fma.MultiplyAdd(WidenHalf8(pW + i + 24), Vector256.LoadUnsafe(ref input[i + 24]), vacc3);
        }
        for (; i <= inFeatures - 8; i += 8)
            vacc0 = Fma.MultiplyAdd(WidenHalf8(pW + i), Vector256.LoadUnsafe(ref input[i]), vacc0);
        float sum = MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc0, vacc1), Avx.Add(vacc2, vacc3)));
        for (; i < inFeatures; i++)
            sum += input[i] * HalfToFloat_F16C(pW[i]);
        return sum;
    }

    public static unsafe void QuantizedMatMulF32_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            QuantizedMatMul_Serial_Wrapper(VecDotF32_FMA, input, rawWeights, output, M, K, N);
            return;
        }
        F32BlockedColumns(input, (float*)rawWeights, output, M, K, N, 0, N);
    }

    public static unsafe void QuantizedMatMulF32_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotF32_FMA, input, rawWeights, output, K, N);
            return;
        }

        // Same split as the F16 path: by column, 16-column quanta, so each thread
        // owns a contiguous span of every output row and no two threads share an
        // output cache line.
        int target = Math.Max(1, N / Environment.ProcessorCount);
        int chunkSize = (target + 15) & ~15;
        int numChunks = (N + chunkSize - 1) / chunkSize;

        long inputAddr = (long)input, weightsAddr = (long)rawWeights, outputAddr = (long)output;
        Parallel.For(0, numChunks, chunkIdx =>
        {
            int colStart = chunkIdx * chunkSize;
            int colEnd = Math.Min(colStart + chunkSize, N);
            F32BlockedColumns((float*)inputAddr, (float*)weightsAddr, (float*)outputAddr,
                M, K, N, colStart, colEnd);
        });
    }

    /// <summary>
    /// The F32 twin of <see cref="F16BlockedColumns"/>: columns
    /// [<paramref name="colStart"/>, <paramref name="colEnd"/>) of an F32 matmul,
    /// four rows per weight vector, rows tiled by <see cref="F16RowTile"/>.
    ///
    /// The M &gt; 1 F32 path used to run M independent GEMVs (one row at a time,
    /// every column), so a 256-row training step streamed the whole weight set 256
    /// times — in the forward and again in the backward's dInput. Training is the
    /// only caller with M &gt; 1 here (inference prefill runs the quantized or F16
    /// kernels), which is why it lagged PR #7's F16 work. Each weight vector now
    /// feeds four rows before it is discarded, and the row tile keeps those four
    /// rows' inputs resident while the columns stream past.
    /// </summary>
    private static unsafe void F32BlockedColumns(
        float* input, float* w, float* output,
        int M, int K, int N, int colStart, int colEnd)
    {
        for (int rowTile = 0; rowTile < M; rowTile += F16RowTile)
        {
            int tileEnd = Math.Min(rowTile + F16RowTile, M);

            for (int col = colStart; col < colEnd; col++)
            {
                float* pW = w + (long)col * K;
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
                    output[(long)r * N + col] = VecDotF32_FMA(input + (long)r * K, (byte*)w, col, K);
            }
        }
    }

    /// <summary>Rows processed together in the blocked F16 path. See <see cref="F16BlockedColumns"/>.</summary>
    private const int F16RowBlock = 4;

    /// <summary>
    /// Rows held resident while sweeping columns. A tile of 16 rows of a
    /// 896-deep hidden state is 56 KB, which stays in L1/L2 across the whole
    /// column sweep; the untiled form re-read a 128-row block from L2 once per
    /// column. Must be a multiple of <see cref="F16RowBlock"/>.
    /// </summary>
    private const int F16RowTile = 16;

    /// <summary>
    /// Computes columns [<paramref name="colStart"/>, <paramref name="colEnd"/>) of
    /// an F16 matmul, four rows at a time.
    ///
    /// The row-at-a-time form widens every weight once per row: a 128-token prefill
    /// chunk converts each fp16 weight 128 times to use it 128 times. Here each
    /// widened vector feeds <see cref="F16RowBlock"/> FMAs before it is discarded.
    ///
    /// Four is measured, not assumed — R=8 was no faster on any shape, so past four
    /// rows the widening has stopped being the limit and the input-row loads have
    /// become it. Four also keeps the working set to 4 accumulators plus one weight
    /// vector, well inside the 16 available YMM registers.
    ///
    /// The outer <see cref="F16RowTile"/> loop is what bounds those input loads.
    /// Sweeping all M rows inside each column re-reads the whole input block once
    /// per column — for one gate/up projection of a 128-token chunk that is 2.2 GB
    /// through L2, against 8.7 MB of weights. Tiling to 16 rows keeps the working
    /// set small enough to stay put while the columns stream past it:
    ///     K=896  N=896     85.5 GF/s -> 162.8 GF/s
    ///     K=896  N=4864   168.1 GF/s -> 185.4 GF/s
    ///     K=4864 N=896    136.5 GF/s -> 192.1 GF/s
    /// Tile height is a broad optimum — anything from 8 to 64 lands within a few
    /// percent — so this is not tuned to one cache size.
    /// </summary>
    private static unsafe void F16BlockedColumns(
        float* input, ushort* w, float* output,
        int M, int K, int N, int colStart, int colEnd)
    {
        for (int rowTile = 0; rowTile < M; rowTile += F16RowTile)
        {
            int tileEnd = Math.Min(rowTile + F16RowTile, M);

            for (int col = colStart; col < colEnd; col++)
            {
                ushort* pW = w + (long)col * K;
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
                        var vw = WidenHalf8(pW + k);
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
                        float wf = HalfToFloat_F16C(pW[k]);
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

                // Rows past the last full block fall back to the single-row kernel.
                // F16RowTile is a multiple of F16RowBlock, so this only runs in the
                // final tile, and only when M is not a multiple of four.
                for (; r < tileEnd; r++)
                    output[(long)r * N + col] = VecDotF16_FMA(input + (long)r * K, (byte*)w, col, K);
            }
        }
    }

    public static unsafe void QuantizedMatMulF16_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            QuantizedMatMul_Serial_Wrapper(VecDotF16_FMA, input, rawWeights, output, M, K, N);
            return;
        }
        F16BlockedColumns(input, (ushort*)rawWeights, output, M, K, N, 0, N);
    }

    public static unsafe void QuantizedMatMulF16_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        // Decode has a single row, so there is nothing to block over; that path
        // parallelises across columns instead.
        if (M <= 1)
        {
            DecodeParallel(VecDotF16_FMA, input, rawWeights, output, K, N);
            return;
        }

        // Split by column rather than by row, so each thread owns a contiguous
        // span of every output row and the widening is amortised inside it.
        // Chunks are rounded to 16 columns — one 64-byte line of float output —
        // so two threads never write the same cache line.
        int target = Math.Max(1, N / Environment.ProcessorCount);
        int chunkSize = (target + 15) & ~15;
        int numChunks = (N + chunkSize - 1) / chunkSize;

        long inputAddr = (long)input, weightsAddr = (long)rawWeights, outputAddr = (long)output;
        Parallel.For(0, numChunks, chunkIdx =>
        {
            int colStart = chunkIdx * chunkSize;
            int colEnd = Math.Min(colStart + chunkSize, N);
            F16BlockedColumns((float*)inputAddr, (ushort*)weightsAddr, (float*)outputAddr,
                M, K, N, colStart, colEnd);
        });
    }

    public static void ReadF32_Scalar(BinaryReader reader, Span<float> data, int _)
    {
        Span<byte> byteView = MemoryMarshal.AsBytes(data);
        reader.Read(byteView);
    }

    public static void ReadF16_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = HalfToFloat_Scalar(reader.ReadUInt16());
    }
    public static void ReadF16_F16C(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = HalfToFloat_F16C(reader.ReadUInt16());
    }

    // ===== I8 (signed byte) =====

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotI8_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        sbyte* w = (sbyte*)rawWeights;
        double sum = 0;
        sbyte* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotI8_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        sbyte* w = (sbyte*)rawWeights;
        sbyte* pW = w + (long)col * inFeatures;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 16; i += 16)
        {
            var vi0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pW + i));
            var vi1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pW + i + 8));
            var vw0 = Vector256.LoadUnsafe(ref input[i]);
            var vw1 = Vector256.LoadUnsafe(ref input[i + 8]);
            vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
            vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
        }
        for (; i <= inFeatures - 8; i += 8)
        {
            var vi = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pW + i));
            var vw = Vector256.LoadUnsafe(ref input[i]);
            vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
        }
        float sum = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        for (; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return sum;
    }

    public static unsafe void QuantizedMatMulI8_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotI8_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI8_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotI8_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI8_Serial_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotI8_FMA, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI8_Parallel_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotI8_FMA, input, rawWeights, output, M, K, N);

    public static void ReadI8_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = (sbyte)reader.ReadByte();
    }

    // ===== I16 (signed short) =====

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotI16_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        short* w = (short*)rawWeights;
        double sum = 0;
        short* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotI16_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        short* w = (short*)rawWeights;
        short* pW = w + (long)col * inFeatures;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 16; i += 16)
        {
            var vi0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pW + i));
            var vi1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pW + i + 8));
            var vw0 = Vector256.LoadUnsafe(ref input[i]);
            var vw1 = Vector256.LoadUnsafe(ref input[i + 8]);
            vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
            vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
        }
        for (; i <= inFeatures - 8; i += 8)
        {
            var vi = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pW + i));
            var vw = Vector256.LoadUnsafe(ref input[i]);
            vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
        }
        float sum = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        for (; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return sum;
    }

    public static unsafe void QuantizedMatMulI16_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotI16_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI16_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotI16_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI16_Serial_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotI16_FMA, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI16_Parallel_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotI16_FMA, input, rawWeights, output, M, K, N);

    public static void ReadI16_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = reader.ReadInt16();
    }

    // ===== I32 (signed int) =====

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotI32_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int* w = (int*)rawWeights;
        double sum = 0;
        int* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotI32_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int* w = (int*)rawWeights;
        int* pW = w + (long)col * inFeatures;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 16; i += 16)
        {
            var vi0 = Avx.ConvertToVector256Single(Avx.LoadVector256(pW + i));
            var vi1 = Avx.ConvertToVector256Single(Avx.LoadVector256(pW + i + 8));
            var vw0 = Vector256.LoadUnsafe(ref input[i]);
            var vw1 = Vector256.LoadUnsafe(ref input[i + 8]);
            vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
            vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
        }
        for (; i <= inFeatures - 8; i += 8)
        {
            var vi = Avx.ConvertToVector256Single(Avx.LoadVector256(pW + i));
            var vw = Vector256.LoadUnsafe(ref input[i]);
            vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
        }
        float sum = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        for (; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return sum;
    }

    public static unsafe void QuantizedMatMulI32_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotI32_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI32_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotI32_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI32_Serial_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotI32_FMA, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulI32_Parallel_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotI32_FMA, input, rawWeights, output, M, K, N);

    public static void ReadI32_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = reader.ReadInt32();
    }

    // ===== TQ2_0 (ternary 2-bit, block_tq2_0: d[2]+qs[64]=66B, 256 el) =====

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotTQ2_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 66;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int byteIdx = i / 4;
                int shift = (i % 4) * 2;
                int v = (qs[byteIdx] >> shift) & 3;
                sum += input[b * qk + i] * ((v - 1) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotTQ2_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 66;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;

        float* tmp = stackalloc float[16]; // hoisted out of both loops: allocated once per call, reused every iteration

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            float* pIn = input + b * qk;
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                int byteBase = i / 4;
                for (int j = 0; j < 16; j++)
                {
                    int bIdx = j / 4;
                    int shift = (j % 4) * 2;
                    int v = (qs[byteBase + bIdx] >> shift) & 3;
                    tmp[j] = (v - 1) * d;
                }
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Vector256.Load(tmp);
                var vw1 = Vector256.Load(tmp + 8);
                vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
            }
            for (; i < blockEnd; i++)
            {
                int byteIdx = i / 4;
                int shift = (i % 4) * 2;
                int v = (qs[byteIdx] >> shift) & 3;
                sum += pIn[i] * ((v - 1) * d);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }
    public static unsafe void QuantizedMatMulTQ2_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotTQ2_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulTQ2_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotTQ2_0_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotTQ2_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadTQ2_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        const int blockBytes = 66;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int byteIdx = j / 4;
                int shift = (j % 4) * 2;
                int v = (buf[2 + byteIdx] >> shift) & 3;
                data[blockStart + j] = (v - 1) * d;
            }
        }
    }

    // ===== Q1_0 (1-bit sign quant, block_q1_0: d[2]+qs[32]=34B, 256 el) =====
    // Each element is 1 bit: value = d * (bit == 0 ? -1 : +1)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ1_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 34;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int byteIdx = i / 8;
                int bitIdx = i % 8;
                int bit = (qs[byteIdx] >> bitIdx) & 1;
                sum += input[b * qk + i] * ((2 * bit - 1) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ1_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 34;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;

        float* tmp = stackalloc float[16];

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            float* pIn = input + b * qk;
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int j = 0; j < 16; j++)
                {
                    int globalIdx = i + j;
                    int byteIdx = globalIdx / 8;
                    int bitIdx = globalIdx % 8;
                    int bit = (qs[byteIdx] >> bitIdx) & 1;
                    tmp[j] = (2 * bit - 1) * d;
                }
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Vector256.Load(tmp);
                var vw1 = Vector256.Load(tmp + 8);
                vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
            }
            for (; i < blockEnd; i++)
            {
                int byteIdx = i / 8;
                int bitIdx = i % 8;
                int bit = (qs[byteIdx] >> bitIdx) & 1;
                sum += pIn[i] * ((2 * bit - 1) * d);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulQ1_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ1_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ1_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ1_0_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ1_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ1_0_Serial_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ1_0_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ1_0_Parallel_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ1_0_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ1_0_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadQ1_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        const int blockBytes = 34;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int byteIdx = j / 8;
                int bitIdx = j % 8;
                int bit = (buf[2 + byteIdx] >> bitIdx) & 1;
                data[blockStart + j] = (2 * bit - 1) * d;
            }
        }
    }

    // ===== TQ1_0 (ternary 1-bit, base-3 packing: qs0[32]+qs1[16]+qh[4]+d[2]=54B, 256 el) =====

    private static int DecodeTQ1_0_Digit(byte b, int pos)
    {
        int div = 1;
        for (int i = 0; i < pos; i++) div *= 3;
        return (b / div) % 3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotTQ1_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 54;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)(block + 52));
            byte* qs0 = block;
            byte* qs1 = block + 32;
            byte* qh = block + 48;
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int v;
                if (i < 160)
                {
                    int byteIdx = i / 5;
                    int subIdx = i % 5;
                    v = DecodeTQ1_0_Digit(qs0[byteIdx], subIdx);
                }
                else if (i < 240)
                {
                    int idx = i - 160;
                    int byteIdx = idx / 5;
                    int subIdx = idx % 5;
                    v = DecodeTQ1_0_Digit(qs1[byteIdx], subIdx);
                }
                else
                {
                    int idx = i - 240;
                    int byteIdx = idx / 4;
                    int subIdx = idx % 4;
                    v = DecodeTQ1_0_Digit(qh[byteIdx], subIdx);
                }
                sum += input[b * qk + i] * ((v - 1) * d);
            }
        }
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulTQ1_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotTQ1_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulTQ1_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotTQ1_0_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotTQ1_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadTQ1_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        const int blockBytes = 54;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[52]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int v;
                if (j < 160)
                {
                    int byteIdx = j / 5;
                    int subIdx = j % 5;
                    v = DecodeTQ1_0_Digit(buf[byteIdx], subIdx);
                }
                else if (j < 240)
                {
                    int idx = j - 160;
                    int byteIdx = idx / 5;
                    int subIdx = idx % 5;
                    v = DecodeTQ1_0_Digit(buf[32 + byteIdx], subIdx);
                }
                else
                {
                    int idx = j - 240;
                    int byteIdx = idx / 4;
                    int subIdx = idx % 4;
                    v = DecodeTQ1_0_Digit(buf[48 + byteIdx], subIdx);
                }
                data[blockStart + j] = (v - 1) * d;
            }
        }
    }

    // ===== IQ1_S (1.56 bpw importance quant, grid-codebook: d[2]+qs[32]+qh[16]=50B, 256 el) =====

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ1_S_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 50;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        var grid = QuantizationGrids.IQ1S_Grid;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            ushort* qhp = (ushort*)(block + 34);
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int g = 0; g < 8 && g * 32 < blockEnd; g++)
            {
                ushort qhWord = qhp[g];
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, blockEnd - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = qs[g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        sum += input[b * qk + g * 32 + s * 8 + k] * (dl * (gv + delta));
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ1_M_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 56;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        var grid = QuantizationGrids.IQ1S_Grid;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * blockBytes + b * blockBytes;
            byte* qs = block;
            ushort* qhp = (ushort*)(block + 32);
            byte* scales = block + 48;
            float d = IQ1_M_DecodeScale(scales);
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int g = 0; g < 8 && g * 32 < blockEnd; g++)
            {
                ushort qhWord = qhp[g];
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, blockEnd - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = qs[g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        sum += input[b * qk + g * 32 + s * 8 + k] * (dl * (gv + delta));
                    }
                }
            }
        }
        return (float)sum;
    }

    private static unsafe float IQ1_M_DecodeScale(byte* scales)
    {
        uint lo = *(uint*)scales;
        uint hi = *(ushort*)(scales + 4);
        uint packed = lo | (hi << 16);
        ushort halfBits = (ushort)(((packed >> 12) & 0x000F) | ((packed >> 8) & 0x00F0) | ((packed >> 4) & 0x0F00) | (packed & 0xF000));
        return HalfToFloat_Scalar(halfBits);
    }

    public static unsafe void QuantizedMatMulIQ1_S_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotIQ1_S_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulIQ1_S_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotIQ1_S_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotIQ1_S_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulIQ1_M_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotIQ1_M_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulIQ1_M_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotIQ1_M_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotIQ1_M_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadIQ1_S_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        const int blockBytes = 50;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];
        var grid = QuantizationGrids.IQ1S_Grid;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int g = 0; g < 8 && g * 32 < valid; g++)
            {
                ushort qhWord = Unsafe.ReadUnaligned<ushort>(ref buf[34 + g * 2]);
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, valid - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = buf[2 + g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        data[blockStart + g * 32 + s * 8 + k] = dl * (gv + delta);
                    }
                }
            }
        }
    }

    public static void ReadIQ1_M_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        const int blockBytes = 56;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];
        var grid = QuantizationGrids.IQ1S_Grid;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d;
            unsafe
            {
                fixed (byte* pBuf = buf)
                    d = IQ1_M_DecodeScale(pBuf + 48);
            }
            int valid = Math.Min(qk, n - blockStart);

            for (int g = 0; g < 8 && g * 32 < valid; g++)
            {
                ushort qhWord = Unsafe.ReadUnaligned<ushort>(ref buf[32 + g * 2]);
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, valid - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = buf[g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        data[blockStart + g * 32 + s * 8 + k] = dl * (gv + delta);
                    }
                }
            }
        }
    }

    // ===== TQ2_0 FMA QuantizedMatMul stubs =====

    public static unsafe void QuantizedMatMulTQ2_0_Serial_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotTQ2_0_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulTQ2_0_Parallel_FMA(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotTQ2_0_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotTQ2_0_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
