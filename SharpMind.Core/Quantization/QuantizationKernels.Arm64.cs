using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    /// <summary>
    /// Q8_0 vec-dot on Arm64 NEON, the 128-bit counterpart of
    /// <see cref="VecDotQ8_0_AVX2"/>: eight int8 weights are widened to two
    /// <see cref="Vector128{T}"/> of float and fused into two four-lane
    /// accumulators, so a full 32-value block is four widening loads and eight
    /// multiply-adds. The block scale is applied to the widened weights rather
    /// than to the input, so <c>d</c> is broadcast once per block.
    /// </summary>
    /// <remarks>
    /// A block's lanes accumulate in a different order from the scalar kernel, so
    /// results agree to float precision rather than bit-for-bit; every caller
    /// compares with a tolerance. The tail is scalar because the last block of a
    /// row can be shorter than 32 when <paramref name="inFeatures"/> is not a
    /// multiple of QK.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_0_Arm64(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vd = Vector128.Create(d);
            var vacc0 = Vector128<float>.Zero;
            var vacc1 = Vector128<float>.Zero;

            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                (Vector128<float> lo, Vector128<float> hi) = WidenQ8_8ToFloat(values + i);
                vacc0 = AdvSimd.Add(vacc0,
                    AdvSimd.Multiply(AdvSimd.LoadVector128(pIn + i), AdvSimd.Multiply(lo, vd)));
                vacc1 = AdvSimd.Add(vacc1,
                    AdvSimd.Multiply(AdvSimd.LoadVector128(pIn + i + 4), AdvSimd.Multiply(hi, vd)));
            }

            sum += Vector128.Sum(AdvSimd.Add(vacc0, vacc1));

            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }

        return (float)sum;
    }

    /// <summary>
    /// Eight packed int8 values as two float vectors (low four, high four).
    /// sbyte -> short -> int -> float is two SXTL pairs and two SCVTF, which is
    /// cheaper than going through int16 arithmetic: the product has to end up in
    /// float anyway because the input is float.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe (Vector128<float> Lo, Vector128<float> Hi) WidenQ8_8ToFloat(sbyte* values)
    {
        Vector128<short> shorts = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(values));
        Vector128<int> lo = AdvSimd.SignExtendWideningLower(shorts.GetLower());
        Vector128<int> hi = AdvSimd.SignExtendWideningUpper(shorts);
        return (AdvSimd.ConvertToSingle(lo), AdvSimd.ConvertToSingle(hi));
    }

    /// <summary>
    /// Serial Q8_0 matmul. Rows run in order and each row is one vec-dot per
    /// column; there is no blocked microkernel yet, so a multi-row (prefill) call
    /// costs the same per row as decode. That is the next NEON step, not a
    /// correctness gap.
    /// </summary>
    public static unsafe void QuantizedMatMulQ8_0_Serial_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_0_Arm64(pInRow, rawWeights, col, K);
        }
    }

    /// <summary>
    /// Parallel Q8_0 matmul: decode (M=1) splits the columns through
    /// <see cref="DecodeParallel"/>, exactly like the FMA and AVX2 variants, and
    /// prefill splits the rows. Column splitting is what keeps a single-token
    /// call on every core, which is the case that matters for chat latency.
    /// </summary>
    public static unsafe void QuantizedMatMulQ8_0_Parallel_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_0_Arm64, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_Arm64(pInRow, rawWeights, col, K);
            });
        }
    }

    /// <summary>
    /// The eight values in one byte group as two float vectors, for Q4_K and Q6_K.
    /// The codes fit in a byte, so they widen byte -> ushort -> uint -> float;
    /// the caller has already masked or shifted the nibble it wants.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector128<float> Lo, Vector128<float> Hi) WidenU16ToFloat(Vector128<ushort> v)
    {
        Vector128<uint> lo = AdvSimd.ZeroExtendWideningLower(v.GetLower());
        Vector128<uint> hi = AdvSimd.ZeroExtendWideningUpper(v);
        return (AdvSimd.ConvertToSingle(lo.AsInt32()), AdvSimd.ConvertToSingle(hi.AsInt32()));
    }

    /// <summary>Eight bytes widened to eight ushort lanes, ready for a mask or shift.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<ushort> Load8BytesToU16(byte* p) =>
        AdvSimd.ZeroExtendWideningLower(AdvSimd.LoadVector64(p));

    /// <summary>
    /// Eight consecutive Q4_K nibbles as floats. A 32-value sub-block that starts
    /// on a 32 boundary lives in one contiguous run of 32 bytes: chunk
    /// <c>basePos/64</c> of <c>qs</c>, low nibbles for the first 32 values of the
    /// chunk and high nibbles for the next 32, so group <paramref name="g"/> is
    /// bytes <c>[8g, 8g+8)</c> of that run.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe (Vector128<float> Lo, Vector128<float> Hi) Q4KNibbles8ToFloat(byte* qs, int basePos, int g)
    {
        Vector128<ushort> v = Load8BytesToU16(qs + (basePos >> 6) * 32 + g * 8);
        v = (basePos & 32) == 0
            ? AdvSimd.And(v, Vector128.Create((ushort)0x0F))
            : AdvSimd.ShiftRightLogical(v, (byte)4);   // bytes are < 256, so this is the high nibble
        return WidenU16ToFloat(v);
    }

    /// <summary>
    /// Q4_K vec-dot on Arm64 NEON. Same block walk as <see cref="VecDotQ4K_AVX2"/>:
    /// a 256-value super-block is four 64-value chunks, each two 32-value
    /// sub-blocks with their own 6-bit scale and min, decoded eight values at a
    /// time. Value = <c>input * (scale*d_super*v - min*d_min)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_Arm64(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector128<float>.Zero;
        var vacc1 = Vector128<float>.Zero;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_Scalar(*(ushort*)(block + 0));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScaleMinK4_Scale_Scalar(isc, scales);
                    float m = GetScaleMinK4_Min_Scalar(isc, scales);
                    var vs = Vector128.Create(s * dSuper);
                    var vm = Vector128.Create(m * minSuper);

                    int subRem = Math.Min(32, blockEnd - basePos);
                    int l = 0;
                    if ((basePos & 31) == 0)
                    {
                        for (; l <= subRem - 8; l += 8)
                        {
                            (Vector128<float> lo, Vector128<float> hi) = Q4KNibbles8ToFloat(qs, basePos, l >> 3);
                            var w0 = AdvSimd.Subtract(AdvSimd.Multiply(lo, vs), vm);
                            var w1 = AdvSimd.Subtract(AdvSimd.Multiply(hi, vs), vm);
                            vacc0 = AdvSimd.Add(vacc0,
                                AdvSimd.Multiply(AdvSimd.LoadVector128(pIn + basePos + l), w0));
                            vacc1 = AdvSimd.Add(vacc1,
                                AdvSimd.Multiply(AdvSimd.LoadVector128(pIn + basePos + l + 4), w1));
                        }
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s * v * dSuper - m * minSuper);
                    }
                }
            }
        }

        sum += Vector128.Sum(AdvSimd.Add(vacc0, vacc1));
        return (float)sum;
    }

    /// <summary>
    /// Q6_K vec-dot on Arm64 NEON. One 128-value half of a super-block is four
    /// interleaved 32-value streams: low nibbles of <c>ql</c> plus two bits of
    /// <c>qh</c>, at bit offsets 0, 2, 4 and 6, and the half is walked eight
    /// values at a time so all four streams share the same <c>qh</c> load.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ6K_Arm64(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector128<float>.Zero;
        var vacc1 = Vector128<float>.Zero;
        var vacc2 = Vector128<float>.Zero;
        var vacc3 = Vector128<float>.Zero;
        var vacc4 = Vector128<float>.Zero;
        var vacc5 = Vector128<float>.Zero;
        var vacc6 = Vector128<float>.Zero;
        var vacc7 = Vector128<float>.Zero;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d = HalfToFloat_Scalar(*(ushort*)(block + 208));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128)
            {
                byte* pql = ql + (nOff == 0 ? 0 : 64);
                byte* pqh = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, blockEnd - nOff);

                int l = 0;
                for (; l + 103 < halfRem && l < 32; l += 8)
                {
                    int is_ = l / 16;
                    Vector128<ushort> qh8 = Load8BytesToU16(pqh + l);
                    Vector128<ushort> qlLo = Load8BytesToU16(pql + l);
                    Vector128<ushort> qlHi = Load8BytesToU16(pql + l + 32);

                    var m0F = Vector128.Create((ushort)0x0F);
                    var m03 = Vector128.Create((ushort)0x03);
                    Vector128<ushort> q1 = AdvSimd.Or(AdvSimd.And(qlLo, m0F),
                        AdvSimd.ShiftLeftLogical(AdvSimd.And(qh8, m03), (byte)4));
                    Vector128<ushort> q2 = AdvSimd.Or(AdvSimd.And(qlHi, m0F),
                        AdvSimd.ShiftLeftLogical(
                            AdvSimd.And(AdvSimd.ShiftRightLogical(qh8, (byte)2), m03), (byte)4));
                    Vector128<ushort> q3 = AdvSimd.Or(AdvSimd.ShiftRightLogical(qlLo, (byte)4),
                        AdvSimd.ShiftLeftLogical(
                            AdvSimd.And(AdvSimd.ShiftRightLogical(qh8, (byte)4), m03), (byte)4));
                    Vector128<ushort> q4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(qlHi, (byte)4),
                        AdvSimd.ShiftLeftLogical(
                            AdvSimd.And(AdvSimd.ShiftRightLogical(qh8, (byte)6), m03), (byte)4));

                    float s1 = d * psc[is_ + 0], s2 = d * psc[is_ + 2];
                    float s3 = d * psc[is_ + 4], s4 = d * psc[is_ + 6];

                    (Vector128<float> q1Lo, Vector128<float> q1Hi) = WidenU16ToFloat(q1);
                    (Vector128<float> q2Lo, Vector128<float> q2Hi) = WidenU16ToFloat(q2);
                    (Vector128<float> q3Lo, Vector128<float> q3Hi) = WidenU16ToFloat(q3);
                    (Vector128<float> q4Lo, Vector128<float> q4Hi) = WidenU16ToFloat(q4);

                    var v1 = Vector128.Create(s1); var m1 = Vector128.Create(32f * s1);
                    var v2 = Vector128.Create(s2); var m2 = Vector128.Create(32f * s2);
                    var v3 = Vector128.Create(s3); var m3 = Vector128.Create(32f * s3);
                    var v4 = Vector128.Create(s4); var m4 = Vector128.Create(32f * s4);

                    vacc0 = AdvSimd.Add(vacc0, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l),
                        AdvSimd.Subtract(AdvSimd.Multiply(q1Lo, v1), m1)));
                    vacc1 = AdvSimd.Add(vacc1, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 4),
                        AdvSimd.Subtract(AdvSimd.Multiply(q1Hi, v1), m1)));
                    vacc2 = AdvSimd.Add(vacc2, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 32),
                        AdvSimd.Subtract(AdvSimd.Multiply(q2Lo, v2), m2)));
                    vacc3 = AdvSimd.Add(vacc3, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 36),
                        AdvSimd.Subtract(AdvSimd.Multiply(q2Hi, v2), m2)));
                    vacc4 = AdvSimd.Add(vacc4, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 64),
                        AdvSimd.Subtract(AdvSimd.Multiply(q3Lo, v3), m3)));
                    vacc5 = AdvSimd.Add(vacc5, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 68),
                        AdvSimd.Subtract(AdvSimd.Multiply(q3Hi, v3), m3)));
                    vacc6 = AdvSimd.Add(vacc6, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 96),
                        AdvSimd.Subtract(AdvSimd.Multiply(q4Lo, v4), m4)));
                    vacc7 = AdvSimd.Add(vacc7, AdvSimd.Multiply(
                        AdvSimd.LoadVector128(pIn + nOff + l + 100),
                        AdvSimd.Subtract(AdvSimd.Multiply(q4Hi, v4), m4)));
                }

                for (; l < halfRem && l < 32; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = nOff + l;
                    int i2 = nOff + l + 32;
                    int i3 = nOff + l + 64;
                    int i4 = nOff + l + 96;

                    float v1 = d * psc[is_ + 0] * (q1v - 32);
                    float v2 = d * psc[is_ + 2] * (q2v - 32);
                    float v3 = d * psc[is_ + 4] * (q3v - 32);
                    float v4 = d * psc[is_ + 6] * (q4v - 32);

                    sum += pIn[i1] * v1;
                    if (i2 < blockEnd) sum += pIn[i2] * v2;
                    if (i3 < blockEnd) sum += pIn[i3] * v3;
                    if (i4 < blockEnd) sum += pIn[i4] * v4;
                }
            }
        }

        Vector128<float> top = AdvSimd.Add(AdvSimd.Add(vacc0, vacc1), AdvSimd.Add(vacc2, vacc3));
        Vector128<float> bottom = AdvSimd.Add(AdvSimd.Add(vacc4, vacc5), AdvSimd.Add(vacc6, vacc7));
        sum += Vector128.Sum(AdvSimd.Add(top, bottom));
        return (float)sum;
    }

    /// <summary>Serial Q4_K matmul: a row loop of <see cref="VecDotQ4K_Arm64"/>.</summary>
    public static unsafe void QuantizedMatMulQ4K_Serial_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_Arm64(pInRow, rawWeights, col, K);
        }
    }

    /// <summary>Parallel Q4_K matmul: columns for decode, rows for prefill.</summary>
    public static unsafe void QuantizedMatMulQ4K_Parallel_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4K_Arm64, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_Arm64(pInRow, rawWeights, col, K);
            });
        }
    }

    /// <summary>Serial Q6_K matmul: a row loop of <see cref="VecDotQ6K_Arm64"/>.</summary>
    public static unsafe void QuantizedMatMulQ6K_Serial_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ6K_Arm64(pInRow, rawWeights, col, K);
        }
    }

    /// <summary>Parallel Q6_K matmul: columns for decode, rows for prefill.</summary>
    public static unsafe void QuantizedMatMulQ6K_Parallel_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ6K_Arm64, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ6K_Arm64(pInRow, rawWeights, col, K);
            });
        }
    }

    /// <summary>
    /// One E2M1 nibble lane pair as floats, built as float bits rather than
    /// looked up. A nibble is <c>s eem</c>, and every codebook value is exact in
    /// float: <c>e &gt; 0</c> is <c>2^(e-1) * (1 + m/2)</c>, which is the normal
    /// float <c>(e + 126) &lt;&lt; 23 | m &lt;&lt; 22</c>; <c>e == 0</c> is the
    /// subnormal pair {0, 0.5}. The only special case is that 0.5, whose bit
    /// pattern (0x3F000000) is selected by <c>m</c> when <c>e</c> is zero.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> E2M1ToFloat(Vector128<uint> n)
    {
        Vector128<uint> e = (AdvSimd.ShiftRightLogical(n, (byte)1)) & Vector128.Create(3u);
        Vector128<uint> m = n & Vector128.Create(1u);

        Vector128<uint> sign = AdvSimd.ShiftLeftLogical(n & Vector128.Create(8u), (byte)28);
        Vector128<uint> normal = AdvSimd.ShiftLeftLogical(AdvSimd.Add(e, Vector128.Create(126u)), (byte)23)
            | AdvSimd.ShiftLeftLogical(m, (byte)22);
        Vector128<uint> sub = AdvSimd.CompareEqual(m, Vector128.Create(1u)) & Vector128.Create(0x3F000000u);

        Vector128<uint> bits = sign | Vector128.ConditionalSelect(
            AdvSimd.CompareEqual(e, Vector128<uint>.Zero), sub, normal);
        return bits.AsSingle();
    }

    /// <summary>
    /// Eight consecutive MXFP4 nibbles as two float vectors. Elements 0..15 of a
    /// block are the low nibbles of its sixteen bytes and 16..31 the high
    /// nibbles, so an eight-value group either masks or shifts all eight.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe (Vector128<float> Lo, Vector128<float> Hi) MXFP4Nibbles8ToFloat(byte* qs, int i)
    {
        Vector128<ushort> v = Load8BytesToU16(qs + (i & 15));
        v = (i & 16) == 0
            ? AdvSimd.And(v, Vector128.Create((ushort)0x0F))
            : AdvSimd.ShiftRightLogical(v, (byte)4);
        return (E2M1ToFloat(AdvSimd.ZeroExtendWideningLower(v.GetLower())),
                E2M1ToFloat(AdvSimd.ZeroExtendWideningUpper(v)));
    }

    /// <summary>
    /// MXFP4 vec-dot on Arm64 NEON: eight nibbles decode to floats as integer
    /// bit patterns, the E8M0 scale is broadcast once per block, and the block
    /// tail stays scalar because the last block of a row can be shorter than 32.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotMXFP4_Arm64(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 17;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector128<float>.Zero;
        var vacc1 = Vector128<float>.Zero;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = E8M0ToFloat_Scalar(block[0]);
            byte* qs = block + 1;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector128.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var (w0, w1) = MXFP4Nibbles8ToFloat(qs, i);
                vacc0 = AdvSimd.Add(vacc0, AdvSimd.Multiply(
                    AdvSimd.LoadVector128(pIn + i), AdvSimd.Multiply(w0, vd)));
                vacc1 = AdvSimd.Add(vacc1, AdvSimd.Multiply(
                    AdvSimd.LoadVector128(pIn + i + 4), AdvSimd.Multiply(w1, vd)));
            }
            for (; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (d * kvalues_mxfp4[nib]);
            }
        }

        return (float)sum + Vector128.Sum(AdvSimd.Add(vacc0, vacc1));
    }

    /// <summary>Serial MXFP4 matmul: one vec-dot per column, rows in order.</summary>
    public static unsafe void QuantizedMatMulMXFP4_Serial_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotMXFP4_Arm64(pInRow, rawWeights, col, K);
        }
    }

    /// <summary>Parallel MXFP4 matmul: columns for decode, rows for prefill.</summary>
    public static unsafe void QuantizedMatMulMXFP4_Parallel_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotMXFP4_Arm64, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotMXFP4_Arm64(pInRow, rawWeights, col, K);
            });
        }
    }
}
