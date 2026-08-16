using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    internal const int QK_K = 256;
    internal const int QK = 32;

    public static unsafe void QuantizedMatMul_Serial_Wrapper(VecDotFn vecdot,
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = vecdot(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMul_Parallel_Wrapper(VecDotFn vecdot,
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(vecdot, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = vecdot(pInRow, rawWeights, col, K);
            });
        }
    }

    /// <summary>One 64-byte cache line of float output. Chunks never split a line.</summary>
    private const int MinColsPerChunk = 16;

    /// <summary>
    /// Work below which spreading across cores costs more than it saves,
    /// measured as columns × inFeatures. See <see cref="DecodeParallel"/>.
    /// </summary>
    private const long MinWorkForDecodeParallel = 65_536;

    /// <summary>
    /// Column-parallel path for M == 1 (decode). Chunks N across cores.
    ///
    /// The previous rule refused to parallelise below 512 columns and then chose
    /// ceil(N/512) chunks. Both halves misfire at real transformer widths: at
    /// hidden 896 the K and V projections (N=128) ran single-threaded, and the
    /// output and down projections (N=896) got two chunks — two cores of sixteen
    /// — because 896 is barely over 512. Columns are a poor proxy for work in any
    /// case, since a column costs inFeatures multiply-accumulates.
    ///
    /// Decide on N × K, then cut into as many chunks as there are cores, subject
    /// to a whole cache line of output each. Measured on 16-core Zen 5, shipped
    /// rule vs this one, times for the whole matmul:
    ///     K=896  N=128     24.4 us -> 13.3 us   (was serial)
    ///     K=896  N=896      109 us -> 58.9 us   (was 2 chunks)
    ///     K=4864 N=896      585 us ->  280 us   (was 2 chunks)
    ///     K=4864 N=4864    1494 us ->  962 us
    /// Below the threshold the loop stays serial: at N=32, K=896 splitting the
    /// work actually cost 1.4x, which is the case the old 512 floor got right.
    /// </summary>
    private static unsafe void DecodeParallel(VecDotFn vecdot,
        float* input, byte* rawWeights, float* output,
        int K, int N)
    {
        int numChunks = (long)N * K >= MinWorkForDecodeParallel
            ? Math.Min(Environment.ProcessorCount, N / MinColsPerChunk)
            : 1;

        if (numChunks <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = vecdot(input, rawWeights, col, K);
            return;
        }

        int chunkSize = (N + numChunks - 1) / numChunks;
        chunkSize = (chunkSize + MinColsPerChunk - 1) / MinColsPerChunk * MinColsPerChunk;
        numChunks = (N + chunkSize - 1) / chunkSize;

        long inputAddr = (long)input;
        long weightsAddr = (long)rawWeights;
        long outputAddr = (long)output;

        Parallel.For(0, numChunks, chunkIdx =>
        {
            float* pInput = (float*)inputAddr;
            byte* pWeights = (byte*)weightsAddr;
            float* pOutput = (float*)outputAddr;

            int colStart = chunkIdx * chunkSize;
            int colEnd = Math.Min(colStart + chunkSize, N);

            for (int col = colStart; col < colEnd; col++)
                pOutput[col] = vecdot(pInput, pWeights, col, K);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HalfToFloat_F16C(ushort half)
    {
        return (float)BitConverter.UInt16BitsToHalf(half);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float HalfToFloat_Scalar(ushort half)
    {
        int exp5 = (half >> 10) & 0x1F;

        if (exp5 == 0)
        {
            uint mant10 = (uint)(half & 0x3FF);
            if (mant10 == 0)
                return (half & 0x8000) == 0 ? 0f : -0f;

            int lz = BitOperations.LeadingZeroCount(mant10);
            int k = 31 - lz;
            uint e = (uint)(k + 103);
            uint m = (mant10 - (1u << k)) << (23 - k);
            uint bitsSub = ((uint)(half & 0x8000) << 16) | (e << 23) | m;
            return *(float*)&bitsSub;
        }

        if (exp5 == 31)
        {
            if ((half & 0x3FF) == 0)
                return (half & 0x8000) == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            return float.NaN;
        }

        uint eBits = (uint)(exp5 + 112);
        uint mMant = (uint)(half & 0x3FF) << 13;
        uint bitsNrm = ((uint)(half & 0x8000) << 16) | (eBits << 23) | mMant;
        return *(float*)&bitsNrm;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToHalf_F16C(float f) => BitConverter.HalfToUInt16Bits((Half)f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ushort FloatToHalf_Scalar(float f)
    {
        uint bits = *(uint*)&f;
        uint sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mant = bits & 0x007FFFFF;

        if (exp <= 0)
        {
            if (exp < -10) return (ushort)sign;
            mant = (mant | 0x00800000) >> (1 - exp);
            return (ushort)(sign | (mant >> 13));
        }

        if (exp >= 31)
        {
            if (exp > 31) return (ushort)(sign | 0x7C00 | (mant >> 13));
            return (ushort)(sign | 0x7C00 | (mant != 0 ? 0x200u : 0));
        }

        uint eBits = (uint)(exp << 10);
        uint mMant = mant >> 13;
        return (ushort)(sign | eBits | mMant);
    }

    public static unsafe byte GetScaleMinK4_Scale_Scalar(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j] & 0x3F);
        return (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
    }

    public static unsafe byte GetScaleMinK4_Min_Scalar(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j + 4] & 0x3F);
        return (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }
}
