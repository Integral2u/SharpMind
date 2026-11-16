using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace SharpMind.Core.Quantization;

/// <summary>
/// The grid-based 2- and 3-bit formats: IQ2_XXS (16), IQ2_XS (17) and IQ3_XXS
/// (18). Each 256-value super-block carries eight 32-value sub-blocks; a
/// sub-block is four 8-value grid lookups (IQ2) or eight 4-value lookups (IQ3)
/// plus one packed scale nibble and four 7-bit sign patterns per sub-block.
///
/// These are decode-first formats: the kernels here are scalar, like IQ1_S /
/// IQ1_M / TQ1_0, because the grid lookup is a gather with no vector form on
/// x86. That is a real cost on 2-bit models and the next thing to measure, not
/// a correctness gap.
/// </summary>
public static partial class QuantizationKernels
{
    private const int IQ2XXS_BlockBytes = 66;
    private const int IQ2XS_BlockBytes = 74;
    private const int IQ3XXS_BlockBytes = 98;
    private const int IQ2S_BlockBytes = 82;
    private const int IQ3S_BlockBytes = 110;

    /// <summary>Bit <c>j</c> set means "negate value <c>j</c>".</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float IQSign(int signs, int j) => (signs & (1 << j)) == 0 ? 1f : -1f;

    /// <summary>
    /// The S variants add high index bits and separate sign bytes: IQ2_S keeps
    /// them in the upper half of <c>qs</c>, IQ3_S in its own array, and both
    /// hide the extra index bits in one <c>qh</c> byte per 32 values.
    /// </summary>
    public static void ReadIQ2_S_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[IQ2S_BlockBytes];
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2S_Grid;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int ib = 0; ib < qk / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= valid) break;

                byte scale = buf[74 + ib];
                float db0 = d * (0.5f + (scale & 0x0F)) * 0.25f;
                float db1 = d * (0.5f + (scale >> 4)) * 0.25f;
                byte qh = buf[66 + ib];
                int subEnd = Math.Min(32, valid - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int index = buf[2 + ib * 4 + l] | ((qh << (8 - 2 * l)) & 0x300);
                    int signs = buf[2 + 32 + ib * 4 + l];
                    float db = l < 2 ? db0 : db1;
                    int gridOffset = index * 8;
                    for (int j = 0; j < 8; j++)
                    {
                        int idx = l * 8 + j;
                        if (idx >= subEnd) break;
                        data[blockStart + basePos + idx] = db * grid[gridOffset + j] * IQSign(signs, j);
                    }
                }
            }
        }
    }

    public static void ReadIQ3_S_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[IQ3S_BlockBytes];
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ3S_Grid;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int ib = 0; ib < qk / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= valid) break;

                // One scale nibble covers two 32-value sub-blocks.
                byte scale = buf[106 + ib / 2];
                float db = d * (1 + 2 * ((ib % 2) == 0 ? (scale & 0x0F) : (scale >> 4)));
                byte qh = buf[66 + ib];
                int subEnd = Math.Min(32, valid - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int signs = buf[74 + ib * 4 + l];
                    int g1 = (buf[2 + ib * 8 + 2 * l] | ((qh << (8 - 2 * l)) & 256)) * 4;
                    int g2 = (buf[2 + ib * 8 + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)) * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        int lo = l * 8 + j;
                        int hi = lo + 4;
                        if (lo < subEnd)
                            data[blockStart + basePos + lo] = db * grid[g1 + j] * IQSign(signs, j);
                        if (hi < subEnd)
                            data[blockStart + basePos + hi] = db * grid[g2 + j] * IQSign(signs, j + 4);
                    }
                }
            }
        }
    }

    public static void ReadIQ2_XXS_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[IQ2XXS_BlockBytes];
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2XXS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int ib = 0; ib < qk / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= valid) break;

                uint word = BinaryPrimitives.ReadUInt32LittleEndian(buf[(2 + ib * 8 + 4)..]);
                float db = d * (0.5f + (word >> 28)) * 0.25f;
                int subEnd = Math.Min(32, valid - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int signs = signTable[(int)((word >> (7 * l)) & 127)];
                    int gridOffset = buf[2 + ib * 8 + l] * 8;
                    for (int j = 0; j < 8; j++)
                    {
                        int idx = l * 8 + j;
                        if (idx >= subEnd) break;
                        data[blockStart + basePos + idx] = db * grid[gridOffset + j] * IQSign(signs, j);
                    }
                }
            }
        }
    }

    public static void ReadIQ2_XS_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[IQ2XS_BlockBytes];
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2XS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int ib = 0; ib < qk / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= valid) break;

                byte scale = buf[2 + 64 + ib];
                float db0 = d * (0.5f + (scale & 0x0F)) * 0.25f;
                float db1 = d * (0.5f + (scale >> 4)) * 0.25f;
                int subEnd = Math.Min(32, valid - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int entry = BinaryPrimitives.ReadUInt16LittleEndian(buf[(2 + (ib * 4 + l) * 2)..]);
                    int signs = signTable[entry >> 9];
                    int gridOffset = (entry & 511) * 8;
                    float db = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                    {
                        int idx = l * 8 + j;
                        if (idx >= subEnd) break;
                        data[blockStart + basePos + idx] = db * grid[gridOffset + j] * IQSign(signs, j);
                    }
                }
            }
        }
    }

    public static void ReadIQ3_XXS_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[IQ3XXS_BlockBytes];
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ3XXS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int ib = 0; ib < qk / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= valid) break;

                uint word = BinaryPrimitives.ReadUInt32LittleEndian(buf[(2 + 64 + ib * 4)..]);
                float db = d * (0.5f + (word >> 28)) * 0.5f;
                int subEnd = Math.Min(32, valid - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int signs = signTable[(int)((word >> (7 * l)) & 127)];
                    int g1 = buf[2 + ib * 8 + 2 * l] * 4;
                    int g2 = buf[2 + ib * 8 + 2 * l + 1] * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        int lo = l * 8 + j;
                        int hi = lo + 4;
                        if (lo < subEnd)
                            data[blockStart + basePos + lo] = db * grid[g1 + j] * IQSign(signs, j);
                        if (hi < subEnd)
                            data[blockStart + basePos + hi] = db * grid[g2 + j] * IQSign(signs, j + 4);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ2_XXS_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2XXS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ2XXS_BlockBytes + b * IQ2XXS_BlockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= blockEnd) break;

                uint word = Unsafe.ReadUnaligned<uint>(qs + ib * 8 + 4);
                float db = d * (0.5f + (word >> 28)) * 0.25f;
                int subEnd = Math.Min(32, blockEnd - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int signs = signTable[(int)((word >> (7 * l)) & 127)];
                    int gridOffset = qs[ib * 8 + l] * 8;
                    for (int j = 0; j < 8; j++)
                    {
                        int idx = l * 8 + j;
                        if (idx >= subEnd) break;
                        sum += pIn[basePos + idx] * (db * grid[gridOffset + j] * IQSign(signs, j));
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ2_XS_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2XS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ2XS_BlockBytes + b * IQ2XS_BlockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            byte* scales = block + 66;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= blockEnd) break;

                float db0 = d * (0.5f + (scales[ib] & 0x0F)) * 0.25f;
                float db1 = d * (0.5f + (scales[ib] >> 4)) * 0.25f;
                int subEnd = Math.Min(32, blockEnd - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int entry = Unsafe.ReadUnaligned<ushort>(qs + (ib * 4 + l) * 2);
                    int signs = signTable[entry >> 9];
                    int gridOffset = (entry & 511) * 8;
                    float db = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                    {
                        int idx = l * 8 + j;
                        if (idx >= subEnd) break;
                        sum += pIn[basePos + idx] * (db * grid[gridOffset + j] * IQSign(signs, j));
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ3_XXS_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ3XXS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ3XXS_BlockBytes + b * IQ3XXS_BlockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            byte* scalesAndSigns = block + 66;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= blockEnd) break;

                uint word = Unsafe.ReadUnaligned<uint>(scalesAndSigns + ib * 4);
                float db = d * (0.5f + (word >> 28)) * 0.5f;
                int subEnd = Math.Min(32, blockEnd - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int signs = signTable[(int)((word >> (7 * l)) & 127)];
                    int g1 = qs[ib * 8 + 2 * l] * 4;
                    int g2 = qs[ib * 8 + 2 * l + 1] * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        int lo = l * 8 + j;
                        int hi = lo + 4;
                        if (lo < subEnd)
                            sum += pIn[basePos + lo] * (db * grid[g1 + j] * IQSign(signs, j));
                        if (hi < subEnd)
                            sum += pIn[basePos + hi] * (db * grid[g2 + j] * IQSign(signs, j + 4));
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ2_S_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2S_Grid;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ2S_BlockBytes + b * IQ2S_BlockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            byte* qh = block + 66;
            byte* scales = block + 74;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= blockEnd) break;

                float db0 = d * (0.5f + (scales[ib] & 0x0F)) * 0.25f;
                float db1 = d * (0.5f + (scales[ib] >> 4)) * 0.25f;
                int subEnd = Math.Min(32, blockEnd - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int index = qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                    int signs = qs[32 + ib * 4 + l];
                    float db = l < 2 ? db0 : db1;
                    int gridOffset = index * 8;
                    for (int j = 0; j < 8; j++)
                    {
                        int idx = l * 8 + j;
                        if (idx >= subEnd) break;
                        sum += pIn[basePos + idx] * (db * grid[gridOffset + j] * IQSign(signs, j));
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ3_S_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        ReadOnlySpan<byte> grid = QuantizationGrids.IQ3S_Grid;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ3S_BlockBytes + b * IQ3S_BlockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            byte* qh = block + 66;
            byte* signs = block + 74;
            byte* scales = block + 106;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / 32; ib++)
            {
                int basePos = ib * 32;
                if (basePos >= blockEnd) break;

                byte scale = scales[ib / 2];
                float db = d * (1 + 2 * ((ib % 2) == 0 ? (scale & 0x0F) : (scale >> 4)));
                int subEnd = Math.Min(32, blockEnd - basePos);

                for (int l = 0; l < 4; l++)
                {
                    int signByte = signs[ib * 4 + l];
                    int g1 = (qs[ib * 8 + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256)) * 4;
                    int g2 = (qs[ib * 8 + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256)) * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        int lo = l * 8 + j;
                        int hi = lo + 4;
                        if (lo < subEnd)
                            sum += pIn[basePos + lo] * (db * grid[g1 + j] * IQSign(signByte, j));
                        if (hi < subEnd)
                            sum += pIn[basePos + hi] * (db * grid[g2 + j] * IQSign(signByte, j + 4));
                    }
                }
            }
        }
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulIQ2_XXS_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalar(VecDotIQ2_XXS_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ2_XXS_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalarParallel(VecDotIQ2_XXS_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ2_XS_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalar(VecDotIQ2_XS_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ2_XS_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalarParallel(VecDotIQ2_XS_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ3_XXS_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalar(VecDotIQ3_XXS_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ3_XXS_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalarParallel(VecDotIQ3_XXS_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ2_S_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalar(VecDotIQ2_S_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ2_S_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalarParallel(VecDotIQ2_S_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ3_S_Serial_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalar(VecDotIQ3_S_Scalar, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulIQ3_S_Parallel_Scalar(
        float* input, byte* rawWeights, float* output, int M, int K, int N)
        => QuantizedMatMulScalarParallel(VecDotIQ3_S_Scalar, input, rawWeights, output, M, K, N);

    private static unsafe void QuantizedMatMulScalar(
        VecDotFn vecdot, float* input, byte* rawWeights, float* output, int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = vecdot(pInRow, rawWeights, col, K);
        }
    }

    private static unsafe void QuantizedMatMulScalarParallel(
        VecDotFn vecdot, float* input, byte* rawWeights, float* output, int M, int K, int N)
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
}
