using JigSawDotNet;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Block formats repacked once into the uniform signed-byte layout consumed by
/// <see cref="WideByteOps"/>: eight rows, a float scale/bias per sub-block, and
/// signed int8 values. The repack is load-time only; the decode branch on the
/// format happens here, never in the matmul loop.
/// </summary>
public sealed unsafe class WideByteWeights : IDisposable
{
    public const int RowsPerGroup = 8;
    private const int ScaleBytes = 8 * sizeof(ushort) * 2;

    /// <summary>Total repack bytes across all live layers; the layer falls back when it would exceed this.</summary>
    public static long MaxRepackBytes
    {
        get => WideRepackBudget.MaxBytes;
        set => WideRepackBudget.MaxBytes = value;
    }

    public static bool Enabled { get; set; } = true;

    /// <summary>
    /// Opt-in: run Q4_K through the generic byte kernel instead of the packed
    /// kernel. The byte repack is about 1.8x the packed copy (2.0x raw), so this
    /// trades memory for the wider kernel; the global budget still applies.
    /// Set it before the first forward; an already wide-only layer keeps its copy.
    /// </summary>
    public static bool UseForQ4K { get; set; }

    /// <summary>Diagnostic A/B knob; 0 uses the production chunk heuristic.</summary>
    public static int ParallelChunkOverride { get; set; }

    private readonly byte[] _data;
    private readonly byte* _p;
    private readonly WideByteOps _ops;
    private readonly int _groups;
    private readonly int _blocks;
    private readonly int _subBlocks;
    private readonly int _valuesPerSub;
    private readonly long _repackBytes;
    private readonly WideMatMulFn _matMul;
    private int _disposed;

    private delegate void WideMatMulFn(
        byte* data, int from, int to, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n);

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public QuantDType Dtype { get; }

    public static bool IsSupported(QuantDType dtype) => Describe(dtype, out _, out _, out _, out _);

    /// <summary>
    /// True when the generic byte kernel may handle this dtype. Q4_K is opt-in
    /// because its byte repack is about 1.8x the packed copy.
    /// </summary>
    public static bool Allowed(QuantDType dtype) =>
        Enabled && IsSupported(dtype) && (UseForQ4K || !IsQ4K(dtype));

    private static bool IsQ4K(QuantDType dtype) =>
        dtype is QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M;

    public static WideByteWeights? TryCreate(byte[]? raw, int inFeatures, int outFeatures, QuantDType dtype)
    {
        if (raw is null || !Enabled || !Describe(dtype, out int blockSize, out int rawBlockBytes,
                out int valuesPerSub, out int subBlocks))
            return null;

        if (inFeatures % blockSize != 0 || outFeatures % RowsPerGroup != 0)
            return null;

        int blocks = inFeatures / blockSize;
        long expected = (long)outFeatures * blocks * rawBlockBytes;
        if (expected != raw.Length)
            return null;

        long repackBytes = (long)(outFeatures / RowsPerGroup) * blocks * subBlocks
            * (ScaleBytes + RowsPerGroup * valuesPerSub);
        if (repackBytes > Array.MaxLength || !WideRepackBudget.TryReserve(repackBytes))
            return null;

        try
        {
            return new WideByteWeights(raw, inFeatures, outFeatures, dtype,
                blocks, subBlocks, valuesPerSub, repackBytes);
        }
        catch
        {
            WideRepackBudget.Release(repackBytes);
            throw;
        }
    }

    private WideByteWeights(byte[] raw, int inFeatures, int outFeatures, QuantDType dtype,
        int blocks, int subBlocks, int valuesPerSub, long repackBytes)
    {
        _repackBytes = repackBytes;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        Dtype = dtype;
        _blocks = blocks;
        _subBlocks = subBlocks;
        _valuesPerSub = valuesPerSub;
        _groups = outFeatures / RowsPerGroup;
        _ops = WideByteFactory.Create();
        _matMul = valuesPerSub == 16 ? _ops.MatMul16 : _ops.MatMul32;

        int subBlockBytes = ScaleBytes + RowsPerGroup * valuesPerSub;
        _data = GC.AllocateUninitializedArray<byte>(
            _groups * blocks * subBlocks * subBlockBytes, pinned: true);
        _p = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_data));

        fixed (byte* pRaw = raw)
        {
            long src = (long)pRaw;
            long dst = (long)_p;
            Parallel.For(0, _groups, g =>
            {
                for (int b = 0; b < blocks; b++)
                {
                    byte* blockDst = (byte*)dst + ((long)g * blocks + b) * subBlocks * subBlockBytes;
                    RepackBlock((byte*)src, g, b, blocks, dtype, subBlocks, valuesPerSub, blockDst);
                }
            });
        }
    }

    private static void RepackBlock(byte* raw, int group, int block, int blocks,
        QuantDType dtype, int subBlocks, int valuesPerSub, byte* dst)
    {
        int subBlockBytes = ScaleBytes + RowsPerGroup * valuesPerSub;
        int rawBlockBytes = RawBlockBytes(dtype);
        float* rowScales = stackalloc float[RowsPerGroup];
        float* rowBiases = stackalloc float[RowsPerGroup];

        for (int row = 0; row < RowsPerGroup; row++)
        {
            byte* src = raw + ((long)(group * RowsPerGroup + row) * blocks + block) * rawBlockBytes;
            for (int s = 0; s < subBlocks; s++)
            {
                byte* sub = dst + s * subBlockBytes;
                ushort* scales = (ushort*)sub;
                ushort* biases = scales + RowsPerGroup;
                byte* values = sub + ScaleBytes;
                Decode(dtype, src, s, valuesPerSub, row, rowScales, rowBiases, values);
                scales[row] = BitConverter.HalfToUInt16Bits((Half)rowScales[row]);
                biases[row] = BitConverter.HalfToUInt16Bits((Half)rowBiases[row]);
            }
        }
    }

    private static void Decode(QuantDType dtype, byte* block, int sub, int valuesPerSub,
        int row, float* scales, float* biases, byte* values)
    {
        if (IsQ4K(dtype))
        {
            DecodeQ4K(block, sub, row, scales, biases, values);
            return;
        }

        if (valuesPerSub == 32 && sub == 0 &&
            dtype is QuantDType.Q4_0 or QuantDType.Q4_1 or QuantDType.Q5_0
                or QuantDType.Q5_1 or QuantDType.IQ4_NL)
        {
            switch (dtype)
            {
                case QuantDType.Q4_0:
                {
                    float d = Half(block);
                    scales[row] = d;
                    biases[row] = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int nib = i < 16 ? (block[2 + i] & 0x0F) : (block[2 + i - 16] >> 4);
                        Write(values, row, i, nib - 8);
                    }
                    return;
                }
                case QuantDType.Q4_1:
                {
                    float d = Half(block);
                    float m = Half(block + 2);
                    scales[row] = d;
                    biases[row] = -m;
                    for (int i = 0; i < 32; i++)
                    {
                        int nib = i < 16 ? (block[4 + i] & 0x0F) : (block[4 + i - 16] >> 4);
                        Write(values, row, i, nib);
                    }
                    return;
                }
                case QuantDType.Q5_0:
                {
                    float d = Half(block);
                    scales[row] = d;
                    biases[row] = 0;
                    uint qh = Unsafe.ReadUnaligned<uint>(block + 2);
                    for (int i = 0; i < 32; i++)
                    {
                        int nib = i < 16 ? (block[6 + i] & 0x0F) : (block[6 + i - 16] >> 4);
                        int q = nib | (int)((qh >> i) & 1) << 4;
                        Write(values, row, i, q - 16);
                    }
                    return;
                }
                case QuantDType.Q5_1:
                {
                    float d = Half(block);
                    float m = Half(block + 2);
                    scales[row] = d;
                    biases[row] = -m;
                    uint qh = Unsafe.ReadUnaligned<uint>(block + 4);
                    for (int i = 0; i < 32; i++)
                    {
                        int nib = i < 16 ? (block[8 + i] & 0x0F) : (block[8 + i - 16] >> 4);
                        int q = nib | (int)((qh >> i) & 1) << 4;
                        Write(values, row, i, q);
                    }
                    return;
                }
                case QuantDType.IQ4_NL:
                {
                    float d = Half(block);
                    scales[row] = d;
                    biases[row] = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int nib = i < 16 ? (block[2 + i] & 0x0F) : (block[2 + i - 16] >> 4);
                        Write(values, row, i, Iq4Nl[nib]);
                    }
                    return;
                }
            }
        }

        switch (dtype)
        {
            case QuantDType.Q5_K:
                DecodeQ5K(block, sub, row, scales, biases, values);
                return;
            case QuantDType.Q2_K:
                DecodeQ2K(block, sub, row, scales, biases, values);
                return;
            case QuantDType.Q3_K:
                DecodeQ3K(block, sub, row, scales, biases, values);
                return;
            case QuantDType.IQ4_XS:
                DecodeIq4Xs(block, sub, row, scales, biases, values);
                return;
            case QuantDType.IQ3_XXS:
                DecodeIq3Xxs(block, sub, row, scales, biases, values);
                return;
            case QuantDType.IQ2_XXS:
                DecodeIq2Xxs(block, sub, row, scales, biases, values);
                return;
            case QuantDType.IQ2_XS:
                DecodeIq2Xs(block, sub, row, scales, biases, values);
                return;
            case QuantDType.IQ2_S:
                DecodeIq2S(block, sub, row, scales, biases, values);
                return;
            case QuantDType.IQ3_S:
                DecodeIq3S(block, sub, row, scales, biases, values);
                return;
            case QuantDType.MXFP4:
                DecodeMxfp4(block, row, scales, biases, values);
                return;
        }

        throw new NotSupportedException($"{dtype} is not repacked yet.");
    }

    private static unsafe void DecodeQ4K(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        float dmin = Half(block + 2);
        byte* sc = block + 4;
        byte* qs = block + 16;
        int scale = QuantizationKernels.GetScaleMinK4_Scale_Scalar(sub, sc);
        int min = QuantizationKernels.GetScaleMinK4_Min_Scalar(sub, sc);
        scales[row] = d * scale;
        biases[row] = dmin * min;

        int chunk = sub / 2;
        int shift = (sub % 2) * 4;
        byte* chunkQs = qs + chunk * 32;
        for (int i = 0; i < 32; i++)
            Write(values, row, i, (chunkQs[i] >> shift) & 0x0F);
    }

    private static unsafe void DecodeQ5K(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        float dmin = Half(block + 2);
        byte* sc = block + 4;
        byte* qh = block + 16;
        byte* qs = block + 48;
        int scale = QuantizationKernels.GetScaleMinK4_Scale_Scalar(sub, sc);
        int min = QuantizationKernels.GetScaleMinK4_Min_Scalar(sub, sc);
        scales[row] = d * scale;
        biases[row] = dmin * min;

        int group = sub / 2;
        int half = sub % 2;
        for (int i = 0; i < 32; i++)
        {
            int high = ((qh[i] >> sub) & 1) << 4;
            int nib = half == 0 ? (qs[group * 32 + i] & 0x0F) : (qs[group * 32 + i] >> 4);
            Write(values, row, i, nib | high);
        }
    }

    private static unsafe void DecodeQ2K(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block + 80);
        float dmin = Half(block + 82);
        byte* scaleByte = block + sub;
        byte* qs = block + 16;
        scales[row] = d * (scaleByte[0] & 0x0F);
        biases[row] = dmin * (scaleByte[0] >> 4);

        int basePos = sub * 16;
        for (int i = 0; i < 16; i++)
        {
            int idx = basePos + i;
            int qb = (idx / 128) * 32 + (idx % 32);
            int shift = ((idx % 128) / 32) * 2;
            Write(values, row, i, (qs[qb] >> shift) & 3);
        }
    }

    private static unsafe void DecodeQ3K(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block + 108);
        byte* hmask = block;
        byte* qs = block + 32;

        byte* scaleBuf = stackalloc byte[16];
        const uint mask1 = 0x03030303u;
        const uint mask2 = 0x0f0f0f0fu;
        uint a0 = ReadU32(block + 96);
        uint a1 = ReadU32(block + 100);
        uint tmp = ReadU32(block + 104);
        *(uint*)scaleBuf = (a0 & mask2) | (((tmp >> 0) & mask1) << 4);
        *(uint*)(scaleBuf + 4) = (a1 & mask2) | (((tmp >> 2) & mask1) << 4);
        *(uint*)(scaleBuf + 8) = ((a0 >> 4) & mask2) | (((tmp >> 4) & mask1) << 4);
        *(uint*)(scaleBuf + 12) = ((a1 >> 4) & mask2) | (((tmp >> 6) & mask1) << 4);
        sbyte* sc8 = (sbyte*)scaleBuf;

        scales[row] = d * (sc8[sub] - 32);
        biases[row] = 0;

        int basePos = sub * 16;
        for (int i = 0; i < 16; i++)
        {
            int idx = basePos + i;
            int qb = (idx / 128) * 32 + (idx % 32);
            int shift = ((idx % 128) / 32) * 2;
            int q2 = (qs[qb] >> shift) & 3;
            int hBit = (hmask[idx % 32] >> (idx / 32)) & 1;
            Write(values, row, i, q2 - (hBit == 0 ? 4 : 0));
        }
    }

    private static unsafe void DecodeIq4Xs(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        ushort scalesH = ReadU16(block + 2);
        byte* scalesL = block + 4;
        byte* qs = block + 8;
        int scale = ((scalesL[sub / 2] >> (4 * (sub % 2))) & 0x0F)
            | (((scalesH >> (2 * sub)) & 0x03) << 4);
        scales[row] = d * (scale - 32);
        biases[row] = 0;

        byte* subQs = qs + sub * 16;
        for (int i = 0; i < 32; i++)
        {
            int nib = i < 16 ? (subQs[i] & 0x0F) : (subQs[i - 16] >> 4);
            Write(values, row, i, Iq4Nl[nib]);
        }
    }

    private static unsafe void DecodeIq3Xxs(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        byte* qs = block + 2;
        byte* scalesAndSigns = block + 66;
        uint word = ReadU32(scalesAndSigns + sub * 4);
        scales[row] = d * (0.5f + (word >> 28)) * 0.5f;
        biases[row] = 0;

        ReadOnlySpan<byte> grid = QuantizationGrids.IQ3XXS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;
        for (int l = 0; l < 4; l++)
        {
            int signs = signTable[(int)((word >> (7 * l)) & 127)];
            int g1 = qs[sub * 8 + 2 * l] * 4;
            int g2 = qs[sub * 8 + 2 * l + 1] * 4;
            for (int j = 0; j < 4; j++)
            {
                int lo = l * 8 + j;
                int hi = lo + 4;
                Write(values, row, lo, Signed(grid[g1 + j], signs, j));
                Write(values, row, hi, Signed(grid[g2 + j], signs, j + 4));
            }
        }
    }

    private static unsafe void DecodeIq2Xxs(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        byte* qs = block + 2;
        uint word = ReadU32(qs + sub * 8 + 4);
        scales[row] = d * (0.5f + (word >> 28)) * 0.25f;
        biases[row] = 0;

        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2XXS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;
        for (int l = 0; l < 4; l++)
        {
            int signs = signTable[(int)((word >> (7 * l)) & 127)];
            int offset = qs[sub * 8 + l] * 8;
            for (int j = 0; j < 8; j++)
                Write(values, row, l * 8 + j, Signed(grid[offset + j], signs, j));
        }
    }

    private static unsafe void DecodeIq2Xs(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        byte* qs = block + 2;
        byte* scaleBytes = block + 66;
        int ib = sub / 2;
        int half = sub % 2;
        int scale = half == 0 ? (scaleBytes[ib] & 0x0F) : (scaleBytes[ib] >> 4);
        scales[row] = d * (0.5f + scale) * 0.25f;
        biases[row] = 0;

        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2XS_Grid;
        ReadOnlySpan<byte> signTable = QuantizationGrids.KSignsIQ2XS;
        for (int i = 0; i < 16; i++)
        {
            int global = sub * 16 + i;
            int l = (global % 32) / 8;
            int j = global % 8;
            int entry = ReadU16(qs + (ib * 4 + l) * 2);
            int signs = signTable[entry >> 9];
            int offset = (entry & 511) * 8;
            Write(values, row, i, Signed(grid[offset + j], signs, j));
        }
    }

    private static unsafe void DecodeIq2S(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        byte* qs = block + 2;
        byte* qh = block + 66;
        byte* scaleBytes = block + 74;
        int ib = sub / 2;
        int half = sub % 2;
        int scale = half == 0 ? (scaleBytes[ib] & 0x0F) : (scaleBytes[ib] >> 4);
        scales[row] = d * (0.5f + scale) * 0.25f;
        biases[row] = 0;

        ReadOnlySpan<byte> grid = QuantizationGrids.IQ2S_Grid;
        for (int i = 0; i < 16; i++)
        {
            int global = sub * 16 + i;
            int l = (global % 32) / 8;
            int j = global % 8;
            int index = qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
            int signs = qs[32 + ib * 4 + l];
            int offset = index * 8;
            Write(values, row, i, Signed(grid[offset + j], signs, j));
        }
    }

    private static unsafe void DecodeIq3S(byte* block, int sub, int row,
        float* scales, float* biases, byte* values)
    {
        float d = Half(block);
        byte* qs = block + 2;
        byte* qh = block + 66;
        byte* signs = block + 74;
        byte* scaleBytes = block + 106;
        byte scale = scaleBytes[sub / 2];
        int scaleNibble = (sub % 2) == 0 ? (scale & 0x0F) : (scale >> 4);
        scales[row] = d * (1 + 2 * scaleNibble);
        biases[row] = 0;

        ReadOnlySpan<byte> grid = QuantizationGrids.IQ3S_Grid;
        for (int l = 0; l < 4; l++)
        {
            int signByte = signs[sub * 4 + l];
            int g1 = (qs[sub * 8 + 2 * l] | ((qh[sub] << (8 - 2 * l)) & 256)) * 4;
            int g2 = (qs[sub * 8 + 2 * l + 1] | ((qh[sub] << (7 - 2 * l)) & 256)) * 4;
            for (int j = 0; j < 4; j++)
            {
                int lo = l * 8 + j;
                int hi = lo + 4;
                Write(values, row, lo, Signed(grid[g1 + j], signByte, j));
                Write(values, row, hi, Signed(grid[g2 + j], signByte, j + 4));
            }
        }
    }

    private static unsafe void DecodeMxfp4(byte* block, int row,
        float* scales, float* biases, byte* values)
    {
        float d = QuantizationKernels.E8M0ToFloat_Scalar(block[0]);
        scales[row] = d * 0.5f;
        biases[row] = 0;
        byte* qs = block + 1;
        for (int i = 0; i < 32; i++)
        {
            int nib = i < 16 ? (qs[i] & 0x0F) : (qs[i - 16] >> 4);
            Write(values, row, i, Mxfp4[nib]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Signed(int value, int signs, int bit)
        => (signs & (1 << bit)) == 0 ? value : -value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe uint ReadU32(byte* p)
        => (uint)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe ushort ReadU16(byte* p)
        => (ushort)(p[0] | (p[1] << 8));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Write(byte* values, int row, int index, int value)
        => values[(index / 4) * 32 + row * 4 + (index % 4)] = unchecked((byte)(sbyte)value);

    private static readonly sbyte[] Iq4Nl =
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    private static readonly int[] Mxfp4 =
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    private static float Half(byte* p)
        => (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(p));

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        WideRepackBudget.Release(_repackBytes);
    }

    private static int RawBlockBytes(QuantDType dtype) => dtype switch
    {
        QuantDType.Q4_0 => 18,
        QuantDType.Q4_1 => 20,
        QuantDType.Q5_0 => 22,
        QuantDType.Q5_1 => 24,
        QuantDType.IQ4_NL => 18,
        QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => 144,
        QuantDType.Q5_K => 176,
        QuantDType.Q2_K => 84,
        QuantDType.Q3_K => 110,
        QuantDType.IQ4_XS => 136,
        QuantDType.IQ3_XXS => 98,
        QuantDType.IQ2_XXS => 66,
        QuantDType.IQ2_XS => 74,
        QuantDType.IQ2_S => 82,
        QuantDType.IQ3_S => 110,
        QuantDType.MXFP4 => 17,
        _ => throw new NotSupportedException(dtype.ToString()),
    };

    private static bool Describe(QuantDType dtype, out int blockSize, out int rawBlockBytes,
        out int valuesPerSub, out int subBlocks)
    {
        (blockSize, valuesPerSub, subBlocks) = dtype switch
        {
            QuantDType.Q4_0 or QuantDType.Q4_1 or QuantDType.Q5_0 or QuantDType.Q5_1
                or QuantDType.IQ4_NL or QuantDType.MXFP4 => (32, 32, 1),
            QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => (256, 32, 8),
            QuantDType.Q2_K or QuantDType.Q3_K
                or QuantDType.IQ2_XS or QuantDType.IQ2_S => (256, 16, 16),
            QuantDType.Q5_K or QuantDType.IQ4_XS or QuantDType.IQ3_XXS
                or QuantDType.IQ2_XXS or QuantDType.IQ3_S => (256, 32, 8),
            _ => (0, 0, 0),
        };

        if (blockSize == 0)
        {
            rawBlockBytes = 0;
            return false;
        }

        rawBlockBytes = RawBlockBytes(dtype);
        return true;
    }

    public void MatMul(float* input, float* output, int m)
    {
        int k = InFeatures, n = OutFeatures;
        int blocks32 = k / 32;
        int blocks16 = k / 16;
        int qBytes = checked(m * k);
        int dBytes = checked(m * blocks32 * sizeof(float));
        int s16Bytes = checked(m * blocks16 * sizeof(int));
        int s32Bytes = checked(m * blocks32 * sizeof(int));
        int scratchBytes = checked(qBytes + dBytes + s16Bytes + s32Bytes + 16);

        byte[] rented = ArrayPool<byte>.Shared.Rent(scratchBytes);
        try
        {
            fixed (byte* scratch = rented)
            {
                byte* p = scratch;
                sbyte* q = (sbyte*)p;
                p += qBytes;
                float* d32 = (float*)p;
                p += dBytes;
                int* s16 = (int*)p;
                p += s16Bytes;
                int* s32 = (int*)p;

                if (m >= 16)
                {
                    long pin = (long)input, pq = (long)q, pd = (long)d32,
                        p16 = (long)s16, p32 = (long)s32;
                    Parallel.For(0, m, row =>
                        QuantizeRow((float*)pin + (long)row * k, k,
                            (sbyte*)pq + (long)row * k,
                            (float*)pd + (long)row * blocks32,
                            (int*)p16 + (long)row * blocks16,
                            (int*)p32 + (long)row * blocks32));
                }
                else
                {
                    for (int row = 0; row < m; row++)
                        QuantizeRow(input + (long)row * k, k, q + (long)row * k,
                            d32 + (long)row * blocks32,
                            s16 + (long)row * blocks16,
                            s32 + (long)row * blocks32);
                }

                int chunks = ParallelChunkOverride > 0
                    ? Math.Min(_groups, ParallelChunkOverride)
                    : Math.Min(_groups, Environment.ProcessorCount * 2);
                if (chunks <= 1)
                {
                    Dispatch(0, _groups, q, d32, s16, s32, output, m, n);
                }
                else
                {
                    int perChunk = (_groups + chunks - 1) / chunks;
                    chunks = (_groups + perChunk - 1) / perChunk;
                    long pw = (long)_p, pq = (long)q, pd = (long)d32,
                        po = (long)output;
                    long p16 = (long)s16, p32 = (long)s32;
                    Parallel.For(0, chunks, c =>
                        Dispatch((byte*)pw, c * perChunk,
                            Math.Min(_groups, (c + 1) * perChunk),
                            (sbyte*)pq, (float*)pd, (int*)p16, (int*)p32,
                            (float*)po, m, n));
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private void Dispatch(int from, int to, sbyte* q, float* d32, int* s16, int* s32,
        float* output, int m, int n)
        => Dispatch(_p, from, to, q, d32, s16, s32, output, m, n);

    private void Dispatch(byte* data, int from, int to, sbyte* q, float* d32, int* s16,
        int* s32, float* output, int m, int n)
        => _matMul(data, from, to, _blocks, _subBlocks, q, d32, s16, s32, output, m, n);

    private static void QuantizeRow(float* x, int inFeatures, sbyte* q, float* d32,
        int* s16, int* s32)
    {
        int blocks32 = inFeatures / 32;
        for (int b = 0; b < blocks32; b++)
        {
            int at = b * 32;
            float amax = 0;
            for (int i = 0; i < 32; i++)
                amax = MathF.Max(amax, MathF.Abs(x[at + i]));

            float d = amax / 127f;
            float id = amax >= 1e-30f ? 127f / amax : 0f;
            d32[b] = d;

            int sum0 = 0, sum1 = 0;
            for (int i = 0; i < 32; i++)
            {
                int v = (int)MathF.Round(x[at + i] * id, MidpointRounding.ToEven);
                q[at + i] = (sbyte)v;
                if (i < 16) sum0 += v;
                else sum1 += v;
            }
            s16[b * 2] = sum0;
            s16[b * 2 + 1] = sum1;
            s32[b] = sum0 + sum1;
        }
    }
}

/// <summary>Assembles <see cref="WideByteOps"/> once per mapping, like the quant factory.</summary>
public static class WideByteFactory
{
    private static readonly Lazy<WideByteOps> Auto = new(() =>
    {
        var mapping = new MappingBuilder(HardwareTierHelpers.DetectBestTier())
            .ApplyQuantPreset()
            .Build();
        return Assembler.CreateInstance<WideByteOps>(mapping);
    });

    public static WideByteOps Create() => Auto.Value;
}
