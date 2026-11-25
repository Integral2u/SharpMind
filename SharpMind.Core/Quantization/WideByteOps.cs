using JigSawDotNet;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Integer matmul over a repacked byte stream. Every block format is normalised
/// during repack to eight output rows, a Half scale and bias per sub-block, and
/// signed int8 values. The hot loop therefore has no format branch, no ISA
/// check, and no sub-block-size branch: JigSawDotNet selects the FMA/scalar body
/// once, and <see cref="MatMul16"/> or <see cref="MatMul32"/> is chosen once when
/// the repack is built.
/// </summary>
public abstract unsafe class WideByteOps
{
    private const string NS =
        $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Quantization)}.{nameof(WideByteKernels)}";

    [PuzzleCornerPiece(QuantizationKeys.KeyWideByte16, true, null,
        "widebyte16_fma", $"{NS}.{nameof(WideByteKernels.MatMul16_Fma)}",
        "widebyte16_avx2", $"{NS}.{nameof(WideByteKernels.MatMul16_Scalar)}",
        "widebyte16_sse", $"{NS}.{nameof(WideByteKernels.MatMul16_Scalar)}",
        "widebyte16_scalar", $"{NS}.{nameof(WideByteKernels.MatMul16_Scalar)}",
        "widebyte16_arm64", $"{NS}.{nameof(WideByteKernels.MatMul16_Scalar)}")]
    public abstract void MatMul16(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyWideByte32, true, null,
        "widebyte32_fma", $"{NS}.{nameof(WideByteKernels.MatMul32_Fma)}",
        "widebyte32_avx2", $"{NS}.{nameof(WideByteKernels.MatMul32_Scalar)}",
        "widebyte32_sse", $"{NS}.{nameof(WideByteKernels.MatMul32_Scalar)}",
        "widebyte32_scalar", $"{NS}.{nameof(WideByteKernels.MatMul32_Scalar)}",
        "widebyte32_arm64", $"{NS}.{nameof(WideByteKernels.MatMul32_Scalar)}")]
    public abstract void MatMul32(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n);
}

/// <summary>Concrete FMA and scalar bodies selected by <see cref="WideByteOps"/>.</summary>
public static unsafe class WideByteKernels
{
    private const int ScaleBytes = 8 * sizeof(ushort) * 2;

    /// <summary>Diagnostic A/B knob for the two-group microkernel; production keeps it on.</summary>
    public static bool UsePairKernel { get; set; } = true;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> LoadScales(byte* p)
        => QuantizationKernels.WidenHalf8((ushort*)p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HalfToFloat(ushort value)
        => QuantizationKernels.HalfToFloat_F16C(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatMul16_Fma(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n)
    {
        const int V = 16;
        long groupBytes = (long)blocks * subBlocks * (ScaleBytes + 8 * V);
        if (!UsePairKernel)
        {
            MatMul16Fma(data, fromGroup, toGroup, blocks, subBlocks,
                activation, activationScale, activationSum16, output, m, n);
            return;
        }

        int g = fromGroup;
        for (; g + 1 < toGroup; g += 2)
            MatMul16FmaPair(data + (long)g * groupBytes, data + (long)(g + 1) * groupBytes,
                blocks, subBlocks, activation, activationScale, activationSum16,
                output + (long)g * 8, m, n);
        if (g < toGroup)
            MatMul16Fma(data + (long)g * groupBytes, 0, 1, blocks, subBlocks,
                activation, activationScale, activationSum16, output + (long)g * 8, m, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatMul32_Fma(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n)
    {
        const int V = 32;
        long groupBytes = (long)blocks * subBlocks * (ScaleBytes + 8 * V);
        if (!UsePairKernel)
        {
            MatMul32Fma(data, fromGroup, toGroup, blocks, subBlocks,
                activation, activationScale, activationSum32, output, m, n);
            return;
        }

        int g = fromGroup;
        for (; g + 1 < toGroup; g += 2)
            MatMul32FmaPair(data + (long)g * groupBytes, data + (long)(g + 1) * groupBytes,
                blocks, subBlocks, activation, activationScale, activationSum32,
                output + (long)g * 8, m, n);
        if (g < toGroup)
            MatMul32Fma(data + (long)g * groupBytes, 0, 1, blocks, subBlocks,
                activation, activationScale, activationSum32, output + (long)g * 8, m, n);
    }

    public static void MatMul16_Scalar(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n)
        => MatMulScalar(data, fromGroup, toGroup, blocks, subBlocks,
            valuesPerSub: 16, activation, activationScale, activationSum16, output, m, n);

    public static void MatMul32_Scalar(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n)
        => MatMulScalar(data, fromGroup, toGroup, blocks, subBlocks,
            valuesPerSub: 32, activation, activationScale, activationSum32, output, m, n);

    private static unsafe void MatMul16Fma(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum,
        float* output, int m, int n)
    {
        const int V = 16;
        const int SubBlockBytes = ScaleBytes + 8 * V;
        int blockValues = subBlocks * V;
        long groupBytes = (long)blocks * subBlocks * SubBlockBytes;
        var ones = Vector256.Create((short)1);

        for (int g = fromGroup; g < toGroup; g++)
        {
            byte* groupData = data + (long)g * groupBytes;
            for (int p = 0; p < m; p++)
            {
                sbyte* act = activation + (long)p * blocks * blockValues;
                float* dq = activationScale + (long)p * blocks * (blockValues / 32);
                int* sums = activationSum + (long)p * blocks * (blockValues / 16);
                float* outRow = output + (long)p * n + g * 8;
                var acc = Vector256<float>.Zero;

                for (int b = 0; b < blocks; b++)
                {
                    byte* blockData = groupData + (long)b * subBlocks * SubBlockBytes;
                    sbyte* blockAct = act + (long)b * blockValues;
                    float* blockDq = dq + (long)b * (blockValues / 32);
                    int* blockSum = sums + (long)b * (blockValues / 16);

                    for (int s = 0; s < subBlocks; s++)
                    {
                        byte* sub = blockData + (long)s * SubBlockBytes;
                        ushort* scales = (ushort*)sub;
                        ushort* biases = scales + 8;
                        byte* values = sub + ScaleBytes;
                        sbyte* subAct = blockAct + (long)s * V;
                        float d = blockDq[s >> 1];
                        int sum = blockSum[s];

                        var dot = Vector256<int>.Zero;
                        for (int t = 0; t < V; t += 4)
                        {
                            var v = Avx.LoadVector256(values + t * 8).AsSByte();
                            var vq = Vector256.Create(
                                Unsafe.ReadUnaligned<int>(subAct + t)).AsByte().AsSByte();
                            dot = Avx2.Add(dot, Avx2.MultiplyAddAdjacent(
                                Avx2.MultiplyAddAdjacent(Avx2.Abs(v), Avx2.Sign(vq, v)), ones));
                        }

                        var scale = Avx.Multiply(LoadScales((byte*)scales), Vector256.Create(d));
                        var bias = Avx.Multiply(LoadScales((byte*)biases),
                            Vector256.Create(d * sum));
                        acc = Fma.MultiplyAdd(scale, Avx.ConvertToVector256Single(dot), acc);
                        acc = Avx.Subtract(acc, bias);
                    }
                }

                Avx.Store(outRow, acc);
            }
        }
    }

    private static unsafe void MatMul32Fma(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum,
        float* output, int m, int n)
    {
        const int V = 32;
        const int SubBlockBytes = ScaleBytes + 8 * V;
        int blockValues = subBlocks * V;
        long groupBytes = (long)blocks * subBlocks * SubBlockBytes;
        var ones = Vector256.Create((short)1);

        for (int g = fromGroup; g < toGroup; g++)
        {
            byte* groupData = data + (long)g * groupBytes;
            for (int p = 0; p < m; p++)
            {
                sbyte* act = activation + (long)p * blocks * blockValues;
                float* dq = activationScale + (long)p * blocks * (blockValues / 32);
                int* sums = activationSum + (long)p * blocks * (blockValues / 32);
                float* outRow = output + (long)p * n + g * 8;
                var acc = Vector256<float>.Zero;

                for (int b = 0; b < blocks; b++)
                {
                    byte* blockData = groupData + (long)b * subBlocks * SubBlockBytes;
                    sbyte* blockAct = act + (long)b * blockValues;
                    float* blockDq = dq + (long)b * (blockValues / 32);
                    int* blockSum = sums + (long)b * (blockValues / 32);

                    for (int s = 0; s < subBlocks; s++)
                    {
                        byte* sub = blockData + (long)s * SubBlockBytes;
                        ushort* scales = (ushort*)sub;
                        ushort* biases = scales + 8;
                        byte* values = sub + ScaleBytes;
                        sbyte* subAct = blockAct + (long)s * V;
                        float d = blockDq[s];
                        int sum = blockSum[s];

                        var dot = Vector256<int>.Zero;
                        for (int t = 0; t < V; t += 4)
                        {
                            var v = Avx.LoadVector256(values + t * 8).AsSByte();
                            var vq = Vector256.Create(
                                Unsafe.ReadUnaligned<int>(subAct + t)).AsByte().AsSByte();
                            dot = Avx2.Add(dot, Avx2.MultiplyAddAdjacent(
                                Avx2.MultiplyAddAdjacent(Avx2.Abs(v), Avx2.Sign(vq, v)), ones));
                        }

                        var scale = Avx.Multiply(LoadScales((byte*)scales), Vector256.Create(d));
                        var bias = Avx.Multiply(LoadScales((byte*)biases),
                            Vector256.Create(d * sum));
                        acc = Fma.MultiplyAdd(scale, Avx.ConvertToVector256Single(dot), acc);
                        acc = Avx.Subtract(acc, bias);
                    }
                }

                Avx.Store(outRow, acc);
            }
        }
    }

    /// <summary>
    /// Two row groups against one activation block for the 16-value layout. One
    /// accumulator per group shares the activation decode and the d/sum lookups;
    /// each row keeps the same sub-block accumulation order as the single path.
    /// </summary>
    private static unsafe void MatMul16FmaPair(
        byte* group0, byte* group1, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum,
        float* output, int m, int n)
    {
        const int V = 16;
        const int SubBlockBytes = ScaleBytes + 8 * V;
        int blockValues = subBlocks * V;
        var ones = Vector256.Create((short)1);

        for (int p = 0; p < m; p++)
        {
            sbyte* act = activation + (long)p * blocks * blockValues;
            float* dq = activationScale + (long)p * blocks * (blockValues / 32);
            int* sums = activationSum + (long)p * blocks * (blockValues / 16);
            float* out0 = output + (long)p * n;
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;

            for (int b = 0; b < blocks; b++)
            {
                byte* block0 = group0 + (long)b * subBlocks * SubBlockBytes;
                byte* block1 = group1 + (long)b * subBlocks * SubBlockBytes;
                sbyte* blockAct = act + (long)b * blockValues;
                float* blockDq = dq + (long)b * (blockValues / 32);
                int* blockSum = sums + (long)b * (blockValues / 16);

                for (int s = 0; s < subBlocks; s++)
                {
                    byte* sub0 = block0 + (long)s * SubBlockBytes;
                    byte* sub1 = block1 + (long)s * SubBlockBytes;
                    ushort* scales0 = (ushort*)sub0;
                    ushort* biases0 = scales0 + 8;
                    ushort* scales1 = (ushort*)sub1;
                    ushort* biases1 = scales1 + 8;
                    byte* values0 = sub0 + ScaleBytes;
                    byte* values1 = sub1 + ScaleBytes;
                    sbyte* subAct = blockAct + (long)s * V;
                    float d = blockDq[s >> 1];
                    int sum = blockSum[s];

                    var dot0 = Vector256<int>.Zero;
                    var dot1 = Vector256<int>.Zero;
                    for (int t = 0; t < V; t += 4)
                    {
                        var vq = Vector256.Create(
                            Unsafe.ReadUnaligned<int>(subAct + t)).AsByte().AsSByte();
                        var v0 = Avx.LoadVector256(values0 + t * 8).AsSByte();
                        var v1 = Avx.LoadVector256(values1 + t * 8).AsSByte();
                        dot0 = Avx2.Add(dot0, Avx2.MultiplyAddAdjacent(
                            Avx2.MultiplyAddAdjacent(Avx2.Abs(v0), Avx2.Sign(vq, v0)), ones));
                        dot1 = Avx2.Add(dot1, Avx2.MultiplyAddAdjacent(
                            Avx2.MultiplyAddAdjacent(Avx2.Abs(v1), Avx2.Sign(vq, v1)), ones));
                    }

                    var scale0 = Avx.Multiply(LoadScales((byte*)scales0), Vector256.Create(d));
                    var scale1 = Avx.Multiply(LoadScales((byte*)scales1), Vector256.Create(d));
                    var bias0 = Avx.Multiply(LoadScales((byte*)biases0), Vector256.Create(d * sum));
                    var bias1 = Avx.Multiply(LoadScales((byte*)biases1), Vector256.Create(d * sum));
                    acc0 = Fma.MultiplyAdd(scale0, Avx.ConvertToVector256Single(dot0), acc0);
                    acc1 = Fma.MultiplyAdd(scale1, Avx.ConvertToVector256Single(dot1), acc1);
                    acc0 = Avx.Subtract(acc0, bias0);
                    acc1 = Avx.Subtract(acc1, bias1);
                }
            }

            Avx.Store(out0, acc0);
            Avx.Store(out0 + 8, acc1);
        }
    }

    /// <summary>Two row groups for the 32-value layout; same structure as the 16-value pair.</summary>
    private static unsafe void MatMul32FmaPair(
        byte* group0, byte* group1, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum,
        float* output, int m, int n)
    {
        const int V = 32;
        const int SubBlockBytes = ScaleBytes + 8 * V;
        int blockValues = subBlocks * V;
        var ones = Vector256.Create((short)1);

        for (int p = 0; p < m; p++)
        {
            sbyte* act = activation + (long)p * blocks * blockValues;
            float* dq = activationScale + (long)p * blocks * (blockValues / 32);
            int* sums = activationSum + (long)p * blocks * (blockValues / 32);
            float* out0 = output + (long)p * n;
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;

            for (int b = 0; b < blocks; b++)
            {
                byte* block0 = group0 + (long)b * subBlocks * SubBlockBytes;
                byte* block1 = group1 + (long)b * subBlocks * SubBlockBytes;
                sbyte* blockAct = act + (long)b * blockValues;
                float* blockDq = dq + (long)b * (blockValues / 32);
                int* blockSum = sums + (long)b * (blockValues / 32);

                for (int s = 0; s < subBlocks; s++)
                {
                    byte* sub0 = block0 + (long)s * SubBlockBytes;
                    byte* sub1 = block1 + (long)s * SubBlockBytes;
                    ushort* scales0 = (ushort*)sub0;
                    ushort* biases0 = scales0 + 8;
                    ushort* scales1 = (ushort*)sub1;
                    ushort* biases1 = scales1 + 8;
                    byte* values0 = sub0 + ScaleBytes;
                    byte* values1 = sub1 + ScaleBytes;
                    sbyte* subAct = blockAct + (long)s * V;
                    float d = blockDq[s];
                    int sum = blockSum[s];

                    var dot0 = Vector256<int>.Zero;
                    var dot1 = Vector256<int>.Zero;
                    for (int t = 0; t < V; t += 4)
                    {
                        var vq = Vector256.Create(
                            Unsafe.ReadUnaligned<int>(subAct + t)).AsByte().AsSByte();
                        var v0 = Avx.LoadVector256(values0 + t * 8).AsSByte();
                        var v1 = Avx.LoadVector256(values1 + t * 8).AsSByte();
                        dot0 = Avx2.Add(dot0, Avx2.MultiplyAddAdjacent(
                            Avx2.MultiplyAddAdjacent(Avx2.Abs(v0), Avx2.Sign(vq, v0)), ones));
                        dot1 = Avx2.Add(dot1, Avx2.MultiplyAddAdjacent(
                            Avx2.MultiplyAddAdjacent(Avx2.Abs(v1), Avx2.Sign(vq, v1)), ones));
                    }

                    var scale0 = Avx.Multiply(LoadScales((byte*)scales0), Vector256.Create(d));
                    var scale1 = Avx.Multiply(LoadScales((byte*)scales1), Vector256.Create(d));
                    var bias0 = Avx.Multiply(LoadScales((byte*)biases0), Vector256.Create(d * sum));
                    var bias1 = Avx.Multiply(LoadScales((byte*)biases1), Vector256.Create(d * sum));
                    acc0 = Fma.MultiplyAdd(scale0, Avx.ConvertToVector256Single(dot0), acc0);
                    acc1 = Fma.MultiplyAdd(scale1, Avx.ConvertToVector256Single(dot1), acc1);
                    acc0 = Avx.Subtract(acc0, bias0);
                    acc1 = Avx.Subtract(acc1, bias1);
                }
            }

            Avx.Store(out0, acc0);
            Avx.Store(out0 + 8, acc1);
        }
    }

    private static unsafe void MatMulScalar(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks, int valuesPerSub,
        sbyte* activation, float* activationScale, int* activationSum,
        float* output, int m, int n)
    {
        int subBlockBytes = ScaleBytes + 8 * valuesPerSub;
        int blockValues = subBlocks * valuesPerSub;
        int scaleStride = blockValues / 32;
        int sumStride = subBlocks;
        int subScaleShift = valuesPerSub == 16 ? 1 : 0;
        long groupBytes = (long)blocks * subBlocks * subBlockBytes;

        for (int g = fromGroup; g < toGroup; g++)
        {
            byte* groupData = data + (long)g * groupBytes;
            for (int p = 0; p < m; p++)
            {
                sbyte* act = activation + (long)p * blocks * blockValues;
                float* dq = activationScale + (long)p * blocks * scaleStride;
                int* sums = activationSum + (long)p * blocks * sumStride;
                float* outRow = output + (long)p * n + g * 8;

                for (int row = 0; row < 8; row++)
                {
                    float acc = 0;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* blockData = groupData + (long)b * subBlocks * subBlockBytes;
                        sbyte* blockAct = act + (long)b * blockValues;
                        float* blockDq = dq + (long)b * scaleStride;
                        int* blockSum = sums + (long)b * sumStride;

                        for (int s = 0; s < subBlocks; s++)
                        {
                            byte* sub = blockData + (long)s * subBlockBytes;
                            ushort* scales = (ushort*)sub;
                            ushort* biases = scales + 8;
                            byte* values = sub + ScaleBytes;
                            sbyte* subAct = blockAct + (long)s * valuesPerSub;
                            float d = blockDq[s >> subScaleShift];
                            int sum = blockSum[s];

                            int dot = 0;
                            for (int t = 0; t < valuesPerSub; t++)
                            {
                                int at = (t / 4) * 32 + row * 4 + (t % 4);
                                dot += (sbyte)values[at] * subAct[t];
                            }
                            acc += HalfToFloat(scales[row]) * d * dot
                                - HalfToFloat(biases[row]) * d * sum;
                        }
                    }
                    outRow[row] = acc;
                }
            }
        }
    }
}
