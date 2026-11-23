using JigSawDotNet;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Integer matmul over a repacked byte stream. Every block format is normalised
/// during repack to eight output rows, a float scale and bias per sub-block, and
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
    private const int ScaleBytes = 8 * sizeof(float) * 2;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatMul16_Fma(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n)
        => MatMul16Fma(data, fromGroup, toGroup, blocks, subBlocks,
            activation, activationScale, activationSum16, output, m, n);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatMul32_Fma(
        byte* data, int fromGroup, int toGroup, int blocks, int subBlocks,
        sbyte* activation, float* activationScale, int* activationSum16, int* activationSum32,
        float* output, int m, int n)
        => MatMul32Fma(data, fromGroup, toGroup, blocks, subBlocks,
            activation, activationScale, activationSum32, output, m, n);

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
                        float* scales = (float*)sub;
                        float* biases = scales + 8;
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

                        var scale = Avx.Multiply(Avx.LoadVector256(scales), Vector256.Create(d));
                        var bias = Avx.Multiply(Avx.LoadVector256(biases),
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
                        float* scales = (float*)sub;
                        float* biases = scales + 8;
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

                        var scale = Avx.Multiply(Avx.LoadVector256(scales), Vector256.Create(d));
                        var bias = Avx.Multiply(Avx.LoadVector256(biases),
                            Vector256.Create(d * sum));
                        acc = Fma.MultiplyAdd(scale, Avx.ConvertToVector256Single(dot), acc);
                        acc = Avx.Subtract(acc, bias);
                    }
                }

                Avx.Store(outRow, acc);
            }
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
                            float* scales = (float*)sub;
                            float* biases = scales + 8;
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
                            acc += scales[row] * d * dot - biases[row] * d * sum;
                        }
                    }
                    outRow[row] = acc;
                }
            }
        }
    }
}
