using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.GPU;
using Xunit;

namespace SharpMind.Tests.GPU;

/// <summary>
/// On-device Q8_0 dequant matmul/gather oracle tests. The contract is that the OpenCL/CPU
/// accelerator (gfx902 here, ILGPU CPU in CI) reproduces the CPU scalar semantics on identical
/// raw GGUF Q8_0 bytes — <see cref="VecDotQ8_0_Scalar"/> for the fused matmul, a block decode
/// for the embedding gather. These are the kernels <see cref="GpuInferenceEngine"/> uses to run
/// a real quantized-resident model (e.g. qwen2-0.5b-q8_0), where the block linears, the
/// embedding gather, and the weight-tied LM head all read the same [N, K] Q8_0 layout.
/// </summary>
[Collection("GPU")]
public sealed class GpuQuantMatmulTests
{
    private const int BlockBytes = 34;   // Q8_0: f16 scale + 32 sbyteds
    private const int QK = 32;

    private static GpuDevice Dev => GpuTestDevice.Device;

    /// <summary>Reference decode of one [row, cols] Q8_0 cell, mirroring VecDotQ8_0_Scalar's layout:
    /// raw[row][block over cols], block = f16 LE scale + 32 sbyte values.</summary>
    private static float DecodeCell(byte[] raw, int row, int col, int cols)
    {
        int nBlocks = (cols + QK - 1) / QK;
        int baseIdx = row * nBlocks * BlockBytes + (col / QK) * BlockBytes;
        float scale = HalfToFloat((ushort)(raw[baseIdx] | (raw[baseIdx + 1] << 8)));
        byte v = raw[baseIdx + 2 + (col % QK)];
        return (v < 128 ? (int)v : v - 256) * scale;
    }

    private static float HalfToFloat(ushort half)
    {
        int exp5 = (half >> 10) & 0x1F;
        float sign = (half & 0x8000) != 0 ? -1f : 1f;
        if (exp5 == 0) return sign * (((half & 0x3FF) == 0) ? 0f : (float)(half & 0x3FF) * 5.960464477539063e-8f);
        return sign * (1.0f + (float)(half & 0x3FF) / 1024.0f) * MathF.Pow(2f, exp5 - 15);
    }

    [Fact]
    public void Q8_0Matmul_AgreesWithVecDotScalar()
    {
        int M = 4, K = 128, N = 96;   // K and N multiples of 32 (TensorQuantizer block layout)
        var rnd = new Random(7);

        // Weight matrix [N, K] (row = output, col = input contract), matching VecDot's col arg.
        var w = new float[N * K];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.5f;
        byte[] raw = TensorQuantizer.Quantize(w, [N, K], QuantDType.Q8_0);

        var x = new float[M * K];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.5f;

        // CPU oracle: y[i,o] = VecDotQ8_0_Scalar(x[i], raw, o, K).
        var want = new float[M * N];
        unsafe
        {
            fixed (float* px = x)
            fixed (byte* praw = raw)
                for (int i = 0; i < M; i++)
                    for (int o = 0; o < N; o++)
                        want[i * N + o] = QuantizationKernels.VecDotQ8_0_Scalar(px + i * K, praw, o, K);
        }

        using var xb = new DeviceBuffer(Dev, M, K);
        using var yb = new DeviceBuffer(Dev, M, N);
        using var wb = new DeviceByteBuffer(Dev.Accelerator, raw);
        xb.Tensor.Upload(x);
        Dev.Kernels.Q8_0Matmul(yb.Tensor, xb.Tensor, wb, K, N);
        var got = yb.Tensor.ToArray();

        GpuTestDevice.AssertClose(want, got, 1e-5, "Q8_0Matmul vs VecDotQ8_0_Scalar");
    }

    [Fact]
    public void EmbedGatherQ8_0_AgreesWithBlockDecode()
    {
        int V = 64, K = 128;   // [V, K] embedding, K multiple of 32
        var rnd = new Random(11);
        var emb = new float[V * K];
        for (int i = 0; i < emb.Length; i++) emb[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.5f;
        byte[] raw = TensorQuantizer.Quantize(emb, [V, K], QuantDType.Q8_0);

        var ids = new int[] { 3, 40, 9, 1, 63 };
        int M = ids.Length;
        var want = new float[M * K];
        for (int i = 0; i < M; i++)
            for (int d = 0; d < K; d++)
                want[i * K + d] = DecodeCell(raw, ids[i], d, K);

        using var xb = new DeviceBuffer(Dev, M, K);
        using var tb = new DeviceByteBuffer(Dev.Accelerator, raw);
        using var idsDev = Dev.UploadInts(ids);
        Dev.Kernels.EmbedGatherQ8_0(xb.Tensor, tb, idsDev.View, K);
        var got = xb.Tensor.ToArray();

        GpuTestDevice.AssertClose(want, got, 1e-5, "EmbedGatherQ8_0 vs block decode");
    }

    [Fact]
    public void Q8_0Matmul_ShapeMismatch_IsRejected()
    {
        var w = new float[32 * 32];
        using var wb = new DeviceByteBuffer(Dev.Accelerator, TensorQuantizer.Quantize(w, [32, 32], QuantDType.Q8_0));
        using var xb = new DeviceBuffer(Dev, 2, 32);
        using var yb = new DeviceBuffer(Dev, 2, 32);
        // x cols (32) != K (16) -> rejected before any kernel launch.
        Assert.Throws<ArgumentException>(() => Dev.Kernels.Q8_0Matmul(yb.Tensor, xb.Tensor, wb, K: 16, N: 32));
    }
}
