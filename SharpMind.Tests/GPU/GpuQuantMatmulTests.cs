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

    private static int BlockBytesOf(QuantDType q) => q switch
    {
        QuantDType.Q8_0 => 34, QuantDType.Q4_0 => 18, QuantDType.Q4_1 => 20, QuantDType.Q5_0 => 22, QuantDType.Q5_1 => 24,
        _ => throw new InvalidOperationException(),
    };

    /// <summary>Encodes one 32-float block to the target quant's raw bytes. The oracle
    /// <c>VecDot*_Scalar</c> and the device kernel read the exact same bytes, so agreement validates
    /// the kernel against the CPU source of truth on identical input — the quantization itself only
    /// needs to be internal-consistent (the CPU half converter is reused so f16 matches exactly).</summary>
    private static void EncodeBlock(byte[] raw, int blockStart, ReadOnlySpan<float> w, QuantDType q)
    {
        static void PutHalf(byte[] r, int o, float v) { var h = QuantizationKernels.FloatToHalf_F16C(v); r[o] = (byte)h; r[o + 1] = (byte)(h >> 8); }

        if (q == QuantDType.Q8_0)
        {
            float amax8 = 0; for (int i = 0; i < 32; i++) amax8 = Math.Max(amax8, Math.Abs(w[i]));
            float d8 = amax8 > 0 ? amax8 / 127f : 1f;
            PutHalf(raw, blockStart, d8);
            for (int i = 0; i < 32; i++)
            {
                int qi = (int)Math.Round(w[i] / d8, MidpointRounding.AwayFromZero);
                qi = Math.Clamp(qi, sbyte.MinValue, sbyte.MaxValue);
                raw[blockStart + 2 + i] = unchecked((byte)(sbyte)qi);
            }
            return;
        }

        bool isQ5 = q is QuantDType.Q5_0 or QuantDType.Q5_1;
        bool hasMin = q is QuantDType.Q4_1 or QuantDType.Q5_1;
        float wmax = float.MinValue, wmin = float.MaxValue, amax = 0;
        for (int i = 0; i < 32; i++) { wmax = Math.Max(wmax, w[i]); wmin = Math.Min(wmin, w[i]); amax = Math.Max(amax, Math.Abs(w[i])); }
        int maxQ = isQ5 ? 31 : 15;
        int center = q is QuantDType.Q4_0 ? 8 : isQ5 ? 16 : 0;
        float d = hasMin ? (wmax - wmin) / maxQ : amax / (isQ5 ? 16f : 7f);
        if (d <= 0) d = 1f;
        float m = wmin;

        int off = 0;
        PutHalf(raw, blockStart + off, d); off += 2;
        if (hasMin) { PutHalf(raw, blockStart + off, m); off += 2; }
        int nibOff = q is QuantDType.Q5_0 ? 6 : q is QuantDType.Q5_1 ? 8 : off;

        var qs = new int[32];
        for (int i = 0; i < 32; i++)
        {
            int qi = hasMin ? (int)Math.Round((w[i] - m) / d, MidpointRounding.AwayFromZero)
                            : (int)Math.Round(w[i] / d, MidpointRounding.AwayFromZero) + center;
            qs[i] = Math.Clamp(qi, 0, maxQ);
        }

        if (isQ5)
        {
            int qhOff = hasMin ? 4 : 2;
            uint qh = 0;
            for (int i = 0; i < 32; i++) if ((qs[i] & 16) != 0) qh |= 1u << i;
            raw[blockStart + qhOff] = (byte)qh; raw[blockStart + qhOff + 1] = (byte)(qh >> 8);
            raw[blockStart + qhOff + 2] = (byte)(qh >> 16); raw[blockStart + qhOff + 3] = (byte)(qh >> 24);
        }
        for (int i = 0; i < 32; i++)
        {
            int nib = qs[i] & 0x0F;
            int slot = i < 16 ? i : i - 16;
            int b = blockStart + nibOff + slot;
            if (i < 16) raw[b] = (byte)((raw[b] & 0xF0) | nib);
            else raw[b] = (byte)((raw[b] & 0x0F) | (nib << 4));
        }
    }

    private static byte[] QuantizeMatrix(float[] w, int n, int k, QuantDType q)
    {
        int blocksPerRow = k / QK;
        var raw = new byte[n * blocksPerRow * BlockBytesOf(q)];
        for (int row = 0; row < n; row++)
            for (int b = 0; b < blocksPerRow; b++)
                EncodeBlock(raw, (row * blocksPerRow + b) * BlockBytesOf(q), w.AsSpan(row * k + b * QK, QK), q);
        return raw;
    }

    /// <summary>The on-device dequant matmul must reproduce the matching <see cref="VecDot*_Scalar"/>
    /// on identical raw bytes for every block quant the engine claims to support (not just Q8_0).</summary>
    [Theory]
    [InlineData(QuantDType.Q4_0)]
    [InlineData(QuantDType.Q4_1)]
    [InlineData(QuantDType.Q5_0)]
    [InlineData(QuantDType.Q5_1)]
    [InlineData(QuantDType.Q8_0)]
    public void DequantMatmul_AgreesWithVecDotScalar(QuantDType q)
    {
        int M = 4, K = 128, N = 96;
        var rnd = new Random(7 + (int)q);
        var w = new float[N * K];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rnd.NextDouble() * 2 - 1) * 2f;
        byte[] raw = QuantizeMatrix(w, N, K, q);

        var x = new float[M * K];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.5f;

        var want = new float[M * N];
        unsafe
        {
            fixed (float* px = x)
            fixed (byte* praw = raw)
                for (int i = 0; i < M; i++)
                    for (int o = 0; o < N; o++)
                        want[i * N + o] = VecDotFor(px + i * K, praw, o, K, q);
        }

        using var xb = new DeviceBuffer(Dev, M, K);
        using var yb = new DeviceBuffer(Dev, M, N);
        using var wb = new DeviceByteBuffer(Dev.Accelerator, raw);
        xb.Tensor.Upload(x);
        Dev.Kernels.DequantMatmul(yb.Tensor, xb.Tensor, wb, K, N, q);
        var got = yb.Tensor.ToArray();

        GpuTestDevice.AssertClose(want, got, 2e-5, $"{q} DequantMatmul vs VecDot*_Scalar");
    }

    /// <summary>The on-device DequantGather reproduces the CPU per-cell block decode of the embedding
    /// table rows (the same VecDot*_Scalar layout the weight-tied LM head matmul consumes).
    /// Kernel-row selection is validated independently by the matmul theory; this checks the gather's
    /// row addressing and per-block decode agree with the CPU reference on the identical raw bytes.</summary>
    [Theory]
    [InlineData(QuantDType.Q4_0)]
    [InlineData(QuantDType.Q4_1)]
    [InlineData(QuantDType.Q5_0)]
    [InlineData(QuantDType.Q5_1)]
    [InlineData(QuantDType.Q8_0)]
    public void DequantGather_MatchesCpuBlockDecode(QuantDType q)
    {
        int V = 64, K = 128;
        var rnd = new Random(11 + (int)q);
        var emb = new float[V * K];
        for (int i = 0; i < emb.Length; i++) emb[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.5f;
        byte[] raw = QuantizeMatrix(emb, V, K, q);

        var ids = new int[] { 3, 40, 9, 1, 63 };
        int M = ids.Length;

        var want = new float[M * K];
        for (int i = 0; i < M; i++)
            for (int d = 0; d < K; d++)
                want[i * K + d] = CpuDecodeCell(q, raw, ids[i], d, K);

        using var gxb = new DeviceBuffer(Dev, M, K);
        using var tb = new DeviceByteBuffer(Dev.Accelerator, raw);
        using var idsDev = Dev.UploadInts(ids);
        Dev.Kernels.DequantGather(gxb.Tensor, tb, idsDev.View, K, q);
        var got = gxb.Tensor.ToArray();

        GpuTestDevice.AssertClose(want, got, 2e-5, $"{q} DequantGather vs CPU block decode");
    }

    /// <summary>CPU per-cell decode of the raw [N, K] block layout for any supported quant, mirroring
    /// the VecDot*_Scalar byte access so it can serve as the independent gather oracle.</summary>
    private static float CpuDecodeCell(QuantDType q, byte[] raw, int row, int col, int cols)
    {
        int bb = BlockBytesOf(q);
        int nb = (cols + QK - 1) / QK;
        int bs = row * nb * bb + (col / QK) * bb;
        float d = HalfToFloat((ushort)(raw[bs] | (raw[bs + 1] << 8)));
        int qt = col % QK;
        if (q == QuantDType.Q8_0)
        {
            int v = raw[bs + 2 + qt];
            return (v < 128 ? v : v - 256) * d;
        }
        int nibOff = q is QuantDType.Q4_0 ? 2 : q is QuantDType.Q4_1 ? 4 : q is QuantDType.Q5_0 ? 6 : 8;
        int nib = qt < 16 ? raw[bs + nibOff + qt] & 0x0F : raw[bs + nibOff + qt - 16] >> 4;
        if (q is QuantDType.Q5_0 or QuantDType.Q5_1)
        {
            int qhOff = q == QuantDType.Q5_0 ? 2 : 4;
            int qh = raw[bs + qhOff] | (raw[bs + qhOff + 1] << 8) | (raw[bs + qhOff + 2] << 16) | (raw[bs + qhOff + 3] << 24);
            if ((qh >> qt & 1) != 0) nib |= 16;
        }
        int off = q is QuantDType.Q4_0 ? 8 : q is QuantDType.Q5_0 ? 16 : 0;
        float mthalf = q is QuantDType.Q4_1 or QuantDType.Q5_1 ? HalfToFloat((ushort)(raw[bs + 2] | (raw[bs + 3] << 8))) : 0f;
        return (nib - off) * d + mthalf;
    }

    private static unsafe float VecDotFor(float* input, byte* raw, int col, int features, QuantDType q)
        => q switch
        {
            QuantDType.Q4_0 => QuantizationKernels.VecDotQ4_0_Scalar(input, raw, col, features),
            QuantDType.Q4_1 => QuantizationKernels.VecDotQ4_1_Scalar(input, raw, col, features),
            QuantDType.Q5_0 => QuantizationKernels.VecDotQ5_0_Scalar(input, raw, col, features),
            QuantDType.Q5_1 => QuantizationKernels.VecDotQ5_1_Scalar(input, raw, col, features),
            QuantDType.Q8_0 => QuantizationKernels.VecDotQ8_0_Scalar(input, raw, col, features),
            _ => throw new InvalidOperationException(),
        };
}
