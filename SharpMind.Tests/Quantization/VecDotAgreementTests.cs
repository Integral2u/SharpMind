using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Tests.Quantization;

public class VecDotAgreementTests
{
    private const string ExternalAssets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

    private static unsafe bool TryLoadTensorData(string modelFile, string tensorName,
        out byte[] rawData, out int inFeatures, out int outFeatures, out GgufDtype dtype)
    {
        rawData = null!;
        inFeatures = outFeatures = 0;
        dtype = default;
        string path = Path.Combine(ExternalAssets, modelFile);
        if (!File.Exists(path)) return false;

        var loader = GgufLoaderFactory.Create();
        var meta = loader.LoadMeta(path);
        var info = meta.Tensors.FirstOrDefault(t => t.Name == tensorName);
        if (info.Shape == null || info.Shape.Length < 2) return false;

        inFeatures = info.Shape[0];
        outFeatures = info.Shape[1];
        dtype = info.Dtype;

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);
        fs.Position = meta.DataOffset + info.Offset;
        long size = loader.GetRawTensorByteCount(info.Shape, info.Dtype);
        rawData = new byte[size];
        br.ReadExactly(rawData);
        return true;
    }

    private static unsafe void DispatchQuantizedMatMul(
        QuantizationOps qOps, GgufDtype dtype,
        float* pIn, byte* pRaw, float* pOut, int m, int k, int n)
    {
        switch (dtype)
        {
            case GgufDtype.Q8_0: qOps.QuantizedMatMulQ8_0(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q8_1: qOps.QuantizedMatMulQ8_1(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q4_0: qOps.QuantizedMatMulQ4_0(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q4_1: qOps.QuantizedMatMulQ4_1(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q5_0: qOps.QuantizedMatMulQ5_0(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q5_1: qOps.QuantizedMatMulQ5_1(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.IQ4_NL: qOps.QuantizedMatMulQ4_NL(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q2_K or GgufDtype.Q2_K_S: qOps.QuantizedMatMulQ2K(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L: qOps.QuantizedMatMulQ3K(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M: qOps.QuantizedMatMulQ4K(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M: qOps.QuantizedMatMulQ5K(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q6_K or GgufDtype.Q6_K_S: qOps.QuantizedMatMulQ6K(pIn, pRaw, pOut, m, k, n); break;
            case GgufDtype.Q8_K: qOps.QuantizedMatMulQ8K(pIn, pRaw, pOut, m, k, n); break;
        }
    }

    /// <summary>
    /// Runs QuantizedMatMul with all available SIMD tiers and verifies they
    /// produce identical output for the same input + weight data.
    /// </summary>
    [Theory]
    [InlineData("qwen2-0.5b-instruct-q2_k.gguf", "blk.0.attn_q.weight")]
    [InlineData("qwen2-0_5b-instruct-q8_0.gguf", "blk.0.attn_q.weight")]
    [InlineData("Qwen3-0.6B-Q4_K_M.gguf", "blk.0.attn_q.weight")]
    public void AllTiers_ProduceIdenticalOutput(string modelFile, string tensorName)
    {
        if (!TryLoadTensorData(modelFile, tensorName, out var rawData, out var inF, out var outF, out var dtype))
        {
            Assert.True(true, $"SKIP: {modelFile} or tensor {tensorName} not found");
            return;
        }

        var rng = new Random(42);
        var input = new float[2 * inF];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        var tiers = new List<(string name, HardwareTier hw)>();
        tiers.Add(("Scalar", HardwareTier.Scalar));
        if (Sse.IsSupported) tiers.Add(("SSE", HardwareTier.SSE));
        if (Avx2.IsSupported) tiers.Add(("AVX2", HardwareTier.AVX2));
        if (Fma.IsSupported) tiers.Add(("FMA", HardwareTier.FMA));

        if (tiers.Count < 2)
        {
            Assert.True(true, "Only one tier available, cannot cross-check");
            return;
        }

        float[]? reference = null;
        string refName = "";

        foreach (var (name, hw) in tiers)
        {
            var qOps = QuantizationFactory.Create(hw);
            var result = new float[2 * outF];

            unsafe
            {
                fixed (byte* pRaw = rawData)
                fixed (float* pIn = input)
                fixed (float* pOut = result)
                {
                    DispatchQuantizedMatMul(qOps, dtype, pIn, pRaw, pOut, 2, inF, outF);
                }
            }

            if (reference == null)
            {
                reference = result;
                refName = name;
            }
            else
            {
                double maxDiff = 0;
                for (int i = 0; i < result.Length; i++)
                {
                    double diff = Math.Abs((double)result[i] - reference[i]);
                    if (diff > maxDiff) maxDiff = diff;
                }
                Assert.True(maxDiff < 1e-4,
                    $"{name} tier differs from {refName} maxDiff={maxDiff:F8}");
            }
        }
    }

    /// <summary>
    /// Verifies each QuantizedMatMul implementation agrees with a plain
    /// VecDot-based loop (the old WrapVecDotAsMatMul pattern).
    /// </summary>
    [Theory]
    [InlineData("qwen2-0.5b-instruct-q2_k.gguf", "blk.0.attn_q.weight")]
    [InlineData("qwen2-0_5b-instruct-q8_0.gguf", "blk.0.attn_q.weight")]
    [InlineData("Qwen3-0.6B-Q4_K_M.gguf", "blk.0.attn_q.weight")]
    [InlineData("Qwen3-0.6B-Q5_K_M.gguf", "blk.0.attn_q.weight")]
    [InlineData("Qwen3-0.6B-Q6_K.gguf", "blk.0.attn_q.weight")]
    [InlineData("Qwen3-0.6B-Q8_0.gguf", "blk.0.attn_q.weight")]
    public void QuantizedMatMul_Agrees_With_VecDotLoop(string modelFile, string tensorName)
    {
        if (!TryLoadTensorData(modelFile, tensorName, out var rawData, out var inF, out var outF, out var dtype))
        {
            Assert.True(true, $"SKIP: {modelFile} not found");
            return;
        }

        var rng = new Random(42);
        var input = new float[inF];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        var vecDotResult = new float[outF];
        var matMulResult = new float[outF];

        unsafe
        {
            fixed (byte* pRaw = rawData)
            fixed (float* pIn = input)
            fixed (float* pVec = vecDotResult)
            fixed (float* pMat = matMulResult)
            {
                DispatchQuantizedMatMul(qOps, dtype, pIn, pRaw, pMat, 1, inF, outF);

                for (int col = 0; col < outF; col++)
                {
                    pVec[col] = dtype switch
                    {
                        GgufDtype.Q8_0 => qOps.VecDotQ8_0(pIn, pRaw, col, inF),
                        GgufDtype.Q8_1 => qOps.VecDotQ8_1(pIn, pRaw, col, inF),
                        GgufDtype.Q4_0 => qOps.VecDotQ4_0(pIn, pRaw, col, inF),
                        GgufDtype.Q4_1 => qOps.VecDotQ4_1(pIn, pRaw, col, inF),
                        GgufDtype.Q5_0 => qOps.VecDotQ5_0(pIn, pRaw, col, inF),
                        GgufDtype.Q5_1 => qOps.VecDotQ5_1(pIn, pRaw, col, inF),
                        GgufDtype.IQ4_NL => qOps.VecDotQ4_NL(pIn, pRaw, col, inF),
                        GgufDtype.Q2_K or GgufDtype.Q2_K_S => qOps.VecDotQ2K(pIn, pRaw, col, inF),
                        GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L => qOps.VecDotQ3K(pIn, pRaw, col, inF),
                        GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M => qOps.VecDotQ4K(pIn, pRaw, col, inF),
                        GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M => qOps.VecDotQ5K(pIn, pRaw, col, inF),
                        GgufDtype.Q6_K or GgufDtype.Q6_K_S => qOps.VecDotQ6K(pIn, pRaw, col, inF),
                        GgufDtype.Q8_K => qOps.VecDotQ8K(pIn, pRaw, col, inF),
                        _ => 0f
                    };
                }
            }
        }

        double maxDiff = 0;
        for (int i = 0; i < outF; i++)
        {
            double diff = Math.Abs((double)vecDotResult[i] - matMulResult[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 1e-5,
            $"[{dtype}] QuantizedMatMul differs from VecDot loop. MaxDiff: {maxDiff:F8}. " +
            "This indicates a bug in QuantizedMatMul vs VecDot implementation.");
    }
}
