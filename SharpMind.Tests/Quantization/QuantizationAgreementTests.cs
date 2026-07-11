using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Quantization;

namespace SharpMind.Tests.Quantization;

public class QuantizationAgreementTests
{
    public static IEnumerable<object[]> UniqueDtypes()
    {
        yield return new object[] { QuantDType.F32 };
        yield return new object[] { QuantDType.F16 };
        yield return new object[] { QuantDType.Q8_0 };
        yield return new object[] { QuantDType.Q8_1 };
        yield return new object[] { QuantDType.Q4_0 };
        yield return new object[] { QuantDType.Q4_1 };
        yield return new object[] { QuantDType.Q5_0 };
        yield return new object[] { QuantDType.Q5_1 };
        yield return new object[] { QuantDType.IQ4_NL };
        yield return new object[] { QuantDType.Q2_K };
        yield return new object[] { QuantDType.Q3_K };
        yield return new object[] { QuantDType.Q4_K };
        yield return new object[] { QuantDType.Q5_K };
        yield return new object[] { QuantDType.Q6_K };
        yield return new object[] { QuantDType.Q8_K };
    }

    private static int GetBlockSize(QuantDType dtype) => dtype switch
    {
        QuantDType.F32 or QuantDType.F16 => 1,
        QuantDType.Q8_0 or QuantDType.Q8_1 or QuantDType.Q4_0 or QuantDType.Q4_1
            or QuantDType.Q5_0 or QuantDType.Q5_1 or QuantDType.IQ4_NL => 32,
        _ => 256
    };

    private static List<(string name, HardwareTier tier)> GetAvailableTiers()
    {
        var tiers = new List<(string, HardwareTier)> { ("Scalar", HardwareTier.Scalar) };
        if (Sse.IsSupported) tiers.Add(("SSE", HardwareTier.SSE));
        if (Avx2.IsSupported) tiers.Add(("AVX2", HardwareTier.AVX2));
        if (Fma.IsSupported) tiers.Add(("FMA", HardwareTier.FMA));
        return tiers;
    }

    private static int GetBlockBytes(QuantDType dtype)
    {
        int blockSize = GetBlockSize(dtype);
        return (int)QuantizationOps.GetRawTensorByteCount([blockSize, 1], dtype);
    }

    private static byte[] GenerateRawData(QuantDType dtype, int K, int N, int seed)
    {
        int totalBytes = (int)QuantizationOps.GetRawTensorByteCount([K, N], dtype);
        var data = new byte[totalBytes];
        new Random(seed).NextBytes(data);
        SanitizeScaleValues(dtype, data);
        return data;
    }

    private static void SanitizeScaleValues(QuantDType dtype, byte[] data)
    {
        int blockBytes = GetBlockBytes(dtype);
        if (blockBytes <= 0) return;
        int totalBlocks = data.Length / blockBytes;

        for (int b = 0; b < totalBlocks; b++)
        {
            int off = b * blockBytes;
            switch (dtype)
            {
                case QuantDType.Q8_0:
                case QuantDType.Q4_0:
                case QuantDType.Q5_0:
                case QuantDType.IQ4_NL:
                    data[off + 0] = 0x00; data[off + 1] = 0x3C;
                    break;
                case QuantDType.Q8_1:
                case QuantDType.Q4_1:
                case QuantDType.Q5_1:
                    data[off + 0] = 0x00; data[off + 1] = 0x3C;
                    data[off + 2] = 0x00; data[off + 3] = 0x00;
                    break;
                case QuantDType.Q2_K:
                    data[off + 80] = 0x00; data[off + 81] = 0x3C;
                    data[off + 82] = 0x00; data[off + 83] = 0x00;
                    break;
                case QuantDType.Q3_K:
                    data[off + 108] = 0x00; data[off + 109] = 0x3C;
                    break;
                case QuantDType.Q4_K:
                case QuantDType.Q5_K:
                    data[off + 0] = 0x00; data[off + 1] = 0x3C;
                    data[off + 2] = 0x00; data[off + 3] = 0x00;
                    break;
                case QuantDType.Q6_K:
                    data[off + 208] = 0x00; data[off + 209] = 0x3C;
                    break;
                case QuantDType.Q8_K:
                    BitConverter.GetBytes(1.0f).CopyTo(data, off);
                    break;
            }
        }
    }

    private static QuantizationOps CreateSerialOps(HardwareTier hw)
    {
        var mapping = new MappingBuilder(hw).ApplyQuantPreset(parallel: false).Build();
        return QuantizationFactory.Create(mapping);
    }

    private static QuantizationOps CreateParallelOps(HardwareTier hw)
    {
        var mapping = new MappingBuilder(hw).ApplyQuantPreset(parallel: true).Build();
        return QuantizationFactory.Create(mapping);
    }

    [Theory]
    [MemberData(nameof(UniqueDtypes))]
    public unsafe void VecDot_AllTiers_Agree(QuantDType dtype)
    {
        var tiers = GetAvailableTiers();
        if (tiers.Count < 2) return;

        int blockSize = GetBlockSize(dtype);
        int K = blockSize * 8;
        int N = blockSize * 4;
        if (dtype is QuantDType.F32 or QuantDType.F16)
        {
            K = 256;
            N = 128;
        }

        var rawData = GenerateRawData(dtype, K, N, 42);
        var input = new float[K];
        var rng = new Random(42);
        for (int i = 0; i < K; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var reference = new float[N];
        string refName = tiers[0].name;
        fixed (float* pIn = input) fixed (byte* pRaw = rawData) fixed (float* pRef = reference)
        {
            var refOps = CreateSerialOps(tiers[0].tier);
            for (int col = 0; col < N; col++)
                pRef[col] = refOps.VecDotFor(dtype, pIn, pRaw, col, K);
        }

        for (int t = 1; t < tiers.Count; t++)
        {
            var results = new float[N];
            fixed (float* pIn = input) fixed (byte* pRaw = rawData) fixed (float* pOut = results)
            {
                var qOps = CreateSerialOps(tiers[t].tier);
                for (int col = 0; col < N; col++)
                    pOut[col] = qOps.VecDotFor(dtype, pIn, pRaw, col, K);
            }

            double maxErr = 0;
            double maxRel = 0;
            for (int i = 0; i < N; i++)
            {
                double err = Math.Abs(results[i] - reference[i]);
                double rel = err / Math.Max(1.0, Math.Abs(reference[i]));
                if (err > maxErr) maxErr = err;
                if (rel > maxRel) maxRel = rel;
            }

            Assert.True(maxErr < 1.0 || maxRel < 1e-3,
                $"VecDot [{dtype}] {tiers[t].name} differs from {refName}. " +
                $"MaxErr={maxErr:G4} MaxRel={maxRel:G6}");
        }
    }

    [Theory]
    [MemberData(nameof(UniqueDtypes))]
    public unsafe void QuantizedMatMul_AllTiers_Agree(QuantDType dtype)
    {
        var tiers = GetAvailableTiers();
        if (tiers.Count < 2) return;

        int blockSize = GetBlockSize(dtype);
        int K = blockSize * 8;
        int N = blockSize * 4;
        int M = 3;
        if (dtype is QuantDType.F32 or QuantDType.F16)
        {
            K = 256;
            N = 128;
        }

        var rawData = GenerateRawData(dtype, K, N, 42);
        var input = new float[M * K];
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        string refName = $"Serial {tiers[0].name}";
        var reference = new float[M * N];
        fixed (float* pIn = input) fixed (byte* pRaw = rawData) fixed (float* pRef = reference)
        {
            var refOps = CreateSerialOps(tiers[0].tier);
            refOps.QuantizedMatMulFor(dtype, pIn, pRaw, pRef, M, K, N);
        }

        for (int serialOrParallel = 0; serialOrParallel < 2; serialOrParallel++)
        {
            bool isParallel = serialOrParallel == 1;
            string mode = isParallel ? "Parallel" : "Serial";

            for (int t = 0; t < tiers.Count; t++)
            {
                var results = new float[M * N];
                fixed (float* pIn = input) fixed (byte* pRaw = rawData) fixed (float* pOut = results)
                {
                    var qOps = isParallel
                        ? CreateParallelOps(tiers[t].tier)
                        : CreateSerialOps(tiers[t].tier);
                    qOps.QuantizedMatMulFor(dtype, pIn, pRaw, pOut, M, K, N);
                }

                double maxErr = 0;
                double maxRel = 0;
                for (int i = 0; i < M * N; i++)
                {
                    double err = Math.Abs(results[i] - reference[i]);
                    double rel = err / Math.Max(1.0, Math.Abs(reference[i]));
                    if (err > maxErr) maxErr = err;
                    if (rel > maxRel) maxRel = rel;
                }

                Assert.True(maxErr < 1.0 || maxRel < 1e-3,
                    $"MatMul [{dtype}] {mode} {tiers[t].name} differs from {refName}. " +
                    $"MaxErr={maxErr:G4} MaxRel={maxRel:G6}");
            }
        }
    }
}
