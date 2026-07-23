using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

public class VecDotTests
{
    public static IEnumerable<object[]> AllQuantizationOps()
    {
        yield return new object[] { QuantizationFactory.Create(HardwareTier.Scalar) };
        if (Sse.IsSupported)
        {
            yield return new object[] { QuantizationFactory.Create(HardwareTier.SSE) };
            if (Avx2.IsSupported)
            {
                yield return new object[] { QuantizationFactory.Create(HardwareTier.AVX2) };
                if (Fma.IsSupported)
                    yield return new object[] { QuantizationFactory.Create(HardwareTier.FMA) };
            }
        }
    }

    [Fact]
    public unsafe void VecDotQ3K_MultiBlockAgrees()
    {
        const int blockBytes = 110, qk = 256, nBlocks = 4;
        var rng = new Random(42);
        var weights = new byte[nBlocks * blockBytes];
        rng.NextBytes(weights);
        var input = new float[nBlocks * qk];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        int inF = nBlocks * qk;
        var all = new List<QuantizationOps> { QuantizationFactory.Create(HardwareTier.Scalar) };
        if (Sse.IsSupported)
        {
            all.Add(QuantizationFactory.Create(HardwareTier.SSE));
            if (Avx2.IsSupported)
            {
                all.Add(QuantizationFactory.Create(HardwareTier.AVX2));
                if (Fma.IsSupported)
                    all.Add(QuantizationFactory.Create(HardwareTier.FMA));
            }
        }
        var results = new List<float>();
        fixed (float* pIn = input) fixed (byte* pW = weights)
            foreach (var q in all) results.Add(q.VecDotQ3K(pIn, pW, 0, inF));
        var b = (double)results[0];
        for (int i = 1; i < results.Count; i++)
        {
            double err = Math.Abs(b - results[i]);
            Assert.True(err < 1.0 || err / Math.Max(1.0, Math.Abs(b)) < 1e-5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ3K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[110];
        block[108] = 0x00;
        block[109] = 0x40;
        for (int j = 0; j < 4; j++) block[96 + j] = 0x00;
        for (int j = 0; j < 4; j++) block[100 + j] = 0x00;
        for (int j = 0; j < 4; j++) block[104 + j] = 0xAA;
        block[32] = 0x00;
        block[0] = 0x00;

        var input = new float[] { 1.0f };
        var weights = new byte[110];
        Array.Copy(block, weights, 110);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ3K(pInput, pWeights, 0, 1);
            Assert.Equal(0.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[144];
        // dSuper = 1.0 at block[0..1]
        block[0] = 0x00; block[1] = 0x3C;
        // scales[0] = 0x11 → scale=17, scales[4] = 0 → min=0
        block[4] = 0x11;
        // qs[0] low nibble = 1 → v=1
        block[16] = 0x01;

        var input = new float[] { 1.0f };
        var weights = new byte[144];
        Array.Copy(block, weights, 144);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ4K(pInput, pWeights, 0, 1);
            Assert.Equal(17.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ5K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[176];
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        block[4] = 0x11;
        block[16] = 0x00;
        block[48] = 0x11;

        var input = new float[] { 1.0f };
        var weights = new byte[176];
        Array.Copy(block, weights, 176);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ5K(pInput, pWeights, 0, 1);
            Assert.Equal(17.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ8_0_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[34];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x03;

        var input = new float[] { 1.0f };
        var weights = new byte[34];
        Array.Copy(block, weights, 34);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ8_0(pInput, pWeights, 0, 1);
            Assert.Equal(6.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ6K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[210];
        block[208] = 0x00;
        block[209] = 0x3C;

        var input = new float[] { 1.0f };
        var weights = new byte[210];
        Array.Copy(block, weights, 210);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ6K(pInput, pWeights, 0, 1);
            Assert.Equal(0.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[18];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x34;
        var input = new float[] { 1.0f };
        var weights = new byte[18];
        Array.Copy(block, weights, 18);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(-8.0f, qOps.VecDotQ4_0(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_1_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[20];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x00; block[3] = 0x00;
        block[4] = 0x34;
        var input = new float[] { 1.0f };
        var weights = new byte[20];
        Array.Copy(block, weights, 20);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(8.0f, qOps.VecDotQ4_1(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ5_0_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[22];
        block[0] = 0x00; block[1] = 0x40;
        block[6] = 0x01;
        var input = new float[] { 1.0f };
        var weights = new byte[22];
        Array.Copy(block, weights, 22);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(-30.0f, qOps.VecDotQ5_0(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ5_1_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[24];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x00; block[3] = 0x00;
        block[8] = 0x01;
        var input = new float[] { 1.0f };
        var weights = new byte[24];
        Array.Copy(block, weights, 24);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(2.0f, qOps.VecDotQ5_1(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ8_1_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[36];
        block[0] = 0x00; block[1] = 0x40;
        block[4] = 0x05;
        var input = new float[] { 1.0f };
        var weights = new byte[36];
        Array.Copy(block, weights, 36);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(10.0f, qOps.VecDotQ8_1(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ2K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[84];
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        var input = new float[] { 1.0f };
        var weights = new byte[84];
        Array.Copy(block, weights, 84);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(0.0f, qOps.VecDotQ2K(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ8K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[292];
        BitConverter.GetBytes(1.0f).CopyTo(block, 0);
        block[4] = 0x05;
        var input = new float[] { 1.0f };
        var weights = new byte[292];
        Array.Copy(block, weights, 292);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(5.0f, qOps.VecDotQ8K(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_FullBlock32_Elements(QuantizationOps qOps)
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        const int N_COLS = 4;
        const int IN_FEATURES = QK;
        var rng = new Random(42);
        var rawWeights = new byte[N_COLS * BLOCK_BYTES];
        var input = new float[IN_FEATURES];

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int c = 0; c < N_COLS; c++)
        {
            ushort scaleHalf = FloatToHalf(0.5f + rng.NextSingle());
            rawWeights[c * BLOCK_BYTES + 0] = (byte)(scaleHalf & 0xFF);
            rawWeights[c * BLOCK_BYTES + 1] = (byte)(scaleHalf >> 8);
            for (int b = 2; b < BLOCK_BYTES; b++)
                rawWeights[c * BLOCK_BYTES + b] = (byte)rng.Next(256);
        }

        for (int c = 0; c < N_COLS; c++)
        {
            float expected = 0;
            float d = HalfToFloatTest((ushort)(rawWeights[c * BLOCK_BYTES] | ((ushort)rawWeights[c * BLOCK_BYTES + 1] << 8)));
            fixed (byte* qs = &rawWeights[c * BLOCK_BYTES + 2])
            {
                for (int i = 0; i < IN_FEATURES; i++)
                {
                    int nib = (i < 16) ? (qs[i] & 0x0F) : (qs[i - 16] >> 4);
                    expected += input[i] * ((nib - 8) * d);
                }
            }

            fixed (float* pIn = input) fixed (byte* pW = rawWeights)
            {
                float result = qOps.VecDotQ4_0(pIn, pW, c, IN_FEATURES);
                Assert.Equal(expected, result, 4);
            }
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_MultiBlock64_Elements(QuantizationOps qOps)
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        const int N_COLS = 2;
        const int N_BLOCKS = 2;
        const int IN_FEATURES = N_BLOCKS * QK;
        var rng = new Random(99);
        var rawWeights = new byte[N_COLS * N_BLOCKS * BLOCK_BYTES];
        var input = new float[IN_FEATURES];

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int c = 0; c < N_COLS; c++)
        {
            for (int b = 0; b < N_BLOCKS; b++)
            {
                int off = c * N_BLOCKS * BLOCK_BYTES + b * BLOCK_BYTES;
                ushort scaleHalf = FloatToHalf(0.3f + rng.NextSingle());
                rawWeights[off + 0] = (byte)(scaleHalf & 0xFF);
                rawWeights[off + 1] = (byte)(scaleHalf >> 8);
                for (int j = 2; j < BLOCK_BYTES; j++)
                    rawWeights[off + j] = (byte)rng.Next(256);
            }
        }

        for (int c = 0; c < N_COLS; c++)
        {
            float expected = 0;
            float d = HalfToFloatTest((ushort)(rawWeights[c * BLOCK_BYTES] | ((ushort)rawWeights[c * BLOCK_BYTES + 1] << 8)));
            for (int b2 = 0; b2 < N_BLOCKS; b2++)
            {
                int off2 = c * N_BLOCKS * BLOCK_BYTES + b2 * BLOCK_BYTES;
                d = HalfToFloatTest((ushort)(rawWeights[off2] | ((ushort)rawWeights[off2 + 1] << 8)));
                fixed (byte* qs = &rawWeights[off2 + 2])
                {
                    for (int i = 0; i < QK; i++)
                    {
                        int nib = (i < 16) ? (qs[i] & 0x0F) : (qs[i - 16] >> 4);
                        expected += input[b2 * QK + i] * ((nib - 8) * d);
                    }
                }
            }

            fixed (float* pIn = input) fixed (byte* pW = rawWeights)
            {
                float result = qOps.VecDotQ4_0(pIn, pW, c, IN_FEATURES);
                Assert.Equal(expected, result, 3);
            }
        }
    }

    [Fact]
    public unsafe void TestVecDotQ4_0_AgreesAcrossTiers()
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        const int N_COLS = 4;
        const int N_BLOCKS = 2;
        const int IN_FEATURES = N_BLOCKS * QK;
        var rng = new Random(7);
        var rawWeights = new byte[N_COLS * N_BLOCKS * BLOCK_BYTES];
        var input = new float[IN_FEATURES];

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int c = 0; c < N_COLS; c++)
            for (int b = 0; b < N_BLOCKS; b++)
            {
                int off = c * N_BLOCKS * BLOCK_BYTES + b * BLOCK_BYTES;
                rawWeights[off + 0] = 0x00; rawWeights[off + 1] = 0x3C;
                for (int j = 2; j < BLOCK_BYTES; j++)
                    rawWeights[off + j] = (byte)rng.Next(256);
            }

        float expected = RunCReference((int)QuantDType.Q4_0, input, rawWeights, 0, IN_FEATURES);

        foreach (var tier in Enum.GetValues<HardwareTier>())
        {
            var q = QuantizationFactory.Create(tier);
            fixed (float* pIn = input) fixed (byte* pW = rawWeights)
                Assert.Equal(expected, q.VecDotQ4_0(pIn, pW, 0, IN_FEATURES), 4);
        }
    }

    // ===== C reference cross-validation =====

    private static string? _refProjectDir;

    private static string GetRefProjectDir()
    {
        if (_refProjectDir == null)
        {
            // Walk up from the test assembly output to find SharpMind.Tests/Reference/Reference.csproj
            var dir = AppDomain.CurrentDomain.BaseDirectory;
            while (dir != null)
            {
                var testMarker = Path.Combine(dir, "SharpMind.Tests.csproj");
                if (File.Exists(testMarker))
                {
                    var candidate = Path.Combine(dir, "Reference", "Reference.csproj");
                    if (File.Exists(candidate))
                    {
                        _refProjectDir = Path.Combine(dir, "Reference");
                        break;
                    }
                }
                var parent = Directory.GetParent(dir);
                dir = parent?.FullName;
            }
            if (_refProjectDir == null)
                throw new FileNotFoundException(
                    "Reference project not found. SharpMind.Tests/Reference/Reference.csproj must exist.");
        }
        return _refProjectDir;
    }

    private static float RunCReference(int dtype, float[] input, byte[] weights, int col, int inFeatures)
    {
        var refDir = GetRefProjectDir();
        var refDll = Path.Combine(refDir, "bin", "Debug", "net10.0", "Reference.dll");

        // Write input data to temp file and pipe it to the reference process
        var tempInput = Path.Combine(Path.GetTempPath(), $"ref_{dtype}_c{col}_{Guid.NewGuid():N}.bin");
        using (var fs = File.Create(tempInput))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write(dtype);
            bw.Write(inFeatures);
            bw.Write(col);
            foreach (var f in input) bw.Write(f);
            bw.Write(weights, 0, weights.Length);
        }

        var psi = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        psi.ArgumentList.Add("exec");
        psi.ArgumentList.Add(refDll);
        psi.ArgumentList.Add(tempInput);

        using var proc = Process.Start(psi)!;
        proc.WaitForExit(15000);
        var result = proc.StandardOutput.ReadToEnd().Trim();
        var err = proc.StandardError.ReadToEnd().Trim();
        File.Delete(tempInput);

        if (proc.ExitCode != 0 || string.IsNullOrEmpty(result))
            throw new InvalidOperationException(
                $"Reference process failed (exit={proc.ExitCode}, output=\"{result}\", stderr=\"{err}\")");

        return float.Parse(result, System.Globalization.CultureInfo.InvariantCulture);
    }

    public static IEnumerable<object[]> AllRefQuantTypes()
    {
        // QuantDType enum values: F32=0, F16=1, Q4_0=2, Q4_1=3,
        // Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9, Q2_K=10, Q3_K=11,
        // Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, I8=16, I16=17, I32=18,
        // IQ1_S=19, IQ4_NL=20, IQ1_M=21, TQ1_0=22, TQ2_0=23
        int[] dtypes = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22];
        string[] names = ["VecDotF32", "VecDotF16", "VecDotQ4_0", "VecDotQ4_1",
                          "VecDotQ5_0", "VecDotQ5_1", "VecDotQ8_0", "VecDotQ8_1",
                          "VecDotQ2K", "VecDotQ3K", "VecDotQ4K", "VecDotQ5K",
                          "VecDotQ6K", "VecDotQ8K", "VecDotI8", "VecDotI16", "VecDotI32",
                          "VecDotIQ1_S", "VecDotIQ1_M", "VecDotIQ4_NL", "VecDotTQ2_0", "VecDotTQ1_0"];
        for (int i = 0; i < dtypes.Length; i++)
            yield return new object[] { dtypes[i], names[i] };
    }

    private static int RefQkForType(QuantDType dtype) => dtype switch
    {
        QuantDType.F32 => 1,
        QuantDType.F16 => 1,
        QuantDType.I8 => 1,
        QuantDType.I16 => 1,
        QuantDType.I32 => 1,
        QuantDType.IQ4_NL => 32,
        QuantDType.IQ1_S or QuantDType.IQ1_M or QuantDType.TQ1_0 or QuantDType.TQ2_0 => 256,
        _ => dtype >= QuantDType.Q2_K ? 256 : 32
    };

    private static int BlockBytesForType(QuantDType dtype) => dtype switch
    {
        QuantDType.F32 => 4,
        QuantDType.F16 => 2,
        QuantDType.Q4_0 => 18,
        QuantDType.Q4_1 => 20,
        QuantDType.Q5_0 => 22,
        QuantDType.Q5_1 => 24,
        QuantDType.Q8_0 => 34,
        QuantDType.Q8_1 => 36,
        QuantDType.Q2_K => 84,
        QuantDType.Q3_K => 110,
        QuantDType.Q4_K => 144,
        QuantDType.Q5_K => 176,
        QuantDType.Q6_K => 210,
        QuantDType.Q8_K => 292,
        QuantDType.I8 => 1,
        QuantDType.I16 => 2,
        QuantDType.I32 => 4,
        QuantDType.IQ4_NL => 18,
        QuantDType.IQ1_S => 50,
        QuantDType.IQ1_M => 56,
        QuantDType.TQ2_0 => 66,
        QuantDType.TQ1_0 => 54,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null)
    };

    [Theory]
    [MemberData(nameof(AllRefQuantTypes))]
    public unsafe void VecDot_AgreesWithCReference(int dtypeInt, string name)
    {
        var dtype = (QuantDType)dtypeInt;
        int blockBytes = BlockBytesForType(dtype);
        int qk = RefQkForType(dtype);
        int nBlocks = 4;
        int nCols = 2;
        int inFeatures = nBlocks * qk;
        var rng = new Random(42);
        var input = new float[inFeatures];
        for (int i = 0; i < inFeatures; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int totalBlockBytes = nBlocks * blockBytes;
        var rawWeights = new byte[nCols * totalBlockBytes];
        rng.NextBytes(rawWeights);

        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        for (int c = 0; c < nCols; c++)
        {
            fixed (float* pIn = input)
            fixed (byte* pW = rawWeights)
            {
                float smResult = dtype switch
                {
                    QuantDType.F32 => qOps.VecDotF32(pIn, pW, c, inFeatures),
                    QuantDType.F16 => qOps.VecDotF16(pIn, pW, c, inFeatures),
                    QuantDType.Q4_0 => qOps.VecDotQ4_0(pIn, pW, c, inFeatures),
                    QuantDType.Q4_1 => qOps.VecDotQ4_1(pIn, pW, c, inFeatures),
                    QuantDType.Q5_0 => qOps.VecDotQ5_0(pIn, pW, c, inFeatures),
                    QuantDType.Q5_1 => qOps.VecDotQ5_1(pIn, pW, c, inFeatures),
                    QuantDType.Q8_0 => qOps.VecDotQ8_0(pIn, pW, c, inFeatures),
                    QuantDType.Q8_1 => qOps.VecDotQ8_1(pIn, pW, c, inFeatures),
                    QuantDType.Q2_K => qOps.VecDotQ2K(pIn, pW, c, inFeatures),
                    QuantDType.Q3_K => qOps.VecDotQ3K(pIn, pW, c, inFeatures),
                    QuantDType.Q4_K => qOps.VecDotQ4K(pIn, pW, c, inFeatures),
                    QuantDType.Q5_K => qOps.VecDotQ5K(pIn, pW, c, inFeatures),
                    QuantDType.Q6_K => qOps.VecDotQ6K(pIn, pW, c, inFeatures),
                    QuantDType.Q8_K => qOps.VecDotQ8K(pIn, pW, c, inFeatures),
                    QuantDType.I8 => qOps.VecDotI8(pIn, pW, c, inFeatures),
                    QuantDType.I16 => qOps.VecDotI16(pIn, pW, c, inFeatures),
                    QuantDType.I32 => qOps.VecDotI32(pIn, pW, c, inFeatures),
                    QuantDType.IQ4_NL => qOps.VecDotQ4_NL(pIn, pW, c, inFeatures),
                    QuantDType.IQ1_S => qOps.VecDotIQ1_S(pIn, pW, c, inFeatures),
                    QuantDType.IQ1_M => qOps.VecDotIQ1_M(pIn, pW, c, inFeatures),
                    QuantDType.TQ2_0 => qOps.VecDotTQ2_0(pIn, pW, c, inFeatures),
                    QuantDType.TQ1_0 => qOps.VecDotTQ1_0(pIn, pW, c, inFeatures),
                    _ => throw new InvalidOperationException()
                };

                float refResult = RunCReference(dtypeInt, input, rawWeights, c, inFeatures);

                Assert.Equal(smResult, refResult, 4);
            }
        }
    }

    private static ushort FloatToHalf(float f)
    {
        unsafe
        {
            uint bits = *(uint*)&f;
            uint sign = (bits >> 16) & 0x8000;
            int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
            uint mantissa = bits & 0x7FFFFF;
            if (exp <= 0) return (ushort)sign;
            if (exp >= 31) return (ushort)(sign | 0x7C00);
            return (ushort)(sign | ((uint)exp << 10) | (mantissa >> 13));
        }
    }

    private static float HalfToFloatTest(ushort h)
    {
        unsafe
        {
            uint sign = (uint)(h & 0x8000) << 16;
            int exp = (h >> 10) & 0x1F;
            uint mantissa = (uint)(h & 0x3FF) << 13;
            if (exp == 0) { float z = 0; return *(float*)&sign; }
            if (exp == 31) { uint inf = sign | 0x7F800000u; return *(float*)&inf; }
            uint bits = sign | ((uint)(exp - 15 + 127) << 23) | mantissa;
            return *(float*)&bits;
        }
    }
}
